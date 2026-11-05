#!/usr/bin/env bash
# A source-module install/upgrade must be usable and preserve the old program on failure.
# No weights, services, SSH or network sockets are needed for this test.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/build"
work=$(mktemp -d "$root/build/backend-modules-test.XXXXXX")
trap 'rm -rf "$work"' EXIT
python3 "$root/tools/release/export-source.py" "$work/source" >/dev/null
cd "$work/source"
python3 tools/backend-module.py new alpha "$work/alpha source" >/dev/null
python3 tools/backend-module.py new beta "$work/beta" >/dev/null
manager=tools/install/fronk-install.sh
prefix=$work/install
install_node() {
    printf '%s\n' module-test | bash "$manager" --prefix "$prefix" --role node --name module-test \
        --yes --secret-stdin --no-service --no-measure "$@"
}
install_node --backend-module "$work/alpha source" --backend-module "$work/beta" --backend beta > "$work/install.log" 2>&1 || { cat "$work/install.log"; exit 1; }
[[ $(cat "$prefix/state/backend") == beta ]]
[[ -f $prefix/modules/alpha/backend.json && -f $prefix/modules/beta/backend.json ]]
"$prefix/src/fronk" backends --list | grep -q 'alpha.*runs here'
"$prefix/src/fronk" backends --list | grep -q 'beta.*runs here'
install_node --backend alpha > "$work/upgrade.log" 2>&1 || { cat "$work/upgrade.log"; exit 1; }
[[ $(cat "$prefix/state/backend") == alpha ]]
"$prefix/bin/fronk-manager" --role node --yes --no-service --no-measure --backend beta > "$work/local-upgrade.log" 2>&1 || { cat "$work/local-upgrade.log"; exit 1; }
[[ $(cat "$prefix/state/backend") == beta ]]
cp "$prefix/src/fronk" "$work/retained"
python3 - "$work/beta/backend.c" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); p.write_text(p.read_text().replace('return loaded()?FR_STATE:fr_native_self_test(&fr_module_beta);', 'return FR_NUMERIC;'))
PY
if install_node --backend-module "$work/beta" > "$work/failure.log" 2>&1; then
    echo 'Bad module was installed'; exit 1
fi
grep -q 'failed SELFTEST' "$work/failure.log"
cmp "$prefix/src/fronk" "$work/retained"
mv "$prefix/modules/beta" "$work/removed-beta"
install_node --backend alpha > "$work/removal.log" 2>&1 || { cat "$work/removal.log"; exit 1; }
[[ ! -e $prefix/src/backends/modules/beta/backend.json ]]
"$prefix/bin/fronk-manager" --role node --yes --no-service --no-measure --backend alpha > "$work/after-removal.log" 2>&1 || { cat "$work/after-removal.log"; exit 1; }
if "$prefix/src/fronk" backends --check beta > /dev/null 2>&1; then echo 'Removed backend returned on upgrade'; exit 1; fi
# A platform-excluded source must not reach the compiler. Removing a module
# must also rebuild the registry without a forced make -B.
python3 tools/backend-module.py copy "$work/alpha source" backends/modules >/dev/null
python3 tools/backend-module.py new excluded >/dev/null
python3 - <<'PY'
import json, platform
from pathlib import Path
p=Path('backends/modules/excluded/backend.json'); d=json.loads(p.read_text())
d['systems']=['Linux' if platform.system()=='Darwin' else 'Darwin']; p.write_text(json.dumps(d))
Path('backends/modules/excluded/backend.c').write_text('#error Must not compile on this platform\n')
PY
make DS4= fronk > "$work/build.log" 2>&1 || { cat "$work/build.log"; exit 1; }
./fronk backends --check alpha
rm -rf backends/modules/alpha
make DS4= fronk >> "$work/build.log" 2>&1
if ./fronk backends --check alpha > /dev/null 2>&1; then echo 'Removed module still registered'; exit 1; fi
bash "$manager" --prefix "$prefix" --uninstall --purge --yes > "$work/uninstall.log" 2>&1
[[ -f $prefix/modules/alpha/backend.c && ! -f $prefix/src/fronk ]]
echo '[ok] backend modules: two choices, upgrade, numerical failure rollback, platform exclusion, removal, source preservation'
