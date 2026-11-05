#!/usr/bin/env bash
# Package only runtime sources. Never collect run directories, secrets or GGUFs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DS4=${DS4:-$ROOT/build/ds4-source}
VERSION=$(sed -n 's/^VERSION := //p' "$ROOT/Makefile")
OUT=$ROOT/build/fronk-install-$VERSION.sh
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
export COPYFILE_DISABLE=1
mkdir -p "$STAGE/fronk/ds4" "$ROOT/build"
(cd "$ROOT" && tar --no-xattrs --no-acls --exclude __pycache__ --exclude .DS_Store -cf - -T tools/install/runtime-files) | tar -xf - -C "$STAGE/fronk"
[[ -f $DS4/Makefile ]] || { echo "Set DS4 to the patched ds4-fronk checkout" >&2; exit 1; }
"$ROOT/tools/mtp/stage-ds4.sh" "$DS4" "$STAGE/fronk/ds4"
# Prebuilt runtimes include the Qwen expert hook, the hardware probe and
# their licenses. Other Linux hosts can build the pinned Vulkan engine locally.
for target in Darwin-x86_64 Linux-x86_64 Linux-aarch64 Linux-arm64; do
    runtime=$ROOT/build/gpu-runtime/$target
    if [[ $target == Darwin-x86_64 && ! -x $runtime/fronk-llama-server ]]; then
        echo "The multiplatform release needs build/gpu-runtime/Darwin-x86_64 from make gpu-head on macOS." >&2; exit 1
    fi
    [[ -x $runtime/fronk-llama-server ]] || continue
    dest=$STAGE/fronk/gpu-runtime/$target
    mkdir -p "$dest"
    cp "$runtime/fronk-llama-server" "$runtime/fronk-gpu-probe" "$runtime/LICENSE-llama.cpp" "$dest/"
    if [[ $target == Darwin-* ]]; then cp "$runtime/libMoltenVK.dylib" "$runtime/LICENSE-MoltenVK" "$dest/"; fi
    if [[ -d $runtime/licenses ]]; then cp -R "$runtime/licenses" "$dest/"; fi
done
python3 "$ROOT/tools/install/package-manifest.py" "$STAGE/fronk"
cat >"$STAGE/manager" <<'HEADER'
#!/usr/bin/env bash
# Fronk Installer Manager. Run with --gui for a window or --help for terminal options.
set -euo pipefail
dir=$(mktemp -d); trap 'rm -rf "$dir"' EXIT
line=$(awk '/^__FRONK_PAYLOAD__$/ {print NR + 1; exit}' "$0")
tail -n +"$line" "$0" | tar -xzf - -C "$dir"
bash "$dir/fronk/tools/install/fronk-install.sh" "$@"
exit $?
__FRONK_PAYLOAD__
HEADER
tar --no-xattrs --no-acls -czf - -C "$STAGE" fronk >> "$STAGE/manager"
chmod +x "$STAGE/manager"
cp "$STAGE/manager" "$OUT"
# Finder opens the same manager directly in graphical mode. Linux desktops
# can extract the archive and run its graphical launcher or use the shell file.
app="$STAGE/Fronk Installer Manager.app/Contents"
mkdir -p "$app/MacOS" "$app/Resources"
cp "$OUT" "$app/Resources/installer.sh"
sed "s/FRONK_VERSION/$VERSION/g" "$ROOT/tools/install/Info.plist" > "$app/Info.plist"
cat > "$app/MacOS/Fronk" <<'MAC'
#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
exec /bin/bash "$(dirname "$0")/../Resources/installer.sh" --gui "$@"
MAC
chmod +x "$app/MacOS/Fronk"
mkdir -p "$STAGE/desktop/Fronk Installer Manager" "$ROOT/build/site/downloads"
cp "$OUT" "$STAGE/desktop/Fronk Installer Manager/installer.sh"
printf '%s\n' '#!/bin/bash' 'exec bash "$(dirname "$0")/installer.sh" --gui "$@"' > "$STAGE/desktop/Fronk Installer Manager/Fronk Setup"
chmod +x "$STAGE/desktop/Fronk Installer Manager/Fronk Setup"
(cd "$STAGE" && zip -qr "$STAGE/fronk-manager-macos.zip" 'Fronk Installer Manager.app')
cp "$STAGE/fronk-manager-macos.zip" "$ROOT/build/site/downloads/"
tar --no-xattrs --no-acls -czf "$ROOT/build/site/downloads/fronk-manager-linux.tar.gz" -C "$STAGE/desktop" 'Fronk Installer Manager'
cp "$OUT" "$ROOT/build/site/downloads/fronk-manager.sh"
# The source download is the same reviewed tree used for a public repository,
# without this machine's configuration or Git history.
python3 "$ROOT/tools/release/export-source.py" "$STAGE/source/fronk-$VERSION"
cp "$STAGE/source/fronk-$VERSION.tar.gz" "$ROOT/build/site/downloads/fronk-source-$VERSION.tar.gz"
# The download site is self-contained. Its guides point into the same reviewed
# source tree as the archive, so a helper brief never becomes a broken link.
# The website's page and media are maintained outside the public repository.
if [[ -f $ROOT/site/index.html ]]; then
    sed -e "s/FRONK_VERSION/$VERSION/g" -e 's|href="../|href="source/|g' "$ROOT/site/index.html" > "$ROOT/build/site/index.html"
    cp -R "$ROOT/site/media" "$ROOT/build/site/"
fi
python3 - "$STAGE/source/fronk-$VERSION" "$ROOT/build/site/source" <<'SITE_SOURCE'
import shutil
import sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
if destination.exists():
    shutil.rmtree(destination)
shutil.copytree(source, destination)
SITE_SOURCE
(cd "$ROOT/build/site/downloads" && python3 - "$VERSION" <<'CHECKSUMS'
from pathlib import Path
import hashlib
import sys
names = ["fronk-manager-macos.zip", "fronk-manager-linux.tar.gz", "fronk-manager.sh"]
names.append(f"fronk-source-{sys.argv[1]}.tar.gz")
Path("SHA256SUMS").write_text("".join(hashlib.sha256(Path(n).read_bytes()).hexdigest()+"  "+n+"\n" for n in names))
CHECKSUMS
)
printf '%s\n' "$OUT" "$ROOT/build/site/downloads/"*
