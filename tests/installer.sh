#!/usr/bin/env bash
# Exercise the downloaded manager, including the head handoff, without a
# launch agent, large weights, or traffic to any physical cluster.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
manager=${FRONK_TEST_MANAGER:-$root/build/site/downloads/fronk-manager.sh}
# CI can exercise the same workflow directly from source. Release qualification
# leaves this unset and must use the self-extracting download.
[[ -x $manager ]] || { echo "Manager not found: $manager" >&2; exit 1; }
bash "$root/tests/installer-choices.sh"
bash "$root/tests/installer-storage.sh"
if [[ ${1:-} == check-peers ]]; then
    prefix=$2
    cp "$FRONK_CLUSTER_DIR/head.conf" "$prefix/src/fronk.conf"
    "$manager" --prefix "$prefix" --join-nodes --demo-later --yes --nodes 'local-1 local-2' > "$prefix/join.log"
    grep -Fq "All 2 planned nodes authenticated" "$prefix/join.log"
    if "$manager" --prefix "$prefix" --join-nodes --demo-later --yes --nodes missing > "$prefix/missing.log" 2>&1; then
        echo "An absent node passed verification" >&2; exit 1
    fi
    grep -Fq "node verification failed" "$prefix/missing.log"
    if "$manager" --prefix "$prefix" --configure-model --download --yes > "$prefix/download.log" 2>&1; then
        echo "Download allowed before a successful demo" >&2; exit 1
    fi
    grep -Fq "finish the synthetic demo" "$prefix/download.log"
    [[ ! -e $prefix/models ]]
    exit
fi
# Real installations need persistent storage; /tmp is tmpfs on some Linux hosts.
# The manager refuses a prefix inside its own source tree, so a source-mode run
# (CI, FRONK_TEST_MANAGER) must test outside it. Both stay on persistent storage.
testbase=$root/build
[[ $manager == "$root"/* ]] && testbase=$(dirname "$root")
mkdir -p "$testbase"
testdir=$(mktemp -d "$testbase/installer-test.XXXXXX")
trap 'rm -rf "$testdir"' EXIT
# A tiny digest fixture exercises file verification and publication of Fronk's
# existing manifest format without reading the 177 GB model in unit runs.
weights=$testdir/weights
mkdir -p "$weights/state"
cp "$root/tools/install/verify-model.py" "$weights/"
printf GGUF > "$weights/test.gguf"
digest=$(openssl dgst -sha256 < "$weights/test.gguf"); digest=${digest##* }
{ echo "gguf-sha256 $digest"; for ((i=0;i<20;i++)); do printf 'unit-%02d %s\n' "$i" "$digest"; done; } > "$weights/model-units.sha256"
python3 "$weights/verify-model.py" "$weights/test.gguf" "$weights/state" 4 "$digest"
grep -q '^gguf 4 ' "$weights/state/manifest.txt"
python3 "$weights/verify-model.py" "$weights/test.gguf" "$weights/state" 4 "$digest"
printf FAKE > "$weights/test.gguf"
if python3 "$weights/verify-model.py" "$weights/test.gguf" "$weights/state" 4 "$digest"; then
    echo "replaced GGUF passed digest verification" >&2; exit 1
fi
prefix=$testdir/node
"$manager" --help | grep -q -- --join-nodes
{ "$manager" --terminal --prefix not/absolute 2>&1 || true; } | grep -Fq "llm/DEBUG.md"
printf '%s\n' fronk-v1:2:local-fixture | "$manager" --yes --role node --name worker-one --prefix "$prefix" --secret-stdin --no-service --no-measure
[[ $(cat "$prefix/state/cluster.secret") == local-fixture ]]
grep -qx "burst 2" "$prefix/src/fronk.conf"
# Front-end edits keep compute measurements; kernel and build-option edits must invalidate them.
bench_key() {
    printf '%s\n' 'key:' '	@echo $(BENCH_DIGEST)' | make -s --no-print-directory -C "$prefix/src" -f Makefile -f - "$@" key
}
first_key=$(bench_key)
[[ $first_key =~ ^[0-9a-f]{64}$ ]]
printf '\n/* controller-only fixture */\n' >> "$prefix/src/src/head.c"
[[ $(bench_key) == "$first_key" && $(bench_key CFLAGS=-O0) != "$first_key" ]]
printf '\n/* kernel fixture */\n' >> "$prefix/src/backends/ref.c"
[[ $(bench_key) != "$first_key" ]]

grep -Fxq 'name worker-one' "$prefix/src/fronk.conf"
[[ ! -e $prefix/src/.cluster && ! -e $prefix/src/tools/cluster.sh && ! -d $prefix/src/ds4 ]]
"$prefix/bin/fronk" --version | grep -Fq "fronk $(sed -n 's/^VERSION := //p' "$root/Makefile")"

# Exercise a head's installed manager using real nodes on private test ports.
head=$testdir/head
mkdir -p "$head/src/tools" "$head/state"
ln -s "$prefix/src/fronk" "$head/src/fronk"
ln -s "$prefix/src/tools/install" "$head/src/tools/install"
echo head > "$head/state/install-role"
FRONK_BIN="$prefix/src/fronk" FRONK_BASE_PORT=29700 "$root/tests/local_cluster.sh" 2 bash "$0" check-peers "$head"

# Keep state, customized settings and unrelated user data through maintenance.
echo 'mem-margin-gb 2.5' >> "$prefix/src/fronk.conf"
echo 'user model fixture' > "$prefix/src/local-model.gguf"
# A legacy node without an explicit name already identifies by short hostname.
sed '/^name /d' "$prefix/src/fronk.conf" > "$prefix/src/fronk.conf.tmp"
mv "$prefix/src/fronk.conf.tmp" "$prefix/src/fronk.conf"
"$manager" --role node --prefix "$prefix" --yes --no-service --no-measure
grep -Fxq 'mem-margin-gb 2.5' "$prefix/src/fronk.conf"
grep -Fxq "name $(hostname -s)" "$prefix/src/fronk.conf"
[[ $(cat "$prefix/state/cluster.secret") == local-fixture ]]
"$prefix/bin/fronk-uninstall" --yes
[[ ! -e $prefix/src/fronk && -f $prefix/src/local-model.gguf && -f $prefix/fronk.conf.saved ]]
[[ $(cat "$prefix/state/cluster.secret") == local-fixture ]]
mkdir -p "$prefix/models"
echo 'existing weights' > "$prefix/models/keep.gguf"
"$manager" --prefix "$prefix" --uninstall --purge --yes
[[ ! -e $prefix/state/cluster.secret && -f $prefix/models/keep.gguf ]]
printf '%s\n' new-instance | "$manager" --yes --role node --name fresh --prefix "$prefix" --secret-stdin --no-service --no-measure
[[ $(cat "$prefix/state/cluster.secret") == new-instance && -f $prefix/models/keep.gguf ]]
echo '[ok] installer: packaged node, authenticated head handoff, missing peer, model download gate, upgrade, uninstall and fresh reinstall'
