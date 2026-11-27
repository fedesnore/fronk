#!/usr/bin/env bash
# install.sh, the simple entry point: its report, its refusals, and a whole node
# install through it in an isolated prefix. No weights, services, sudo or network.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/build"
work=$(mktemp -d "$root/build/install-simple-test.XXXXXX")
trap 'rm -rf "$work"' EXIT
# The installer refuses a prefix inside its own source: work from an exported copy,
# which also shows that the public source carries everything install.sh needs.
python3 "$root/tools/release/export-source.py" "$work/source" >/dev/null
cd "$work/source"

# The report names this computer and ends with a verdict; it changes nothing.
bash install.sh check > "$work/check.log" 2>&1 || { cat "$work/check.log"; exit 1; }
grep -q '^== 1 of 3 · This computer' "$work/check.log"
grep -q 'has everything a Fronk node needs' "$work/check.log"
[[ ! -e $work/prefix ]]

# No role and no terminal: it says what to type instead of guessing.
if bash install.sh --yes --prefix "$work/prefix" < /dev/null > "$work/norole.log" 2>&1; then cat "$work/norole.log"; exit 1; fi
grep -q 'bash install.sh head, or bash install.sh node' "$work/norole.log"
if bash install.sh sideways > "$work/word.log" 2>&1; then cat "$work/word.log"; exit 1; fi
grep -q 'unknown word' "$work/word.log"

# A head on a computer that cannot be one stops before installing anything.
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    if bash install.sh head --yes --prefix "$work/prefix" < /dev/null > "$work/head.log" 2>&1; then cat "$work/head.log"; exit 1; fi
    grep -q 'needs an Apple Silicon Mac' "$work/head.log"
    [[ ! -e $work/prefix ]]
fi

# Linux without a compiler: it names what is missing and the command that installs it,
# and with --yes it stops there instead of running anything.
if [[ $(uname -s) == Linux ]]; then
    mkdir "$work/fakebin"
    for tool in bash sh uname hostname df awk grep sed tr cat dirname id wc tar mktemp rm mv mkdir python3 apt-get dnf yum pacman zypper apk; do
        found=$(command -v "$tool" 2>/dev/null) && ln -s "$found" "$work/fakebin/$tool"
    done
    if PATH=$work/fakebin bash install.sh node --yes --prefix "$work/prefix" < /dev/null > "$work/nocc.log" 2>&1; then cat "$work/nocc.log"; exit 1; fi
    grep -q 'MISSING  cc' "$work/nocc.log" && grep -q 'MISSING  make' "$work/nocc.log" || { cat "$work/nocc.log"; exit 1; }
    grep -Eq 'apt-get install|dnf install|yum install|pacman -S|zypper install|apk add|package manager is not one' "$work/nocc.log" || { cat "$work/nocc.log"; exit 1; }
    [[ ! -e $work/prefix ]]
fi

# A node, start to finish: tools found, source used in place, built, paired, ready to launch.
printf '%s\n' simple-test-code | bash install.sh node --yes --prefix "$work/prefix" -- \
    --name simple-node --secret-stdin --no-service > "$work/node.log" 2>&1 ||
    { cat "$work/node.log"; [[ ! -f $work/prefix/state/build.log ]] || tail -n 40 "$work/prefix/state/build.log"; exit 1; }
grep -q 'Using the source this file is in' "$work/node.log"
grep -q 'Node simple-node · manual launch' "$work/node.log"
[[ -x $work/prefix/src/fronk && -x $work/prefix/bin/fronk-node ]]
[[ $(cat "$work/prefix/state/install-role") == node ]]
[[ $(awk '$1=="name" {print $2}' "$work/prefix/src/fronk.conf") == simple-node ]]
"$work/prefix/src/fronk" backends --list | grep -q 'gsq.*runs here'
# The pairing code is stored, not shown.
! grep -q simple-test-code "$work/node.log"

# The first settings of a GSQ head. A head is an Apple Silicon Mac, so the part of
# the installer that writes them is checked here by itself, on any system.
settings=tools/install/gsq-head-settings.sh
gib=$((1024 * 1024 * 1024))
value() { cat "$1/$2"; }
# 24 GiB: all eight, with the values of the guide.
mkdir "$work/s24"
bash "$settings" "$work/s24" $((24 * gib)) > "$work/s24.log"
[[ $(value "$work/s24" gsq-threads) == 2 && $(value "$work/s24" gsq-draft) == 12 ]]
[[ $(value "$work/s24" gsq-draft-p-min) == 0.05 && $(value "$work/s24" gsq-draft-chain) == 0.75 ]]
[[ $(value "$work/s24" gsq-copy-drafts) == 31 && $(value "$work/s24" ds4-ctx) == 32768 ]]
[[ $(value "$work/s24" gsq-cache-ram-mb) == 0 && $(value "$work/s24" gsq-ctx-checkpoints) == 4 ]]
[[ $(wc -l < "$work/s24/gsq-settings.installed") -eq 8 && $(wc -l < "$work/s24.log") -eq 8 ]]
# Written once: a file the person removed or changed is not touched by a later run.
rm "$work/s24/gsq-threads"; echo 9 > "$work/s24/gsq-draft"
bash "$settings" "$work/s24" $((24 * gib)) > "$work/s24-again.log"
[[ ! -e $work/s24/gsq-threads && $(value "$work/s24" gsq-draft) == 9 && ! -s $work/s24-again.log ]]
# A file that is already there is kept and said so; the others are written.
mkdir "$work/kept"; echo 16384 > "$work/kept/ds4-ctx"
bash "$settings" "$work/kept" $((24 * gib)) > "$work/kept.log"
[[ $(value "$work/kept" ds4-ctx) == 16384 && $(value "$work/kept" gsq-threads) == 2 ]]
grep -q 'ds4-ctx *kept as it was' "$work/kept.log"
[[ $(wc -l < "$work/kept/gsq-settings.installed") -eq 7 ]]
# 64 GiB: the three that trade context and chat switching for memory are left alone.
mkdir "$work/s64"
bash "$settings" "$work/s64" $((64 * gib)) > /dev/null
[[ $(value "$work/s64" ds4-ctx) == 65536 && $(value "$work/s64" gsq-draft) == 12 ]]
[[ ! -e $work/s64/gsq-cache-ram-mb && ! -e $work/s64/gsq-ctx-checkpoints ]]
# It refuses what it cannot use, and writes nothing.
if bash "$settings" "$work/missing" $((24 * gib)) 2> /dev/null; then exit 1; fi
mkdir "$work/bad"
if bash "$settings" "$work/bad" lots 2> /dev/null; then exit 1; fi
[[ -z $(ls -A "$work/bad") ]]

# The engine's launcher accepts those files and turns them into the engine's
# arguments. A stand-in takes the place of Python, which would start the engine.
mkdir -p "$work/stand-in" "$work/models/IQ3_XXS"
: > "$work/models/mtp-Qwen3.8-Flash-Next.gguf"
for name in python3 python3.15; do
    printf '#!/bin/sh\necho "chain=$FRONK_DRAFT_P_CHAIN"\necho "$@"\n' > "$work/stand-in/$name"
    chmod +x "$work/stand-in/$name"
done
mkdir "$work/run"
bash "$settings" "$work/run" $((24 * gib)) > /dev/null
PATH=$work/stand-in:$PATH KEEPALIVE=0 FRONK_GSQ_ENGINE=/bin/sh FRONK_DS4_PORT=1 \
    FRONK_MOE_SOCKET=$work/run/moe.sock FRONK_MODEL=$work/models/IQ3_XXS/shard-1.gguf \
    bash tools/engine/gsq/run.sh > "$work/run.log" 2>&1 || { cat "$work/run.log"; exit 1; }
for wanted in 'chain=0.75' '-c 32768 ' '-t 2 ' '--cache-ram 0 ' '--ctx-checkpoints 4 ' \
    '--spec-ngram-mod-n-max 31 ' '--spec-draft-n-max 12 ' '--spec-draft-p-min 0.05 '; do
    grep -q -e "$wanted" "$work/run.log" || { echo "launcher did not pass: $wanted"; cat "$work/run.log"; exit 1; }
done
grep -q 'draft 12 (p>=0.05), lenience 0, 2 threads, cache 0 MiB, 4 checkpoints' "$work/run.log"
echo "[ok] simple install: report, refusals, a complete node install and a GSQ head's first settings"
