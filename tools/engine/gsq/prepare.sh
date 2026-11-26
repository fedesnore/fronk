#!/usr/bin/env bash
# Reconstruct the llama.cpp head engine for Qwen3.8-Flash-Next GSQ-RCO from a
# public llama.cpp commit and our patch (the routed-expert hook in
# build_moe_ffn, src/llama-fronk.{h,cpp}, and a copy of client/fronk_client.h).
# Build:  cmake -S DEST -B DEST/build -DGGML_METAL=ON -DLLAMA_CURL=OFF
#         cmake --build DEST/build --target llama-server -j
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
revision=cd959bae99ff4bed22bef2604f45241c09bbb8c8   # llama.cpp 0.9.0
destination=${1:?usage: prepare.sh DESTINATION}
key="gsq $revision $(cksum < "$here/llama-fronk.patch")"
if [[ -f $destination/.fronk-source && $(cat "$destination/.fronk-source") == "$key" ]]; then
    printf 'Engine source already prepared: %s\n' "$destination"; exit
fi
[[ ! -e $destination ]] || { echo "Refusing to replace $destination; choose a new destination." >&2; exit 1; }
mkdir -p "$(dirname "$destination")"
work=$(mktemp -d "${destination}.tmp.XXXXXX")
trap 'rm -rf "$work"' EXIT
git -C "$work" init -q source
git -C "$work/source" fetch -q --depth=1 https://github.com/ggml-org/llama.cpp.git "$revision"
git -C "$work/source" checkout -q --detach FETCH_HEAD
(cd "$work/source" && git apply --check "$here/llama-fronk.patch" && git apply "$here/llama-fronk.patch")
printf '%s\n' "$key" > "$work/source/.fronk-source"
mv "$work/source" "$destination"
printf 'Prepared llama.cpp head engine: %s\n' "$destination"
