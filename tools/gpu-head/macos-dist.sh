#!/usr/bin/env bash
# Release build on macOS; the end user only needs the packaged runtime.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
deps=$root/build/gpu-deps
mkdir -p "$deps"
sum=f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e
archive=$deps/MoltenVK-macos.tar
if [[ ! -f $archive ]]; then
    curl -fL --retry 2 -o "$archive.part" https://github.com/KhronosGroup/MoltenVK/releases/download/v1.4.2/MoltenVK-macos.tar
    printf '%s  %s\n' "$sum" "$archive.part" | shasum -a 256 -c -
    mv "$archive.part" "$archive"
fi
printf '%s  %s\n' "$sum" "$archive" | shasum -a 256 -c -
library=MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib
tar -xf "$archive" -C "$deps" "$library" MoltenVK/LICENSE
export FRONK_GPU_ARCH=x86_64 MOLTENVK_DYLIB="$deps/$library"
export FRONK_GPU_BUILD=${FRONK_GPU_BUILD:-$root/build/gpu-head-work}
if [[ -z ${VULKAN_SDK:-} ]] && command -v brew >/dev/null; then export VULKAN_SDK=$(brew --prefix); fi
out=$root/build/gpu-runtime/Darwin-x86_64
"$root/tools/gpu-head/build.sh" "$out"
cp "$deps/MoltenVK/LICENSE" "$out/LICENSE-MoltenVK"
