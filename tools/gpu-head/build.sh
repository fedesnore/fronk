#!/usr/bin/env bash
# Build in a private source tree. The revision fixes Qwen graph and hook ABI.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
revision=a6ea155d3d38b3f6f43d0c0dc29c2d592413ef44
platform=${FRONK_GPU_PLATFORM:-$(uname -s)}; arch=${FRONK_GPU_ARCH:-$(uname -m)}
out=${1:?usage: build.sh OUTPUT_DIRECTORY}
mkdir -p "$out"; out=$(cd "$out" && pwd)
work=${FRONK_GPU_BUILD:-$(mktemp -d)}
if [[ -z ${FRONK_GPU_BUILD:-} ]]; then trap 'rm -rf "$work"' EXIT; fi
mkdir -p "$work/source"
src=$work/source
cmake=${CMAKE:-cmake}
patch_key="$revision $(cksum < "$here/llama-fronk.patch")"
if [[ ! -f $src/.fronk-patched || $(cat "$src/.fronk-patched") != "$patch_key" ]]; then
    if [[ -n ${LLAMA_CPP:-} ]]; then git -C "$LLAMA_CPP" archive "$revision" | tar -xf - -C "$src"
    else
        git -C "$src" init -q
        git -C "$src" fetch --depth=1 https://github.com/ggml-org/llama.cpp.git "$revision"
        git -C "$src" checkout --detach --force FETCH_HEAD
    fi
    (cd "$src" && patch -p1 < "$here/llama-fronk.patch")
    cat >> "$src/CMakeLists.txt" <<'CMAKE'
add_executable(fronk-gpu-probe fronk-probe.cpp)
target_link_libraries(fronk-gpu-probe PRIVATE ggml)
target_compile_features(fronk-gpu-probe PRIVATE cxx_std_17)
CMAKE
    printf '%s\n' "$patch_key" > "$src/.fronk-patched"
fi
cp "$here/fronk-llama.inc" "$root/client/fronk_client.h" "$src/src/"
cp "$here/probe.cpp" "$src/fronk-probe.cpp"
args=(-DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_METAL=OFF -DGGML_NATIVE=OFF
      -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF -DGGML_CCACHE=OFF)
if [[ $platform == Darwin ]]; then
    mvk=${MOLTENVK_DYLIB:?Set MOLTENVK_DYLIB to the official universal libMoltenVK.dylib}
    args+=(-DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR="$arch" -DCMAKE_OSX_ARCHITECTURES="$arch"
           -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DVulkan_LIBRARY="$mvk" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
           -DCMAKE_INSTALL_RPATH=@executable_path)
fi
if [[ -n ${VULKAN_SDK:-} ]]; then
    args+=(-DVulkan_INCLUDE_DIR="$VULKAN_SDK/include" -DCMAKE_PREFIX_PATH="$VULKAN_SDK"
           -DVulkan_GLSLC_EXECUTABLE="$VULKAN_SDK/bin/glslc")
fi
"$cmake" -S "$src" -B "$work/objects" "${args[@]}"
"$cmake" --build "$work/objects" --parallel "${FRONK_BUILD_JOBS:-4}" --target fronk-gpu-probe llama-server
cp "$work/objects/bin/llama-server" "$out/fronk-llama-server"
cp "$work/objects/bin/fronk-gpu-probe" "$out/"
cp "$src/LICENSE" "$out/LICENSE-llama.cpp"
(cd "$src" && find vendor -type f \( -iname 'license*' -o -iname 'copying*' \) | tar -cf - -T -) | (mkdir -p "$out/licenses" && tar -xf - -C "$out/licenses")
if [[ $platform == Darwin ]]; then
    lipo "$mvk" -thin "$arch" -output "$out/libMoltenVK.dylib"
    codesign --force --sign - "$out/libMoltenVK.dylib"
fi
