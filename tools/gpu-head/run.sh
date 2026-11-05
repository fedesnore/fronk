#!/usr/bin/env bash
# Launched by fronk head after opening its authenticated cluster plane's socket.
set -euo pipefail
: "${FRONK_MOE_SOCKET:?start this program with fronk head}"
: "${FRONK_MODEL:?missing model in the head config}" "${FRONK_DS4_PORT:?missing chat port}"
cd "$(dirname "$0")/../.."
state=$(dirname "$FRONK_MOE_SOCKET")
device=$(cat "$state/gpu-device")
[[ $device =~ ^Vulkan[0-9]+$ ]] || { echo "Run the GPU check again in the installer manager" >&2; exit 1; }
ctx=8192
if [[ -f $state/ds4-ctx ]]; then read -r ctx < "$state/ds4-ctx"; fi
# The 100+ GB n-gram table stays mapped on the host. The hook replaces
# routed matmuls; the dense and shared-expert graph can use the GPU.
printf 'Fronk dense head: %s, all dense layers in GPU memory, context %s. Local expert blocks use system RAM.\n' "$device" "$ctx" >&2
# On the four-core Intel iMac, ggml CPU helpers otherwise compete with the
# AVX2 expert pool and the receive loop while waiting for the GPU.
cpu_args=(--cache-ram 0)
if [[ $(uname -s) == Darwin && $(sysctl -n hw.logicalcpu) -le 4 ]]; then
    cpu_args+=(--threads 1 --threads-batch 1 --poll 0 --poll-batch 0)
fi
export MVK_CONFIG_LOG_LEVEL=1
exec ./gpu/fronk-llama-server --model "$FRONK_MODEL" --alias ds4 --host 127.0.0.1 --port "$FRONK_DS4_PORT" \
    --device "$device" --split-mode none --gpu-layers all --fit off --ctx-size "$ctx" --parallel 1 --batch-size 32 --ubatch-size 1 --flash-attn off -lv 4 \
    "${cpu_args[@]}" --load-mode mmap --lazy-mode on --cpu-moe --override-tensor 'per_layer_token_embd.*=CPU' --no-warmup --no-repack --no-webui
