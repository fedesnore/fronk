#!/usr/bin/env bash
# The dense engine for Qwen3.8-Flash-Next GSQ-RCO, launched by fronk head (config
# `ds4 tools/engine/gsq/run.sh`, `model` = the GSQ shard 1).
#
#   chat client -> tools/fronk_engine.py -> ONE llama.cpp engine -> moe.sock -> fronk head -> nodes
#
# llama.cpp with the Fronk hook (llama-fronk.patch) runs attention, the shared
# expert and the logits on this Mac's GPU; every routed expert goes straight to
# the head's socket. fronk_engine.py supervises it: API, queue, tools prefix,
# memory guard, emergency chat save, status for the console.
# Settings are one-line files in the state directory; remove a file for its default:
#   ds4-ctx             context length (65536)
#   gsq-draft           MTP draft tokens per pass, 1-15 (7)
#   gsq-draft-p-min     stop a draft below this probability (0.5)
#   gsq-copy-drafts     longest draft copied from the context, in tokens (0 = off; 31 fills one
#                       32-token expert call per layer)
#   gsq-draft-sampling  greedy (default) or probabilistic: the draft is sampled and the full model
#                       verifies it by rejection sampling, which keeps its output distribution exact
#   gsq-lenience        keep a drafted token the full model rates at least this likely even when
#                       it would have written another (0 = exact decode; 0.3 trades exactness for
#                       fewer passes; never applied inside tool calls)
#   gsq-cache-ram-mb    RAM for chats other than the current one (3072)
#   gsq-ctx-checkpoints rollback points kept per chat, ~112 MiB each (16)
#   gsq-threads         CPU threads (performance cores)
#   gsq-listen          ADDRESS:PORT on a private cable where the API is also served
#   gsq-thinking        present: the model reasons before answering
#   gsq-thinking-budget N: at most N tokens of reasoning per answer (with gsq-thinking; empty = no limit)
#   gsq-vision-cpu      present: the image encoder stays on the CPU
#   gsq-no-vision       present: no image input; the 0.9 GB image encoder is not loaded and
#                       that memory goes to the expert cache (a request with images gets a clear error)
#   gsq-profile         present: per-layer head/cluster times in the log
#   gsq-no-overlap      present: do not overlap the shared expert with the cluster call
set -euo pipefail
: "${FRONK_MOE_SOCKET:?start this program with fronk head}"
: "${FRONK_MODEL:?missing model in the head config}" "${FRONK_DS4_PORT:?missing chat port}"
cd "$(dirname "$0")/../../.."
state=$(dirname "$FRONK_MOE_SOCKET")
setting() {  # NAME DEFAULT PATTERN
    local value=$2
    if [[ -f $state/$1 ]]; then read -r value < "$state/$1"; fi
    [[ $value =~ ^$3$ ]] || { echo "GSQ engine: invalid $state/$1: $value" >&2; exit 1; }
    printf '%s' "$value"
}
engine=${FRONK_GSQ_ENGINE:-build/gsq-engine/fronk-llama-server}
[[ -x $engine ]] || { echo "GSQ engine missing: $engine (make gsq-engine)" >&2; exit 1; }
models=$(dirname "$FRONK_MODEL")/..
ctx=$(setting ds4-ctx 65536 '[0-9]{4,6}')
draft=$(setting gsq-draft 7 '([1-9]|1[0-5])')
p_min=$(setting gsq-draft-p-min 0.5 '0?\.[0-9]+')
sampling=$(setting gsq-draft-sampling greedy '(greedy|probabilistic)')
copy=$(setting gsq-copy-drafts 0 '[0-9]{1,2}')
lenience=$(setting gsq-lenience 0 '(0|0?\.[0-9]+)')
[[ $lenience == 0 ]] || export FRONK_SPEC_LENIENCE=$lenience
cache_ram=$(setting gsq-cache-ram-mb 3072 '[0-9]{1,5}')
checkpoints=$(setting gsq-ctx-checkpoints 16 '[0-9]{1,2}')
threads=$(setting gsq-threads "$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN)" '[1-9][0-9]?')
[[ -f $state/gsq-no-overlap ]] || export FRONK_MOE_OVERLAP=1
# gsq-draft-log present: the engine logs every verified draft (probabilities, tokens kept); gsq-draft-chain X: chain threshold
[[ ! -f $state/gsq-draft-log ]] || export FRONK_DRAFT_LOG=1
if [[ -f $state/gsq-draft-chain ]]; then v=$(head -c 8 "$state/gsq-draft-chain" | tr -cd 0-9.); [[ -z $v ]] || export FRONK_DRAFT_P_CHAIN=$v; fi
# gsq-draft-vocab K: the MTP drafts score only the first K tokens (diagnostic; output unchanged)
if [[ -f $state/gsq-draft-vocab ]]; then n=$(head -c 8 "$state/gsq-draft-vocab" | tr -cd 0-9); [[ -z $n ]] || export FRONK_MTP_VOCAB=$n; fi
[[ ! -f $state/gsq-profile ]] || export FRONK_MOE_PROFILE=1 FRONK_PASS_PROFILE=1
# gsq-lifeboat-check (diagnostic, reads the SSD): N in the file checks one expert call in N of every layer against the model file; empty = the first only
if [[ -f $state/gsq-lifeboat-check ]]; then n=$(head -c 8 "$state/gsq-lifeboat-check" | tr -cd 0-9); export FRONK_MOE_LIFEBOAT_CHECK=${n:-1}; fi
think=false; [[ -f $state/gsq-thinking ]] && think=true
arguments=(-m "$FRONK_MODEL" --alias qwen3.8-flash-next --host 127.0.0.1 --slot-save-path "$state/slot-cache"
    --jinja -c "$ctx" -np 1 -b 64 -ub 64 -ngl 99 -fa on -ot 'exps=CPU' --load-mode mmap --lazy-mode on --no-repack
    -t "$threads" --no-webui --cache-ram "$cache_ram" --ctx-checkpoints "$checkpoints"
    --chat-template-kwargs "{\"enable_thinking\": $think}")
if [[ $think == true && -f $state/gsq-thinking-budget ]]; then n=$(head -c 8 "$state/gsq-thinking-budget" | tr -cd 0-9); [[ -z $n ]] || arguments+=(--reasoning-budget "$n"); fi
if [[ -f $models/mtp-Qwen3.8-Flash-Next.gguf ]]; then
    # Drafts come from two sources, tried in this order. Where the text being written
    # repeats something already in the context (code under edit, a document being
    # reworked), up to gsq-copy-drafts tokens are copied from there; elsewhere the
    # model's MTP head proposes up to gsq-draft. Every drafted token is still verified
    # by the full model in the same pass, so the output is the same model's output.
    types=draft-mtp
    if [[ $copy != 0 ]]; then
        types=ngram-mod,draft-mtp
        arguments+=(--spec-ngram-mod-n-match 16 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max "$copy")
    fi
    arguments+=(--spec-type "$types" --model-draft "$models/mtp-Qwen3.8-Flash-Next.gguf"
                --spec-draft-n-max "$draft" --spec-draft-p-min "$p_min" --spec-draft-sampling "$sampling" -ngld 99)
else
    echo "GSQ engine: no MTP head in $models; n-gram drafts instead" >&2
    arguments+=(--spec-type ngram-mod --spec-ngram-mod-n-max 4)
fi
if [[ ! -f $state/gsq-no-vision && -f $models/mmproj-Qwen3.8-Flash-Next-BF16.gguf ]]; then
    arguments+=(--mmproj "$models/mmproj-Qwen3.8-Flash-Next-BF16.gguf" --image-max-tokens 1024)
    [[ ! -f $state/gsq-vision-cpu ]] || arguments+=(--no-mmproj-offload)
fi
# Keep the GPU in its fast state while it waits for the nodes; it exits with us
# (exec keeps this PID, which fronk records as the engine's).
keepalive=build/gsq-engine/gpu-keepalive
[[ -x $keepalive && ${KEEPALIVE:-1} == 1 ]] && "$keepalive" --watch $$ 20000 &
python=$(command -v python3.15 || command -v python3)
printf 'Fronk GSQ engine: context %s, draft %s (p>=%s), lenience %s, %s threads, cache %s MiB, %s checkpoints\n' \
    "$ctx" "$draft" "$p_min" "$lenience" "$threads" "$cache_ram" "$checkpoints" >&2
mkdir -p "$state/slot-cache"
exec "$python" tools/fronk_engine.py -- "$engine" "${arguments[@]}"
