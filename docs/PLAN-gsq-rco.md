# Plan: Qwen3.8-Flash-Next GSQ-RCO (IQ3) on Fronk

Goal: run Qwen3.8-Flash-Next with ISTA-DASLab's GSQ-RCO weights (near the BF16
model's quality at about a seventh of its size) on the Fronk cluster with the M1 Pro
as head, so the M4 Pro stays free. Nothing here has run yet.

## Weights

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, Apache 2.0.

| Variant | Resident shard | n-gram shard (stays on disk) | Task avg (BF16 93.12) |
|---|---|---|---|
| IQ3_XXS (chosen first) | 47.0 GB, `219ea929...56d15` | 28.8 GB, `316b46f3...e113` | 92.57 |
| IQ3_S | 54.8 GB, `4c1eb2ce...2aca3` | same file | 93.26 |

RCO gives every tensor its own type. In the IQ3_XXS file (header read 2026-11-07):

| Tensor | Types |
|---|---|
| ffn_gate_exps, ffn_up_exps `[2560, 640, 512]` | IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S (per layer) |
| ffn_down_exps `[640, 2560, 512]` | IQ4_NL (18 layers), Q2_0 (30 layers) |
| shared expert, attention, SSM out | IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K, IQ4_NL, Q2_0 |
| router, hyper-connections, PLE | BF16 / F32 |

## Why llama.cpp, not ds4, on the head

Neither Fronk's pinned ds4 nor upstream ds4 (`b3f50180ec`, 2026-11-03) handles
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS or Q2_0, and the dense part uses them
too. llama.cpp 0.9.0 (`cd959bae99ff4bed22bef2604f45241c09bbb8c8`) has the
`qwen4exp` architecture, every type, MTP (`--spec-type draft-mtp`) and lazy n-gram
loading (`--lazy-mode`).

## Design

1. **Head hook (llama.cpp patch).** In `src/models/qwen4exp.cpp`,
   `build_layer_ffn` calls `build_moe_ffn` for the routed experts, then adds the
   gated shared expert. The patch keeps the router in the graph (softmax, top 10,
   normalized weights, `expert_weights_scale`) and replaces the routed expert
   computation by a `ggml_custom_4d` op whose CPU callback sends x, ids and weights
   to Fronk and returns the routed sum. The shared expert stays in llama.cpp, so the
   request's shared gate is 0. The MTP layer's experts stay local at first.
2. **Per-layer types in Fronk.** `src/model.h` fixes one type per projection today.
   The unit format gets a per-layer table (gate/up type, down type) read from the
   GGUF and recorded in the unit manifest. Units carry routed experts only.
3. **Lossless slicing.** gate/up rows span the hidden size (2560 = 10 blocks of
   256), so a 32-neuron block copies whole rows. down rows span the 640 neurons:
   IQ4_NL blocks are 32 weights, so a slice copies its block unchanged. Q2_0 blocks
   are 64 weights (fp16 d, then 2 bits per weight in order, 4 per byte, value
   (q - 1) * d), so a 32-neuron slice is d plus the 8 bytes of its half: 10 bytes,
   no weight changes. That half-block needs a small kernel of our own (ggml's
   vec_dot works on whole 64-weight blocks).
4. **Node backend on ggml kernels.** A `ggml` backend computes each slice with
   ggml's per-type `vec_dot` (activations quantized to the type's `vec_dot_type`),
   which already has AVX2 code for iMac, Shuttle and HP. ggml is MIT; it is built on
   each node. The scalar oracle uses ggml's reference `dequantize_row_*`, and the
   existing tolerances apply.

## Memory (no M4 Pro, today's RAM)

Computed from the IQ3_XXS header's per-layer types: the routed experts are 42.91 GB
in the file; one unit (32 neurons of every routed expert in all 48 layers) is
2.185 GB, 20 units 43.70 GB (the half-block Q2_0 slices repeat their fp16 scale).
iMac (48 GB, 8 GB kept) holds 18 units, Shuttle about 6: no HP and no M4 Pro needed. The M1 Pro as head holds the dense part (about 5 GB) and maps
the n-gram shard from disk.

## Steps

1. Download IQ3_XXS and verify SHA-256 (in progress).
2. Patch llama.cpp: written. `tools/engine/gsq/llama-fronk.patch` (applies to
   `cd959bae9`, rebuilt by `tools/engine/gsq/prepare.sh`) adds the hook in
   `build_moe_ffn` with three modes (socket, `FRONK_MOE_LOOPBACK=ref`,
   `FRONK_MOE_LOOPBACK=zero`); the changed files pass a syntax check. Building needs
   CMake. `~/fronk/gsq-test.sh parity` compares plain llama.cpp with the reference
   path (CPU); `~/fronk/gsq-test.sh cost` measures the GPU dense part with the hook
   returning zeros and the experts kept off the GPU (`-ot exps=CPU`).
3. Fronk: per-layer type table, extraction, manifest, oracle via ggml reference.
4. `ggml` node backend, its self-test against the oracle, build on Linux nodes.
5. Cluster bring-up with the M1 Pro head; compare answers with plain llama.cpp.
6. MTP: local first, then decide whether its experts need the cluster.

## Measured (2026-11-07, M4 Pro 24 GB, IQ3_XXS, both shards SHA-256 checked)

- Head ceiling (`gsq-test.sh cost`): dense part on Metal, experts kept off the GPU
  (`-ot exps=CPU`), hook on all 48 layers returning zeros: 22.1 tok/s generation
  (45 ms per token), no MTP. Load about 40 s.
- Parity: greedy, 12 tokens, CPU, `--no-repack`: plain llama.cpp and the hook's
  reference path give the same text (": 1. **First law** – energy cannot be").
  Without `--no-repack` the reference path read repacked IQ4_NL weights and failed;
  the socket path never reads expert weights, and units come from the GGUF file.

## Measured on the cluster (2026-11-08)

Head on the M4 Pro (llama.cpp, MTP draft head from `tools/gsq-mtp-build.py`,
`--spec-draft-n-max 3 --spec-draft-p-min 0.5`), units over M1 Pro (0-2, 6 threads),
HP (5-8, 12; 16 threads), Shuttle (9-11; 8 threads), iMac (3, 4, 13-19; 4 threads),
`gsq-node` + `gsq-broker` over TCP. Three Italian chat prompts:
2.9 tok/s at the first run, 19.3 / 13.8 / 14.8 tok/s now. What changed, measured
with the broker's per-node times and `FRONK_MOE_PROFILE=1` in the hook:

- M1 Pro (16 GB) with 5 units (10.9 GB) was evicting pages: 3.4 ms per request,
  0.37 ms with 3 units. On Linux `pkill -f` matched the starting shell and the
  Shuttle node never ran; the scripts use `pkill -x gsq-node`.
- Node workers spin between layers (condvar wake-ups cost milliseconds on macOS)
  and ask for performance cores.
- While the hook waited, the head's GPU idled and lowered its clock: a simulated
  3 ms wait raised the head from 0.76 to 1.88 ms per layer. `gpu-keepalive` (a
  tiny kernel in a loop) brings it to 0.72; on the cluster 1.44 -> 0.95 ms per
  layer, DwarfStar's level on this Mac (0.93-1.03). `--poll`, `-t`, `-bs` changed
  nothing.
- Gate/up is 85% of a slice on x86 (iMac 20 of 23.7 us). `src/gsq_multi.c` decodes
  each 32-weight group once for every token routed to the expert: bit-identical
  to ggml's dot on all 30,720 rows and 480 slices (`gsq-multicheck`), 1.56x / 1.94x
  / 2.23x for 2 / 3 / 4 tokens on the iMac. A pass has about 23 distinct experts
  of 36 uses, so the cluster went 2.72 -> 2.49 ms per layer.
- Per pass now: about 120 ms in the nodes (iMac and Shuttle 2.1 ms per layer), 46 ms
  in the head's layers, 20 ms between passes (logits, drafts).

## Risks

- The hook moves activations GPU -> CPU -> network -> CPU -> GPU 48 times per
  token; its cost must be measured in step 2.
- Lookup-table kernels (IQ2/IQ3) on the iMac's CPU may be slower than today's Q4_K.
- Q2_0 layout read from llama.cpp `cd959bae9` (`ggml-common.h`,
  `dequantize_row_q2_0`); the half-block slice must be tested against it.
