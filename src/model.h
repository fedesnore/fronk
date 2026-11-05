/* model.h - fixed Qwen3.8-Flash-Next geometry.
 *
 * The same neuron blocks are owned across every expert and layer. Keeping the
 * dimensions here prevents backends and the planner from inventing shapes.
 * Byte counts use 64-bit arithmetic: one stored block already exceeds 2 GiB.
 */
#ifndef FR_MODEL_H
#define FR_MODEL_H

#include <stdint.h>

#define FR_MODEL_NAME "Qwen3.8-Flash-Next"
#define FR_LAYERS 48
#define FR_DELTANET_LAYERS 36
#define FR_QSA_LAYERS 12
#define FR_HIDDEN 2560
#define FR_ROUTED_EXPERTS 512
#define FR_SHARED_EXPERTS 1
#define FR_EXPERTS (FR_ROUTED_EXPERTS + FR_SHARED_EXPERTS)
#define FR_SHARED_EXPERT FR_ROUTED_EXPERTS
#define FR_ACTIVE_ROUTED 10
#define FR_ACTIVE_EXPERTS (FR_ACTIVE_ROUTED + FR_SHARED_EXPERTS)
#define FR_INTERMEDIATE 640
#define FR_BLOCK_NEURONS 32
#define FR_BLOCKS (FR_INTERMEDIATE / FR_BLOCK_NEURONS)
#define FR_MAX_BURST 8
#define FR_BENCH_BURST_MAX 4

/* Canonical ggml Q4_0: one fp16 scale and sixteen packed nibble bytes. */
#define FR_Q4_WEIGHTS 32
#define FR_Q4_BYTES 18
#define FR_EXPERT_PARAMS \
    (UINT64_C(3) * FR_LAYERS * FR_EXPERTS * FR_HIDDEN * FR_INTERMEDIATE)
#define FR_EXPERT_BYTES (FR_EXPERT_PARAMS / FR_Q4_WEIGHTS * FR_Q4_BYTES)
#define FR_WEIGHT_BLOCK_BYTES (FR_EXPERT_BYTES / FR_BLOCKS)

_Static_assert(FR_DELTANET_LAYERS + FR_QSA_LAYERS == FR_LAYERS,
               "attention layer counts must cover the model");
_Static_assert(FR_INTERMEDIATE % FR_BLOCK_NEURONS == 0,
               "neuron blocks must cover every expert exactly");
_Static_assert(FR_HIDDEN % FR_Q4_WEIGHTS == 0 &&
               FR_BLOCK_NEURONS % FR_Q4_WEIGHTS == 0,
               "all sliced projection rows must contain whole Q4_0 groups");

#endif
