/* gsq.h - Qwen3.8-Flash-Next with ISTA-DASLab's GSQ-RCO IQ3_XXS weights.
 *
 * RCO gives every layer its own expert types: gate and up rows (2560 weights)
 * in IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS or IQ3_S, down rows (640 weights) in IQ4_NL
 * or Q2_0. A unit holds the routed experts' 32-neuron block b of every layer:
 *
 *   for each layer, for each of the 512 routed experts, one slice:
 *     gate rows of neurons 32b..32b+31  (32 rows, copied whole)
 *     up rows of the same neurons       (32 rows, copied whole)
 *     down: for each of the 2560 output rows, the 32 weights of those neurons
 *       IQ4_NL: the row's 32-weight block, copied (18 bytes)
 *       Q2_0:   half of the row's 64-weight block: its fp16 scale and the
 *               32 two-bit codes of those weights, regrouped in planes: code j
 *               in byte j%8 at bits 2(j/8), so SIMD unpacks them with one shift
 *               and mask (10 bytes; value (q-1)*d). Codes and scale are copied
 *               unchanged, only their bit positions differ from ggml's block.
 *
 * No weight changes. The shared expert is not in the unit: llama.cpp on the head
 * computes it, so requests carry a shared gate of 0. The type table is the
 * pinned file's (revision ed59f92, IQ3_XXS); fr_gsq_open refuses any other.
 */
#ifndef FR_GSQ_H
#define FR_GSQ_H
#include <stddef.h>
#include <stdint.h>

#define FR_GSQ_LAYERS 48
#define FR_GSQ_EXPERTS 512
#define FR_GSQ_HIDDEN 2560
#define FR_GSQ_INTERMEDIATE 640
#define FR_GSQ_NEURONS 32
#define FR_GSQ_BLOCKS (FR_GSQ_INTERMEDIATE / FR_GSQ_NEURONS)
#define FR_GSQ_Q8K_BYTES 2920   /* x as 10 Q8_K blocks of 292 bytes */

/* ggml type ids used here. */
enum { FR_GSQ_IQ2_XXS = 16, FR_GSQ_IQ2_XS = 17, FR_GSQ_IQ3_XXS = 18, FR_GSQ_IQ4_NL = 20,
       FR_GSQ_IQ3_S = 21, FR_GSQ_IQ2_S = 22, FR_GSQ_Q2_0 = 42 };

extern const uint8_t fr_gsq_gate_type[FR_GSQ_LAYERS]; /* gate and up share it */
extern const uint8_t fr_gsq_down_type[FR_GSQ_LAYERS];

size_t fr_gsq_row_bytes(unsigned layer);    /* one gate/up row of 2560 weights */
size_t fr_gsq_piece_bytes(unsigned layer);  /* one down piece of 32 weights */
size_t fr_gsq_slice_bytes(unsigned layer);  /* one (layer, expert) slice */
uint64_t fr_gsq_layer_offset(unsigned layer);
uint64_t fr_gsq_unit_bytes(void);
static inline uint64_t fr_gsq_offset(unsigned layer, unsigned expert) {
    return fr_gsq_layer_offset(layer) + (uint64_t)expert * fr_gsq_slice_bytes(layer);
}

/* The first GGUF shard, memory-mapped read-only, with the routed tensors found. */
typedef struct {
    int fd;
    const unsigned char *map;
    size_t size;
    uint64_t gate[FR_GSQ_LAYERS], up[FR_GSQ_LAYERS], down[FR_GSQ_LAYERS]; /* absolute offsets */
} fr_gsq_gguf;
int fr_gsq_open(const char *path, fr_gsq_gguf *g, char *error, size_t capacity);
void fr_gsq_close(fr_gsq_gguf *g);
/* For a cache of whole experts (hot.h): the offset of each expert's slice in every
 * loaded unit, instead of its place in a full unit; NULL restores the unit layout. */
void fr_gsq_backend_slots(const uint64_t (*table)[512]);
/* Copy slice (block, layer, expert) into out (fr_gsq_slice_bytes(layer)). */
int fr_gsq_slice(const fr_gsq_gguf *g, unsigned block, unsigned layer, unsigned expert, unsigned char *out);
/* The pages of the model file that hold one expert leave the active memory. A cache that
 * has copied the expert into its own wired slot (hot.c) has no use for them, and left
 * alone they stay resident beside the copy: 4.9 GB of them beside a 3.3 GB cache,
 * measured 2026-11-11. The file is simply read again if the expert is wanted later. */
void fr_gsq_forget(const fr_gsq_gguf *g, unsigned layer, unsigned expert);
/* Synthetic slice/unit for self-tests and benchmarks: every fp16 scale finite
 * (about 0.004-0.008, either sign), every other byte pseudo-random from seed. */
void fr_gsq_fill_slice(unsigned layer, unsigned char *out, uint32_t seed);
void fr_gsq_fill_unit(unsigned char *out, uint32_t seed);
/* Weight j (0..31) of a Q2_0 half piece in the planar layout above. */
float fr_gsq_q2_half(const unsigned char *piece, unsigned j);

/* Oracle: y[d] += gate * sum_n down[d][n] * silu(gate_n . x) * (up_n . x), with
 * ggml's reference dequantizers, fp32 x, double accumulation. */
int fr_gsq_ref_slice(unsigned layer, const unsigned char *slice, const float *x, float gate, double *y);
/* Fast: x quantized once per token with fr_gsq_quantize_x (Q8_K, as llama.cpp's
 * CPU path does), ggml's optimized dot products; h quantized to Q8_0 for IQ4_NL
 * down pieces. Adds gate * contribution to y. */
void fr_gsq_quantize_x(const float *x, void *xq);
/* Fills ggml's fp16 table (read by the x86 kernels); call once before computing. */
void fr_gsq_init(void);
int fr_gsq_fast_slice(unsigned layer, const unsigned char *slice, const void *xq, float gate, float *y);
/* The same for nt tokens routed to this expert (nt <= 16): the gate/up weights
 * are decoded once for all of them (gsq_multi.h); y[t] += gate[t] * slice(x[t]). */
int fr_gsq_fast_slice_multi(unsigned layer, const unsigned char *slice, const void *const *xq, const float *gate,
                            float *const *y, unsigned nt);
#endif
