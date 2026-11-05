/* Lossless 32-neuron units from the existing GGUF. No quantization or padding.
 * Each layer stores 512 Q4_K/Q4_K/MXFP4 routed slices, then one Q8_0 shared
 * slice. The extra MTP layer stays with ds4, outside these 48 layer units.
 */
#ifndef FR_NATIVE_H
#define FR_NATIVE_H
#include "backend.h"
enum { FR_Q8_0=8, FR_Q4_K=12, FR_MXFP4=39 };
#define FR_NATIVE_GATE_BYTES ((size_t)FR_BLOCK_NEURONS * (FR_HIDDEN / 256) * 144)
#define FR_NATIVE_DOWN_BYTES ((size_t)FR_HIDDEN * 17)
#define FR_NATIVE_SLICE_BYTES (2 * FR_NATIVE_GATE_BYTES + FR_NATIVE_DOWN_BYTES)
#define FR_NATIVE_SHARED_PROJECTION ((size_t)FR_HIDDEN * 34)
#define FR_NATIVE_SHARED_BYTES (3 * FR_NATIVE_SHARED_PROJECTION)
#define FR_NATIVE_LAYER_BYTES (FR_ROUTED_EXPERTS * FR_NATIVE_SLICE_BYTES + FR_NATIVE_SHARED_BYTES)
#define FR_NATIVE_BLOCK_BYTES ((uint64_t)FR_LAYERS * FR_NATIVE_LAYER_BYTES)
static inline size_t fr_native_offset(unsigned layer,unsigned expert) {
    return (size_t)layer * FR_NATIVE_LAYER_BYTES + (size_t)expert * FR_NATIVE_SLICE_BYTES;
}
static inline size_t fr_native_row(unsigned type,unsigned n) {
    return type==FR_Q4_K?(size_t)n/256*144:(size_t)n/32*(type==FR_MXFP4?17:34);
}
int fr_native_dot(const unsigned char *p,unsigned type,const float *x,unsigned n,double *out);
void fr_native_fill(unsigned char *data,uint32_t seed); /* Fully resident BENCH input. */
/* Sparse, deterministic correctness fixture. Not a memory/performance test. */
void fr_native_fixture(unsigned char *data,unsigned layer,uint32_t seed);
/* Frozen backend tolerances, independent scalar oracle, sparse correctness
 * fixture only. Never use its storage for performance measurements. */
int fr_native_self_test(const fr_backend *backend);
#endif
