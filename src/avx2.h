/* avx2.h - preserved Shuttle microkernels behind the fixed backend interface.
 * Native Q4_K/MXFP4 and shared Q8_0 are read losslessly; canonical Q4_0 remains
 * supported for the frozen legacy vectors. Borrowed weights are never rewritten.
 * Runtime AVX2/FMA/F16C and OS YMM support precede every optional instruction.
 * Lifecycle calls are externally serialized, as required by backend.h.
 */
#ifndef FR_AVX2_H
#define FR_AVX2_H
#include "backend.h"
#include <stdio.h>
typedef struct {
    uint64_t runs, weight_bytes, hot_allocations;
    unsigned blocks, unique_experts, workers, features, prefetch;
    double total_ns;
} fr_avx2_sample;
#ifdef FR_AVX2_COMPUTE
extern const fr_backend fr_avx2;
int fr_avx2_sample_get(fr_avx2_sample *out);
int fr_avx2_close(void); /* Refuses to destroy a pool with borrowed blocks. */
int fr_avx2_tune(const fr_req *req,FILE *csv); /* BENCH selects runtime knobs on resident units. */
int fr_avx2_configure(unsigned blocks,unsigned burst);
void fr_avx2_set_tuning(unsigned blocks,unsigned burst,unsigned workers,unsigned prefetch); /* from cached BENCH */ /* Restore measured knobs before loading a session. */
#endif
#endif
