/* metal.h - Apple Silicon diagnostics; the binding backend vtable is unchanged.
 * Setup resources belong to load; only framework submission objects may be
 * transient during run (user decision recorded in NOTES.md, milestone 6).
 */
#ifndef FR_METAL_H
#define FR_METAL_H
#include "backend.h"
#include "ssd.h"
enum { FR_METAL_ENCODE, FR_METAL_QUEUE, FR_METAL_GPU, FR_METAL_RETURN,
       FR_METAL_OVERHEAD, FR_METAL_TOTAL, FR_METAL_PHASES };
typedef struct {
    double ns[FR_METAL_PHASES];
    uint64_t weight_bytes, runs, hot_allocations, resources;
    unsigned blocks, unique_experts;
} fr_metal_sample;
#ifdef FR_METAL_COMPUTE
extern const fr_backend fr_metal;
extern const fr_backend fr_metal_ssd;
/* Bounded native-slice streaming uses the same Metal kernels and reduction. */
int fr_metal_load_ssd(unsigned block,int fd,uint64_t cache_bytes);
int fr_metal_load_ssd_fixture(unsigned block,int fd,uint64_t cache_bytes);
void fr_metal_ssd_clear(void);
int fr_metal_ssd_pipeline(int enabled); /* idle-only A/B and qualification */
void fr_metal_ssd_sample(fr_ssd_sample *out);
int fr_metal_sample_get(fr_metal_sample *out);
/* Releases idle backend resources; refuses while blocks are still loaded. */
int fr_metal_close(void);
/* fronk demo: GPU time to read `bytes` once (a dense layer's weights per pass). */
int fr_metal_stream(uint64_t bytes,double *gpu_ns);
#endif
#endif
