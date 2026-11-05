/* neon.h - ARM CPU eligibility and diagnostics without changing backend.h.
 * Q4_0 is the only weight layout. I8MM is reported, never required or used.
 * All calls except feature discovery are externally serialized like the vtable.
 */
#ifndef FR_NEON_H
#define FR_NEON_H
#include "backend.h"
#define FR_ARM_DOTPROD 1u
#define FR_ARM_I8MM 2u
unsigned fr_neon_features(void);
typedef struct {
    uint64_t runs, weight_bytes, hot_allocations;
    unsigned blocks, unique_experts, workers, features;
    double total_ns;
} fr_neon_sample;
#ifdef FR_NEON_COMPUTE
extern const fr_backend fr_neon;
int fr_neon_sample_get(fr_neon_sample *out);
int fr_neon_close(void); /* Refuses to destroy a pool with borrowed blocks. */
#endif
#endif
