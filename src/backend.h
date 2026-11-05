/* backend.h - fixed-shape compute contract (design section 7).
 * All calls are synchronous and externally serialized per backend. Requests
 * contain decoded fp32 activations; network fp16 conversion belongs elsewhere.
 * Callers own separate, valid req/out objects. Success replaces out completely;
 * failure leaves it unchanged. run allocates no memory and does no I/O.
 */
#ifndef FR_BACKEND_H
#define FR_BACKEND_H
#include "model.h"
#include "hw.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum { FR_BACKENDS_MAX=16, FR_BACKEND_NAME_MAX=15, FR_BACKEND_MODULE_API=1 };

enum { FR_OK=0, FR_INVALID=-1, FR_STATE=-2, FR_NUMERIC=-3,
       FR_IO=-4, FR_UNSUPPORTED=-5, FR_MISMATCH=-6 };

typedef struct {
    unsigned layer, burst;              /* layer 0..47, burst 1..8 */
    uint16_t expert[FR_MAX_BURST][FR_ACTIVE_ROUTED]; /* distinct ids 0..511 */
    float gate[FR_MAX_BURST][FR_ACTIVE_EXPERTS];    /* finite, >=0; shared last */
    float x[FR_MAX_BURST][FR_HIDDEN];     /* finite; unused tokens ignored */
} fr_req;
typedef struct {
    unsigned layer, burst;
    float y[FR_MAX_BURST][FR_HIDDEN];     /* unused tokens zeroed */
} fr_part;
typedef struct {
    unsigned burst_max;
    double t0_ns, bw_eff;               /* bytes/second; t = t0 + 1e9*bytes/bw */
    double ns[FR_MAX_BURST];
    uint64_t bytes[FR_MAX_BURST];
} fr_costmodel;

/* Legacy oracle ownership block: [layer][expert][gate, up, down], Q4_0.
 * Production GGUF layout and byte length are specified separately in native.h.
 * Gate/up: 32 rows x 2560 columns. Down: 2560 rows x 32 columns.
 * All rows are contiguous. Scale is little-endian fp16; low/high nibbles
 * contain columns 0..15 / 16..31. Negative scales are valid.
 */
#define FR_PROJECTION_BYTES ((size_t)FR_HIDDEN * FR_BLOCK_NEURONS / FR_Q4_WEIGHTS * FR_Q4_BYTES)
#define FR_SLICE_BYTES (3 * FR_PROJECTION_BYTES)
_Static_assert((uint64_t)FR_SLICE_BYTES * FR_LAYERS * FR_EXPERTS == FR_WEIGHT_BLOCK_BYTES,
               "backend layout must cover an ownership block");
_Static_assert(SIZE_MAX >= FR_WEIGHT_BLOCK_BYTES, "backends require a 64-bit address space");
static inline size_t fr_slice_offset(unsigned layer, unsigned expert) {
    return ((size_t)layer * FR_EXPERTS + expert) * FR_SLICE_BYTES;
}

typedef struct fr_backend {
    const char *name;
    int (*probe)(fr_hwinfo *out);
    /* Ref/AVX2 borrow immutable bytes until unload; len distinguishes the legacy
     * FR_WEIGHT_BLOCK_BYTES layout from native.h's FR_NATIVE_BLOCK_BYTES.
     * Loading an occupied slot or unloading an empty slot fails. No copy,
     * conversion or eager scan of multi-GB input. Caller validates file hash.
     * Referenced scales/arithmetic are checked by run, including zero gates.
     */
    int (*load)(int block, const void *data, size_t len);
    int (*unload)(int block);
    int (*run)(const fr_req *req, fr_part *out);
    int (*bench)(int burst_max, fr_costmodel *out);
    /* Requires no loaded blocks; always releases its temporary fixtures. */
    int (*self_test)(const char *vectors_dir);
} fr_backend;

extern const fr_backend fr_ref, fr_ref_q8;
#ifdef FR_USER_BACKEND
extern const fr_backend fr_user; /* backends/user.c; its name is its own */
#endif
#ifdef FR_BACKEND_MODULES
#define FR_BACKEND_MODULE(name) extern const fr_backend fr_module_##name;
#include "registry.h"
#undef FR_BACKEND_MODULE
#endif
const fr_backend *fr_backend_find(const char *name);
const fr_backend *fr_backend_at(unsigned index); /* NULL past the last compiled backend */
const char *fr_backends_compiled(void);
/* Setup-only validation: names are unique, tables complete, module names match. */
int fr_backends_validate(FILE *error);

typedef struct { double relative_l2, normalized_max, cosine; } fr_error;
typedef struct { double relative_l2, normalized_max, cosine_min; } fr_tolerance;
extern const fr_tolerance fr_reference_tolerance, fr_backend_tolerance;
/* Metrics compare one token's whole partial (2560 dimensions), in double.
 * Zero ref requires exactly zero candidate. Nonfinite input always fails.
 */
int fr_compare(const float *ref, const float *actual, size_t n, fr_error *out);
int fr_conforms(const fr_error *error, const fr_tolerance *limit);
#endif
