/* gsq_multi.h - gate/up dot products for several tokens that share one expert.
 * The weights of a row are decoded once and applied to each token's q8_K
 * activations: bit-identical to ggml's single-token vec_dot (same integer sums,
 * same float order). AVX2 and NEON kernels; elsewhere it loops over ggml's own. */
#ifndef FR_GSQ_MULTI_H
#define FR_GSQ_MULTI_H
#include <stddef.h>

#define FR_GSQ_MULTI_MAX 16

/* s[t] = dot(row, y[t]) for t < nt; n = 2560 weights, type = an FR_GSQ IQ type. */
void fr_gsq_dot_multi(unsigned type, int n, float *s, const void *row, const void *const *y, unsigned nt);
#endif
