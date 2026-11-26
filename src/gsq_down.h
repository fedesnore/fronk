/* gsq_down.h - down projection kernels for GSQ slices (see gsq_down.c). */
#ifndef FR_GSQ_DOWN_H
#define FR_GSQ_DOWN_H
#include "gsq.h"
#include <stdint.h>

/* h[32] -> int8 hq[32] with scale (returned) and the sum of hq. */
float fr_gsq_quantize_h(const float *h, int8_t *hq, int32_t *hsum);
/* rows pieces of the given down type starting at p: y[r] += scale * d_r * (row . hq). */
void fr_gsq_down(unsigned type, const unsigned char *p, unsigned rows, const int8_t *hq, int32_t hsum,
                 float scale, float *y);
/* Scalar version of the same arithmetic: the reference for the SIMD paths. */
void fr_gsq_down_ref(unsigned type, const unsigned char *p, unsigned rows, const int8_t *hq, int32_t hsum,
                     float scale, float *y);
#endif
