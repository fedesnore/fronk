/* Scalar numerical primitives, also tested independently of the MoE loop. */
#ifndef FR_Q4_H
#define FR_Q4_H
#include <stddef.h>
#include <stdint.h>
float fr_f16(uint16_t bits);
int fr_q4_dot(const unsigned char *q, const float *x, size_t n, double *out);
/* Round-to-nearest-even, signed [-127,127], fp32 scale, groups of 64.
 * A final 32-element group is equivalent to padding with 32 zeros. Input and
 * output may be identical. Requires finite input, FE_TONEAREST, n%32 == 0.
 */
int fr_q8_activation(const float *x, float *out, size_t n);
#endif
