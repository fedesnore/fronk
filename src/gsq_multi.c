/* gsq_multi.c - see gsq_multi.h.
 *
 * Every GSQ gate/up type (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) has the same
 * shape in ggml's AVX2 kernel: for each group of 32 weights it builds an unsigned
 * grid vector, a sign vector and a 16-lane scale vector, then for the activations
 * does sign -> maddubs -> madd with the scale, summing int32 per 256-weight block
 * and folding each block into a float accumulator with d_x * d_y.
 *
 * Here a group is decoded once (grid lookups, sign bits, scales) and applied to
 * up to four tokens, each with its own accumulators. The integer sums per block
 * are the same (integer addition is exact in any order), the float steps are the
 * same, so each token's result equals ggml's single-token dot bit for bit
 * (checked by tools/gsq-multicheck.c).
 *
 * ggml's NEON kernels have another shape, so the NEON branch below follows that
 * one and is equal bit for bit to ggml's NEON dot, not to the AVX2 one. */
#include "gsq.h"
#include "gsq_multi.h"

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include "ggml-impl.h"
#include "ggml-cpu/simd-mappings.h"

#include <string.h>

typedef void (*dot1_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
static dot1_fn single(unsigned type) {
    switch (type) {
        case FR_GSQ_IQ2_XXS: return ggml_vec_dot_iq2_xxs_q8_K;
        case FR_GSQ_IQ2_XS:  return ggml_vec_dot_iq2_xs_q8_K;
        case FR_GSQ_IQ2_S:   return ggml_vec_dot_iq2_s_q8_K;
        case FR_GSQ_IQ3_XXS: return ggml_vec_dot_iq3_xxs_q8_K;
        case FR_GSQ_IQ3_S:   return ggml_vec_dot_iq3_s_q8_K;
        default:             return NULL;
    }
}

#if defined(__AVX2__)
#include <immintrin.h>
#include "gsq_signs.inc"

typedef struct { __m256i q, s, sc; } group;

static inline float hsum8(__m256 x) {  /* ggml's hsum_float_8, same order */
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}
static inline __m256i scale2(unsigned lo, unsigned hi) {  /* 8 lanes for each half of the group */
    return _mm256_set_m128i(_mm_set1_epi16((short)(2 * hi + 1)), _mm_set1_epi16((short)(2 * lo + 1)));
}
/* signs packed as 32 bits, one per weight (IQ2_S, IQ3_S): -1/+1 per byte */
static inline __m256i bit_signs(uint32_t bits) {
    static const uint8_t k1[32] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3};
    static const uint8_t k2[32] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const __m256i m1 = _mm256_loadu_si256((const __m256i *)k1), m2 = _mm256_loadu_si256((const __m256i *)k2);
    __m256i a = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), m1), m2);
    return _mm256_or_si256(_mm256_cmpeq_epi8(a, m2), _mm256_set1_epi8(1));
}
static inline __m256i even_signs(uint32_t a) {  /* four 7-bit sign indices (IQ2_XXS, IQ2_XS, IQ3_XXS) */
    const uint64_t *s64 = (const uint64_t *)keven_signs_q2xs;
    return _mm256_set_epi64x((long long)s64[(a >> 21) & 127], (long long)s64[(a >> 14) & 127],
                             (long long)s64[(a >> 7) & 127], (long long)s64[a & 127]);
}

static inline group dec_iq2_xxs(const block_iq2_xxs *x, unsigned g) {
    uint32_t a[2]; memcpy(a, x->qs + 4 * g, 8);
    const uint8_t *b = (const uint8_t *)a;
    group r;
    r.q = _mm256_set_epi64x((long long)iq2xxs_grid[b[3]], (long long)iq2xxs_grid[b[2]], (long long)iq2xxs_grid[b[1]], (long long)iq2xxs_grid[b[0]]);
    r.s = even_signs(a[1]);
    r.sc = _mm256_set1_epi16((short)(2 * (a[1] >> 28) + 1));
    return r;
}
static inline group dec_iq2_xs(const block_iq2_xs *x, unsigned g) {
    const uint16_t *q = x->qs + 4 * g;
    group r;
    r.q = _mm256_set_epi64x((long long)iq2xs_grid[q[3] & 511], (long long)iq2xs_grid[q[2] & 511],
                            (long long)iq2xs_grid[q[1] & 511], (long long)iq2xs_grid[q[0] & 511]);
    const uint64_t *s64 = (const uint64_t *)keven_signs_q2xs;
    r.s = _mm256_set_epi64x((long long)s64[q[3] >> 9], (long long)s64[q[2] >> 9], (long long)s64[q[1] >> 9], (long long)s64[q[0] >> 9]);
    r.sc = scale2(x->scales[g] & 15, x->scales[g] >> 4);
    return r;
}
static inline group dec_iq2_s(const block_iq2_s *x, unsigned g) {
    const uint8_t *qs = x->qs + 4 * g; const unsigned qh = x->qh[g];
    group r;
    r.q = _mm256_set_epi64x((long long)iq2s_grid[qs[3] | ((qh << 2) & 0x300)], (long long)iq2s_grid[qs[2] | ((qh << 4) & 0x300)],
                            (long long)iq2s_grid[qs[1] | ((qh << 6) & 0x300)], (long long)iq2s_grid[qs[0] | ((qh << 8) & 0x300)]);
    const uint16_t *sg = (const uint16_t *)(x->qs + QK_K / 8) + 2 * g;
    r.s = bit_signs(sg[0] | ((uint32_t)sg[1] << 16));
    r.sc = scale2(x->scales[g] & 15, x->scales[g] >> 4);
    return r;
}
static inline group dec_iq3_xxs(const block_iq3_xxs *x, unsigned g) {
    const uint8_t *q3 = x->qs + 8 * g; uint32_t a; memcpy(&a, x->qs + QK_K / 4 + 4 * g, 4);
    group r;
    r.q = _mm256_set_epi32((int)iq3xxs_grid[q3[7]], (int)iq3xxs_grid[q3[6]], (int)iq3xxs_grid[q3[5]], (int)iq3xxs_grid[q3[4]],
                           (int)iq3xxs_grid[q3[3]], (int)iq3xxs_grid[q3[2]], (int)iq3xxs_grid[q3[1]], (int)iq3xxs_grid[q3[0]]);
    r.s = even_signs(a);
    r.sc = _mm256_set1_epi16((short)(2 * (a >> 28) + 1));
    return r;
}
static inline group dec_iq3_s(const block_iq3_s *x, unsigned g) {
    const uint8_t *qs = x->qs + 8 * g; const unsigned qh = x->qh[g];
    group r;
#define I3(j) (int)iq3s_grid[qs[j] | ((qh << (8 - (j))) & 256)]
    r.q = _mm256_set_epi32(I3(7), I3(6), I3(5), I3(4), I3(3), I3(2), I3(1), I3(0));
#undef I3
    const uint16_t *sg = (const uint16_t *)x->signs + 2 * g;
    r.s = bit_signs(sg[0] | ((uint32_t)sg[1] << 16));
    const unsigned ls = (x->scales[g / 2] >> (4 * (g & 1))) & 15;
    r.sc = _mm256_set1_epi16((short)(2 * ls + 1));
    return r;
}

static inline __m256i apply(group G, const int8_t *q8) {
    const __m256i v = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)q8), G.s);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(G.q, v), G.sc);
}

/* NT tokens (1..4, a constant after inlining) against one row */
#define MULTI(NAME, BT, DEC, FACTOR)                                                                  \
static inline __attribute__((always_inline)) void NAME##_n(int n, float *s, const BT *x,            \
                                                           const block_q8_K *const *y, const unsigned NT) { \
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()}; \
    for (int i = 0; i < n / QK_K; ++i) {                                                              \
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);                                               \
        __m256i sum[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()}; \
        for (unsigned g = 0; g < QK_K / 32; ++g) {                                                    \
            const group G = DEC(&x[i], g);                                                            \
            for (unsigned t = 0; t < NT; ++t) sum[t] = _mm256_add_epi32(sum[t], apply(G, y[t][i].qs + 32 * g)); \
        }                                                                                             \
        for (unsigned t = 0; t < NT; ++t)                                                             \
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sum[t]), acc[t]); \
    }                                                                                                 \
    for (unsigned t = 0; t < NT; ++t) s[t] = (FACTOR) * hsum8(acc[t]);                                \
}                                                                                                     \
static void NAME(int n, float *s, const void *vx, const block_q8_K *const *y, unsigned nt) {          \
    const BT *x = vx;                                                                                 \
    switch (nt) {                                                                                     \
        case 2: NAME##_n(n, s, x, y, 2); break;                                                       \
        case 3: NAME##_n(n, s, x, y, 3); break;                                                       \
        default: NAME##_n(n, s, x, y, 4); break;                                                      \
    }                                                                                                 \
}
MULTI(multi_iq2_xxs, block_iq2_xxs, dec_iq2_xxs, 0.125f)
MULTI(multi_iq2_xs,  block_iq2_xs,  dec_iq2_xs,  0.125f)
MULTI(multi_iq2_s,   block_iq2_s,   dec_iq2_s,   0.125f)
MULTI(multi_iq3_xxs, block_iq3_xxs, dec_iq3_xxs, 0.25f)
MULTI(multi_iq3_s,   block_iq3_s,   dec_iq3_s,   1.0f)
#undef MULTI

typedef void (*multi_fn)(int, float *, const void *, const block_q8_K *const *, unsigned);
static multi_fn multi(unsigned type) {
    switch (type) {
        case FR_GSQ_IQ2_XXS: return multi_iq2_xxs;
        case FR_GSQ_IQ2_XS:  return multi_iq2_xs;
        case FR_GSQ_IQ2_S:   return multi_iq2_s;
        case FR_GSQ_IQ3_XXS: return multi_iq3_xxs;
        case FR_GSQ_IQ3_S:   return multi_iq3_s;
        default:             return NULL;
    }
}

void fr_gsq_dot_multi(unsigned type, int n, float *s, const void *row, const void *const *y, unsigned nt) {
    if (nt == 1) { single(type)(n, s, 0, row, 0, y[0], 0, 1); return; }  /* ggml's own is tuned for one */
    const multi_fn f = multi(type);
    for (unsigned t = 0; t < nt; t += 4) {
        const unsigned k = nt - t < 4 ? nt - t : 4;
        if (k == 1) single(type)(n, s + t, 0, row, 0, y[t], 0, 1);
        else f(n, s + t, row, (const block_q8_K *const *)(y + t), k);
    }
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#include "gsq_signs.inc"

/* ggml's NEON kernels are not the AVX2 ones in other clothes: they multiply the
 * signs into the grid bytes and take signed dot products of 16 weights
 * (vdotq_s32), 64 weights (two groups) per step. So here a step is decoded once
 * into 64 signed weights plus its scales, and each token only loads its
 * activations and does the dots. What ggml then does with the dots depends on
 * the type and is kept as it is:
 *
 *   IQ2_XS, IQ2_S, IQ3_S: integers. Each dot is multiplied by an odd scale and
 *     added into one int per 256-weight block, then sumf += d * int. Integer
 *     sums are exact in any order, so they stay in a vector until the block ends.
 *   IQ2_XXS, IQ3_XXS: floats. In a block sumf1 takes the even groups and sumf2
 *     the odd ones, each as sumf1 += int * (0.5f + scale), then
 *     sumf += d * (sumf1 + sumf2). Same accumulators, same order here.
 *
 * The float statements are written as ggml writes them. ggml's results depend on
 * the compiler turning sumf += d * v into one fused multiply-add (it does with
 * the flags of this library; gsq-multicheck fails if only one side is fused), so
 * this file must be compiled with the flags of ggml's quants.c, as the Makefile
 * does, and checked with gsq-multicheck on each compiler. */
typedef struct { int8x16_t w[4]; int32x4_t sc; } pair_i;  /* sc: the scale of each pair of dot lanes */
typedef struct { int8x16_t w[4]; float ls[2]; } pair_f;   /* ls: the float scale of each group */

static inline int8x16_t two8(const void *a, const void *b) {  /* two 8-byte table entries */
    return vcombine_s8(vld1_s8((const int8_t *)a), vld1_s8((const int8_t *)b));
}
static inline int8x16_t even2(unsigned i, unsigned j) {  /* two 7-bit sign indices (IQ2_XXS, IQ2_XS, IQ3_XXS) */
    return two8(keven_signs_q2xs + 8 * i, keven_signs_q2xs + 8 * j);
}
static inline int8x16_t grid4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {  /* four 4-byte grid entries */
    const uint32x4_t v = ggml_vld1q_u32(a, b, c, d);
    return vreinterpretq_s8_u32(v);
}
/* Signs packed as bits, one per weight (IQ2_S, IQ3_S): the 64 of a step, applied
 * to its four vectors of grid bytes. Bit set: the weight is negative. */
static inline void bit_signs(const uint8_t *bits, int8x16_t w[4]) {
    static const uint8_t k1[64] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,
                                   4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7};
    static const uint8_t k2[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x16_t v = vcombine_u8(vld1_u8(bits), vdup_n_u8(0)), m2 = vld1q_u8(k2);
    for (unsigned k = 0; k < 4; ++k) {
        const int8x16_t neg = vreinterpretq_s8_u8(vtstq_u8(ggml_vqtbl1q_u8(v, vld1q_u8(k1 + 16 * k)), m2));
        w[k] = vsubq_s8(veorq_s8(w[k], neg), neg);  /* -w where the bit is set */
    }
}
/* The scales of two groups, a low and a high nibble each: one per 16 weights. */
static inline int32x4_t scale4(unsigned a, unsigned b) {
    const int32_t v[4] = {(int32_t)(2 * (a & 15) + 1), (int32_t)(2 * (a >> 4) + 1), (int32_t)(2 * (b & 15) + 1), (int32_t)(2 * (b >> 4) + 1)};
    return vld1q_s32(v);
}

/* Decoders. h: which 64 weights (groups 2h and 2h+1) of the block. */
static inline pair_f dec_iq2_xxs(const block_iq2_xxs *x, unsigned h) {
    uint32_t a[4]; memcpy(a, x->qs + 8 * h, 16);  /* per group: four grid indices, then signs and scale */
    const uint8_t *b = (const uint8_t *)a;
    pair_f r;
    r.w[0] = vmulq_s8(two8(iq2xxs_grid + b[0], iq2xxs_grid + b[1]), even2(a[1] & 127, (a[1] >> 7) & 127));
    r.w[1] = vmulq_s8(two8(iq2xxs_grid + b[2], iq2xxs_grid + b[3]), even2((a[1] >> 14) & 127, (a[1] >> 21) & 127));
    r.w[2] = vmulq_s8(two8(iq2xxs_grid + b[8], iq2xxs_grid + b[9]), even2(a[3] & 127, (a[3] >> 7) & 127));
    r.w[3] = vmulq_s8(two8(iq2xxs_grid + b[10], iq2xxs_grid + b[11]), even2((a[3] >> 14) & 127, (a[3] >> 21) & 127));
    r.ls[0] = 0.5f + (a[1] >> 28);
    r.ls[1] = 0.5f + (a[3] >> 28);
    return r;
}
static inline pair_i dec_iq2_xs(const block_iq2_xs *x, unsigned h) {
    uint16_t q[8]; memcpy(q, x->qs + 8 * h, 16);  /* 9 bits of grid index, 7 of sign index */
    pair_i r;
    for (unsigned k = 0; k < 4; ++k)
        r.w[k] = vmulq_s8(two8(iq2xs_grid + (q[2 * k] & 511), iq2xs_grid + (q[2 * k + 1] & 511)), even2(q[2 * k] >> 9, q[2 * k + 1] >> 9));
    r.sc = scale4(x->scales[2 * h], x->scales[2 * h + 1]);
    return r;
}
static inline pair_i dec_iq2_s(const block_iq2_s *x, unsigned h) {
    const uint8_t *qs = x->qs + 8 * h; const unsigned q0 = x->qh[2 * h], q1 = x->qh[2 * h + 1];
    pair_i r;
    r.w[0] = two8(iq2s_grid + (qs[0] | ((q0 << 8) & 0x300)), iq2s_grid + (qs[1] | ((q0 << 6) & 0x300)));
    r.w[1] = two8(iq2s_grid + (qs[2] | ((q0 << 4) & 0x300)), iq2s_grid + (qs[3] | ((q0 << 2) & 0x300)));
    r.w[2] = two8(iq2s_grid + (qs[4] | ((q1 << 8) & 0x300)), iq2s_grid + (qs[5] | ((q1 << 6) & 0x300)));
    r.w[3] = two8(iq2s_grid + (qs[6] | ((q1 << 4) & 0x300)), iq2s_grid + (qs[7] | ((q1 << 2) & 0x300)));
    bit_signs(x->qs + QK_K / 8 + 8 * h, r.w);
    r.sc = scale4(x->scales[2 * h], x->scales[2 * h + 1]);
    return r;
}
static inline pair_f dec_iq3_xxs(const block_iq3_xxs *x, unsigned h) {
    uint64_t q3[2]; memcpy(q3, x->qs + 16 * h, 16);  /* 16 grid indices: two loads, not 16 */
    uint32_t a[2]; memcpy(a, x->qs + QK_K / 4 + 8 * h, 8);  /* per group: signs and scale */
    pair_f r;
#define G3(j) iq3xxs_grid[(q3[(j) / 8] >> (8 * ((j) & 7))) & 255]
    r.w[0] = vmulq_s8(even2(a[0] & 127, (a[0] >> 7) & 127), grid4(G3(0), G3(1), G3(2), G3(3)));
    r.w[1] = vmulq_s8(even2((a[0] >> 14) & 127, (a[0] >> 21) & 127), grid4(G3(4), G3(5), G3(6), G3(7)));
    r.w[2] = vmulq_s8(even2(a[1] & 127, (a[1] >> 7) & 127), grid4(G3(8), G3(9), G3(10), G3(11)));
    r.w[3] = vmulq_s8(even2((a[1] >> 14) & 127, (a[1] >> 21) & 127), grid4(G3(12), G3(13), G3(14), G3(15)));
#undef G3
    r.ls[0] = 0.5f + (a[0] >> 28);
    r.ls[1] = 0.5f + (a[1] >> 28);
    return r;
}
static inline pair_i dec_iq3_s(const block_iq3_s *x, unsigned h) {
    /* sixteen 9-bit grid indices, a byte of qs and one bit of the group's qh, put
     * together as ggml does and read back from two general registers per group */
    static const int16_t k_shift[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    const uint8x16_t lo = vld1q_u8(x->qs + 16 * h);
    const int16x8_t shift = vld1q_s16(k_shift);
    const uint16x8_t m256 = vdupq_n_u16(256);
    const uint64x2_t i0 = vreinterpretq_u64_u16(vorrq_u16(vmovl_u8(vget_low_u8(lo)), vandq_u16(vshlq_u16(vdupq_n_u16(x->qh[2 * h]), shift), m256)));
    const uint64x2_t i1 = vreinterpretq_u64_u16(vorrq_u16(vmovl_u8(vget_high_u8(lo)), vandq_u16(vshlq_u16(vdupq_n_u16(x->qh[2 * h + 1]), shift), m256)));
    const uint64_t a = vgetq_lane_u64(i0, 0), b = vgetq_lane_u64(i0, 1), c = vgetq_lane_u64(i1, 0), d = vgetq_lane_u64(i1, 1);
    pair_i r;
#define I3(v, j) iq3s_grid[((v) >> (16 * (j))) & 0xffff]
    r.w[0] = grid4(I3(a, 0), I3(a, 1), I3(a, 2), I3(a, 3));
    r.w[1] = grid4(I3(b, 0), I3(b, 1), I3(b, 2), I3(b, 3));
    r.w[2] = grid4(I3(c, 0), I3(c, 1), I3(c, 2), I3(c, 3));
    r.w[3] = grid4(I3(d, 0), I3(d, 1), I3(d, 2), I3(d, 3));
#undef I3
    bit_signs(x->signs + 8 * h, r.w);
    const int32_t sa = 2 * (x->scales[h] & 15) + 1, sb = 2 * (x->scales[h] >> 4) + 1;  /* one scale per group */
    const int32_t sc[4] = {sa, sa, sb, sb};
    r.sc = vld1q_s32(sc);
    return r;
}

/* One token against a decoded step. Four dots of 16 weights, each with its scale
 * (IQ2_XS, IQ2_S): the pairwise adds leave one dot per lane. */
static inline int32x4_t apply_i4(int32x4_t sum, pair_i G, const int8_t *q8) {
    const int32x4_t z = vdupq_n_s32(0);
    const int32x4_t p1 = ggml_vdotq_s32(z, G.w[0], vld1q_s8(q8)), p2 = ggml_vdotq_s32(z, G.w[1], vld1q_s8(q8 + 16));
    const int32x4_t p3 = ggml_vdotq_s32(z, G.w[2], vld1q_s8(q8 + 32)), p4 = ggml_vdotq_s32(z, G.w[3], vld1q_s8(q8 + 48));
    return vmlaq_s32(sum, vpaddq_s32(vpaddq_s32(p1, p2), vpaddq_s32(p3, p4)), G.sc);
}
/* Two dots of 32 weights, each with its scale (IQ3_S): half a dot per lane. */
static inline int32x4_t apply_i2(int32x4_t sum, pair_i G, const int8_t *q8) {
    const int32x4_t z = vdupq_n_s32(0);
    const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(z, G.w[0], vld1q_s8(q8)), G.w[1], vld1q_s8(q8 + 16));
    const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(z, G.w[2], vld1q_s8(q8 + 32)), G.w[3], vld1q_s8(q8 + 48));
    return vmlaq_s32(sum, vpaddq_s32(p1, p2), G.sc);
}

#define MULTI_BY_COUNT(NAME, BT)                                                                      \
static void NAME(int n, float *s, const void *vx, const block_q8_K *const *y, unsigned nt) {          \
    const BT *x = vx;                                                                                 \
    switch (nt) {                                                                                     \
        case 1: NAME##_n(n, s, x, y, 1); break;                                                       \
        case 2: NAME##_n(n, s, x, y, 2); break;                                                       \
        case 3: NAME##_n(n, s, x, y, 3); break;                                                       \
        default: NAME##_n(n, s, x, y, 4); break;                                                      \
    }                                                                                                 \
}
/* NT tokens (1..4, a constant after inlining) against one row: the integer types.
 * LAST is ggml's closing statement. */
#define MULTI_I(NAME, BT, DEC, APPLY, LAST)                                                           \
static inline __attribute__((always_inline)) void NAME##_n(int n, float *s, const BT *x,            \
                                                           const block_q8_K *const *y, const unsigned NT) { \
    float sumf[4] = {0, 0, 0, 0};                                                                     \
    for (int i = 0; i < n / QK_K; ++i) {                                                              \
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);                                               \
        int32x4_t sum[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};          \
        for (unsigned h = 0; h < QK_K / 64; ++h) {                                                    \
            const pair_i G = DEC(&x[i], h);                                                           \
            for (unsigned t = 0; t < NT; ++t) sum[t] = APPLY(sum[t], G, y[t][i].qs + 64 * h);         \
        }                                                                                             \
        for (unsigned t = 0; t < NT; ++t) {                                                           \
            const float d = dx * y[t][i].d;                                                           \
            sumf[t] += d * vaddvq_s32(sum[t]);                                                        \
        }                                                                                             \
    }                                                                                                 \
    for (unsigned t = 0; t < NT; ++t) s[t] = LAST(sumf[t]);                                           \
}                                                                                                     \
MULTI_BY_COUNT(NAME, BT)
/* The same for the types whose group sums go through floats. */
#define MULTI_F(NAME, BT, DEC, LAST)                                                                  \
static inline __attribute__((always_inline)) void NAME##_n(int n, float *s, const BT *x,            \
                                                           const block_q8_K *const *y, const unsigned NT) { \
    float sumf[4] = {0, 0, 0, 0};                                                                     \
    for (int i = 0; i < n / QK_K; ++i) {                                                              \
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);                                               \
        float sumf1[4] = {0, 0, 0, 0}, sumf2[4] = {0, 0, 0, 0};                                       \
        for (unsigned h = 0; h < QK_K / 64; ++h) {                                                    \
            const pair_f G = DEC(&x[i], h);                                                           \
            for (unsigned t = 0; t < NT; ++t) {                                                       \
                const int8_t *q8 = y[t][i].qs + 64 * h;                                               \
                const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), G.w[0], vld1q_s8(q8)), G.w[1], vld1q_s8(q8 + 16)); \
                const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), G.w[2], vld1q_s8(q8 + 32)), G.w[3], vld1q_s8(q8 + 48)); \
                sumf1[t] += vaddvq_s32(p1) * G.ls[0];                                                 \
                sumf2[t] += vaddvq_s32(p2) * G.ls[1];                                                 \
            }                                                                                         \
        }                                                                                             \
        for (unsigned t = 0; t < NT; ++t) {                                                           \
            const float d = dx * y[t][i].d;                                                           \
            sumf[t] += d * (sumf1[t] + sumf2[t]);                                                     \
        }                                                                                             \
    }                                                                                                 \
    for (unsigned t = 0; t < NT; ++t) s[t] = LAST(sumf[t]);                                           \
}                                                                                                     \
MULTI_BY_COUNT(NAME, BT)
#define LAST_QUARTER(v) (0.25f * (v))
#define LAST_EIGHTH(v) (0.125f * (v))
#define LAST_HALF(v) (0.5f * (v))
#define LAST_AS_IS(v) (v)
MULTI_F(multi_iq2_xxs, block_iq2_xxs, dec_iq2_xxs, LAST_QUARTER)
MULTI_I(multi_iq2_xs,  block_iq2_xs,  dec_iq2_xs,  apply_i4, LAST_EIGHTH)
MULTI_I(multi_iq2_s,   block_iq2_s,   dec_iq2_s,   apply_i4, LAST_EIGHTH)
MULTI_F(multi_iq3_xxs, block_iq3_xxs, dec_iq3_xxs, LAST_HALF)
MULTI_I(multi_iq3_s,   block_iq3_s,   dec_iq3_s,   apply_i2, LAST_AS_IS)
#undef MULTI_I
#undef MULTI_F
#undef MULTI_BY_COUNT

typedef void (*multi_fn)(int, float *, const void *, const block_q8_K *const *, unsigned);
static multi_fn multi(unsigned type) {
    switch (type) {
        case FR_GSQ_IQ2_XXS: return multi_iq2_xxs;
        case FR_GSQ_IQ2_XS:  return multi_iq2_xs;
        case FR_GSQ_IQ2_S:   return multi_iq2_s;
        case FR_GSQ_IQ3_XXS: return multi_iq3_xxs;
        case FR_GSQ_IQ3_S:   return multi_iq3_s;
        default:             return NULL;
    }
}
/* A single token: ggml's kernel where it is the faster one. For IQ2_S, IQ3_XXS and
 * IQ3_S the decoders above are (M4 Pro and M1 Pro, 2026-11-10), with the same bits. */
static int own_for_one(unsigned type) {
    return type == FR_GSQ_IQ2_S || type == FR_GSQ_IQ3_XXS || type == FR_GSQ_IQ3_S;
}

void fr_gsq_dot_multi(unsigned type, int n, float *s, const void *row, const void *const *y, unsigned nt) {
    const multi_fn f = multi(type);
    const dot1_fn one = own_for_one(type) ? NULL : single(type);
    for (unsigned t = 0; t < nt; t += 4) {
        const unsigned k = nt - t < 4 ? nt - t : 4;
        if (k == 1 && one) one(n, s + t, 0, row, 0, y[t], 0, 1);
        else f(n, s + t, row, (const block_q8_K *const *)(y + t), k);
    }
}
#else
void fr_gsq_dot_multi(unsigned type, int n, float *s, const void *row, const void *const *y, unsigned nt) {
    const dot1_fn f = single(type);
    for (unsigned t = 0; t < nt; ++t) f(n, s + t, 0, row, 0, y[t], 0, 1);
}
#endif
