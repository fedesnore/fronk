/* gsq_down.c - the down projection of one GSQ slice: 2560 output rows, each a
 * 32-weight piece (IQ4_NL or planar Q2_0 half, see gsq.h), times the slice's 32
 * hidden activations h. h is quantized once to int8 (hq, scale dh; the Q8_0
 * scheme llama.cpp's CPU path uses), each row is an integer dot product, and
 * y[r] += scale * d_r * dot, with scale = gate * dh.
 * AVX2 handles 8 rows per step and NEON 4, so the horizontal sums are shared;
 * the scalar loop is the reference both are tested against. */
#include "gsq_down.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>

#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#define FR_GSQ_AVX2 1
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define FR_GSQ_NEON 1
#endif

static const int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

float fr_gsq_quantize_h(const float *h, int8_t *hq, int32_t *hsum) {
    float amax = 0;
    for (unsigned j = 0; j < 32; ++j) if (fabsf(h[j]) > amax) amax = fabsf(h[j]);
    const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
    int32_t s = 0;
    for (unsigned j = 0; j < 32; ++j) { hq[j] = (int8_t)roundf(h[j] * id); s += hq[j]; }
    *hsum = s;
    return d;
}

static float half_to_float(const unsigned char *p) { ggml_half h; memcpy(&h, p, 2); return GGML_FP16_TO_FP32(h); }

void fr_gsq_down_ref(unsigned type, const unsigned char *p, unsigned rows, const int8_t *hq, int32_t hsum,
                     float scale, float *y) {
    for (unsigned r = 0; r < rows; ++r) {
        int32_t s = 0;
        if (type == FR_GSQ_IQ4_NL) {
            const unsigned char *q = p + 2;
            for (unsigned j = 0; j < 16; ++j) s += kv_iq4nl[q[j] & 15] * hq[j] + kv_iq4nl[q[j] >> 4] * hq[j + 16];
            y[r] += scale * half_to_float(p) * (float)s;
            p += 18;
        } else {
            for (unsigned j = 0; j < 32; ++j) s += (int32_t)((p[2 + j % 8] >> (2 * (j / 8))) & 3) * hq[j];
            y[r] += scale * half_to_float(p) * (float)(s - hsum);
            p += 10;
        }
    }
}

#ifdef FR_GSQ_AVX2
/* Eight int32x8 partial sums -> one vector of the eight row totals. */
static inline __m256i sum8(const __m256i a[8]) {
    const __m256i t0 = _mm256_hadd_epi32(_mm256_hadd_epi32(a[0], a[1]), _mm256_hadd_epi32(a[2], a[3]));
    const __m256i t1 = _mm256_hadd_epi32(_mm256_hadd_epi32(a[4], a[5]), _mm256_hadd_epi32(a[6], a[7]));
    return _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20), _mm256_permute2x128_si256(t0, t1, 0x31));
}
static inline __m256 scales8(const unsigned char *p, size_t stride) {
    uint16_t h[8];
    for (unsigned i = 0; i < 8; ++i) memcpy(&h[i], p + i * stride, 2);
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)h));
}
#endif

void fr_gsq_down(unsigned type, const unsigned char *p, unsigned rows, const int8_t *hq, int32_t hsum,
                 float scale, float *y) {
    unsigned r = 0;
#ifdef FR_GSQ_AVX2
    const __m256i hv = _mm256_loadu_si256((const __m256i *)hq), ones = _mm256_set1_epi16(1);
    const __m256 sv = _mm256_set1_ps(scale);
    if (type == FR_GSQ_IQ4_NL) {
        const __m256i kv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)kv_iq4nl));
        const __m256i m4 = _mm256_set1_epi8(0x0F);
        for (; r + 8 <= rows; r += 8, p += 8 * 18, y += 8) {
            __m256i acc[8];
            for (unsigned i = 0; i < 8; ++i) {
                const __m128i q = _mm_loadu_si128((const __m128i *)(p + i * 18 + 2));
                const __m256i idx = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), m4);
                const __m256i w = _mm256_shuffle_epi8(kv, idx);
                acc[i] = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(hv, w)), ones);
            }
            const __m256 s = _mm256_cvtepi32_ps(sum8(acc));
            _mm256_storeu_ps(y, _mm256_fmadd_ps(_mm256_mul_ps(sv, scales8(p, 18)), s, _mm256_loadu_ps(y)));
        }
    } else {
        const __m256i shifts = _mm256_set_epi64x(6, 4, 2, 0), m2 = _mm256_set1_epi8(3);
        const __m256i hs = _mm256_set1_epi32(hsum);
        for (; r + 8 <= rows; r += 8, p += 8 * 10, y += 8) {
            __m256i acc[8];
            for (unsigned i = 0; i < 8; ++i) {
                uint64_t u; memcpy(&u, p + i * 10 + 2, 8);
                const __m256i c = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x((long long)u), shifts), m2);
                acc[i] = _mm256_madd_epi16(_mm256_maddubs_epi16(c, hv), ones);
            }
            const __m256 s = _mm256_cvtepi32_ps(_mm256_sub_epi32(sum8(acc), hs));
            _mm256_storeu_ps(y, _mm256_fmadd_ps(_mm256_mul_ps(sv, scales8(p, 10)), s, _mm256_loadu_ps(y)));
        }
    }
#elif defined(FR_GSQ_NEON)
    const int8x16_t h0 = vld1q_s8(hq), h1 = vld1q_s8(hq + 16);
    if (type == FR_GSQ_IQ4_NL) {
        const int8x16_t kv = vld1q_s8(kv_iq4nl);
        const uint8x16_t m4 = vdupq_n_u8(0x0F);
        for (; r < rows; ++r, p += 18, ++y) {
            const uint8x16_t q = vld1q_u8(p + 2);
            const int8x16_t lo = vqtbl1q_s8(kv, vandq_u8(q, m4)), hi = vqtbl1q_s8(kv, vshrq_n_u8(q, 4));
            const int32_t s = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, h0), hi, h1));
            *y += scale * half_to_float(p) * (float)s;
        }
    } else {
        const int64x2_t sh01 = {0, -2}, sh23 = {-4, -6};
        const uint8x16_t m2 = vdupq_n_u8(3);
        for (; r < rows; ++r, p += 10, ++y) {
            uint64_t u; memcpy(&u, p + 2, 8);
            const uint64x2_t v = vdupq_n_u64(u);
            const int8x16_t c01 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_u64(vshlq_u64(v, sh01)), m2));
            const int8x16_t c23 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_u64(vshlq_u64(v, sh23)), m2));
            const int32_t s = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), c01, h0), c23, h1));
            *y += scale * half_to_float(p) * (float)(s - hsum);
        }
    }
#endif
    if (r < rows) fr_gsq_down_ref(type, p, rows - r, hq, hsum, scale, y);
}
