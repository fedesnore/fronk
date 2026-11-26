/* gsq-multicheck [SLICES|-] [SYNTHETIC_PER_LAYER] - check the multi-token gate/up
 * kernels (src/gsq_multi.c) against ggml's single-token dot, bit for bit.
 *
 *   SLICES: the 480 real slices that gsq-nodebench dump writes (10 per layer);
 *           "-" or nothing: synthetic slices only.
 *   SYNTHETIC_PER_LAYER: random slices per layer from fr_gsq_fill_slice (default 10):
 *           every grid index, sign pattern and scale turns up, which real weights
 *           do not guarantee. Every other one gets the largest scale everywhere.
 *
 *   1. every gate/up row of every slice, for nt = 2, 3, 4, 5, 7, 16 tokens taken
 *      from 16 different activations in two different orders:
 *      fr_gsq_dot_multi(nt)[t] == ggml_vec_dot_<type>_q8_K for that token alone.
 *      The reference is ggml's function itself, not fr_gsq_dot_multi(1). The
 *      activations are quantized vectors of several magnitudes, raw Q8_K blocks
 *      at the limits (all 127, all -127, alternating, any byte, zero) and three
 *      made for the row: 127 or near it with the sign of each weight, or the
 *      opposite one. Those give the largest sums a row can produce; with the
 *      largest scales the float sums of IQ2_XXS and IQ3_XXS stop being exact, so
 *      a kernel that adds them in another order than ggml is caught.
 *   2. every slice: fr_gsq_fast_slice_multi == fr_gsq_fast_slice per token.
 *   3. time per token-slice on one thread: single vs multi at nt = 2, 3, 4.
 * One line per type and nt; exit 1 on any difference.
 *
 * Build against the library the binaries link (after make), in the source folder:
 *   cc -O2 -w -o build/gsq-multicheck tools/gsq-multicheck.c build/gsq/$(uname -m)/libfronk-gsq.a -lm -lpthread */
#include "../src/gsq.h"
#include "../src/gsq_multi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REAL_PER_LAYER 10
#define NTOK 16
#define QTOK 8 /* tokens 0..7 come from fr_gsq_quantize_x and are the ones used for slices */
#define ROWTOK 13 /* tokens 13..15 are made again for every row */

typedef void dot1(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void deq(const void *, float *, int64_t);
dot1 ggml_vec_dot_iq2_xxs_q8_K, ggml_vec_dot_iq2_xs_q8_K, ggml_vec_dot_iq2_s_q8_K, ggml_vec_dot_iq3_xxs_q8_K, ggml_vec_dot_iq3_s_q8_K;
deq dequantize_row_iq2_xxs, dequantize_row_iq2_xs, dequantize_row_iq2_s, dequantize_row_iq3_xxs, dequantize_row_iq3_s;
static const struct { unsigned type; const char *name; dot1 *dot; deq *weights; } types[] = {
    {FR_GSQ_IQ2_XXS, "IQ2_XXS", ggml_vec_dot_iq2_xxs_q8_K, dequantize_row_iq2_xxs},
    {FR_GSQ_IQ2_XS, "IQ2_XS", ggml_vec_dot_iq2_xs_q8_K, dequantize_row_iq2_xs},
    {FR_GSQ_IQ2_S, "IQ2_S", ggml_vec_dot_iq2_s_q8_K, dequantize_row_iq2_s},
    {FR_GSQ_IQ3_XXS, "IQ3_XXS", ggml_vec_dot_iq3_xxs_q8_K, dequantize_row_iq3_xxs},
    {FR_GSQ_IQ3_S, "IQ3_S", ggml_vec_dot_iq3_s_q8_K, dequantize_row_iq3_s}};
enum { NTYPES = sizeof(types) / sizeof(types[0]) };
static const unsigned nts[] = {2, 3, 4, 5, 7, 16};
enum { NNT = sizeof(nts) / sizeof(nts[0]) };

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint32_t rnd_state = 0x9e3779b9u;
static uint32_t rnd(void) { rnd_state ^= rnd_state << 13; rnd_state ^= rnd_state >> 17; rnd_state ^= rnd_state << 5; return rnd_state; }
static float unit(void) { return (float)(rnd() >> 8) / 16777216.0f - 0.5f; }

/* Q8_K block: float d, 256 int8, 16 int16 sums of 16 */
static unsigned char xq[NTOK][FR_GSQ_Q8K_BYTES];
static void raw_token(unsigned t, int kind) {
    for (unsigned b = 0; b < FR_GSQ_HIDDEN / 256; ++b) {
        unsigned char *p = xq[t] + b * 292;
        float d = kind == 4 ? 0.0f : unit() * (b & 1 ? 0.03f : 3.0f);
        if (kind < 3) d = 0.0123f * (float)(b + 1);
        memcpy(p, &d, 4);
        int8_t *q = (int8_t *)p + 4;
        for (unsigned k = 0; k < 256; ++k)
            q[k] = kind == 0 ? 127 : kind == 1 ? -127 : kind == 2 ? (k & 1 ? -127 : 127) : kind == 4 ? 0 : (int8_t)(rnd() >> 24);
        for (unsigned g = 0; g < 16; ++g) {
            int16_t sum = 0;
            for (unsigned k = 0; k < 16; ++k) sum = (int16_t)(sum + q[16 * g + k]);
            memcpy(p + 260 + 2 * g, &sum, 2);
        }
    }
}
static void make_tokens(void) {
    static const float mag[QTOK] = {4.0f, 4.0f, 4.0f, 0.01f, 1000.0f, 1.0f, 60.0f, 0.3f};
    for (unsigned t = 0; t < QTOK; ++t) {
        float x[FR_GSQ_HIDDEN];
        for (unsigned d = 0; d < FR_GSQ_HIDDEN; ++d) x[d] = unit() * mag[t];
        if (t >= 5) for (unsigned k = 0; k < 40; ++k) x[rnd() % FR_GSQ_HIDDEN] *= 25.0f;  /* outliers, as activations have */
        fr_gsq_quantize_x(x, xq[t]);
    }
    for (unsigned t = QTOK; t < ROWTOK; ++t) raw_token(t, (int)(t - QTOK));
}

/* Activations with the sign of each weight of this row (or the opposite one). */
static void row_tokens(unsigned ti, const unsigned char *row) {
    static float w[FR_GSQ_HIDDEN];
    types[ti].weights(row, w, FR_GSQ_HIDDEN);
    for (unsigned t = ROWTOK; t < NTOK; ++t)
        for (unsigned b = 0; b < FR_GSQ_HIDDEN / 256; ++b) {
            unsigned char *p = xq[t] + b * 292;
            const float d = t == ROWTOK ? 0.02f : unit() * 0.5f;
            memcpy(p, &d, 4);
            int8_t *q = (int8_t *)p + 4;
            for (unsigned k = 0; k < 256; ++k) {
                const int mag = t == ROWTOK + 1 ? 100 + (int)(rnd() % 28) : 127, neg = w[256 * b + k] < 0;
                q[k] = (int8_t)((t == ROWTOK + 2) != neg ? -mag : mag);
            }
        }
}

static unsigned long rows_of[NTYPES], results[NTYPES][NNT], differ[NTYPES][NNT], one_differ[NTYPES], shown;
static void check_row(unsigned ti, const unsigned char *row, unsigned l, unsigned r) {
    float ref[NTOK];
    const void *xp[NTOK];
    row_tokens(ti, row);
    for (unsigned t = 0; t < NTOK; ++t) {
        float one;
        types[ti].dot(FR_GSQ_HIDDEN, &ref[t], 0, row, 0, xq[t], 0, 1);
        xp[0] = xq[t];
        fr_gsq_dot_multi(types[ti].type, FR_GSQ_HIDDEN, &one, row, xp, 1);
        if (memcmp(&one, &ref[t], 4)) ++one_differ[ti];
    }
    for (unsigned k = 0; k < NNT; ++k)
        for (unsigned order = 0; order < 2; ++order) {
            unsigned pick[NTOK];
            float m[NTOK];
            for (unsigned t = 0; t < nts[k]; ++t) {
                pick[t] = order ? (11 * t + 5 + r) % NTOK : t;  /* 11 and 16 are coprime: no token twice */
                xp[t] = xq[pick[t]];
                m[t] = -12345.0f;
            }
            fr_gsq_dot_multi(types[ti].type, FR_GSQ_HIDDEN, m, row, xp, nts[k]);
            for (unsigned t = 0; t < nts[k]; ++t) {
                ++results[ti][k];
                if (!memcmp(&m[t], &ref[pick[t]], 4)) continue;
                ++differ[ti][k];
                if (shown++ < 8)
                    fprintf(stderr, "layer %u %s row %u nt %u position %u activation %u: %.9g, ggml %.9g\n", l, types[ti].name, r, nts[k], t, pick[t], m[t], ref[pick[t]]);
            }
        }
    ++rows_of[ti];
}
static unsigned type_index(unsigned l) {
    for (unsigned ti = 0; ti < NTYPES; ++ti) if (types[ti].type == fr_gsq_gate_type[l]) return ti;
    fprintf(stderr, "layer %u: unknown gate type %u\n", l, fr_gsq_gate_type[l]); exit(2);
}

/* The largest scale in every group of the gate/up rows (ggml's block layouts). */
static void saturate(unsigned l, unsigned char *slice) {
    const size_t rb = fr_gsq_row_bytes(l), blk = rb / (FR_GSQ_HIDDEN / 256);
    for (size_t o = 0; o < 2 * FR_GSQ_NEURONS * rb; o += blk) {
        unsigned char *p = slice + o + 2;  /* after the fp16 scale of the block */
        switch (fr_gsq_gate_type[l]) {
            case FR_GSQ_IQ2_XXS: for (unsigned g = 0; g < 8; ++g) p[8 * g + 7] |= 0xf0; break;       /* top of each group's second word */
            case FR_GSQ_IQ3_XXS: for (unsigned g = 0; g < 8; ++g) p[64 + 4 * g + 3] |= 0xf0; break;  /* top of each group's sign word */
            case FR_GSQ_IQ2_XS:  memset(p + 64, 0xff, 8); break;            /* scales after qs */
            case FR_GSQ_IQ2_S:   memset(p + 64 + 8, 0xff, 8); break;        /* after qs and qh */
            case FR_GSQ_IQ3_S:   memset(p + 64 + 8 + 32, 0xff, 4); break;   /* after qs, qh and signs */
            default: break;
        }
    }
}

static unsigned long slices, slices_differ;
static void check_slice(unsigned l, const unsigned char *slice) {
    static float ya[QTOK][FR_GSQ_HIDDEN], yb[QTOK][FR_GSQ_HIDDEN];
    static const unsigned snt[] = {2, 3, 4, 7};
    for (unsigned r = 0; r < 2 * FR_GSQ_NEURONS; ++r) check_row(type_index(l), slice + r * fr_gsq_row_bytes(l), l, r);
    const void *xp[QTOK]; float *yp[QTOK]; float gate[QTOK];
    for (unsigned t = 0; t < QTOK; ++t) { xp[t] = xq[t]; yp[t] = yb[t]; gate[t] = 0.1f + 0.05f * (float)t; }
    for (unsigned k = 0; k < 4; ++k) {
        memset(ya, 0, sizeof(ya)); memset(yb, 0, sizeof(yb));
        for (unsigned t = 0; t < snt[k]; ++t) fr_gsq_fast_slice(l, slice, xq[t], gate[t], ya[t]);
        fr_gsq_fast_slice_multi(l, slice, xp, gate, yp, snt[k]);
        ++slices;
        if (memcmp(ya, yb, sizeof(ya))) ++slices_differ;
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 && strcmp(argv[1], "-") ? argv[1] : NULL;
    const unsigned synthetic = argc > 2 ? (unsigned)atoi(argv[2]) : 10;
    fr_gsq_init();
    make_tokens();
    size_t offs[FR_GSQ_LAYERS * REAL_PER_LAYER], total = 0, largest = 0;
    for (unsigned l = 0; l < FR_GSQ_LAYERS; ++l) {
        if (fr_gsq_slice_bytes(l) > largest) largest = fr_gsq_slice_bytes(l);
        for (unsigned k = 0; k < REAL_PER_LAYER; ++k) { offs[l * REAL_PER_LAYER + k] = total; total += fr_gsq_slice_bytes(l); }
    }
    unsigned char *sl = NULL, *one = malloc(largest);
    if (!one) return 2;
    if (path) {
        FILE *f = fopen(path, "rb");
        sl = malloc(total);
        if (!f || !sl || fread(sl, 1, total, f) != total || fgetc(f) != EOF) { fprintf(stderr, "%s is not the %zu bytes of %u slices\n", path, total, FR_GSQ_LAYERS * REAL_PER_LAYER); return 2; }
        fclose(f);
        for (unsigned i = 0; i < FR_GSQ_LAYERS * REAL_PER_LAYER; ++i) check_slice(i / REAL_PER_LAYER, sl + offs[i]);
        printf("real slices: %u from %s\n", FR_GSQ_LAYERS * REAL_PER_LAYER, path);
    }
    for (unsigned l = 0; l < FR_GSQ_LAYERS; ++l)
        for (unsigned k = 0; k < synthetic; ++k) {
            fr_gsq_fill_slice(l, one, 0xc0ffee01u + 7919u * l + 104729u * k);
            if (k & 1) saturate(l, one);
            check_slice(l, one);
        }
    printf("synthetic slices: %u, %u of them with the largest scales\n", FR_GSQ_LAYERS * synthetic, FR_GSQ_LAYERS * (synthetic / 2));

    unsigned long bad = slices_differ;
    for (unsigned ti = 0; ti < NTYPES; ++ti) {
        printf("%-8s %6lu rows; nt 1: %lu differ;", types[ti].name, rows_of[ti], one_differ[ti]);
        bad += one_differ[ti];
        for (unsigned k = 0; k < NNT; ++k) { printf(" nt %u: %lu of %lu differ%s", nts[k], differ[ti][k], results[ti][k], k + 1 < NNT ? ";" : "\n"); bad += differ[ti][k]; }
        if (!rows_of[ti]) { printf("%-8s not covered\n", types[ti].name); ++bad; }
    }
    printf("slices (fast_slice_multi against fast_slice, nt 2, 3, 4, 7): %lu checked, %lu differ\n", slices, slices_differ);
    printf(bad ? "FAILED: %lu differences\n" : "ok: 0 differences\n", bad);

    if (sl || synthetic) {  /* timing, one thread */
        const unsigned count = sl ? FR_GSQ_LAYERS * REAL_PER_LAYER : FR_GSQ_LAYERS;
        static float ya[4][FR_GSQ_HIDDEN]; float *yp[4]; const void *xp[4]; float gate[4];
        unsigned char *syn = sl ? NULL : malloc(total / REAL_PER_LAYER);
        if (!sl) for (unsigned l = 0; l < FR_GSQ_LAYERS; ++l) fr_gsq_fill_slice(l, syn + offs[l * REAL_PER_LAYER] / REAL_PER_LAYER, 77u + l);
        for (unsigned t = 0; t < 4; ++t) { yp[t] = ya[t]; xp[t] = xq[t]; gate[t] = 0.1f; }
        for (unsigned nt = 2; nt <= 4; ++nt) {
            const unsigned rounds = sl ? 3 : 30;
            double t0 = now();
            for (unsigned r = 0; r < rounds; ++r) for (unsigned i = 0; i < count; ++i) {
                const unsigned l = sl ? i / REAL_PER_LAYER : i; const unsigned char *s = sl ? sl + offs[i] : syn + offs[i * REAL_PER_LAYER] / REAL_PER_LAYER;
                for (unsigned t = 0; t < nt; ++t) fr_gsq_fast_slice(l, s, xq[t], gate[t], ya[t]);
            }
            double t1 = now();
            for (unsigned r = 0; r < rounds; ++r) for (unsigned i = 0; i < count; ++i) {
                const unsigned l = sl ? i / REAL_PER_LAYER : i; const unsigned char *s = sl ? sl + offs[i] : syn + offs[i * REAL_PER_LAYER] / REAL_PER_LAYER;
                fr_gsq_fast_slice_multi(l, s, xp, gate, yp, nt);
            }
            double t2 = now();
            const double per = 1e6 / ((double)rounds * count * nt);
            printf("nt %u: single %.2f us, multi %.2f us per token-slice (%.2fx)\n", nt, (t1 - t0) * per, (t2 - t1) * per, (t1 - t0) / (t2 - t1));
        }
    }
    return bad ? 1 : 0;
}
