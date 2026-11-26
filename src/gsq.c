/* gsq.c - see gsq.h. The quantized formats and their dot products come from
 * ggml (third_party/ggml, MIT); this file only lays out and cuts slices. */
#include "gsq.h"
#include "gsq_down.h"
#include "gsq_multi.h"
#include "backend.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include "ggml-impl.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* From the pinned IQ3_XXS file's header (blk.N.ffn_{gate,up,down}_exps types). */
const uint8_t fr_gsq_gate_type[FR_GSQ_LAYERS] = {
    17,16,17,16,17,17,16,16,21,17,17,16,22,16,17,22,16,22,21,17,21,22,21,21,
    21,17,22,16,21,22,21,16,21,21,21,18,18,21,22,17,22,18,22,21,18,18,22,18};
const uint8_t fr_gsq_down_type[FR_GSQ_LAYERS] = {
    20,42,20,42,20,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,
    42,42,42,42,20,20,20,20,20,42,20,20,42,20,20,20,42,42,20,20,20,42,20,20};

static size_t block_bytes(unsigned type) {
    switch (type) {
        case FR_GSQ_IQ2_XXS: return sizeof(block_iq2_xxs);
        case FR_GSQ_IQ2_XS:  return sizeof(block_iq2_xs);
        case FR_GSQ_IQ2_S:   return sizeof(block_iq2_s);
        case FR_GSQ_IQ3_XXS: return sizeof(block_iq3_xxs);
        case FR_GSQ_IQ3_S:   return sizeof(block_iq3_s);
        default:             return 0;
    }
}
size_t fr_gsq_row_bytes(unsigned l) { return l < FR_GSQ_LAYERS ? FR_GSQ_HIDDEN / QK_K * block_bytes(fr_gsq_gate_type[l]) : 0; }
size_t fr_gsq_piece_bytes(unsigned l) {
    if (l >= FR_GSQ_LAYERS) return 0;
    return fr_gsq_down_type[l] == FR_GSQ_IQ4_NL ? sizeof(block_iq4_nl) : sizeof(ggml_half) + 8;
}
size_t fr_gsq_slice_bytes(unsigned l) {
    return 2 * FR_GSQ_NEURONS * fr_gsq_row_bytes(l) + (size_t)FR_GSQ_HIDDEN * fr_gsq_piece_bytes(l);
}
uint64_t fr_gsq_layer_offset(unsigned layer) {
    static uint64_t offset[FR_GSQ_LAYERS + 1];
    if (!offset[FR_GSQ_LAYERS])
        for (unsigned l = 0; l < FR_GSQ_LAYERS; ++l)
            offset[l + 1] = offset[l] + (uint64_t)FR_GSQ_EXPERTS * fr_gsq_slice_bytes(l);
    return offset[layer <= FR_GSQ_LAYERS ? layer : FR_GSQ_LAYERS];
}
uint64_t fr_gsq_unit_bytes(void) { return fr_gsq_layer_offset(FR_GSQ_LAYERS); }

/* ---------- GGUF ---------- */

typedef struct { const unsigned char *p, *end; } cursor;
static int take(cursor *c, void *out, size_t n) {
    if ((size_t)(c->end - c->p) < n) return -1;
    memcpy(out, c->p, n); c->p += n; return 0;
}
static int string(cursor *c, const char **s, uint64_t *n) {
    if (take(c, n, 8) || (uint64_t)(c->end - c->p) < *n) return -1;
    *s = (const char *)c->p; c->p += *n; return 0;
}
static int skip_value(cursor *c, uint32_t type, int depth) {
    static const uint8_t size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type == 8) { const char *s; uint64_t n; return string(c, &s, &n); }
    if (type == 9) {
        uint32_t et; uint64_t n;
        if (depth > 2 || take(c, &et, 4) || take(c, &n, 8)) return -1;
        for (uint64_t i = 0; i < n; ++i) if (skip_value(c, et, depth + 1)) return -1;
        return 0;
    }
    if (type > 12 || !size[type] || (size_t)(c->end - c->p) < size[type]) return -1;
    c->p += size[type]; return 0;
}

int fr_gsq_open(const char *path, fr_gsq_gguf *g, char *error, size_t cap) {
    memset(g, 0, sizeof(*g)); g->fd = -1;
#define FAIL(...) do { if (error) snprintf(error, cap, __VA_ARGS__); fr_gsq_close(g); return -1; } while (0)
    struct stat st;
    if ((g->fd = open(path, O_RDONLY)) < 0 || fstat(g->fd, &st)) FAIL("cannot open %s", path);
    g->size = (size_t)st.st_size;
    void *m = mmap(NULL, g->size, PROT_READ, MAP_SHARED, g->fd, 0);
    if (m == MAP_FAILED) FAIL("cannot map %s", path);
    g->map = m;
    cursor c = {g->map, g->map + g->size};
    uint32_t magic, version, type; uint64_t tensors, keys, n, align = 32; const char *s;
    if (take(&c, &magic, 4) || memcmp(&magic, "GGUF", 4) || take(&c, &version, 4) || version != 3 ||
        take(&c, &tensors, 8) || take(&c, &keys, 8)) FAIL("not a GGUF v3 file");
    int arch_ok = 0;
    for (uint64_t i = 0; i < keys; ++i) {
        if (string(&c, &s, &n) || take(&c, &type, 4)) FAIL("bad metadata");
        if (n == 20 && !memcmp(s, "general.architecture", 20) && type == 8) {
            const char *v; uint64_t vn;
            if (string(&c, &v, &vn)) FAIL("bad metadata");
            arch_ok = vn == 8 && !memcmp(v, "qwen4exp", 8);
        } else if (n == 17 && !memcmp(s, "general.alignment", 17) && type == 4) {
            uint32_t a; if (take(&c, &a, 4) || !a || (a & (a - 1))) FAIL("bad alignment"); align = a;
        } else if (skip_value(&c, type, 0)) FAIL("bad metadata");
    }
    if (!arch_ok) FAIL("not a qwen4exp (Qwen3.8-Flash-Next) model");
    unsigned found = 0;
    for (uint64_t i = 0; i < tensors; ++i) {
        uint32_t dims, t; uint64_t ne[4] = {0}, off; unsigned l; int at = 0; char name[96];
        if (string(&c, &s, &n) || take(&c, &dims, 4) || dims > 4) FAIL("bad tensor table");
        for (uint32_t d = 0; d < dims; ++d) if (take(&c, &ne[d], 8)) FAIL("bad tensor table");
        if (take(&c, &t, 4) || take(&c, &off, 8)) FAIL("bad tensor table");
        if (n >= sizeof(name)) continue;
        memcpy(name, s, n); name[n] = 0;
        if (sscanf(name, "blk.%u.%n", &l, &at) != 1 || !at || l >= FR_GSQ_LAYERS) continue;
        uint64_t *slot = !strcmp(name + at, "ffn_gate_exps.weight") ? g->gate :
                         !strcmp(name + at, "ffn_up_exps.weight")   ? g->up :
                         !strcmp(name + at, "ffn_down_exps.weight") ? g->down : NULL;
        if (!slot) continue;
        int down = slot == g->down;
        unsigned want = down ? fr_gsq_down_type[l] : fr_gsq_gate_type[l];
        uint64_t n0 = down ? FR_GSQ_INTERMEDIATE : FR_GSQ_HIDDEN, n1 = down ? FR_GSQ_HIDDEN : FR_GSQ_INTERMEDIATE;
        if (dims != 3 || t != want || ne[0] != n0 || ne[1] != n1 || ne[2] != FR_GSQ_EXPERTS || slot[l])
            FAIL("%s: type %u shape %llux%llux%llu, expected type %u (not the pinned GSQ-RCO IQ3_XXS file)",
                 name, t, (unsigned long long)ne[0], (unsigned long long)ne[1], (unsigned long long)ne[2], want);
        slot[l] = off + 1; ++found;
    }
    if (found != 3 * FR_GSQ_LAYERS) FAIL("routed expert tensors missing (%u of %u)", found, 3 * FR_GSQ_LAYERS);
    uint64_t data = ((uint64_t)(c.p - g->map) + align - 1) / align * align;
    for (unsigned l = 0; l < FR_GSQ_LAYERS; ++l) {
        uint64_t bytes[3] = {
            (uint64_t)FR_GSQ_EXPERTS * FR_GSQ_INTERMEDIATE * fr_gsq_row_bytes(l),
            (uint64_t)FR_GSQ_EXPERTS * FR_GSQ_INTERMEDIATE * fr_gsq_row_bytes(l),
            (uint64_t)FR_GSQ_EXPERTS * FR_GSQ_HIDDEN * (FR_GSQ_INTERMEDIATE / 32) * sizeof(block_iq4_nl)};
        if (fr_gsq_down_type[l] == FR_GSQ_Q2_0)
            bytes[2] = (uint64_t)FR_GSQ_EXPERTS * FR_GSQ_HIDDEN * (FR_GSQ_INTERMEDIATE / QK2_0) * sizeof(block_q2_0);
        uint64_t *slots[3] = {&g->gate[l], &g->up[l], &g->down[l]};
        for (unsigned k = 0; k < 3; ++k) {
            uint64_t off = *slots[k] - 1;
            if (off > g->size - data || bytes[k] > g->size - data - off) FAIL("layer %u tensor outside the file", l);
            *slots[k] = data + off;
        }
    }
    return 0;
#undef FAIL
}

void fr_gsq_close(fr_gsq_gguf *g) {
    if (g->map) munmap((void *)g->map, g->size);
    if (g->fd >= 0) close(g->fd);
    memset(g, 0, sizeof(*g)); g->fd = -1;
}

int fr_gsq_slice(const fr_gsq_gguf *g, unsigned b, unsigned l, unsigned e, unsigned char *out) {
    if (!g || !g->map || b >= FR_GSQ_BLOCKS || l >= FR_GSQ_LAYERS || e >= FR_GSQ_EXPERTS || !out) return FR_INVALID;
    const size_t row = fr_gsq_row_bytes(l), rows = FR_GSQ_NEURONS * row;
    const size_t expert = (size_t)FR_GSQ_INTERMEDIATE * row;
    memcpy(out, g->map + g->gate[l] + e * expert + (size_t)b * rows, rows);
    memcpy(out + rows, g->map + g->up[l] + e * expert + (size_t)b * rows, rows);
    unsigned char *p = out + 2 * rows;
    if (fr_gsq_down_type[l] == FR_GSQ_IQ4_NL) {
        const size_t drow = FR_GSQ_INTERMEDIATE / 32 * sizeof(block_iq4_nl);
        const unsigned char *d = g->map + g->down[l] + (size_t)e * FR_GSQ_HIDDEN * drow + (size_t)b * sizeof(block_iq4_nl);
        for (unsigned r = 0; r < FR_GSQ_HIDDEN; ++r, p += sizeof(block_iq4_nl))
            memcpy(p, d + (size_t)r * drow, sizeof(block_iq4_nl));
    } else {
        const size_t drow = FR_GSQ_INTERMEDIATE / QK2_0 * sizeof(block_q2_0);
        const unsigned char *d = g->map + g->down[l] + (size_t)e * FR_GSQ_HIDDEN * drow + (size_t)(b / 2) * sizeof(block_q2_0);
        for (unsigned r = 0; r < FR_GSQ_HIDDEN; ++r, p += 10) {
            const block_q2_0 *blk = (const block_q2_0 *)(d + (size_t)r * drow);
            const uint8_t *qs = blk->qs + (b % 2) * 8;   /* weight j at byte j/4, bits 2(j%4) */
            memcpy(p, &blk->d, sizeof(ggml_half));
            memset(p + 2, 0, 8);
            for (unsigned j = 0; j < 32; ++j)            /* planar: byte j%8, bits 2(j/8) */
                p[2 + j % 8] |= (uint8_t)(((qs[j / 4] >> (2 * (j % 4))) & 3) << (2 * (j / 8)));
        }
    }
    return FR_OK;
}

void fr_gsq_forget(const fr_gsq_gguf *g, unsigned l, unsigned e) {
    if (!g || !g->map || l >= FR_GSQ_LAYERS || e >= FR_GSQ_EXPERTS) return;
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    const size_t expert = (size_t)FR_GSQ_INTERMEDIATE * fr_gsq_row_bytes(l);
    const size_t drow = fr_gsq_down_type[l] == FR_GSQ_IQ4_NL ? FR_GSQ_INTERMEDIATE / 32 * sizeof(block_iq4_nl)
                                                             : FR_GSQ_INTERMEDIATE / QK2_0 * sizeof(block_q2_0);
    const uint64_t at[3] = { g->gate[l] + (uint64_t)e * expert, g->up[l] + (uint64_t)e * expert, g->down[l] + (uint64_t)e * FR_GSQ_HIDDEN * drow };
    const size_t len[3] = { expert, expert, (size_t)FR_GSQ_HIDDEN * drow };
    for (unsigned i = 0; i < 3; ++i) {   /* whole pages inside the expert only: its neighbours keep theirs */
        uintptr_t a = (uintptr_t)g->map + at[i], b = a + len[i];
        a = (a + page - 1) / page * page; b = b / page * page;
        if (b > a) (void)madvise((void *)a, b - a, MADV_DONTNEED);
    }
}

/* ---------- compute ---------- */

static void dequant_row(unsigned type, const void *row, float *out, int64_t n) {
    switch (type) {
        case FR_GSQ_IQ2_XXS: dequantize_row_iq2_xxs(row, out, n); break;
        case FR_GSQ_IQ2_XS:  dequantize_row_iq2_xs(row, out, n); break;
        case FR_GSQ_IQ2_S:   dequantize_row_iq2_s(row, out, n); break;
        case FR_GSQ_IQ3_XXS: dequantize_row_iq3_xxs(row, out, n); break;
        case FR_GSQ_IQ3_S:   dequantize_row_iq3_s(row, out, n); break;
        default:             memset(out, 0, (size_t)n * sizeof(float));
    }
}
float fr_gsq_q2_half(const unsigned char *p, unsigned j) {
    ggml_half h; memcpy(&h, p, sizeof(h));
    const unsigned q = (p[sizeof(h) + j % 8] >> (2 * (j / 8))) & 3;
    return ((int)q - 1) * GGML_FP16_TO_FP32(h);
}

int fr_gsq_ref_slice(unsigned l, const unsigned char *slice, const float *x, float gate, double *y) {
    if (l >= FR_GSQ_LAYERS || !slice || !x || !y || !isfinite(gate)) return FR_INVALID;
    const size_t row = fr_gsq_row_bytes(l), rows = FR_GSQ_NEURONS * row, piece = fr_gsq_piece_bytes(l);
    const unsigned type = fr_gsq_gate_type[l];
    float w[FR_GSQ_HIDDEN], h[FR_GSQ_NEURONS];
    for (unsigned n = 0; n < FR_GSQ_NEURONS; ++n) {
        double a = 0, u = 0;
        dequant_row(type, slice + n * row, w, FR_GSQ_HIDDEN);
        for (unsigned k = 0; k < FR_GSQ_HIDDEN; ++k) a += (double)w[k] * x[k];
        dequant_row(type, slice + rows + n * row, w, FR_GSQ_HIDDEN);
        for (unsigned k = 0; k < FR_GSQ_HIDDEN; ++k) u += (double)w[k] * x[k];
        if (!isfinite(a) || !isfinite(u)) return FR_NUMERIC;
        h[n] = (float)(a / (1 + exp(-a)) * u);
    }
    const unsigned char *p = slice + 2 * rows;
    for (unsigned d = 0; d < FR_GSQ_HIDDEN; ++d, p += piece) {
        float v[32];
        if (fr_gsq_down_type[l] == FR_GSQ_IQ4_NL) dequantize_row_iq4_nl((const block_iq4_nl *)p, v, 32);
        else for (unsigned j = 0; j < 32; ++j) v[j] = fr_gsq_q2_half(p, j);
        double s = 0;
        for (unsigned j = 0; j < 32; ++j) s += (double)v[j] * h[j];
        if (!isfinite(s)) return FR_NUMERIC;
        y[d] += (double)gate * s;
    }
    return FR_OK;
}

void fronk_ggml_init(void);
void fr_gsq_init(void) { fronk_ggml_init(); }
void fr_gsq_quantize_x(const float *x, void *xq) { quantize_row_q8_K(x, xq, FR_GSQ_HIDDEN); }

typedef void (*dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
static dot_fn gate_dot(unsigned type) {
    switch (type) {
        case FR_GSQ_IQ2_XXS: return ggml_vec_dot_iq2_xxs_q8_K;
        case FR_GSQ_IQ2_XS:  return ggml_vec_dot_iq2_xs_q8_K;
        case FR_GSQ_IQ2_S:   return ggml_vec_dot_iq2_s_q8_K;
        case FR_GSQ_IQ3_XXS: return ggml_vec_dot_iq3_xxs_q8_K;
        case FR_GSQ_IQ3_S:   return ggml_vec_dot_iq3_s_q8_K;
        default:             return NULL;
    }
}

int fr_gsq_fast_slice(unsigned l, const unsigned char *slice, const void *xq, float gate, float *y) {
    if (l >= FR_GSQ_LAYERS || !slice || !xq || !y) return FR_INVALID;
    const size_t row = fr_gsq_row_bytes(l), rows = FR_GSQ_NEURONS * row;
    const dot_fn dot = gate_dot(fr_gsq_gate_type[l]);
    float h[FR_GSQ_NEURONS];
    for (unsigned n = 0; n < FR_GSQ_NEURONS; ++n) {
        float a, u;
        dot(FR_GSQ_HIDDEN, &a, 0, slice + n * row, 0, xq, 0, 1);
        dot(FR_GSQ_HIDDEN, &u, 0, slice + rows + n * row, 0, xq, 0, 1);
        h[n] = a / (1.0f + expf(-a)) * u;
    }
    int8_t hq[32]; int32_t hsum;
    const float dh = fr_gsq_quantize_h(h, hq, &hsum);
    fr_gsq_down(fr_gsq_down_type[l], slice + 2 * rows, FR_GSQ_HIDDEN, hq, hsum, gate * dh, y);
    return FR_OK;
}

int fr_gsq_fast_slice_multi(unsigned l, const unsigned char *slice, const void *const *xq, const float *gate,
                            float *const *y, unsigned nt) {
    if (l >= FR_GSQ_LAYERS || !slice || !xq || !y || !nt || nt > FR_GSQ_MULTI_MAX) return FR_INVALID;
    if (nt == 1) return fr_gsq_fast_slice(l, slice, xq[0], gate[0], y[0]);
    const size_t row = fr_gsq_row_bytes(l), rows = FR_GSQ_NEURONS * row;
    const unsigned type = fr_gsq_gate_type[l];
    float h[FR_GSQ_MULTI_MAX][FR_GSQ_NEURONS];
    for (unsigned n = 0; n < FR_GSQ_NEURONS; ++n) {
        float a[FR_GSQ_MULTI_MAX], u[FR_GSQ_MULTI_MAX];
        fr_gsq_dot_multi(type, FR_GSQ_HIDDEN, a, slice + n * row, xq, nt);
        fr_gsq_dot_multi(type, FR_GSQ_HIDDEN, u, slice + rows + n * row, xq, nt);
        for (unsigned t = 0; t < nt; ++t) h[t][n] = a[t] / (1.0f + expf(-a[t])) * u[t];
    }
    for (unsigned t = 0; t < nt; ++t) {
        int8_t hq[32]; int32_t hsum;
        const float dh = fr_gsq_quantize_h(h[t], hq, &hsum);
        fr_gsq_down(fr_gsq_down_type[l], slice + 2 * rows, FR_GSQ_HIDDEN, hq, hsum, gate[t] * dh, y[t]);
    }
    return FR_OK;
}

static uint32_t fill_next(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static void fill_scale(unsigned char *p, uint32_t *s) {
    const uint16_t h = (uint16_t)(0x1c00 | (fill_next(s) & 0x83ff));  /* finite, either sign */
    p[0] = (unsigned char)h; p[1] = (unsigned char)(h >> 8);
}
void fr_gsq_fill_slice(unsigned l, unsigned char *out, uint32_t seed) {
    if (l >= FR_GSQ_LAYERS || !out) return;
    uint32_t s = seed ? seed : 1;
    const size_t bytes = fr_gsq_slice_bytes(l), row = fr_gsq_row_bytes(l), rows = FR_GSQ_NEURONS * row;
    const size_t blk = block_bytes(fr_gsq_gate_type[l]), piece = fr_gsq_piece_bytes(l);
    for (size_t i = 0; i < bytes; ++i) out[i] = (unsigned char)fill_next(&s);
    for (size_t o = 0; o < 2 * rows; o += blk) fill_scale(out + o, &s);              /* gate and up blocks */
    for (size_t o = 2 * rows; o + piece <= bytes; o += piece) fill_scale(out + o, &s); /* down pieces */
}
void fr_gsq_fill_unit(unsigned char *out, uint32_t seed) {
    if (!out) return;
    for (unsigned l = 0; l < FR_GSQ_LAYERS; ++l)
        for (unsigned e = 0; e < FR_GSQ_EXPERTS; ++e)
            fr_gsq_fill_slice(l, out + fr_gsq_offset(l, e), seed ^ (l * 1000003u + e * 7919u + 1u));
}
