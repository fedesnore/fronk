/* fronk_shim.c - the few symbols ggml-quants.c and ggml-cpu/quants.c need from
 * the rest of ggml (ggml.c, ggml-cpu.c), so Fronk can use the quantized dot
 * products alone. Only the types Fronk's GSQ path uses have sizes here. */
#include "ggml.h"
#include "ggml-impl.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void ggml_abort(const char *file, int line, const char *fmt, ...) {
    va_list args; va_start(args, fmt);
    fprintf(stderr, "ggml: %s:%d: ", file, line); vfprintf(stderr, fmt, args); fputc('\n', stderr);
    va_end(args);
    abort();
}

static int64_t blck(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: return 1;
        case GGML_TYPE_Q8_0: case GGML_TYPE_IQ4_NL: return 32;
        case GGML_TYPE_Q2_0: return QK2_0;
        default: return QK_K;
    }
}
size_t ggml_type_size(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:     return 4;
        case GGML_TYPE_F16:     return 2;
        case GGML_TYPE_Q8_0:    return sizeof(block_q8_0);
        case GGML_TYPE_Q8_K:    return sizeof(block_q8_K);
        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_XS:  return sizeof(block_iq2_xs);
        case GGML_TYPE_IQ2_S:   return sizeof(block_iq2_s);
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ3_S:   return sizeof(block_iq3_s);
        case GGML_TYPE_IQ4_NL:  return sizeof(block_iq4_nl);
        case GGML_TYPE_Q2_0:    return sizeof(block_q2_0);
        default: ggml_abort(__FILE__, __LINE__, "type %d not used by Fronk", (int)t); return 0;
    }
}
size_t ggml_row_size(enum ggml_type t, int64_t ne) { return ggml_type_size(t) * (size_t)(ne / blck(t)); }
const char *ggml_type_name(enum ggml_type t) { (void)t; return "fronk-gsq"; }

/* Lookup tables ggml-cpu.c fills in ggml_cpu_init(). The fp16 table is read on
 * x86 builds, so fronk_ggml_init() must run before any dot product; the FP8 and
 * E8M0 tables belong to formats Fronk does not use and stay zero. */
float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_f8_e4m3[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

void fronk_ggml_init(void) {
    static int done;
    if (done) return;
    for (int i = 0; i < (1 << 16); ++i) {
        union { uint16_t u16; ggml_fp16_t fp16; } u = {(uint16_t)i};
        ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
    }
    done = 1;
}
