/* test_gsq.c - the GSQ layout and the gsq backend.
 *
 * Layout: unit size, names, backend ownership. Backend: SELFTEST (fast path
 * against gsq.c's scalar oracle on synthetic slices of every type), then a real
 * run on whole units: two anonymous 2.18 GB units (untouched pages cost nothing)
 * with only the requested slices filled, tokens that share experts, compared with
 * the oracle summed over both units. Refusals: a shared gate, a wrong unit size,
 * a second load into a slot. */
#include "../src/backend.h"
#include "../src/gsq.h"
#include "../src/layout.h"
#include "../src/native.h"
#include "test.h"
#include <math.h>
#include <string.h>
#include <sys/mman.h>

int main(void) {
    test_begin("gsq");
    check(fr_layout() == FR_LAYOUT_NATIVE, "native is the default layout");
    check(fr_layout_unit_bytes(FR_LAYOUT_GSQ) == UINT64_C(2184970240), "GSQ unit size");
    check(fr_layout_unit_bytes(FR_LAYOUT_NATIVE) == FR_NATIVE_BLOCK_BYTES, "native unit size");
    unsigned layout = 9;
    check(!fr_layout_parse("gsq", &layout) && layout == FR_LAYOUT_GSQ, "parse gsq");
    check(fr_layout_parse("q4", &layout), "unknown layout refused");
    check(fr_backend_layout_of("gsq") == FR_LAYOUT_GSQ && fr_backend_layout_of("avx2") == FR_LAYOUT_NATIVE, "backend ownership");
    check(!fr_layout_set(FR_LAYOUT_GSQ) && fr_unit_bytes() == fr_gsq_unit_bytes(), "set gsq");

    const fr_backend *b = fr_backend_find("gsq");
    check(b != NULL, "gsq backend compiled");
    if (!b) return test_end();
    fr_hwinfo hw;
    if (b->probe(&hw) == FR_UNSUPPORTED) { check(1, "gsq unsupported here (no AVX2/NEON)"); return test_end(); }
    check(b->self_test(NULL) == FR_OK, "SELFTEST against the scalar oracle");

    const size_t len = (size_t)fr_gsq_unit_bytes();
    unsigned char *u[2];
    for (unsigned k = 0; k < 2; ++k) {
        u[k] = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(u[k] != MAP_FAILED, "unit mapping");
        if (u[k] == MAP_FAILED) return test_end();
    }
    static fr_req req; static fr_part part; static float x[3][FR_HIDDEN];
    const unsigned layer = 12;  /* IQ2_S gate/up, Q2_0 down */
    req.layer = layer; req.burst = 3;
    for (unsigned t = 0; t < 3; ++t) {
        for (unsigned j = 0; j < FR_ACTIVE_ROUTED; ++j) {
            /* tokens 0 and 1 share five experts, token 2 shares two with token 0 */
            unsigned e = t == 2 && j < 2 ? 40 + j : (t == 1 && j < 5 ? 40 + j : 100 * t + 40 + j);
            req.expert[t][j] = (uint16_t)(e % FR_ROUTED_EXPERTS);
            req.gate[t][j] = 0.05f + 0.01f * (float)j;
        }
        req.gate[t][FR_ACTIVE_ROUTED] = 0;
        for (unsigned d = 0; d < FR_HIDDEN; ++d) x[t][d] = req.x[t][d] = sinf((float)(d * 3 + t * 7)) * 1.5f;
    }
    for (unsigned k = 0; k < 2; ++k)
        for (unsigned t = 0; t < 3; ++t) for (unsigned j = 0; j < FR_ACTIVE_ROUTED; ++j)
            fr_gsq_fill_slice(layer, u[k] + fr_gsq_offset(layer, req.expert[t][j]), 77u * (k + 1) + req.expert[t][j]);
    check(b->load(0, u[0], len - 1) == FR_INVALID, "wrong unit size refused");
    check(b->load(3, u[0], len) == FR_OK && b->load(7, u[1], len) == FR_OK, "two units loaded");
    check(b->load(3, u[0], len) == FR_STATE, "occupied slot refused");
    check(b->run(&req, &part) == FR_OK, "run");
    static double acc[FR_HIDDEN]; static float want[FR_HIDDEN];
    for (unsigned t = 0; t < 3; ++t) {
        memset(acc, 0, sizeof(acc));
        for (unsigned k = 0; k < 2; ++k) for (unsigned j = 0; j < FR_ACTIVE_ROUTED; ++j)
            fr_gsq_ref_slice(layer, u[k] + fr_gsq_offset(layer, req.expert[t][j]), x[t], req.gate[t][j], acc);
        for (unsigned d = 0; d < FR_HIDDEN; ++d) want[d] = (float)acc[d];
        fr_error e;
        check(!fr_compare(want, part.y[t], FR_HIDDEN, &e) && fr_conforms(&e, &fr_backend_tolerance), "run matches the oracle");
    }
    int zero = 1; for (unsigned t = 3; t < FR_MAX_BURST; ++t) for (unsigned d = 0; d < FR_HIDDEN; ++d) zero &= part.y[t][d] == 0;
    check(zero && part.burst == 3 && part.layer == layer, "unused tokens zeroed");
    req.gate[1][FR_ACTIVE_ROUTED] = 0.5f;
    check(b->run(&req, &part) == FR_UNSUPPORTED, "a shared-expert gate is refused");
    check(b->unload(3) == FR_OK && b->unload(7) == FR_OK && b->unload(7) == FR_STATE, "unload");
    check(fr_gsq_backend_close() == FR_OK, "pool stops");
    for (unsigned k = 0; k < 2; ++k) munmap(u[k], len);
    return test_end();
}
