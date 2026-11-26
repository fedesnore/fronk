/* layout.h - which unit format this process works with.
 *
 * A unit is the 32-neuron block b of every routed expert of every layer. Two
 * formats exist, both lossless gathers of a GGUF (no requantization):
 *   native: the Q4 file ds4 reads (Q4_K gate/up, MXFP4 down, plus the Q8_0
 *           shared expert); native.h, 3.3 GB per unit.
 *   gsq:    ISTA-DASLab's GSQ-RCO IQ3_XXS file llama.cpp reads (per-layer IQ2/IQ3
 *           gate/up, IQ4_NL or Q2_0 down, no shared expert: the head computes
 *           it); gsq.h, 2.18 GB per unit.
 * The head takes the format from its model file and tells every node (LAYOUT);
 * a node computes one format at a time and only with backends made for it.
 * Set once at setup, before units, backends or benchmarks are touched.
 */
#ifndef FR_LAYOUT_H
#define FR_LAYOUT_H
#include <stdint.h>

enum { FR_LAYOUT_NATIVE = 0, FR_LAYOUT_GSQ = 1 };

unsigned fr_layout(void);
int fr_layout_set(unsigned layout);            /* 0, or -1 for an unknown layout */
const char *fr_layout_name(unsigned layout);   /* "native", "gsq" */
int fr_layout_parse(const char *name, unsigned *layout);
uint64_t fr_unit_bytes(void);                  /* one unit file in the current layout */
uint64_t fr_layout_unit_bytes(unsigned layout);
/* A backend computes exactly one layout; the GSQ backends are named gsq*. */
unsigned fr_backend_layout_of(const char *backend_name);
/* Where this layout's units (and the head's manifest) live under state-dir:
 * state-dir itself for native, state-dir/gsq for GSQ (created). Static buffer. */
const char *fr_layout_dir(const char *state_dir);
/* Average bytes a routed (layer, expert) slice reads, for cost models. */
uint64_t fr_slice_bytes_mean(void);
#endif
