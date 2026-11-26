/* layout.c - see layout.h. */
#include "layout.h"
#include "gsq.h"
#include "native.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static unsigned current = FR_LAYOUT_NATIVE;

unsigned fr_layout(void) { return current; }
int fr_layout_set(unsigned layout) {
    if (layout != FR_LAYOUT_NATIVE && layout != FR_LAYOUT_GSQ) return -1;
    current = layout;
    return 0;
}
const char *fr_layout_name(unsigned layout) { return layout == FR_LAYOUT_GSQ ? "gsq" : "native"; }
int fr_layout_parse(const char *name, unsigned *layout) {
    if (!name || !layout) return -1;
    if (!strcmp(name, "native")) { *layout = FR_LAYOUT_NATIVE; return 0; }
    if (!strcmp(name, "gsq")) { *layout = FR_LAYOUT_GSQ; return 0; }
    return -1;
}
uint64_t fr_layout_unit_bytes(unsigned layout) {
    return layout == FR_LAYOUT_GSQ ? fr_gsq_unit_bytes() : FR_NATIVE_BLOCK_BYTES;
}
uint64_t fr_unit_bytes(void) { return fr_layout_unit_bytes(current); }
unsigned fr_backend_layout_of(const char *name) {
    return name && !strncmp(name, "gsq", 3) ? FR_LAYOUT_GSQ : FR_LAYOUT_NATIVE;
}
uint64_t fr_slice_bytes_mean(void) {
    return current == FR_LAYOUT_GSQ ? fr_gsq_unit_bytes() / ((uint64_t)FR_GSQ_LAYERS * FR_GSQ_EXPERTS)
                                    : FR_NATIVE_SLICE_BYTES;
}
const char *fr_layout_dir(const char *state_dir) {
    static char path[600];
    if (current == FR_LAYOUT_NATIVE || !state_dir) return state_dir;
    snprintf(path, sizeof(path), "%s/%s", state_dir, fr_layout_name(current));
    mkdir(path, 0700);
    return path;
}
