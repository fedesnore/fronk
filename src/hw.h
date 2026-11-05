/* hw.h - bounded hardware snapshots and platform probes, before backend choice.
 * Capacity is an observation, not a reservation. Probe/parse replace output only
 * on success; finalize derives the explicit margin and weight capacity.
 */
#ifndef FR_HW_H
#define FR_HW_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#define FR_HW_TEXT 2048
#define FR_ISA_AVX2 1u
#define FR_ISA_FMA 2u
#define FR_ISA_F16C 4u
#define FR_ISA_AVX 8u
#define FR_ISA_OSXSAVE 16u
#define FR_ISA_XSAVE 32u
#define FR_ISA_YMM 64u
#define FR_ISA_USABLE 127u
typedef struct {
    char os[16], arch[16], version[64], cpu[128], gpu[128], name[64];
    unsigned cores, logical_cores, isa, metal, translated;
    uint64_t id, total, available, margin, cap, blocks, metal_limit, frequency, wired_mb;
    uint64_t head_reserve, head_cap, head_blocks, head_fixed; /* head_fixed: configured head-blocks */
} fr_hwinfo;
unsigned hw_x86_isa(uint32_t leaf1_ecx, uint32_t leaf7_ebx, uint64_t xcr0);
const char *hw_backend(const fr_hwinfo *hw);
const char *hw_reason(const fr_hwinfo *hw);
int hw_finalize(fr_hwinfo *hw);
int hw_linux_read(FILE *cpuinfo, FILE *meminfo, fr_hwinfo *out);
int hw_macos_read(FILE *sysctl, FILE *vmstat, FILE *version, fr_hwinfo *out);
int hw_pages(fr_hwinfo *hw, uint64_t page_size, uint64_t free_pages, uint64_t inactive, uint64_t purgeable);
/* Memory for units on a node or the head: total minus what the system cannot
 * drop (wired, anonymous, compressed). File cache, kept units included, counts once. */
int hw_unpinned(fr_hwinfo *hw, uint64_t page_size, uint64_t wired, uint64_t anonymous, uint64_t compressed);
int hw_node_available(fr_hwinfo *hw);
int hw_probe(fr_hwinfo *out);
int hw_format(const fr_hwinfo *hw, char *out, size_t cap);
int hw_parse(const char *text, size_t len, fr_hwinfo *out);
#ifdef __APPLE__
int hw_metal_probe(fr_hwinfo *out);
#endif
#endif
