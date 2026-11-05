/* bench.h - measured backend cost and submission distributions.
 * Model fits and text serialization are portable. The resident runner times
 * CPU backends too; GPU phase fields stay zero for CPU-only execution.
 * Samples/weights are allocated before the measured run loop.
 */
#ifndef FR_BENCH_H
#define FR_BENCH_H
#include "metal.h"
#include <stdatomic.h>
#define FR_BENCH_SAMPLES 10000
#define FR_BENCH_TEXT 8192
#define FR_PLAN_SAMPLES 256
typedef struct {
    fr_costmodel cost;
    unsigned blocks,samples,burst_max,selected_burst; /* zero: full 1..4 report */
    unsigned workers,prefetch; /* BENCH-selected CPU knobs; zero when inapplicable. */
    double percentile[2][FR_BENCH_BURST_MAX][FR_METAL_PHASES][2]; /* distinct,reused; p50,p99 */
    double mean[2][FR_BENCH_BURST_MAX],sustained_ratio,burst_slope,elapsed;
    uint64_t rss_start,rss_end,barriers,working_set,resident_bytes,hot_allocations;
} fr_bench_report;
int fr_bench_fit(fr_bench_report *r);
int fr_bench_format(const fr_bench_report *r,char *out,size_t cap);
int fr_bench_parse(const char *text,size_t len,fr_bench_report *out);
/* Bounded planning measurements, distinct from the full sustained report.
 * All four widths share one resident fixture and one CPU tuning choice. */
int fr_bench_quick_format(const fr_bench_report *r,char *out,size_t cap);
int fr_bench_quick_parse(const char *text,size_t len,fr_bench_report *out);
int fr_ref_idle(const fr_backend *backend); /* Guard resident fixture ownership. */
#ifdef FR_BENCH_RUNTIME
int fr_bench_measure_backend(const fr_backend *backend,unsigned blocks,unsigned samples,unsigned burst_max,unsigned selected_burst,double seconds,
                            FILE *progress,FILE *csv,const atomic_int *cancel,fr_bench_report *out);
int fr_bench_execute_backend(const fr_backend *backend,int burst_max,fr_costmodel *out);
int fr_bench_measure(unsigned blocks,unsigned samples,unsigned burst_max,double seconds,
                     FILE *progress,FILE *csv,const atomic_int *cancel,fr_bench_report *out);
/* Serialized setup for the fixed backend bench vtable; defaults to 3 blocks. */
int fr_bench_configure(unsigned blocks,const atomic_int *cancel);
int fr_bench_configure_point(unsigned blocks,unsigned burst,const atomic_int *cancel);
int fr_bench_execute(int burst_max,fr_costmodel *out);
int fr_bench_last(fr_bench_report *out);
#endif
#endif
