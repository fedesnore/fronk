/* stats.h - bounded RTT samples plus a logarithmic histogram.
 * Exact nearest-rank percentiles use successful samples only. Callers retain
 * timeout counts separately; missing replies are never recorded as fast RTTs.
 */
#ifndef FR_STATS_H
#define FR_STATS_H
#include <stdint.h>
#define FR_RTT_SAMPLES 1000
#define FR_HIST_BINS 32
typedef struct {
    uint64_t samples[FR_RTT_SAMPLES], sum, min, max;
    unsigned count, bins[FR_HIST_BINS];
} fr_stats;
int stats_add(fr_stats *s, uint64_t ns);
/* p50/p99/p99.9 in ns. -1 for no samples. Does not reorder stored samples. */
int stats_percentiles(const fr_stats *s, uint64_t out[3]);
/* Bin 0: [0,1us); bin n: [2^(n-1),2^n) us; last bin has no upper bound. */
#endif
