/* stats.c - exact summaries for a fixed 1,000-sample CONNECTION trial.
 * Sorting is outside the timed path; acquisition only updates preallocated data.
 */
#include "stats.h"
#include <stdlib.h>
#include <string.h>
int stats_add(fr_stats *s, uint64_t ns) {
    if (s->count==FR_RTT_SAMPLES || ns>UINT64_MAX-s->sum) return -1;
    unsigned bin=0; uint64_t upper=1000;
    while (bin<FR_HIST_BINS-1 && ns>=upper) { ++bin; upper*=2; }
    ++s->bins[bin]; s->samples[s->count]=ns; s->sum+=ns;
    if (!s->count || ns<s->min) s->min=ns;
    if (!s->count || ns>s->max) s->max=ns;
    ++s->count; return 0;
}
static int compare(const void *a, const void *b) {
    uint64_t x=*(const uint64_t *)a, y=*(const uint64_t *)b;
    return (x>y)-(x<y);
}
int stats_percentiles(const fr_stats *s, uint64_t out[3]) {
    if (!s->count || s->count>FR_RTT_SAMPLES) return -1;
    uint64_t sorted[FR_RTT_SAMPLES];
    memcpy(sorted,s->samples,s->count*sizeof(*sorted));
    qsort(sorted,s->count,sizeof(*sorted),compare);
    unsigned permille[]={500,990,999};
    for (int i=0; i<3; ++i) out[i]=sorted[(s->count*permille[i]+999)/1000-1];
    return 0;
}
