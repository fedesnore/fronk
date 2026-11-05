/* test_stats.c - exact ranks, histogram boundaries and bounded accumulation. */
#include "test.h"
#include "../src/stats.h"
int main(void) {
    test_begin("stats"); fr_stats s={0}; uint64_t p[3];
    check(stats_percentiles(&s,p)<0,"empty sample set has no percentiles");
    check(!stats_add(&s,0) && !stats_add(&s,999) && !stats_add(&s,1000) && !stats_add(&s,2000),"histogram boundary samples");
    check(s.bins[0]==2 && s.bins[1]==1 && s.bins[2]==1 && s.sum==3999 && s.min==0 && s.max==2000,"histogram bounds and exact totals");
    s=(fr_stats){0}; int ok=1;
    for (unsigned i=1000; i>0; --i) if (stats_add(&s,i*1000)) ok=0;
    check(ok && s.count==1000 && s.sum==500500000,"bounded sample collection");
    check(!stats_percentiles(&s,p) && p[0]==500000 && p[1]==990000 && p[2]==999000,"nearest-rank p50/p99/p99.9");
    check(s.samples[0]==1000000 && stats_add(&s,1)<0 && s.count==1000,"summary is nondestructive and capacity enforced");
    s=(fr_stats){0};
    check(!stats_add(&s,UINT64_MAX) && s.bins[31]==1 && stats_add(&s,1)<0 && s.count==1,"sum overflow rejected atomically");
    check(!stats_percentiles(&s,p) && p[0]==UINT64_MAX && p[2]==UINT64_MAX,"single sample ranks");
    return test_end();
}
