#include "test.h"
#include "../src/bench.h"
#include "../src/native.h"
#include <math.h>
#include <string.h>
int main(void) {
    test_begin("bench");
    {
        fr_bench_report q={.blocks=2,.samples=FR_PLAN_SAMPLES,.burst_max=4,.workers=3,.prefetch=6,.resident_bytes=2*FR_NATIVE_BLOCK_BYTES},got={0};
        for (unsigned b=0;b<4;++b) { q.percentile[0][b][FR_METAL_TOTAL][0]=100+b*80; q.percentile[0][b][FR_METAL_TOTAL][1]=150+b*90; }
        char buf[1024]; int n=fr_bench_quick_format(&q,buf,sizeof(buf));
        check(n>0 && !fr_bench_quick_parse(buf,(size_t)n,&got) && got.workers==3 && got.prefetch==6 &&
              got.percentile[0][3][FR_METAL_TOTAL][0]==340,"short planning cache retains independently measured widths and actual tuning");
        fr_bench_report save=got;
        check(fr_bench_quick_parse(buf,(size_t)n-1,&got)<0 && !memcmp(&save,&got,sizeof(got)),"interrupted quick profile is rejected atomically");
        q.percentile[0][1][FR_METAL_TOTAL][1]=NAN;
        check(fr_bench_quick_format(&q,buf,sizeof(buf))<0,"nonfinite quick cost refused");
        q.percentile[0][1][FR_METAL_TOTAL][1]=240; q.resident_bytes--;
        check(fr_bench_quick_format(&q,buf,sizeof(buf))<0,"quick measurement still requires all fixture bytes resident");
    }
    fr_bench_report r={.blocks=2,.samples=10000,.burst_max=4,.sustained_ratio=0.97};
    for (unsigned b=0;b<4;++b) {
        double bytes=(10*(b+1)+1)*2*FR_SLICE_BYTES;
        r.percentile[0][b][FR_METAL_TOTAL][0]=30000+bytes/0.1;
    }
    check(!fr_bench_fit(&r) && fabs(r.cost.t0_ns-30000)<1e-4 && fabs(r.cost.bw_eff-1e8)<1,"fit latency intercept and bytes/second on unique reads");
    check(r.cost.bytes[3]==41*2*FR_SLICE_BYTES && r.cost.burst_max==4,"cost bytes count distinct experts plus shared once");
    for (unsigned m=0;m<2;++m) for (unsigned b=0;b<4;++b) {
        double total=r.cost.ns[b]*(m?0.8:1);
        r.mean[m][b]=total*1.05;
        for (unsigned p=0;p<FR_METAL_PHASES;++p) {
            double factor=p==FR_METAL_TOTAL?1:p==FR_METAL_GPU?0.7:p==FR_METAL_OVERHEAD?0.3:0.1;
            r.percentile[m][b][p][0]=total*factor; r.percentile[m][b][p][1]=total*factor*1.2;
        }
    }
    r.working_set=r.resident_bytes=r.blocks*FR_WEIGHT_BLOCK_BYTES;
    r.rss_start=r.rss_end=r.working_set+1000000; r.elapsed=120; r.barriers=80000;
    char text[FR_BENCH_TEXT]; int len=fr_bench_format(&r,text,sizeof(text)); fr_bench_report parsed={0};
    check(len>0 && !fr_bench_parse(text,(size_t)len,&parsed) && parsed.resident_bytes==r.working_set &&
          fabs(parsed.cost.bw_eff-r.cost.bw_eff)<1 && parsed.percentile[1][3][FR_METAL_OVERHEAD][1]==r.percentile[1][3][FR_METAL_OVERHEAD][1],"strict benchmark wire round trip retains cost and distributions");
    check(fr_bench_format(&r,text,16)==FR_INVALID,"bounded formatter refuses short buffer");
    for (unsigned m=0;m<2;++m) for (unsigned b=0;b<4;++b)
        r.percentile[m][b][FR_METAL_GPU][0]=r.percentile[m][b][FR_METAL_GPU][1]=0;
    len=fr_bench_format(&r,text,sizeof(text));
    check(len>0 && !fr_bench_parse(text,(size_t)len,&parsed),"CPU report has no invented GPU execution time");
    fr_bench_report point=r; point.selected_burst=3; point.barriers=20000;
    for (unsigned m=0;m<2;++m) for (unsigned b=0;b<4;++b) if (b!=2) {
        point.mean[m][b]=0; memset(point.percentile[m][b],0,sizeof(point.percentile[m][b]));
    }
    check(!fr_bench_fit(&point),"single configured burst produces a measured point without inventing a slope");
    len=fr_bench_format(&point,text,sizeof(text));
    check(len>0 && !fr_bench_parse(text,(size_t)len,&parsed) && parsed.selected_burst==3 &&
          parsed.cost.ns[2]>0 && parsed.cost.ns[0]==0,"version two point report round trip retains only the measured burst");
    point.mean[0][0]=1;
    check(fr_bench_format(&point,text,sizeof(text))==FR_INVALID,"point report refuses fabricated unmeasured bursts");
    len=fr_bench_format(&r,text,sizeof(text)); fr_bench_report before=parsed;
    text[len-1]='x';
    check(fr_bench_parse(text,(size_t)len,&parsed)==FR_INVALID && !memcmp(&parsed,&before,sizeof(parsed)),"malformed benchmark is rejected atomically");
    r.resident_bytes=0;
    check(fr_bench_format(&r,text,sizeof(text))==FR_INVALID,"unverified residency cannot become a profile"); r.resident_bytes=r.working_set;
    r.samples=99; check(fr_bench_format(&r,text,sizeof(text))==FR_INVALID,"insufficient sampling cannot become a profile"); r.samples=10000;
    r.hot_allocations=1; check(fr_bench_format(&r,text,sizeof(text))==FR_INVALID,"hot allocation invalidates benchmark"); r.hot_allocations=0;
    r.working_set=r.resident_bytes=r.blocks*FR_NATIVE_BLOCK_BYTES;
    check(!fr_bench_fit(&r) && r.cost.bytes[3]==r.blocks*(40*FR_NATIVE_SLICE_BYTES+FR_NATIVE_SHARED_BYTES),"native cost counts Q8 shared bytes separately");
    len=fr_bench_format(&r,text,sizeof(text));
    check(len>0 && !fr_bench_parse(text,(size_t)len,&parsed) && parsed.resident_bytes==r.working_set,"native residency and byte model survive BENCH round trip");
    r.workers=4; r.prefetch=2;
    len=fr_bench_format(&r,text,sizeof(text));
    check(len>0 && !fr_bench_parse(text,(size_t)len,&parsed) && parsed.workers==4 && parsed.prefetch==2,"BENCH-selected runtime knobs survive report round trip");
    r.percentile[0][0][FR_METAL_TOTAL][0]=NAN;
    check(fr_bench_fit(&r)==FR_NUMERIC,"nonfinite timing cannot become a cost model");
    r.burst_max=0; check(fr_bench_fit(&r)==FR_INVALID,"empty benchmark refused");
    return test_end();
}
