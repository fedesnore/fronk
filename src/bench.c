/* bench.c - real Q4 kernel measurements and a bounded cost model.
 * Distinct routing measures t0/bandwidth; repeated routing separately measures
 * burst reuse. Full, resident synthetic ownership blocks defeat small-fixture
 * cache artifacts. No model weights, system tuning or runtime dependencies.
 */
#include "bench.h"
#include "neon.h"
#include "avx2.h"
#include "native.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static int quick_valid(const fr_bench_report *r) {
    if (!r || !r->blocks || r->blocks>5 /* bounded installer sweep */ ||
        r->samples!=FR_PLAN_SAMPLES || r->selected_burst || r->burst_max!=4 ||
        r->resident_bytes!=r->blocks*FR_NATIVE_BLOCK_BYTES || r->workers>256 || r->prefetch>64) return 0;
    for (unsigned b=0;b<4;++b) {
        double a=r->percentile[0][b][FR_METAL_TOTAL][0],z=r->percentile[0][b][FR_METAL_TOTAL][1];
        if (!isfinite(a) || !isfinite(z) || a<=0 || z<a || z>1e12) return 0;
    }
    return 1;
}
int fr_bench_quick_format(const fr_bench_report *r,char *out,size_t cap) {
    if (!out || !cap || !quick_valid(r)) return -1;
    int n=snprintf(out,cap,"FRQUICK1 %u %u %u %u %llu",r->blocks,r->samples,r->workers,r->prefetch,(unsigned long long)r->resident_bytes);
    if (n<0 || (size_t)n>=cap) return -1;
    size_t used=(size_t)n;
    for (unsigned b=0;b<4;++b) {
        n=snprintf(out+used,cap-used," %.17g %.17g",r->percentile[0][b][FR_METAL_TOTAL][0],r->percentile[0][b][FR_METAL_TOTAL][1]);
        if (n<0 || (size_t)n>=cap-used) return -1;
        used+=(size_t)n;
    }
    if (used+2>cap) return -1;
    out[used++]='\n'; out[used]=0; return (int)used;
}
int fr_bench_quick_parse(const char *text,size_t len,fr_bench_report *out) {
    if (!text || !out || !len || len>=1024 || text[len-1]!='\n' || memchr(text,0,len)) return -1;
    char line[1024]; memcpy(line,text,len); line[len]=0;
    fr_bench_report r={.burst_max=4}; unsigned long long resident=0; int used=0;
    if (sscanf(line,"FRQUICK1 %u %u %u %u %llu%n",&r.blocks,&r.samples,&r.workers,&r.prefetch,&resident,&used)!=5) return -1;
    r.resident_bytes=resident; const char *p=line+used;
    for (unsigned b=0;b<4;++b) {
        used=0;
        if (sscanf(p," %lf %lf%n",&r.percentile[0][b][FR_METAL_TOTAL][0],&r.percentile[0][b][FR_METAL_TOTAL][1],&used)!=2) return -1;
        p+=used;
    }
    if (strcmp(p,"\n") || !quick_valid(&r)) return -1;
    *out=r; return 0;
}

static const char *const phase_names[]={"encode","commit_to_start","gpu","completion_to_return","overhead","total"};
typedef struct { char key[80]; double value; } bench_field;
static unsigned fields(const fr_bench_report *r,bench_field f[128]) {
    unsigned n=0;
#define FIELD(key_,value_) do { snprintf(f[n].key,sizeof(f[n].key),"%s",key_); f[n++].value=(double)(value_); } while (0)
    FIELD("bench_version",(r->workers?2:0)+(r->selected_burst?2:1)); FIELD("bench_blocks",r->blocks); FIELD("bench_samples",r->samples);
    FIELD("bench_burst_max",r->burst_max); FIELD("bench_working_set_bytes",r->working_set);
    FIELD("bench_resident_bytes",r->resident_bytes); FIELD("bench_rss_start",r->rss_start); FIELD("bench_rss_end",r->rss_end);
    FIELD("bench_barriers",r->barriers); FIELD("bench_hot_allocations",r->hot_allocations); FIELD("bench_elapsed_s",r->elapsed);
    FIELD("bw_eff_gbs",r->cost.bw_eff/1e9); FIELD("t0_us",r->cost.t0_ns/1000);
    FIELD("burst_slope",r->burst_slope); FIELD("sustained_ratio",r->sustained_ratio);
    if (r->selected_burst) FIELD("bench_selected_burst",r->selected_burst);
    for (unsigned m=0;m<2;++m) for (unsigned b=0;b<4;++b) {
        snprintf(f[n].key,sizeof(f[n].key),"bench_%s_%u_mean_ns",m?"reused":"distinct",b+1);
        f[n++].value=r->mean[m][b];
        for (unsigned p=0;p<FR_METAL_PHASES;++p) for (unsigned q=0;q<2;++q) {
            snprintf(f[n].key,sizeof(f[n].key),"submission_%s_%u_%s_p%u_ns",m?"reused":"distinct",b+1,phase_names[p],q?99:50);
            f[n++].value=r->percentile[m][b][p][q];
        }
    }
    if (r->workers) { FIELD("bench_workers",r->workers); FIELD("bench_prefetch_groups",r->prefetch); }
#undef FIELD
    return n;
}
static int valid(const fr_bench_report *r) {
    if (!r || r->blocks<1 || r->blocks>FR_BLOCKS || r->samples!=FR_BENCH_SAMPLES || r->burst_max!=4 || r->selected_burst>4 ||
        r->workers>32 || (r->prefetch!=0 && r->prefetch!=2) || (!r->workers && r->prefetch) ||
        (r->working_set!=r->blocks*FR_WEIGHT_BLOCK_BYTES && r->working_set!=r->blocks*FR_NATIVE_BLOCK_BYTES) || r->resident_bytes!=r->working_set ||
        r->rss_start<r->working_set || r->rss_end<r->working_set || r->barriers<(r->selected_burst?2:8)*FR_BENCH_SAMPLES ||
        r->hot_allocations || !isfinite(r->elapsed) || r->elapsed<30 ||
        !isfinite(r->sustained_ratio) || r->sustained_ratio<=0 ||
        !isfinite(r->cost.bw_eff) || r->cost.bw_eff<=0 || !isfinite(r->cost.t0_ns) || r->cost.t0_ns<0 ||
        !isfinite(r->burst_slope)) return 0;
    for (unsigned m=0;m<2;++m) for (unsigned b=0;b<4;++b) {
        if (r->selected_burst && b+1!=r->selected_burst) {
            if (r->mean[m][b]!=0) return 0;
            for (unsigned p=0;p<FR_METAL_PHASES;++p) for (unsigned q=0;q<2;++q) if (r->percentile[m][b][p][q]!=0) return 0;
            continue;
        }
        if (!isfinite(r->mean[m][b]) || r->mean[m][b]<=0) return 0;
        for (unsigned p=0;p<FR_METAL_PHASES;++p) {
            double a=r->percentile[m][b][p][0],z=r->percentile[m][b][p][1];
            if (!isfinite(a) || !isfinite(z) || a<0 || z<a) return 0;
        }
        if (r->percentile[m][b][FR_METAL_TOTAL][0]<=0) return 0;
    }
    return 1;
}
int fr_bench_format(const fr_bench_report *r,char *out,size_t cap) {
    if (!out || !cap || !valid(r)) return FR_INVALID;
    bench_field f[128]; unsigned count=fields(r,f); size_t used=0;
    for (unsigned i=0;i<count;++i) {
        int n=snprintf(out+used,cap-used,"%s:%.17g\r\n",f[i].key,f[i].value);
        if (n<0 || (size_t)n>=cap-used) return FR_INVALID;
        used+=(size_t)n;
    }
    return (int)used;
}
int fr_bench_parse(const char *text,size_t len,fr_bench_report *out) {
    if (!text || !out || !len || len>=FR_BENCH_TEXT || memchr(text,0,len)) return FR_INVALID;
    fr_bench_report r={0};
    if (len>17 && !memcmp(text,"bench_version:",14)) {
        if (text[14]=='2' || text[14]=='4') r.selected_burst=1;
        if (text[14]=='3' || text[14]=='4') r.workers=1;
    }
    bench_field f[128]; unsigned count=fields(&r,f); const char *end=text+len;
    for (unsigned i=0;i<count;++i) {
        size_t key=strlen(f[i].key); if ((size_t)(end-text)<key+3 || memcmp(text,f[i].key,key) || text[key]!=':') return FR_INVALID;
        text+=key+1; const char *e=memchr(text,'\r',(size_t)(end-text));
        if (!e || e+1==end || e[1]!='\n' || e-text>=64 || e==text) return FR_INVALID;
        char number[64],*tail; memcpy(number,text,(size_t)(e-text)); number[e-text]=0;
        if (number[0]==' ' || number[0]=='\t') return FR_INVALID;
        errno=0; f[i].value=strtod(number,&tail);
        if (errno || *tail || !isfinite(f[i].value)) return FR_NUMERIC;
        text=e+2;
    }
    if (text!=end || f[0].value!=(r.workers?2:0)+(r.selected_burst?2:1) || f[1].value<1 || f[1].value>FR_BLOCKS || f[2].value!=FR_BENCH_SAMPLES || f[3].value!=4) return FR_INVALID;
    for (unsigned i=0;i<10;++i) if (f[i].value<0 || f[i].value>0x1fffffffffffffp0 || floor(f[i].value)!=f[i].value) return FR_INVALID;
    r.blocks=(unsigned)f[1].value; r.samples=(unsigned)f[2].value; r.burst_max=4;
    r.working_set=(uint64_t)f[4].value; r.resident_bytes=(uint64_t)f[5].value;
    r.rss_start=(uint64_t)f[6].value; r.rss_end=(uint64_t)f[7].value; r.barriers=(uint64_t)f[8].value;
    r.hot_allocations=(uint64_t)f[9].value; r.elapsed=f[10].value;
    r.cost.bw_eff=f[11].value*1e9; r.cost.t0_ns=f[12].value*1000;
    r.burst_slope=f[13].value; r.sustained_ratio=f[14].value;
    unsigned i=15;
    if (r.selected_burst) {
        double v=f[i++].value; if (v<1 || v>4 || floor(v)!=v) return FR_INVALID;
        r.selected_burst=(unsigned)v;
    }
    for (unsigned m=0;m<2;++m) for (unsigned b=0;b<4;++b) {
        r.mean[m][b]=f[i++].value;
        for (unsigned p=0;p<FR_METAL_PHASES;++p) for (unsigned q=0;q<2;++q) r.percentile[m][b][p][q]=f[i++].value;
    }
    if (r.workers) {
        double workers=f[i++].value,prefetch=f[i++].value;
        if (workers<1 || workers>32 || floor(workers)!=workers || (prefetch!=0 && prefetch!=2)) return FR_INVALID;
        r.workers=(unsigned)workers; r.prefetch=(unsigned)prefetch;
    }
    if (!valid(&r)) return FR_INVALID;
    fr_bench_report fitted=r;
    if (fr_bench_fit(&fitted) || fabs(fitted.cost.bw_eff-r.cost.bw_eff)>r.cost.bw_eff*1e-9 ||
        fabs(fitted.cost.t0_ns-r.cost.t0_ns)>fmax(1e-6,r.cost.t0_ns*1e-9) || fabs(fitted.burst_slope-r.burst_slope)>1e-9) return FR_INVALID;
    r.cost=fitted.cost; *out=r; return FR_OK;
}

static uint64_t bench_bytes(const fr_bench_report *r,unsigned burst) {
    return r->working_set==r->blocks*FR_NATIVE_BLOCK_BYTES?
        r->blocks*(FR_ACTIVE_ROUTED*burst*FR_NATIVE_SLICE_BYTES+FR_NATIVE_SHARED_BYTES):
        (uint64_t)(FR_ACTIVE_ROUTED*burst+1)*r->blocks*FR_SLICE_BYTES;
}
int fr_bench_fit(fr_bench_report *r) {
    if (!r || !r->blocks || r->blocks>FR_BLOCKS || r->burst_max<2 || r->burst_max>4) return FR_INVALID;
    if (r->selected_burst) {
        if (r->selected_burst>r->burst_max) return FR_INVALID;
        unsigned b=r->selected_burst-1;
        double ns=r->percentile[0][b][FR_METAL_TOTAL][0];
        if (!isfinite(ns) || ns<=0) return FR_NUMERIC;
        uint64_t bytes=bench_bytes(r,b+1);
        r->cost=(fr_costmodel){.burst_max=r->burst_max,.bw_eff=1e9*bytes/ns};
        r->cost.ns[b]=ns; r->cost.bytes[b]=bytes; r->burst_slope=0; return FR_OK;
    }
    double sx=0,sy=0,sxx=0,sxy=0,n=r->burst_max;
    for (unsigned b=0;b<r->burst_max;++b) {
        double x=(double)bench_bytes(r,b+1);
        double y=r->percentile[0][b][FR_METAL_TOTAL][0];
        if (!isfinite(y) || y<=0) return FR_NUMERIC;
        sx+=x; sy+=y; sxx+=x*x; sxy+=x*y;
    }
    double slope=(n*sxy-sx*sy)/(n*sxx-sx*sx), t0=(sy-slope*sx)/n;
    if (!isfinite(slope) || slope<=0) return FR_NUMERIC;
    if (t0<0) { t0=0; slope=sxy/sxx; } /* nonnegative least-squares boundary */
    r->cost=(fr_costmodel){.burst_max=r->burst_max,.t0_ns=t0,.bw_eff=1e9/slope};
    for (unsigned b=0;b<r->burst_max;++b) {
        r->cost.ns[b]=r->percentile[0][b][FR_METAL_TOTAL][0];
        r->cost.bytes[b]=bench_bytes(r,b+1);
    }
    r->burst_slope=(r->cost.ns[r->burst_max-1]/r->cost.ns[0]-1)/(r->burst_max-1);
    return FR_OK;
}

#ifdef FR_BENCH_RUNTIME
#include <sys/mman.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
static unsigned configured_blocks=3,configured_burst;
static const atomic_int *configured_cancel;
static fr_bench_report last_report;
static int last_valid;
int fr_bench_configure(unsigned blocks,const atomic_int *cancel) {
    if (blocks<1 || blocks>FR_BLOCKS) return FR_INVALID;
    configured_blocks=blocks; configured_burst=0; configured_cancel=cancel; return FR_OK;
}
int fr_bench_configure_point(unsigned blocks,unsigned burst,const atomic_int *cancel) {
    if (!burst || burst>FR_BENCH_BURST_MAX) return FR_INVALID;
    int rc=fr_bench_configure(blocks,cancel); if (!rc) configured_burst=burst; return rc;
}
int fr_bench_execute_backend(const fr_backend *backend,int burst_max,fr_costmodel *out) {
    if (!backend || !out || burst_max<1 || burst_max>FR_BENCH_BURST_MAX) return FR_INVALID;
    last_valid=0;
    struct timespec stamp; char path[192];
    if ((mkdir("runs",0700) && errno!=EEXIST) || clock_gettime(CLOCK_REALTIME,&stamp)) return FR_IO;
    int n=snprintf(path,sizeof(path),"runs/bench-%s-%u-%u-%lld-%09ld.csv",backend->name,configured_blocks,configured_burst,(long long)stamp.tv_sec,stamp.tv_nsec);
    if (n<0 || (size_t)n>=sizeof(path)) return FR_IO;
    FILE *csv=fopen(path,"wx"); if (!csv) return FR_IO;
    int rc=fr_bench_measure_backend(backend,configured_blocks,FR_BENCH_SAMPLES,4,configured_burst,30,NULL,csv,configured_cancel,&last_report);
    if (fclose(csv) && !rc) rc=FR_IO;
    if (!rc) { last_valid=1; *out=last_report.cost; out->burst_max=(unsigned)burst_max; }
    return rc;
}
int fr_bench_execute(int burst_max,fr_costmodel *out) {
#ifdef FR_METAL_COMPUTE
    return fr_bench_execute_backend(&fr_metal,burst_max,out);
#else
    return fr_bench_execute_backend(&fr_ref,burst_max,out);
#endif
}
int fr_bench_last(fr_bench_report *out) {
    if (!out) return FR_INVALID;
    if (!last_valid) return FR_STATE;
    *out=last_report; return FR_OK;
}
static double seconds_now(void) { struct timespec t; if (clock_gettime(CLOCK_MONOTONIC,&t)) return -1; return t.tv_sec+t.tv_nsec*1e-9; }
static uint64_t rss(void) {
#ifdef __APPLE__
    struct mach_task_basic_info t; mach_msg_type_number_t n=MACH_TASK_BASIC_INFO_COUNT;
    return task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&t,&n)==KERN_SUCCESS?t.resident_size:0;
#else
    FILE *f=fopen("/proc/self/statm","r"); unsigned long total=0,resident=0;
    if (!f) return 0;
    int n=fscanf(f,"%lu %lu",&total,&resident); fclose(f);
    return n==2?(uint64_t)resident*(uint64_t)sysconf(_SC_PAGESIZE):0;
#endif
}
static int compare_double(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
static uint64_t random64(uint64_t *s) { *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
static int resident(unsigned char *const weights[FR_BLOCKS],unsigned blocks,size_t unit,char *pages,size_t page,uint64_t *bytes) {
    *bytes=0;
    for (unsigned b=0;b<blocks;++b) {
        if (mincore(weights[b],unit,(void *)pages)) return FR_IO;
        for (size_t i=0;i<unit/page;++i) if (pages[i]&1) *bytes+=page;
    }
    if (*bytes!=blocks*unit) {
        fprintf(stderr,"BENCH residency failed resident=%llu required=%llu rss=%llu\n",
                (unsigned long long)*bytes,(unsigned long long)(blocks*unit),(unsigned long long)rss());
        return FR_STATE;
    }
    return FR_OK;
}
static int backend_close(const fr_backend *b) {
#ifdef FR_METAL_COMPUTE
    if (b==&fr_metal) return fr_metal_close();
#endif
#ifdef FR_NEON_COMPUTE
    if (b==&fr_neon) return fr_neon_close();
#endif
#ifdef FR_AVX2_COMPUTE
    if (b==&fr_avx2) return fr_avx2_close();
#endif
    /* Reference benches use the same load-state guard as self-test. Any other
     * backend (backends/user.c) refuses the benchmark's load into an occupied slot. */
    return b==&fr_ref || b==&fr_ref_q8 ? fr_ref_idle(b):FR_OK;
}
static void route(fr_req *r,uint64_t pass,unsigned burst,int reused) {
    r->burst=burst; r->layer=(unsigned)(pass%FR_LAYERS);
    unsigned base=(unsigned)((pass/FR_LAYERS)*41%FR_ROUTED_EXPERTS);
    for (unsigned t=0;t<burst;++t) for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e)
        r->expert[t][e]=(uint16_t)((base+(reused?0:t*FR_ACTIVE_ROUTED)+e)%FR_ROUTED_EXPERTS);
}
int fr_bench_measure_backend(const fr_backend *backend,unsigned blocks,unsigned samples,unsigned burst_max,unsigned selected_burst,double duration,
                     FILE *progress,FILE *csv,const atomic_int *cancel,fr_bench_report *out) {
    if (!backend || !out || !blocks || blocks>FR_BLOCKS || !samples || samples>FR_BENCH_SAMPLES || burst_max<2 || burst_max>4 ||
        selected_burst>burst_max || !isfinite(duration) || duration<=0) return FR_INVALID;
    /* Every backend loads the native units the head ships (BACKENDS.md); the
     * legacy Q4_0 layout lives on only in the oracle vectors. */
    int native=1;
    size_t unit=native?FR_NATIVE_BLOCK_BYTES:FR_WEIGHT_BLOCK_BYTES;
    fr_bench_report result={.blocks=blocks,.samples=samples,.burst_max=burst_max,.selected_burst=selected_burst,.working_set=blocks*unit};
    /* A benchmark owns isolated backend state; never evict session weights. */
    int closed=backend_close(backend); if (closed) return closed;
    double *data=calloc((size_t)2*burst_max*samples*FR_METAL_PHASES,sizeof(double));
    double *sorted=malloc(samples*sizeof(double)); unsigned char *weights[FR_BLOCKS]={0};
    long page_size=sysconf(_SC_PAGESIZE); size_t page=page_size>0?(size_t)page_size:16384;
    char *pages=malloc(unit/page);
    int rc=(!data || !sorted || !pages || page_size<=0 || unit%page)?FR_IO:FR_OK; unsigned loaded=0;
    if (!rc) {
        volatile double *touch=data;
        for (size_t i=0;i<(size_t)2*burst_max*samples*FR_METAL_PHASES;++i) touch[i]=0;
        touch=sorted; for (unsigned i=0;i<samples;++i) touch[i]=0;
    }
    for (unsigned b=0;!rc && b<blocks;++b) {
        if (cancel && atomic_load(cancel)) { rc=FR_STATE; break; }
        void *p=mmap(NULL,unit,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (p==MAP_FAILED) { rc=FR_IO; break; } weights[b]=p;
        uint64_t seed=UINT64_C(0x9e3779b97f4a7c15)^(UINT64_C(0x94d049bb133111eb)*(b+1));
        if (native) fr_native_fill(weights[b],(uint32_t)seed);
        else for (size_t q=0;q<unit;q+=FR_Q4_BYTES) {
            uint64_t a=random64(&seed),v=random64(&seed);
            uint16_t scale=(uint16_t)(0x1800|((a>>48)&0x83ff)); /* finite, signed, nonzero */
            weights[b][q]=(unsigned char)scale; weights[b][q+1]=(unsigned char)(scale>>8);
            memcpy(weights[b]+q+2,&a,8); memcpy(weights[b]+q+10,&v,8);
        }
        rc=backend->load((int)b,weights[b],unit); if (!rc) ++loaded;
    }
    /* Correctness fixtures may be sparse. Timing storage never is: every byte
     * above is written with random canonical Q4 data, and every page must be
     * resident before even the warmup submissions begin. */
    if (!rc) {
        /* Earlier blocks may have been compressed while later ones were
         * initialized. Fault all fully written pages back in before checking;
         * the end-of-run check below deliberately does not do this. */
        for (unsigned b=0;b<blocks;++b) for (size_t offset=0;offset<unit;offset+=page)
            (void)*(volatile const unsigned char *)(weights[b]+offset);
    }
    if (!rc) rc=resident(weights,blocks,unit,pages,page,&result.resident_bytes);
    if (!rc && rss()<result.working_set) {
        fprintf(stderr,"BENCH initial RSS below working set rss=%llu required=%llu\n",
                (unsigned long long)rss(),(unsigned long long)result.working_set); rc=FR_STATE;
    }
    fr_req req={0}; fr_part part;
    for (unsigned t=0;t<FR_MAX_BURST;++t) {
        for (unsigned e=0;e<FR_ACTIVE_EXPERTS;++e) req.gate[t][e]=e==FR_ACTIVE_ROUTED?0.5f:0.0625f;
        for (unsigned d=0;d<FR_HIDDEN;++d) req.x[t][d]=(float)((int)((d*37+t*19)%257)-128)/128;
    }
#ifdef FR_AVX2_COMPUTE
    if (!rc && backend==&fr_avx2) {
        if (csv) fputs("mode,burst,sample,encode_ns,commit_to_start_ns,gpu_ns,completion_to_return_ns,overhead_ns,total_ns,workers,prefetch_groups\n",csv);
        route(&req,0,selected_burst?selected_burst:burst_max,0);
        rc=fr_avx2_tune(&req,csv);
        if (!rc) {
            fr_avx2_sample sample; fr_avx2_sample_get(&sample);
            result.workers=sample.workers; result.prefetch=sample.prefetch;
            if (progress) { fprintf(progress,"BENCH avx2 workers=%u prefetch_groups=%u format=Q4_K/MXFP4 shared=Q8_0\n",sample.workers,sample.prefetch); fflush(progress); }
        }
    }
#endif
    for (unsigned warm=0;!rc && warm<512;++warm) {
        route(&req,warm,selected_burst?selected_burst:warm%burst_max+1,(warm/burst_max)%2); rc=backend->run(&req,&part);
    }
    double start=seconds_now(), next_log=start+60,sustained_start=-1;
    result.rss_start=rss();
    if (!rc) rc=resident(weights,blocks,unit,pages,page,&result.resident_bytes);
    if (!rc && (!result.rss_start || start<0)) rc=FR_IO;
    if (!rc && result.rss_start<result.working_set) rc=FR_STATE;
    if (!rc && progress) { fprintf(progress,"start blocks=%u working_set=%llu resident_bytes=%llu residency=mincore_all_pages+RSS samples=%u duration_s=%.0f rss=%llu\n",blocks,
        (unsigned long long)result.working_set,(unsigned long long)result.resident_bytes,samples,duration,(unsigned long long)result.rss_start); fflush(progress); }
    unsigned counts[2][4]={{0}},mode=0,burst=0; uint64_t pass=0;
    double last_sum[4]={0}; uint64_t last_count[4]={0};
    while (!rc) {
        double now=seconds_now();
        int sampling=counts[1][selected_burst?selected_burst-1:burst_max-1]<samples;
        if (!sampling && sustained_start<0) sustained_start=now;
        if (!sampling && now-sustained_start>=duration) break;
        if (cancel && atomic_load(cancel)) { rc=FR_STATE; break; }
        if (sampling) {
            /* Alternate modes and bursts to avoid temperature/order confounds. */
            burst=(unsigned)(pass%burst_max); mode=(unsigned)(pass/burst_max)%2;
        } else { burst=(unsigned)(pass%burst_max); mode=0; }
        if (selected_burst) { burst=selected_burst-1; mode=sampling?(unsigned)(pass%2):0; }
        route(&req,pass,burst+1,(int)mode); double began=seconds_now(); rc=backend->run(&req,&part); if (rc) break;
        fr_metal_sample sample={0};
        sample.ns[FR_METAL_TOTAL]=(seconds_now()-began)*1e9;
        sample.ns[FR_METAL_OVERHEAD]=sample.ns[FR_METAL_TOTAL];
#ifdef FR_METAL_COMPUTE
        if (backend==&fr_metal) fr_metal_sample_get(&sample);
#endif
#ifdef FR_NEON_COMPUTE
        if (backend==&fr_neon) { fr_neon_sample cpu; fr_neon_sample_get(&cpu); sample.hot_allocations=cpu.hot_allocations; }
#endif
#ifdef FR_AVX2_COMPUTE
        if (backend==&fr_avx2) { fr_avx2_sample cpu; fr_avx2_sample_get(&cpu); sample.hot_allocations=cpu.hot_allocations; }
#endif
        if (sample.hot_allocations) { rc=FR_STATE; break; }
        if (counts[mode][burst]<samples) {
            size_t row=((size_t)mode*burst_max+burst)*samples+counts[mode][burst]++;
            memcpy(data+row*FR_METAL_PHASES,sample.ns,sizeof(sample.ns));
            result.mean[mode][burst]+=sample.ns[FR_METAL_TOTAL]/samples;
        }
        if (!sampling && mode==0 && now-sustained_start>=fmax(0,duration-30)) { last_sum[burst]+=sample.ns[FR_METAL_TOTAL]; ++last_count[burst]; }
        ++pass;
        if (progress && now>=next_log) {
            fprintf(progress,"elapsed_s=%.1f barriers=%llu rss=%llu hot_allocations=%llu\n",now-start,
                    (unsigned long long)pass,(unsigned long long)rss(),(unsigned long long)sample.hot_allocations);
            fflush(progress); next_log=now+60;
        }
    }
    result.elapsed=seconds_now()-start; result.rss_end=rss(); result.barriers=pass;
    if (!rc) rc=resident(weights,blocks,unit,pages,page,&result.resident_bytes);
    if (!rc && result.rss_end<result.working_set) rc=FR_STATE;
    if (!rc) for (unsigned m=0;m<2;++m) for (unsigned b=0;b<burst_max;++b) {
        if (selected_burst && b+1!=selected_burst) continue;
        if (counts[m][b]!=samples) { rc=FR_STATE; break; }
        for (unsigned phase=0;phase<FR_METAL_PHASES;++phase) {
            for (unsigned j=0;j<samples;++j) sorted[j]=data[(((size_t)m*burst_max+b)*samples+j)*FR_METAL_PHASES+phase];
            qsort(sorted,samples,sizeof(double),compare_double);
            result.percentile[m][b][phase][0]=sorted[(samples+1)/2-1];
            result.percentile[m][b][phase][1]=sorted[(99*samples+99)/100-1];
        }
    }
    if (!rc) {
        double baseline=0,sustained=0;
        for (unsigned b=0;b<burst_max;++b) {
            if (selected_burst && b+1!=selected_burst) continue;
            baseline+=result.mean[0][b];
            if (!last_count[b]) { rc=FR_STATE; break; }
            sustained+=last_sum[b]/last_count[b];
        }
        result.sustained_ratio=baseline/sustained;
        if (!rc) rc=fr_bench_fit(&result);
    }
    if (!rc && csv) {
        if (!result.workers) fputs("mode,burst,sample,encode_ns,commit_to_start_ns,gpu_ns,completion_to_return_ns,overhead_ns,total_ns,workers,prefetch_groups\n",csv);
        for (unsigned m=0;m<2;++m) for (unsigned b=0;b<burst_max;++b) for (unsigned j=0;j<samples;++j) {
            if (selected_burst && b+1!=selected_burst) continue;
            fprintf(csv,"%s,%u,%u",m?"reused":"distinct",b+1,j);
            for (unsigned k=0;k<FR_METAL_PHASES;++k) fprintf(csv,",%.0f",data[(((size_t)m*burst_max+b)*samples+j)*FR_METAL_PHASES+k]);
            fprintf(csv,",%u,%u\n",result.workers,result.prefetch);
        }
        if (fflush(csv) || ferror(csv)) rc=FR_IO;
    }
    if (progress) { fprintf(progress,"end elapsed_s=%.3f barriers=%llu rss_start=%llu rss_end=%llu status=%d\n",result.elapsed,
        (unsigned long long)pass,(unsigned long long)result.rss_start,(unsigned long long)result.rss_end,rc); fflush(progress); }
    for (unsigned b=0;b<loaded;++b) if (backend->unload((int)b) && !rc) rc=FR_STATE;
    for (unsigned b=0;b<blocks;++b) if (weights[b]) munmap(weights[b],unit);
    (void)backend_close(backend); free(data); free(sorted); free(pages);
    if (!rc) *out=result;
    return rc;
}
int fr_bench_measure(unsigned blocks,unsigned samples,unsigned burst_max,double duration,
                     FILE *progress,FILE *csv,const atomic_int *cancel,fr_bench_report *out) {
#ifdef FR_METAL_COMPUTE
    return fr_bench_measure_backend(&fr_metal,blocks,samples,burst_max,0,duration,progress,csv,cancel,out);
#else
    (void)blocks; (void)samples; (void)burst_max; (void)duration; (void)progress; (void)csv; (void)cancel; (void)out;
    return FR_UNSUPPORTED;
#endif
}
#endif
