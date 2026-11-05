/* Selection is a comparison of measured costs, never an architecture preference.
 * Linear extrapolation is needed only beyond the bounded five-block sweep.
 */
#include "selection.h"
#include "native.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef FR_BENCH_RUNTIME
#include "backend.h"
#include "bench.h"
#include "sha256.h"
#include "avx2.h"
#include <sys/stat.h>
#include <unistd.h>
_Static_assert(FR_CANDIDATES>=FR_BACKENDS_MAX,"selection must hold every compiled backend");
/* The backend this machine was told to use (config "backend"); empty races all. */
static char chosen[16];
int selection_discover(const fr_hwinfo *hw,unsigned burst,int head,fr_selection *out) {
    if (!hw || !out || !burst || burst>FR_BENCH_BURST_MAX) return -1;
    fr_selection s={.burst=burst};
    /* Preserve built-in tie order, then consider every compiled source module. */
    const char *names[FR_CANDIDATES+1]={"ref","ref-q8","neon","metal","avx2"}; unsigned count=5;
    for (unsigned i=0;fr_backend_at(i) && count<FR_CANDIDATES+1;++i) {
        unsigned j=0; while (j<count && strcmp(names[j],fr_backend_at(i)->name)) ++j;
        if (j==count) names[count++]=fr_backend_at(i)->name;
    }
    if (!hw->translated) for (unsigned i=0;i<count;++i) {
        const fr_backend *b=fr_backend_find(names[i]); fr_hwinfo actual;
        if (!b || (chosen[0] && strcmp(chosen,b->name))) continue;
        int rc=b->probe(&actual);
        if (rc==FR_UNSUPPORTED) continue;
        if (rc) return -1;
        if (selection_add(&s,b->name,(unsigned)selection_capacity(hw,b->name,head))) return -1;
    }
    *out=s; return 0;
}
int selection_backend(const char *name,FILE *err) {
    chosen[0]=0;
    if (!name || !name[0]) return 0;
    const fr_backend *b=fr_backend_find(name); fr_hwinfo hw;
    if (!b) { if (err) fprintf(err,"backend %s is not in this build (built: %s); see ./fronk backends\n",name,fr_backends_compiled()); return -1; }
    int rc=b->probe(&hw);
    if (rc) { if (err) fprintf(err,"backend %s cannot run on this machine (%s); see ./fronk backends\n",name,rc==FR_UNSUPPORTED?"unsupported hardware":"probe failed"); return -1; }
    snprintf(chosen,sizeof(chosen),"%s",name); return 0;
}
const char *selection_chosen(void) { return chosen[0]?chosen:NULL; }
/* A rule, not a measurement: the GPU on Apple Silicon, else the CPU's widest
 * built-in kernel. Measured on this cluster, metal beat neon on both M-series
 * Macs and avx2 beat ref by ~100x on the x86 nodes (BENCH.md). */
const char *selection_recommend_cpu(const fr_hwinfo *hw) {
    const char *order[]={"avx2","neon","ref-q8"}; fr_hwinfo probe;
    if (!hw || hw->translated) return NULL;
    for (unsigned i=0;i<sizeof(order)/sizeof(order[0]);++i) {
        const fr_backend *b=fr_backend_find(order[i]);
        if (b && !b->probe(&probe)) return b->name;
    }
    return "ref";
}
const char *selection_recommend(const fr_hwinfo *hw) {
    const fr_backend *metal=fr_backend_find("metal"); fr_hwinfo probe;
    if (hw && !hw->translated && metal && !metal->probe(&probe)) return metal->name;
    return selection_recommend_cpu(hw);
}
/* Cache entries live in DIR/bench/<sha256(key)>.txt; the key binds the compute build,
 * hardware identity, backend, unit format, blocks, burst and sample count. */
static char cache_identity[160];
void selection_cache_identity(const char *identity) { snprintf(cache_identity,sizeof(cache_identity),"%s",identity); }
const char *selection_cache_identity_get(void) { return cache_identity; }
static void cache_path(const char *dir,const char *backend,unsigned b,unsigned burst,unsigned samples,char *out,size_t size) {
    char key[320]; unsigned char d[32]; char hex[65];
    snprintf(key,sizeof(key),"%s|%s|native|%u|%u|%u",cache_identity,backend,b,burst,samples);
    sha256(key,strlen(key),d); for (unsigned i=0;i<32;++i) snprintf(hex+2*i,3,"%02x",d[i]);
    snprintf(out,size,"%s/bench/%s.txt",dir,hex);
}
/* A probe (fewer than FR_BENCH_SAMPLES) is not a valid full report, so only its
 * p50/p99 are saved: unsaved, every run re-probed for ~25 s (BENCH run 42). */
static int probe_parse(const char *text,size_t n,unsigned burst,fr_bench_report *r) {
    char line[128]; double p50,p99; if (n>=sizeof(line)) return -1;
    memcpy(line,text,n); line[n]=0;
    if (sscanf(line,"probe %lg %lg",&p50,&p99)!=2 || !isfinite(p50) || !isfinite(p99) || p50<=0 || p99<p50) return -1;
    memset(r,0,sizeof(*r)); r->percentile[0][burst-1][FR_METAL_TOTAL][0]=p50; r->percentile[0][burst-1][FR_METAL_TOTAL][1]=p99; return 0;
}
static int measure(const fr_backend *backend,unsigned b,unsigned burst,unsigned samples,const char *dir,int refresh,FILE *progress,const atomic_int *cancel,fr_bench_report *r) {
    char path[600],text[FR_BENCH_TEXT]; FILE *f; int probe=samples<FR_BENCH_SAMPLES;
    char quick_path[600]={0};
    if (cancel && atomic_load(cancel)) return -1;
    if (dir && cache_identity[0]) {
        cache_path(dir,backend->name,b,burst,samples,path,sizeof(path));
        if (!refresh && (f=fopen(path,"r"))) {
            size_t n=fread(text,1,sizeof(text),f); fclose(f);
            if (n && (probe?!probe_parse(text,n,burst,r):!fr_bench_parse(text,n,r))) { if (progress) fprintf(progress,"BENCH cached backend=%s blocks=%u burst=%u samples=%u\n",backend->name,b,burst,samples); return 0; }
            /* A share that did not stay resident fails again on the same binary and
             * hardware; retrying it every run loads gigabytes and pages for nothing. */
            if (n>=6 && !memcmp(text,"failed",6)) { if (progress) fprintf(progress,"BENCH cached failure backend=%s blocks=%u burst=%u\n",backend->name,b,burst); return -1; }
        }
    }
    /* A short all-width profile is a planning measurement, not a sustained
     * 10,000-sample BENCH report. Keep legacy full reports above; a cache miss
     * loads weights once and saves all widths for later mode changes. */
    if (!probe && dir && cache_identity[0]) {
        cache_path(dir,backend->name,b,0,FR_PLAN_SAMPLES,quick_path,sizeof(quick_path));
        if (!refresh && (f=fopen(quick_path,"r"))) {
            size_t n=fread(text,1,sizeof(text),f); fclose(f);
            if (!fr_bench_quick_parse(text,n,r) && r->blocks==b) {
                if (progress) fprintf(progress,"BENCH cached quick backend=%s blocks=%u burst=%u (saved widths 1..4; no measurement)\n",backend->name,b,burst);
                return 0;
            }
        }
    }
    if (progress) { fprintf(progress,"BENCH backend=%s blocks=%u burst=%u samples=%u%s (loads %.1f GB once)\n",backend->name,b,burst,probe?samples:FR_PLAN_SAMPLES,probe?"":" per width 1..4; short planning profile",b*(double)FR_NATIVE_BLOCK_BYTES/1e9); fflush(progress); }
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    if (fr_bench_measure_backend(backend,b,probe?samples:FR_PLAN_SAMPLES,4,probe?burst:0,1,NULL,NULL,cancel,r)) {
        if (cancel && atomic_load(cancel)) return -1; /* stopped, not failed: nothing saved */
        if (dir && cache_identity[0]) {
            char bench[600]; snprintf(bench,sizeof(bench),"%s/bench",dir); mkdir(bench,0700);
            if ((f=fopen(path,"w"))) { fputs("failed\n",f); fclose(f); }
        }
        if (progress) { fprintf(progress,"BENCH backend=%s blocks=%u failed; saved, so later runs skip it until BENCH is refreshed\n",backend->name,b); fflush(progress); }
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    if (progress) { fprintf(progress,"BENCH backend=%s blocks=%u done %.1f s\n",backend->name,b,(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9); fflush(progress); }
    int n=probe?snprintf(text,sizeof(text),"probe %.17g %.17g\n",r->percentile[0][burst-1][FR_METAL_TOTAL][0],r->percentile[0][burst-1][FR_METAL_TOTAL][1]):
                fr_bench_quick_format(r,text,sizeof(text));
    if (dir && cache_identity[0] && n>0) {
        char bench[600]; snprintf(bench,sizeof(bench),"%s/bench",dir); mkdir(bench,0700);
        char tmp[640]; const char *dest=probe?path:quick_path;
        snprintf(tmp,sizeof(tmp),"%s.tmp",dest);
        if ((f=fopen(tmp,"w"))) {
            int bad=fwrite(text,1,(size_t)n,f)!=(size_t)n;
            if (fclose(f)) bad=1;
            if (bad || rename(tmp,dest)) unlink(tmp);
        }
    }
    return 0;
}
/* SELFTEST passes are saved like measurements (b=0 is never a measured share).
 * A backend holding loaded blocks refuses SELFTEST, and an unsaved pass made a
 * node with kept units fall back to ref and sweep for many minutes (run 42). */
static int self_test_saved(const char *dir,const char *backend) {
    char path[600],text[16]={0}; if (!dir || !cache_identity[0]) return 0;
    cache_path(dir,backend,0,0,0,path,sizeof(path)); FILE *f=fopen(path,"r"); if (!f) return 0;
    size_t n=fread(text,1,sizeof(text)-1,f); fclose(f); return n>=4 && !memcmp(text,"pass",4);
}
static int self_test(const fr_backend *b,const char *dir,int refresh) {
    if (!refresh && self_test_saved(dir,b->name)) return 0;
    int rc=b->self_test("tests/vectors");
    if (!rc && dir && cache_identity[0]) {
        char path[600],bench[600]; snprintf(bench,sizeof(bench),"%s/bench",dir); mkdir(bench,0700);
        cache_path(dir,b->name,0,0,0,path,sizeof(path)); FILE *f=fopen(path,"w"); if (f) { fputs("pass\n",f); fclose(f); }
    }
    return rc;
}
int selection_self_tests_cached(const fr_hwinfo *hw,unsigned burst,int head,const char *cache) {
    fr_hwinfo stable=*hw; stable.available=hw->total; fr_selection s;
    if (selection_discover(&stable,burst,head,&s)) return 0;
    for (unsigned i=0;i<s.count;++i) if (strncmp(s.candidate[i].name,"ref",3) && !self_test_saved(cache,s.candidate[i].name)) return 0;
    return 1;
}
int selection_sweep_local(const fr_hwinfo *hw,unsigned burst,int head,fr_selection *out,FILE *progress,const char *cache,int refresh,const atomic_int *cancel) {
    fr_selection s; fr_bench_report report;
    /* What gets measured follows total memory, so the cache keys match from run
     * to run; free memory right now only caps the result (below). Sizing the
     * sweep from free memory re-measured everything whenever it moved by a block. */
    fr_hwinfo stable=*hw; stable.available=hw->total;
    if (selection_discover(&stable,burst,head,&s)) return -1;
    /* ref is the oracle: a candidate only if no native backend passes SELFTEST. */
    int native=0; fr_selection kept={.burst=burst};
    for (unsigned i=0;i<s.count;++i) if (strncmp(s.candidate[i].name,"ref",3)) {
        const fr_backend *b=fr_backend_find(s.candidate[i].name);
        if (b && !self_test(b,cache,refresh)) { kept.candidate[kept.count++]=s.candidate[i]; native=1; }
        else if (progress) fprintf(progress,"BENCH backend=%s failed SELFTEST; not a candidate\n",s.candidate[i].name);
    }
    if (!native && chosen[0] && strncmp(chosen,"ref",3)) {
        if (progress) fprintf(progress,"BENCH backend=%s (chosen in the config) failed SELFTEST; fix it or set backend auto\n",chosen);
        return -1;
    }
    if (!native) kept=s;
    /* Short probe at the largest sweep point picks the winner (CPU wins ties). */
    unsigned winner=0; double best=INFINITY;
    for (unsigned i=0;kept.count>1 && i<kept.count;++i) {
        unsigned b=kept.candidate[i].capacity<FR_SWEEP?kept.candidate[i].capacity:FR_SWEEP;
        unsigned live=(unsigned)selection_capacity(hw,kept.candidate[i].name,head);
        if (b>live) b=live;
        /* A probe point that cannot stay resident steps down; none skips the candidate. */
        while (b && measure(fr_backend_find(kept.candidate[i].name),b,burst,200,cache,refresh,progress,cancel,&report)) {
            if (cancel && atomic_load(cancel)) return -1;
            kept.candidate[i].capacity=--b; }
        if (!b) continue;
        double ns=report.percentile[0][burst-1][FR_METAL_TOTAL][0]/b;
        if (ns<best) { best=ns; winner=i; }
    }
    s=(fr_selection){.burst=burst,.count=1}; s.candidate[0]=kept.candidate[winner];
    s.candidate[0].measured=0;
    const fr_backend *backend=fr_backend_find(s.candidate[0].name);
    unsigned live=(unsigned)selection_capacity(hw,s.candidate[0].name,head);
    if (s.candidate[0].capacity>live) s.candidate[0].capacity=live;
    for (unsigned b=1;b<=s.candidate[0].capacity && b<=FR_SWEEP;++b) {
        /* A share that cannot stay resident (BENCH residency check) caps capacity. */
        if (b>1 && measure(backend,b,burst,FR_BENCH_SAMPLES,cache,refresh,progress,cancel,&report)) {
            if (cancel && atomic_load(cancel)) return -1;
            s.candidate[0].capacity=b-1; break; }
        if ((b==1 && measure(backend,b,burst,FR_BENCH_SAMPLES,cache,refresh,progress,cancel,&report)) ||
            selection_record(&s,0,b,report.percentile[0][burst-1][FR_METAL_TOTAL][0],report.percentile[0][burst-1][FR_METAL_TOTAL][1],report.resident_bytes) ||
            selection_settings(&s,0,b,report.workers,report.prefetch)) return -1;
#ifdef FR_AVX2_COMPUTE
        if (backend==&fr_avx2) fr_avx2_set_tuning(b,burst,report.workers,report.prefetch);
#endif
    }
    if (s.candidate[0].capacity>live) {
        if (progress) fprintf(progress,"BENCH %s measured up to %u blocks; free memory now allows %u\n",s.candidate[0].name,s.candidate[0].capacity,live);
        if (selection_limit(&s,0,live)) return -1;
    }
    selection_hybrid_restore(hw,head,&s,cache,progress);
    if (progress) fflush(progress);
    *out=s; return 0;
}
#endif
/* Any well-formed name: a node may run a backend this head was not built with. */
static int known(const char *n) {
    if (!n || !n[0] || strlen(n)>=sizeof(((fr_candidate *)0)->name)) return 0;
    for (const char *p=n;*p;++p) if (!((*p>='a' && *p<='z') || (*p>='0' && *p<='9') || *p=='-' || *p=='_')) return 0;
    return 1;
}
uint64_t selection_capacity(const fr_hwinfo *h,const char *backend,int head) {
    if (!h || !known(backend) || h->translated) return 0;
    uint64_t bytes=h->available,reserve=head?h->head_reserve:h->margin;
    int hybrid=!strcmp(backend,"metal-ssd"),metal=hybrid || !strcmp(backend,"metal");
    if (metal && !h->metal) return 0;
    if (metal) {
        if (bytes>h->metal_limit) bytes=h->metal_limit;
    }
    if (hybrid) {
        uint64_t extra=FR_SSD_DEFAULT_CACHE+FR_SSD_MAX_BLOCKS*FR_NATIVE_LAYER_BYTES+(4ull<<20);
        if (reserve>UINT64_MAX-extra || bytes<=reserve+extra) return 0;
        reserve+=extra;
    }
    uint64_t blocks=bytes>reserve?(bytes-reserve)/FR_NATIVE_BLOCK_BYTES:0;
    if (head && h->head_fixed && h->head_fixed<blocks) blocks=h->head_fixed;
    if (hybrid) blocks+=FR_SSD_MAX_BLOCKS;
    return blocks>FR_BLOCKS?FR_BLOCKS:blocks;
}
int selection_fits(const fr_candidate *c,const fr_hwinfo *h,int head,unsigned blocks) {
    if (!c || !h || !blocks || blocks>c->capacity) return 0;
    if (!c->ssd_blocks) return blocks<=selection_capacity(h,c->name,head);
    uint64_t bytes=h->available,reserve=head?h->head_reserve:h->margin;
    if (!h->metal || h->translated) return 0;
    if (bytes>h->metal_limit) bytes=h->metal_limit;
    unsigned ram=blocks<c->ram_blocks?blocks:c->ram_blocks,ssd=blocks-ram;
    uint64_t need=ram*FR_NATIVE_BLOCK_BYTES+c->cache_bytes+ssd*FR_NATIVE_LAYER_BYTES+(4ull<<20);
    return bytes>reserve && need<=bytes-reserve;
}
uint32_t selection_paged(const fr_candidate *c,uint32_t mask) {
    if (!c || !c->ssd_blocks) return 0;
    unsigned ram=0; uint32_t paged=0;
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask>>b&1) { if (ram<c->ram_blocks) ++ram; else paged|=1u<<b; }
    return paged;
}
int selection_hybrid(fr_selection *s,unsigned ram,unsigned streamed,uint64_t bytes,const double *cold,const double *warm,const double *tail) {
    if (!s || !s->count || s->count>=FR_CANDIDATES || !streamed || streamed>FR_SSD_MAX_BLOCKS ||
        ram+streamed>FR_BLOCKS || bytes<(64ull<<20) || bytes>(2ull<<30) || !cold || !warm || !tail) return -1;
    const fr_candidate *base=NULL;
    for (unsigned i=0;i<s->count;++i) if (!strcmp(s->candidate[i].name,"metal") && s->candidate[i].capacity>=ram) base=&s->candidate[i];
    if (!base) return -1;
    fr_candidate c=*base; strcpy(c.name,"metal-ssd"); c.ram_blocks=ram; c.ssd_blocks=streamed; c.cache_bytes=bytes; c.capacity=ram+streamed;
    unsigned n=ram<FR_SWEEP?ram:FR_SWEEP; c.measured&=(1u<<n)-1;
    for (unsigned i=0;i<streamed;++i) {
        if (!isfinite(cold[i]) || !isfinite(warm[i]) || !isfinite(tail[i]) || cold[i]<=0 || warm[i]<=0 || tail[i]<cold[i]) return -1;
        c.cold_ns[i]=cold[i]; c.warm_ns[i]=warm[i]; c.tail_ns[i]=tail[i];
    }
    for (unsigned i=0;i<s->count;++i) if (!strcmp(s->candidate[i].name,c.name)) return -1;
    s->candidate[s->count++]=c; return 0;
}
int selection_add(fr_selection *s,const char *backend,unsigned capacity) {
    if (!s || !known(backend) || capacity>FR_BLOCKS || s->count>=FR_CANDIDATES || !s->burst || s->burst>FR_BENCH_BURST_MAX) return -1;
    for (unsigned i=0;i<s->count;++i) if (!strcmp(s->candidate[i].name,backend)) return -1;
    fr_candidate *c=&s->candidate[s->count++]; memset(c,0,sizeof(*c)); strcpy(c->name,backend); c->capacity=capacity; return 0;
}
static int valid_resident(unsigned b,uint64_t resident) {
    return resident==b*FR_WEIGHT_BLOCK_BYTES || resident==b*FR_NATIVE_BLOCK_BYTES;
}
int selection_record(fr_selection *s,unsigned i,unsigned b,double p50,double p99,uint64_t resident) {
    if (!s || i>=s->count || !b || b>FR_SWEEP || b>s->candidate[i].capacity ||
        !isfinite(p50) || !isfinite(p99) || p50<=0 || p99<p50 || !valid_resident(b,resident)) return -1;
    fr_candidate *c=&s->candidate[i]; c->ns[b-1]=p50; c->p99[b-1]=p99; c->resident[b-1]=resident; c->measured|=1u<<(b-1); return 0;
}
/* Lower a capacity and forget the points above it: a saved selection never
 * holds a measured share it cannot take (the profile save refused one). */
int selection_limit(fr_selection *s,unsigned i,unsigned capacity) {
    if (!s || i>=s->count || capacity>FR_BLOCKS) return -1;
    fr_candidate *c=&s->candidate[i];
    if (capacity<c->capacity) c->capacity=capacity;
    if (c->capacity<FR_SWEEP) c->measured&=(1u<<c->capacity)-1;
    return 0;
}
int selection_settings(fr_selection *s,unsigned i,unsigned b,unsigned workers,unsigned prefetch) {
    if (!s || i>=s->count || !b || b>FR_SWEEP || !(s->candidate[i].measured&(1u<<(b-1))) ||
        workers>32 || (prefetch!=0 && prefetch!=2) || (!workers && prefetch) ||
        (workers && strcmp(s->candidate[i].name,"avx2"))) return -1;
    s->candidate[i].workers[b-1]=workers; s->candidate[i].prefetch[b-1]=prefetch; return 0;
}
int selection_complete(const fr_selection *s) {
    if (!s || !s->count || s->count>FR_CANDIDATES || !s->burst || s->burst>FR_BENCH_BURST_MAX) return 0;
    for (unsigned i=0;i<s->count;++i) {
        const fr_candidate *c=&s->candidate[i]; unsigned n=c->ssd_blocks?c->ram_blocks:c->capacity; if (n>FR_SWEEP) n=FR_SWEEP;
        if (c->measured!=((1u<<n)-1)) return 0;
        if (c->ssd_blocks) {
            if (strcmp(c->name,"metal-ssd") || c->ssd_blocks>FR_SSD_MAX_BLOCKS || c->ram_blocks+c->ssd_blocks!=c->capacity ||
                c->cache_bytes<(64ull<<20) || c->cache_bytes>(2ull<<30)) return 0;
            for (unsigned j=0;j<c->ssd_blocks;++j) if (!isfinite(c->cold_ns[j]) || !isfinite(c->warm_ns[j]) || !isfinite(c->tail_ns[j]) ||
                c->cold_ns[j]<=0 || c->warm_ns[j]<=0 || c->tail_ns[j]<c->cold_ns[j]) return 0;
        } else if (!strcmp(c->name,"metal-ssd")) return 0;
    }
    return 1;
}
/* ---- Cost evaluation is pure. Measured points, extrapolated RAM, and explicit
 * RAM+SSD points have different validity domains; never invent SSD capacity. ---- */
double selection_cost(const fr_candidate *c,unsigned b) {
    if (!c || !b || b>c->capacity) return INFINITY;
    if (c->ssd_blocks && b>c->ram_blocks) return b-c->ram_blocks<=c->ssd_blocks?c->cold_ns[b-c->ram_blocks-1]:INFINITY;
    if (b<=FR_SWEEP) return c->measured&(1u<<(b-1))?c->ns[b-1]:INFINITY;
    if (c->measured!=(1u<<FR_SWEEP)-1) return INFINITY;
    double sx=0,sy=0,sxx=0,sxy=0;
    for (unsigned i=0;i<FR_SWEEP;++i) { double x=i+1,y=c->ns[i]; sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; }
    double slope=(FR_SWEEP*sxy-sx*sy)/(FR_SWEEP*sxx-sx*sx),t0=(sy-slope*sx)/FR_SWEEP;
    if (slope<=0) { slope=0; t0=sy/FR_SWEEP; }
    if (t0<0) { t0=0; slope=sxy/sxx; }
    return t0+slope*b;
}
int selection_pick(const fr_selection *s,unsigned b) {
    if (!selection_complete(s)) return -1;
    int best=-1; double ns=INFINITY;
    for (unsigned i=0;i<s->count;++i) {
        double cost=selection_cost(&s->candidate[i],b);
        int cpu=strncmp(s->candidate[i].name,"metal",5)!=0;
        if (isfinite(cost) && (cost<ns || (cost==ns && best>=0 && cpu && !strncmp(s->candidate[best].name,"metal",5)))) { ns=cost; best=(int)i; }
    }
    return best;
}
int selection_format(const fr_selection *s,char *out,size_t size) {
    if (!s || !out || !size || !s->burst || s->burst>FR_BENCH_BURST_MAX || s->count>FR_CANDIDATES || s->assigned>FR_BLOCKS) return -1;
    unsigned version=1; for (unsigned i=0;i<s->count;++i) if (s->candidate[i].ssd_blocks) version=2;
    int n=snprintf(out,size,"FRCHOICE%u %u %u %u\n",version,s->burst,s->count,s->assigned); if (n<0 || (size_t)n>=size) return -1;
    size_t used=(size_t)n;
    for (unsigned i=0;i<s->count;++i) {
        const fr_candidate *c=&s->candidate[i];
        if (!known(c->name) || c->capacity>FR_BLOCKS || c->measured>=1u<<FR_SWEEP) return -1;
        if (c->ssd_blocks) n=snprintf(out+used,size-used,"%s %u %u %u %u %llu\n",c->name,c->capacity,c->measured,c->ram_blocks,c->ssd_blocks,(unsigned long long)c->cache_bytes);
        else n=snprintf(out+used,size-used,"%s %u %u\n",c->name,c->capacity,c->measured);
        if (n<0 || (size_t)n>=size-used) return -1;
        used+=(size_t)n;
        for (unsigned b=0;b<FR_SWEEP;++b) if (c->measured&(1u<<b)) {
            if (b>=c->capacity || !isfinite(c->ns[b]) || !isfinite(c->p99[b]) || c->ns[b]<=0 || c->p99[b]<c->ns[b] || !valid_resident(b+1,c->resident[b])) return -1;
            if (c->workers[b]>32 || (c->prefetch[b]!=0 && c->prefetch[b]!=2) || (!c->workers[b] && c->prefetch[b])) return -1;
            if (!strcmp(c->name,"avx2")) n=snprintf(out+used,size-used,"%u %.17g %.17g %llu %u %u\n",b+1,c->ns[b],c->p99[b],(unsigned long long)c->resident[b],c->workers[b],c->prefetch[b]);
            else n=snprintf(out+used,size-used,"%u %.17g %.17g %llu\n",b+1,c->ns[b],c->p99[b],(unsigned long long)c->resident[b]);
            if (n<0 || (size_t)n>=size-used) return -1;
            used+=(size_t)n;
        }
        for (unsigned j=0;j<c->ssd_blocks;++j) {
            if (j>=FR_SSD_MAX_BLOCKS || !isfinite(c->cold_ns[j]) || !isfinite(c->warm_ns[j]) || !isfinite(c->tail_ns[j]) ||
                c->cold_ns[j]<=0 || c->warm_ns[j]<=0 || c->tail_ns[j]<c->cold_ns[j]) return -1;
            n=snprintf(out+used,size-used,"SSD %u %.17g %.17g %.17g\n",j+1,c->cold_ns[j],c->warm_ns[j],c->tail_ns[j]);
            if (n<0 || (size_t)n>=size-used) return -1;
            used+=(size_t)n;
        }
    }
    return (int)used;
}
int selection_parse(const char *text,size_t len,fr_selection *out) {
    if (!text || !out || !len || len>=FR_SELECTION_TEXT || memchr(text,0,len)) return -1;
    char copy[FR_SELECTION_TEXT],canonical[FR_SELECTION_TEXT]; memcpy(copy,text,len); copy[len]=0;
    fr_selection s={0}; unsigned count,version; int n=0;
    if (sscanf(copy,"FRCHOICE%u %u %u %u\n%n",&version,&s.burst,&count,&s.assigned,&n)!=4 || (version!=1 && version!=2) || count>FR_CANDIDATES) return -1;
    const char *p=copy+n;
    for (unsigned i=0;i<count;++i) {
        char name[16]; unsigned cap,mask; n=0;
        if (sscanf(p,"%15s %u %u\n%n",name,&cap,&mask,&n)!=3 || selection_add(&s,name,cap) || mask>=1u<<FR_SWEEP) return -1;
        if (!strcmp(name,"metal-ssd")) {
            unsigned ram,streamed; unsigned long long bytes;
            if (version!=2 || sscanf(p,"%15s %u %u %u %u %llu\n%n",name,&cap,&mask,&ram,&streamed,&bytes,&n)!=6 ||
                !streamed || streamed>FR_SSD_MAX_BLOCKS || ram+streamed!=cap || bytes<(64ull<<20) || bytes>(2ull<<30)) return -1;
            s.candidate[i].ram_blocks=ram; s.candidate[i].ssd_blocks=streamed; s.candidate[i].cache_bytes=bytes;
        }
        p+=n;
        for (unsigned b=0;b<FR_SWEEP;++b) if (mask&(1u<<b)) {
            unsigned block; double a,z; unsigned long long resident; n=0;
            unsigned workers=0,prefetch=0;
            if (!strcmp(name,"avx2")) {
                if (sscanf(p,"%u %lf %lf %llu %u %u\n%n",&block,&a,&z,&resident,&workers,&prefetch,&n)!=6) return -1;
            } else if (sscanf(p,"%u %lf %lf %llu\n%n",&block,&a,&z,&resident,&n)!=4) return -1;
            if (block!=b+1 || selection_record(&s,i,block,a,z,resident) || selection_settings(&s,i,block,workers,prefetch)) return -1;
            p+=n;
        }
        fr_candidate *c=&s.candidate[i];
        for (unsigned j=0;j<c->ssd_blocks;++j) {
            unsigned slot; n=0;
            if (sscanf(p,"SSD %u %lf %lf %lf\n%n",&slot,&c->cold_ns[j],&c->warm_ns[j],&c->tail_ns[j],&n)!=4 || slot!=j+1) return -1;
            p+=n;
        }
        if (c->ssd_blocks && !selection_complete(&(fr_selection){.count=1,.burst=s.burst,.candidate={*c}})) return -1;
    }
    n=selection_format(&s,canonical,sizeof(canonical));
    if (n<0 || (size_t)n!=len || memcmp(canonical,text,len)) return -1;
    *out=s; return 0;
}
int selection_reason(const fr_selection *s,unsigned b,char *out,size_t size) {
    int winner=selection_pick(s,b); if (winner<0 || !out || !size) return -1;
    int n=snprintf(out,size,"%s selected at %u blocks, burst %u%s: ",s->candidate[winner].name,b,s->burst,b>FR_SWEEP && !(s->candidate[winner].ssd_blocks && b>s->candidate[winner].ram_blocks)?" (estimated beyond sweep)":"");
    if (n<0 || (size_t)n>=size) return -1;
    size_t used=(size_t)n;
    for (unsigned i=0;i<s->count;++i) {
        double cost=selection_cost(&s->candidate[i],b);
        if (isfinite(cost)) n=snprintf(out+used,size-used,"%s%s %.3f us",i?" vs ":"",s->candidate[i].name,cost/1000);
        else n=snprintf(out+used,size-used,"%s%s unavailable (capacity %u)",i?" vs ":"",s->candidate[i].name,s->candidate[i].capacity);
        if (n<0 || (size_t)n>=size-used) return -1;
        used+=(size_t)n;
    }
    const fr_candidate *c=&s->candidate[winner];
    if (c->ssd_blocks && b>c->ram_blocks) {
        unsigned j=b-c->ram_blocks-1;
        n=snprintf(out+used,size-used,"; RAM %u + SSD %u, cache %.0f MiB, cold %.3f us warm %.3f us p99 %.3f us",c->ram_blocks,j+1,c->cache_bytes/1048576.0,c->cold_ns[j]/1000,c->warm_ns[j]/1000,c->tail_ns[j]/1000);
        if (n<0 || (size_t)n>=size-used) return -1;
        used+=(size_t)n;
    }
    n=snprintf(out+used,size-used,"; minimum BENCH cost, ties favor CPU");
    return n<0 || (size_t)n>=size-used?-1:(int)used+n;
}
