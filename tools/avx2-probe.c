/* AVX2 diagnosis on one node: streaming read ceiling, then legacy Q4_0 and
 * native Q4_K/MXFP4 on the same resident shape with per-phase timers.
 * Usage: avx2-probe [RUNS]. Includes the backend to reach its engine state. */
#ifndef FR_NO_PHASES /* -DFR_NO_PHASES times the real kernels without per-neuron clock reads */
#define FR_PHASES 1
#endif
#include "../backends/avx2.c"
#include <sys/mman.h>
static uint64_t mix(uint64_t *s) { *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
/* Threads interleave 64 KiB blocks, so no two stream the same DRAM bank offsets. */
typedef struct { const unsigned char *p; size_t n,first,step; double sum; } span;
AVX2 static void *stream(void *arg) {
    span *s=arg; __m256i a=_mm256_setzero_si256(),b=a;
    for (size_t block=s->first;block+65536<=s->n;block+=s->step) for (size_t i=block;i<block+65536;i+=64) {
        a=_mm256_xor_si256(a,_mm256_load_si256((const __m256i *)(s->p+i)));
        b=_mm256_xor_si256(b,_mm256_load_si256((const __m256i *)(s->p+i+32)));
    }
    s->sum=_mm256_extract_epi32(_mm256_xor_si256(a,b),0); return NULL;
}
static double ceiling(unsigned threads,const unsigned char *p,size_t n) {
    double best=0;
    for (unsigned rep=0;rep<5;++rep) {
        pthread_t t[WORKERS]; span s[WORKERS];
        double start=now();
        for (unsigned i=0;i<threads;++i) { s[i]=(span){p,n,(size_t)i*65536,(size_t)threads*65536,0}; pthread_create(&t[i],NULL,stream,&s[i]); }
        for (unsigned i=0;i<threads;++i) pthread_join(t[i],NULL);
        double gbs=n/(now()-start)/1e9; if (gbs>best) best=gbs;
    }
    return best;
}
static void measure(const char *name,int native,unsigned runs,unsigned blocks) {
    size_t unit=native?FR_NATIVE_BLOCK_BYTES:FR_WEIGHT_BLOCK_BYTES; unsigned char *all[FR_BLOCKS];
    for (unsigned k=0;k<blocks;++k) {
    unsigned char *w=all[k]=mmap(NULL,unit,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (w==MAP_FAILED) { perror("mmap"); exit(1); }
    uint64_t seed=0x9e3779b97f4a7c15u*(k+1);
    if (native) fr_native_fill(w,(uint32_t)seed);
    else for (size_t q=0;q<unit;q+=FR_Q4_BYTES) {
        uint64_t a=mix(&seed),v=mix(&seed); uint16_t sc=(uint16_t)(0x1800|((a>>48)&0x83ff));
        w[q]=(unsigned char)sc; w[q+1]=(unsigned char)(sc>>8); memcpy(w+q+2,&a,8); memcpy(w+q+10,&v,8);
    }
    if (fr_avx2.load((int)k,w,unit)) { fputs("load failed\n",stderr); exit(1); }
    }
    static fr_req r; fr_part out; float x=0.01f;
    for (unsigned t=0;t<TOKENS;++t) for (unsigned d=0;d<FR_HIDDEN;++d) r.x[t][d]=(x=-x*1.0007f);
    for (unsigned burst=1;burst<=4;++burst) {
        double phase[6]={0},total=0; uint64_t bytes=0;
        for (unsigned i=0;i<runs+20;++i) {
            r.burst=burst; r.layer=getenv("PROBE_HOT")?0:i%FR_LAYERS; /* PROBE_HOT: same layer and experts every run, cache-resident */
            for (unsigned t=0;t<burst;++t) for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) {
                if (j<FR_ACTIVE_ROUTED) r.expert[t][j]=(uint16_t)(((getenv("PROBE_HOT")?0:i*41)+t*FR_ACTIVE_ROUTED+j)%FR_ROUTED_EXPERTS);
                r.gate[t][j]=0.1f;
            }
            double start=now(); if (fr_avx2.run(&r,&out)) { fputs("run failed\n",stderr); exit(1); }
            if (i<20) continue;
            total+=now()-start; bytes+=state->sample.weight_bytes;
            for (unsigned j=0;j<6;++j) phase[j]+=state->phase[j];
        }
        printf("%s blocks=%u burst=%u workers=%u us=%.1f eff_gbs=%.3f | quant %.1f gate+up %.1f silu %.1f down %.1f wake/join %.1f reduce %.1f\n",
               name,blocks,burst,state->threads,total/runs*1e6,bytes/total/1e9,phase[0]/runs*1e6,phase[1]/runs*1e6,
               phase[2]/runs*1e6,phase[3]/runs*1e6,phase[4]/runs*1e6,phase[5]/runs*1e6);
    }
    for (unsigned k=0;k<blocks;++k) { fr_avx2.unload((int)k); munmap(all[k],unit); }
    fr_avx2_close();
}
int main(int argc,char **argv) {
    unsigned runs=argc>1?(unsigned)atoi(argv[1]):300; fr_hwinfo hw;
    if (probe(&hw)) { puts("avx2 unsupported"); return 0; }
    size_t n=(size_t)2<<30; unsigned char *p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (p==MAP_FAILED) return 1;
    for (size_t i=0;i<n;i+=8) { uint64_t v=i*0x9e3779b97f4a7c15u; memcpy(p+i,&v,8); }
    unsigned choices[4]={1,hw.cores/2?hw.cores/2:1,hw.cores,hw.logical_cores};
    printf("read_ceiling");
    for (unsigned c=0;c<4;++c) printf(" t%u=%.2fGB/s",choices[c],ceiling(choices[c],p,n));
    putchar('\n'); munmap(p,n);
    const char *only=getenv("PROBE_ONLY"),*b=getenv("PROBE_BLOCKS"),*w=getenv("PROBE_WORKERS");
    unsigned blocks=b?(unsigned)atoi(b):1; if (w) configured_workers=(unsigned)atoi(w);
    if (!only || !strcmp(only,"q4_0")) measure("q4_0",0,runs,blocks);
    if (!only || !strcmp(only,"native")) measure("native",1,runs,blocks);
    return 0;
}
