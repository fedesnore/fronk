/* Standalone experiment, deliberately absent from the backend registry.
 * Q4_0 x fp16 (fp32 accumulate) and Q4_0 x Q8 (sdot, fp32 scales).
 * Full SwiGLU and down projection, grouped expert reuse, persistent workers.
 * --check uses sparse correctness-only storage. Timings require every byte of
 * 2/3 ownership blocks randomized, every page resident, and 10,000 samples.
 */
#include "../src/backend.h"
#include "../src/q4.h"
#include <arm_neon.h>
#include <errno.h>
#include <mach/mach.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#define TOKENS 4
#define WORKERS 32
#define EXPERTS (TOKENS*FR_ACTIVE_ROUTED+1)
typedef struct {
    float16_t half[TOKENS][FR_HIDDEN];
    int8_t q8[TOKENS][FR_HIDDEN];
    float scale[TOKENS][FR_HIDDEN/64];
} activation;
typedef struct { unsigned id,mask; float gate[TOKENS]; } expert;
typedef struct engine engine;
typedef struct {
    engine *engine;
    pthread_t thread;
    float out[TOKENS][FR_HIDDEN];
} worker;
struct engine {
    unsigned char *weight[3];
    unsigned blocks,threads,burst,layer,mode,experts,generation,done,stop;
    atomic_uint next;
    pthread_mutex_t mutex;
    pthread_cond_t start,complete;
    expert expert[EXPERTS];
    activation input;
    worker worker[WORKERS];
};
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint64_t rss(void) {
    struct mach_task_basic_info t; mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
    return task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&t,&count)==KERN_SUCCESS?t.resident_size:0;
}
static unsigned setting(const char *name) {
    unsigned value=0; size_t bytes=sizeof(value);
    return sysctlbyname(name,&value,&bytes,NULL,0)?0:value;
}
static uint64_t random64(uint64_t *s) { *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
static void random_weights(unsigned char *p,size_t bytes,uint64_t *seed) {
    for (size_t q=0;q<bytes;q+=18) {
        uint64_t a=random64(seed),v=random64(seed);
        uint16_t scale=(uint16_t)(0x1800|((a>>48)&0x83ff));
        p[q]=(unsigned char)scale; p[q+1]=(unsigned char)(scale>>8);
        memcpy(p+q+2,&a,8); memcpy(p+q+10,&v,8);
    }
}
static int resident(engine *e,int prefault) {
    size_t page=(size_t)sysconf(_SC_PAGESIZE),count=FR_WEIGHT_BLOCK_BYTES/page;
    char *pages=malloc(count); if (!pages || FR_WEIGHT_BLOCK_BYTES%page) { free(pages); return -1; }
    uint64_t bytes=0;
    if (prefault) for (unsigned b=0;b<e->blocks;++b) for (size_t i=0;i<count;++i)
        (void)*(volatile const unsigned char *)(e->weight[b]+i*page);
    for (unsigned b=0;b<e->blocks;++b) {
        if (mincore(e->weight[b],FR_WEIGHT_BLOCK_BYTES,pages)) { free(pages); return -1; }
        for (size_t i=0;i<count;++i) if (pages[i]&1) bytes+=page;
    }
    free(pages);
    printf("residency bytes=%llu required=%llu rss=%llu method=mincore_all_pages+RSS prefault=%d\n",
           (unsigned long long)bytes,(unsigned long long)(e->blocks*FR_WEIGHT_BLOCK_BYTES),(unsigned long long)rss(),prefault);
    fflush(stdout);
    return bytes==e->blocks*FR_WEIGHT_BLOCK_BYTES && rss()>=bytes?0:-1;
}
static void quantize(activation *out,unsigned t,const float *x,unsigned n,unsigned mode) {
    if (!mode) {
        for (unsigned i=0;i<n;i+=4) vst1_f16(out->half[t]+i,vcvt_f16_f32(vld1q_f32(x+i)));
        return;
    }
    for (unsigned i=0;i<n;i+=64) {
        unsigned count=n-i<64?n-i:64; float32x4_t maximum=vdupq_n_f32(0);
        for (unsigned j=0;j<count;j+=4) maximum=vmaxq_f32(maximum,vabsq_f32(vld1q_f32(x+i+j)));
        float max=vmaxvq_f32(maximum); out->scale[t][i/64]=max/127.0f;
        float inv=max?127.0f/max:0;
        for (unsigned j=0;j<count;j+=16) {
            int16x8_t a=vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+i+j),inv))),
                                    vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+i+j+4),inv))));
            int16x8_t b=vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+i+j+8),inv))),
                                    vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+i+j+12),inv))));
            vst1q_s8(out->q8[t]+i+j,vmaxq_s8(vdupq_n_s8(-127),vcombine_s8(vqmovn_s16(a),vqmovn_s16(b))));
        }
    }
}
static float dot16(int8x16_t w,const float16_t *x) {
    int16x8_t lo=vmovl_s8(vget_low_s8(w)),hi=vmovl_s8(vget_high_s8(w));
    float32x4_t sum=vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),vcvt_f32_f16(vld1_f16(x)));
    sum=vfmaq_f32(sum,vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),vcvt_f32_f16(vld1_f16(x+4)));
    sum=vfmaq_f32(sum,vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),vcvt_f32_f16(vld1_f16(x+8)));
    sum=vfmaq_f32(sum,vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),vcvt_f32_f16(vld1_f16(x+12)));
    return vaddvq_f32(sum);
}
static void dot(const unsigned char *p,const activation *a,unsigned n,unsigned mask,unsigned burst,unsigned mode,float out[TOKENS]) {
    for (unsigned t=0;t<burst;++t) out[t]=0;
    for (unsigned k=0;k<n;k+=32,p+=18) {
        float16_t h; memcpy(&h,p,2); float scale=(float)h;
        uint8x16_t packed=vld1q_u8(p+2);
        int8x16_t lo=vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed,vdupq_n_u8(15))),vdupq_n_s8(8));
        int8x16_t hi=vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed,4)),vdupq_n_s8(8));
        for (unsigned t=0;t<burst;++t) if (mask&(1u<<t)) {
            float value;
            if (mode) {
                int32x4_t sum=vdotq_s32(vdupq_n_s32(0),lo,vld1q_s8(a->q8[t]+k));
                sum=vdotq_s32(sum,hi,vld1q_s8(a->q8[t]+k+16));
                value=(float)vaddvq_s32(sum)*a->scale[t][k/64];
            } else value=dot16(lo,a->half[t]+k)+dot16(hi,a->half[t]+k+16);
            out[t]+=scale*value;
        }
    }
}
static void job(worker *w,unsigned index) {
    engine *e=w->engine; unsigned b=index/e->experts; const expert *ex=&e->expert[index%e->experts];
    const unsigned char *p=e->weight[b]+fr_slice_offset(e->layer,ex->id);
    float hidden[TOKENS][FR_BLOCK_NEURONS]; activation h;
    for (unsigned n=0;n<FR_BLOCK_NEURONS;++n) {
        float a[TOKENS],u[TOKENS]; size_t row=(size_t)n*FR_HIDDEN/32*18;
        dot(p+row,&e->input,FR_HIDDEN,ex->mask,e->burst,e->mode,a);
        dot(p+FR_PROJECTION_BYTES+row,&e->input,FR_HIDDEN,ex->mask,e->burst,e->mode,u);
        for (unsigned t=0;t<e->burst;++t) if (ex->mask&(1u<<t)) {
            float z=a[t]>=0?1/(1+expf(-a[t])):expf(a[t])/(1+expf(a[t]));
            hidden[t][n]=(a[t]*z)*u[t];
        }
    }
    for (unsigned t=0;t<e->burst;++t) if (ex->mask&(1u<<t)) quantize(&h,t,hidden[t],FR_BLOCK_NEURONS,e->mode);
    for (unsigned d=0;d<FR_HIDDEN;++d) {
        float y[TOKENS]; dot(p+2*FR_PROJECTION_BYTES+d*18,&h,32,ex->mask,e->burst,e->mode,y);
        for (unsigned t=0;t<e->burst;++t) if (ex->mask&(1u<<t)) w->out[t][d]+=ex->gate[t]*y[t];
    }
}
static void *work(void *arg) {
    worker *w=arg; engine *e=w->engine; unsigned seen=0;
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0)) abort();
    pthread_mutex_lock(&e->mutex);
    for (;;) {
        while (!e->stop && seen==e->generation) pthread_cond_wait(&e->start,&e->mutex);
        if (e->stop) break;
        seen=e->generation; pthread_mutex_unlock(&e->mutex);
        memset(w->out,0,sizeof(w->out));
        unsigned index;
        while ((index=atomic_fetch_add_explicit(&e->next,1,memory_order_relaxed))<e->experts*e->blocks) job(w,index);
        pthread_mutex_lock(&e->mutex);
        if (++e->done==e->threads) pthread_cond_signal(&e->complete);
    }
    pthread_mutex_unlock(&e->mutex); return NULL;
}
static void run(engine *e,const fr_req *r,unsigned mode,fr_part *out) {
    e->burst=r->burst; e->layer=r->layer; e->mode=mode; e->experts=0;
    for (unsigned t=0;t<r->burst;++t) {
        quantize(&e->input,t,r->x[t],FR_HIDDEN,mode);
        for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) {
            unsigned id=j==FR_ACTIVE_ROUTED?FR_SHARED_EXPERT:r->expert[t][j],slot=0;
            for (;slot<e->experts && e->expert[slot].id!=id;++slot) {}
            if (slot==e->experts) { e->expert[slot]=(expert){.id=id}; ++e->experts; }
            e->expert[slot].mask|=1u<<t; e->expert[slot].gate[t]=r->gate[t][j];
        }
    }
    pthread_mutex_lock(&e->mutex); e->done=0; atomic_store(&e->next,0); ++e->generation;
    pthread_cond_broadcast(&e->start);
    while (e->done!=e->threads) pthread_cond_wait(&e->complete,&e->mutex);
    pthread_mutex_unlock(&e->mutex);
    memset(out,0,sizeof(*out)); out->layer=r->layer; out->burst=r->burst;
    for (unsigned t=0;t<r->burst;++t) for (unsigned d=0;d<FR_HIDDEN;d+=4) {
        float32x4_t value=vdupq_n_f32(0);
        for (unsigned i=0;i<e->threads;++i) value=vaddq_f32(value,vld1q_f32(e->worker[i].out[t]+d));
        vst1q_f32(out->y[t]+d,value);
    }
}
static void route(fr_req *r,uint64_t pass,unsigned burst,unsigned reused) {
    r->burst=burst; r->layer=(unsigned)(pass%FR_LAYERS);
    unsigned base=(unsigned)((pass/FR_LAYERS)*41%FR_ROUTED_EXPERTS);
    for (unsigned t=0;t<burst;++t) for (unsigned j=0;j<FR_ACTIVE_ROUTED;++j)
        r->expert[t][j]=(uint16_t)((base+(reused?0:t*FR_ACTIVE_ROUTED)+j)%FR_ROUTED_EXPERTS);
}
static int compare(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
static int check(engine *e,fr_req *r) {
    int rc=0;
    for (unsigned b=0;b<e->blocks;++b)
        if (fr_ref.load((int)b,e->weight[b],FR_WEIGHT_BLOCK_BYTES) || fr_ref_q8.load((int)b,e->weight[b],FR_WEIGHT_BLOCK_BYTES)) return -1;
    for (unsigned reused=0;reused<2;++reused) for (unsigned burst=1;burst<=4;++burst) {
        route(r,0,burst,reused);
        fr_part reference,q8_reference,actual;
        if (fr_ref.run(r,&reference) || fr_ref_q8.run(r,&q8_reference)) { rc=-1; break; }
        for (unsigned mode=0;mode<2;++mode) {
            run(e,r,mode,&actual);
            for (unsigned t=0;t<burst;++t) {
                fr_error error,oracle; fr_compare(reference.y[t],actual.y[t],FR_HIDDEN,&error);
                fr_compare(mode?q8_reference.y[t]:reference.y[t],actual.y[t],FR_HIDDEN,&oracle);
                /* Q8 accumulation-order differences can cross a rounding tie;
                 * compare to both FP32 and the existing Q8 reference. */
                if (!fr_conforms(&error,&fr_backend_tolerance) || !fr_conforms(&oracle,&fr_backend_tolerance)) rc=-1;
                printf("check mode=%s reuse=%u burst=%u token=%u l2=%.9g max=%.9g cosine=%.12g q8_oracle_l2=%.9g\n",
                       mode?"q8":"fp16",reused,burst,t,error.relative_l2,error.normalized_max,error.cosine,oracle.relative_l2);
            }
        }
    }
    memset(r->x,0,sizeof(r->x)); route(r,0,4,1);
    for (unsigned mode=0;mode<2;++mode) {
        fr_part zero; run(e,r,mode,&zero);
        for (unsigned t=0;t<4;++t) for (unsigned d=0;d<FR_HIDDEN;++d) if (zero.y[t][d]!=0) rc=-1;
    }
    puts(rc?"FAIL numerical checks":"PASS all-zero exactness for fp16 and Q8");
    for (unsigned b=0;b<e->blocks;++b) { fr_ref.unload((int)b); fr_ref_q8.unload((int)b); }
    return rc;
}
int main(int argc,char **argv) {
    int checking=argc==2 && !strcmp(argv[1],"--check");
    if (!checking && argc!=4) { fputs("Usage: neon-probe --check | BLOCKS 10000 NEW_CSV\n",stderr); return 2; }
    unsigned blocks=checking?3:!strcmp(argv[1],"2")?2:!strcmp(argv[1],"3")?3:0;
    if (!blocks || (!checking && strcmp(argv[2],"10000"))) return 2;
    unsigned threads=setting("hw.perflevel0.physicalcpu");
    if (!setting("hw.optional.arm.FEAT_DotProd") || !threads || threads>WORKERS) return 1;
    char version[64]={0},level[64]={0}; size_t n=sizeof(version); sysctlbyname("kern.osproductversion",version,&n,NULL,0);
    n=sizeof(level); sysctlbyname("hw.perflevel0.name",level,&n,NULL,0);
    if (strcmp(level,"Performance")) { fputs("perflevel0 is not Performance\n",stderr); return 1; }
    printf("neon-probe macos=%s blocks=%u workers=%u dotprod=1 qos=user_interactive placement=OS_managed samples=%u\n",
           version,blocks,threads,checking?0:10000); fflush(stdout);
    engine *e=calloc(1,sizeof(*e)); if (!e) return 1;
    e->blocks=blocks; e->threads=threads; atomic_init(&e->next,0);
    pthread_mutex_init(&e->mutex,NULL); pthread_cond_init(&e->start,NULL); pthread_cond_init(&e->complete,NULL);
    int rc=0; unsigned started=0; FILE *csv=NULL;
    for (unsigned b=0;b<blocks;++b) {
        e->weight[b]=mmap(NULL,FR_WEIGHT_BLOCK_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
        if (e->weight[b]==MAP_FAILED) { e->weight[b]=NULL; rc=-1; goto done; }
        uint64_t seed=UINT64_C(0x9e3779b97f4a7c15)^(UINT64_C(0x94d049bb133111eb)*(b+1));
        if (!checking) random_weights(e->weight[b],FR_WEIGHT_BLOCK_BYTES,&seed);
        else for (unsigned id=0;id<FR_EXPERTS;++id) if (id<40 || id==FR_SHARED_EXPERT)
            random_weights(e->weight[b]+fr_slice_offset(0,id),FR_SLICE_BYTES,&seed);
    }
    if (!checking && resident(e,1)) { rc=-1; goto done; }
    for (;started<threads;++started) {
        e->worker[started].engine=e;
        if (pthread_create(&e->worker[started].thread,NULL,work,&e->worker[started])) { rc=-1; goto done; }
    }
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    fr_req r={0}; fr_part part;
    for (unsigned t=0;t<TOKENS;++t) {
        for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) r.gate[t][j]=j==FR_ACTIVE_ROUTED?0.5f:0.0625f;
        for (unsigned d=0;d<FR_HIDDEN;++d) r.x[t][d]=(float)((int)((d*37+t*19)%257)-128)/128;
    }
    if (checking) { rc=check(e,&r); puts(rc?"FAIL correctness (never timed)":"PASS correctness (never timed)"); goto done; }
    csv=fopen(argv[3],"wx"); if (!csv) { rc=-1; goto done; }
    fputs("precision,routing,burst,sample,total_ns\n",csv);
    double *samples=malloc(16*10000*sizeof(double)); if (!samples) { rc=-1; goto done; }
    /* Touch sample pages and warm both precision variants before timing. */
    for (unsigned i=0;i<160000;++i) ((volatile double *)samples)[i]=0;
    for (unsigned i=0;i<512;++i) { route(&r,i,i%4+1,(i/4)%2); run(e,&r,(i/8)%2,&part); }
    if (resident(e,0)) { free(samples); rc=-1; goto done; }
    double began=now(),checksum=0; uint64_t initial=rss();
    for (unsigned pass=0;pass<160000;++pass) {
        unsigned burst=pass%4+1,reused=(pass/4)%2,mode=(pass/8)%2,index=pass/16;
        route(&r,pass,burst,reused); double start=now(); run(e,&r,mode,&part); double ns=(now()-start)*1e9;
        samples[((mode*2+reused)*4+burst-1)*10000+index]=ns;
        checksum+=part.y[pass%burst][pass%FR_HIDDEN];
        if (pass && pass%16000==0) { printf("progress barriers=%u seconds=%.3f rss=%llu\n",pass,now()-began,(unsigned long long)rss()); fflush(stdout); }
    }
    if (resident(e,0) || !isfinite(checksum)) rc=-1;
    if (!rc) for (unsigned mode=0;mode<2;++mode) for (unsigned reuse=0;reuse<2;++reuse) for (unsigned b=0;b<4;++b) {
        double *values=samples+((mode*2+reuse)*4+b)*10000;
        for (unsigned j=0;j<10000;++j) fprintf(csv,"%s,%s,%u,%u,%.0f\n",mode?"q8":"fp16",reuse?"reused":"distinct",b+1,j,values[j]);
        qsort(values,10000,sizeof(double),compare);
        printf("total precision=%s routing=%s burst=%u samples=10000 p50_ns=%.0f p99_ns=%.0f\n",mode?"q8":"fp16",reuse?"reused":"distinct",b+1,values[4999],values[9899]);
    }
    printf("end seconds=%.3f rss_start=%llu rss_end=%llu checksum=%.9g status=%d\n",now()-began,
           (unsigned long long)initial,(unsigned long long)rss(),checksum,rc); free(samples);
done:
    pthread_mutex_lock(&e->mutex); e->stop=1; pthread_cond_broadcast(&e->start); pthread_mutex_unlock(&e->mutex);
    for (unsigned i=0;i<started;++i) pthread_join(e->worker[i].thread,NULL);
    if (csv && fclose(csv)) rc=-1;
    for (unsigned b=0;b<blocks;++b) if (e->weight[b]) munmap(e->weight[b],FR_WEIGHT_BLOCK_BYTES);
    pthread_cond_destroy(&e->start); pthread_cond_destroy(&e->complete); pthread_mutex_destroy(&e->mutex); free(e);
    return rc?1:0;
}
