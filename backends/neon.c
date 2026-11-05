/* DotProd Q4_0 x Q8 port of tools/neon-probe.c. Each worker owns partial sums;
 * a job is one block/expert across the burst, so weights are decoded once.
 * The format-specific dot function is the boundary for a future Q3_0 kernel.
 * Setup owns the pool; run only uses fixed storage. I8MM is never emitted.
 */
#include "../src/neon.h"
#include "../src/vectors.h"
#include "../src/bench.h"
#include "../src/native.h"
#include "../src/q4.h"
#include <arm_neon.h>
#include <fenv.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#define TOKENS FR_MAX_BURST
#define WORKERS 32
#define EXPERTS (TOKENS*FR_ACTIVE_ROUTED+1)
typedef struct { int8_t q8[TOKENS][FR_HIDDEN]; float scale[TOKENS][FR_HIDDEN/64]; } activation;
typedef struct { unsigned id,mask; float gate[TOKENS]; } expert;
typedef struct engine engine;
typedef struct { engine *engine; pthread_t thread; unsigned index; float out[TOKENS][FR_HIDDEN]; } worker;
struct engine {
    const unsigned char *weight[FR_BLOCKS];
    unsigned native[FR_BLOCKS];
    const fr_req *request;
    unsigned slot[FR_BLOCKS],blocks,threads,burst,layer,experts;
    atomic_uint sequence,remaining,started;
    atomic_int stop,error;
    pthread_mutex_t mutex;
    pthread_cond_t wake;
    expert expert[EXPERTS];
    activation input;
    worker worker[WORKERS];
    fr_neon_sample sample;
};
static engine *state;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
#ifdef __APPLE__
static unsigned setting(const char *name) {
    unsigned value=0; size_t bytes=sizeof(value);
    return sysctlbyname(name,&value,&bytes,NULL,0)?0:value;
}
#endif
unsigned fr_neon_features(void) {
#ifdef __APPLE__
    return (setting("hw.optional.arm.FEAT_DotProd")?FR_ARM_DOTPROD:0) |
           (setting("hw.optional.arm.FEAT_I8MM")?FR_ARM_I8MM:0);
#else
    return ((getauxval(AT_HWCAP)&HWCAP_ASIMDDP)?FR_ARM_DOTPROD:0) |
           ((getauxval(AT_HWCAP2)&HWCAP2_I8MM)?FR_ARM_I8MM:0);
#endif
}
static int quantize(activation *out,unsigned t,const float *x,unsigned n) {
    for (unsigned i=0;i<n;i+=64) {
        unsigned count=n-i<64?n-i:64; float maximum=0;
        for (unsigned j=0;j<count;++j) {
            if (!isfinite(x[i+j])) return FR_NUMERIC;
            if (fabsf(x[i+j])>maximum) maximum=fabsf(x[i+j]);
        }
        float scale=maximum/127.0f; out->scale[t][i/64]=scale;
        float inv=maximum?127.0f/maximum:0;
        for (unsigned j=0;j<count;++j) {
            float v=!scale?0:isfinite(inv)?x[i+j]*inv:(x[i+j]/maximum)*127.0f;
            v=nearbyintf(v); if (v>127) v=127; if (v< -127) v=-127;
            out->q8[t][i+j]=(int8_t)v;
        }
    }
    return FR_OK;
}
/* Restrict optional instructions to this function, after runtime eligibility.
 * No file-wide -march: unsupported CPUs must be able to execute probe/load. */
__attribute__((target("dotprod")))
static int dot(const unsigned char *p,const activation *a,unsigned n,unsigned mask,unsigned burst,float out[TOKENS]) {
    for (unsigned t=0;t<burst;++t) out[t]=0;
    for (unsigned k=0;k<n;k+=32,p+=18) {
        uint16_t bits=(uint16_t)(p[0]|(uint16_t)p[1]<<8);
        if ((bits&0x7c00)==0x7c00) return FR_NUMERIC;
        float16_t h; memcpy(&h,&bits,2); float scale=(float)h;
        uint8x16_t packed=vld1q_u8(p+2);
        int8x16_t lo=vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed,vdupq_n_u8(15))),vdupq_n_s8(8));
        int8x16_t hi=vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed,4)),vdupq_n_s8(8));
        for (unsigned t=0;t<burst;++t) if (mask&(1u<<t)) {
            int32x4_t sum=vdotq_s32(vdupq_n_s32(0),lo,vld1q_s8(a->q8[t]+k));
            sum=vdotq_s32(sum,hi,vld1q_s8(a->q8[t]+k+16));
            out[t]+=scale*((float)vaddvq_s32(sum)*a->scale[t][k/64]);
        }
    }
    return FR_OK;
}
/* Native weights are decoded once per vector and reused across all positions.
 * Keep fp32 activations for the lossless GGUF path; legacy Q4_0/Q8 is unchanged. */
static int native_dot(const unsigned char *p,unsigned type,unsigned n,const float *const *x,
                      unsigned mask,unsigned burst,float out[TOKENS]) {
    float32x4_t sum[TOKENS]; for (unsigned t=0;t<TOKENS;++t) sum[t]=vdupq_n_f32(0);
    static const int8_t values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    for (unsigned base=0;base<n;base+=type==FR_Q4_K?256:32) {
        float d,dm=0;
        if (type==FR_MXFP4) { if (p[0]==255) return FR_NUMERIC; d=ldexpf(.5f,(int)p[0]-127); }
        else {
            d=fr_f16((uint16_t)(p[0]|(uint16_t)p[1]<<8));
            if (type==FR_Q4_K) dm=fr_f16((uint16_t)(p[2]|(uint16_t)p[3]<<8));
            if (!isfinite(d) || !isfinite(dm)) return FR_NUMERIC;
        }
        for (unsigned g=0;g<(type==FR_Q4_K?8u:1u);++g) {
            float scale=d,minimum=0;
            if (type==FR_Q4_K) {
                const unsigned char *s=p+4; unsigned sc,mi;
                if (g<4) { sc=s[g]&63; mi=s[g+4]&63; }
                else { sc=(s[g+4]&15)|((s[g-4]&192)>>2); mi=(s[g+4]>>4)|((s[g]&192)>>2); }
                scale=d*sc; minimum=dm*mi;
            }
            for (unsigned k=0;k<32;k+=8) {
                int16x8_t q;
                if (type==FR_Q8_0) q=vmovl_s8(vld1_s8((const int8_t *)(p+2+k)));
                else {
                    uint8x8_t packed=vld1_u8(p+(type==FR_Q4_K?16+(g/2)*32+k:1+k%16));
                    if (type==FR_Q4_K?(g&1):k>=16) packed=vshr_n_u8(packed,4);
                    packed=vand_u8(packed,vdup_n_u8(15));
                    if (type==FR_MXFP4) q=vmovl_s8(vget_low_s8(vqtbl1q_s8(vld1q_s8(values),vcombine_u8(packed,vdup_n_u8(0)))));
                    else q=vreinterpretq_s16_u16(vmovl_u8(packed));
                }
                float32x4_t lo=vsubq_f32(vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q))),scale),vdupq_n_f32(minimum));
                float32x4_t hi=vsubq_f32(vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q))),scale),vdupq_n_f32(minimum));
                for (unsigned t=0;t<burst;++t) if (mask&(1u<<t)) {
                    sum[t]=vfmaq_f32(sum[t],lo,vld1q_f32(x[t]+base+g*32+k));
                    sum[t]=vfmaq_f32(sum[t],hi,vld1q_f32(x[t]+base+g*32+k+4));
                }
            }
        }
        p+=type==FR_Q4_K?144:type==FR_MXFP4?17:34;
    }
    for (unsigned t=0;t<burst;++t) out[t]=vaddvq_f32(sum[t]);
    return FR_OK;
}
static int native_job(worker *w,unsigned block,const expert *ex) {
    engine *e=w->engine; const unsigned char *p=e->weight[block]+fr_native_offset(e->layer,ex->id);
    unsigned type=ex->id==FR_SHARED_EXPERT?FR_Q8_0:FR_Q4_K,down=ex->id==FR_SHARED_EXPERT?FR_Q8_0:FR_MXFP4;
    size_t projection=ex->id==FR_SHARED_EXPERT?FR_NATIVE_SHARED_PROJECTION:FR_NATIVE_GATE_BYTES;
    float hidden[TOKENS][32]; const float *input[TOKENS],*mid[TOKENS];
    for (unsigned t=0;t<e->burst;++t) { input[t]=e->request->x[t]; mid[t]=hidden[t]; }
    for (unsigned n=0;n<32;++n) {
        float a[TOKENS],u[TOKENS]; size_t row=n*fr_native_row(type,FR_HIDDEN);
        if (native_dot(p+row,type,FR_HIDDEN,input,ex->mask,e->burst,a) || native_dot(p+projection+row,type,FR_HIDDEN,input,ex->mask,e->burst,u)) return FR_NUMERIC;
        for (unsigned t=0;t<e->burst;++t) if (ex->mask&(1u<<t)) {
            float v=a[t],z=v>=0?1/(1+expf(-v)):expf(v)/(1+expf(v)); hidden[t][n]=(v*z)*u[t];
        }
    }
    for (unsigned d=0;d<FR_HIDDEN;++d) {
        float y[TOKENS];
        if (native_dot(p+2*projection+d*fr_native_row(down,32),down,32,mid,ex->mask,e->burst,y)) return FR_NUMERIC;
        for (unsigned t=0;t<e->burst;++t) if (ex->mask&(1u<<t)) w->out[t][d]+=ex->gate[t]*y[t];
    }
    return FR_OK;
}
__attribute__((target("dotprod")))
static int job(worker *w,unsigned index) {
    engine *e=w->engine; unsigned b=e->slot[index/e->experts]; const expert *ex=&e->expert[index%e->experts];
    if (e->native[b]) return native_job(w,b,ex);
    const unsigned char *p=e->weight[b]+fr_slice_offset(e->layer,ex->id);
    float hidden[TOKENS][FR_BLOCK_NEURONS]; activation h;
    for (unsigned n=0;n<FR_BLOCK_NEURONS;++n) {
        float a[TOKENS],u[TOKENS]; size_t row=(size_t)n*FR_HIDDEN/32*18;
        if (dot(p+row,&e->input,FR_HIDDEN,ex->mask,e->burst,a) ||
            dot(p+FR_PROJECTION_BYTES+row,&e->input,FR_HIDDEN,ex->mask,e->burst,u)) return FR_NUMERIC;
        for (unsigned t=0;t<e->burst;++t) if (ex->mask&(1u<<t)) {
            float z=a[t]>=0?1/(1+expf(-a[t])):expf(a[t])/(1+expf(a[t]));
            hidden[t][n]=(a[t]*z)*u[t];
        }
    }
    for (unsigned t=0;t<e->burst;++t) if ((ex->mask&(1u<<t)) && quantize(&h,t,hidden[t],FR_BLOCK_NEURONS)) return FR_NUMERIC;
    for (unsigned d=0;d<FR_HIDDEN;++d) {
        float y[TOKENS]; if (dot(p+2*FR_PROJECTION_BYTES+d*18,&h,32,ex->mask,e->burst,y)) return FR_NUMERIC;
        for (unsigned t=0;t<e->burst;++t) if (ex->mask&(1u<<t)) w->out[t][d]+=ex->gate[t]*y[t];
    }
    return FR_OK;
}
static void *work(void *arg) {
    worker *w=arg; engine *e=w->engine; unsigned seen=0; double idle=now();
#ifdef __APPLE__
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0)) atomic_store(&e->error,FR_IO);
#endif
    atomic_fetch_add_explicit(&e->started,1,memory_order_release);
    while (!atomic_load_explicit(&e->stop,memory_order_relaxed)) {
        unsigned seq=atomic_load_explicit(&e->sequence,memory_order_acquire);
        if (seq==seen) {
            if (now()-idle<0.050) { __asm__ volatile("yield"); continue; }
            pthread_mutex_lock(&e->mutex);
            while (!atomic_load(&e->stop) && atomic_load(&e->sequence)==seen) pthread_cond_wait(&e->wake,&e->mutex);
            pthread_mutex_unlock(&e->mutex); continue;
        }
        seen=seq; memset(w->out,0,sizeof(w->out));
        /* Fixed assignment makes reduction reproducible across wake-up orders. */
        for (unsigned index=w->index;index<e->experts*e->blocks;index+=e->threads)
            if (job(w,index)) atomic_store_explicit(&e->error,FR_NUMERIC,memory_order_relaxed);
        atomic_fetch_sub_explicit(&e->remaining,1,memory_order_acq_rel); idle=now();
    }
    return NULL;
}
static void destroy(engine *e) {
    pthread_mutex_lock(&e->mutex); atomic_store(&e->stop,1); pthread_cond_broadcast(&e->wake); pthread_mutex_unlock(&e->mutex);
    for (unsigned i=0;i<e->threads;++i) pthread_join(e->worker[i].thread,NULL);
    pthread_cond_destroy(&e->wake); pthread_mutex_destroy(&e->mutex); free(e);
}
static int prepare(void) {
    if (state) return FR_OK;
    if (!(fr_neon_features()&FR_ARM_DOTPROD)) return FR_UNSUPPORTED;
#ifdef __APPLE__
    unsigned threads=setting("hw.perflevel0.physicalcpu");
#else
    long cpus=sysconf(_SC_NPROCESSORS_ONLN); unsigned threads=cpus>0?(unsigned)cpus:0;
#endif
    if (!threads) return FR_UNSUPPORTED;
    if (threads>WORKERS) threads=WORKERS;
    engine *e=calloc(1,sizeof(*e)); if (!e) return FR_IO;
    atomic_init(&e->sequence,0); atomic_init(&e->remaining,0); atomic_init(&e->started,0); atomic_init(&e->stop,0); atomic_init(&e->error,0);
    if (pthread_mutex_init(&e->mutex,NULL)) { free(e); return FR_IO; }
    if (pthread_cond_init(&e->wake,NULL)) { pthread_mutex_destroy(&e->mutex); free(e); return FR_IO; }
    for (unsigned i=0;i<threads;++i) {
        e->worker[i].engine=e; e->worker[i].index=i;
        if (pthread_create(&e->worker[i].thread,NULL,work,&e->worker[i])) { destroy(e); return FR_IO; }
        ++e->threads;
    }
    while (atomic_load_explicit(&e->started,memory_order_acquire)<threads) __asm__ volatile("yield");
    if (atomic_load(&e->error)) { destroy(e); return FR_IO; }
    e->sample.workers=threads; e->sample.features=fr_neon_features(); state=e; return FR_OK;
}
static int probe(fr_hwinfo *out) {
    if (!out) return FR_INVALID;
    if (!(fr_neon_features()&FR_ARM_DOTPROD)) return FR_UNSUPPORTED;
    return hw_probe(out)?FR_IO:FR_OK;
}
static int load(int b,const void *p,size_t n) {
    if (b<0 || b>=FR_BLOCKS || !p || (n!=FR_WEIGHT_BLOCK_BYTES && n!=FR_NATIVE_BLOCK_BYTES)) return FR_INVALID;
    int rc=prepare(); if (rc) return rc;
    if (state->weight[b]) return FR_STATE;
    state->weight[b]=p; state->native[b]=n==FR_NATIVE_BLOCK_BYTES; state->slot[state->blocks++]=(unsigned)b; return FR_OK;
}
static int unload(int b) {
    if (b<0 || b>=FR_BLOCKS) return FR_INVALID;
    if (!state || !state->weight[b]) return FR_STATE;
    state->weight[b]=NULL; state->blocks=0;
    for (unsigned i=0;i<FR_BLOCKS;++i) if (state->weight[i]) state->slot[state->blocks++]=i;
    return FR_OK;
}
static int run(const fr_req *r,fr_part *out) {
    if (!out || !r || r->layer>=FR_LAYERS || !r->burst || r->burst>TOKENS || fegetround()!=FE_TONEAREST) return FR_INVALID;
    for (unsigned t=0;t<r->burst;++t) {
        for (unsigned j=0;j<FR_ACTIVE_ROUTED;++j) {
            if (r->expert[t][j]>=FR_ROUTED_EXPERTS) return FR_INVALID;
            for (unsigned k=0;k<j;++k) if (r->expert[t][k]==r->expert[t][j]) return FR_INVALID;
        }
        for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) {
            if (!isfinite(r->gate[t][j])) return FR_NUMERIC;
            if (r->gate[t][j]<0) return FR_INVALID;
        }
        for (unsigned d=0;d<FR_HIDDEN;++d) if (!isfinite(r->x[t][d])) return FR_NUMERIC;
    }
    engine *e=state; if (!e || !e->blocks) return FR_STATE;
    double began=now(); e->burst=r->burst; e->layer=r->layer; e->experts=0; e->request=r;
    unsigned legacy=0; for (unsigned b=0;b<e->blocks;++b) if (!e->native[e->slot[b]]) legacy=1;
    for (unsigned t=0;t<r->burst;++t) {
        if (legacy) { int rc=quantize(&e->input,t,r->x[t],FR_HIDDEN); if (rc) return rc; }
        for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) {
            if (j==FR_ACTIVE_ROUTED && r->gate[t][j]==0) continue; /* shared expert computed by the head */
            unsigned id=j==FR_ACTIVE_ROUTED?FR_SHARED_EXPERT:r->expert[t][j],slot=0;
            for (;slot<e->experts && e->expert[slot].id!=id;++slot) {}
            if (slot==e->experts) { e->expert[slot]=(expert){.id=id}; ++e->experts; }
            e->expert[slot].mask|=1u<<t; e->expert[slot].gate[t]=r->gate[t][j];
        }
    }
    atomic_store(&e->error,0); atomic_store(&e->remaining,e->threads);
    pthread_mutex_lock(&e->mutex); atomic_fetch_add_explicit(&e->sequence,1,memory_order_release);
    pthread_cond_broadcast(&e->wake); pthread_mutex_unlock(&e->mutex);
    while (atomic_load_explicit(&e->remaining,memory_order_acquire)) __asm__ volatile("yield");
    int rc=atomic_load(&e->error); if (rc) return rc;
    fr_part result={.layer=r->layer,.burst=r->burst};
    for (unsigned t=0;t<r->burst;++t) for (unsigned d=0;d<FR_HIDDEN;++d) {
        float value=0; for (unsigned i=0;i<e->threads;++i) value+=e->worker[i].out[t][d];
        if (!isfinite(value)) return FR_NUMERIC;
        result.y[t][d]=value;
    }
    *out=result; e->sample.total_ns=(now()-began)*1e9; ++e->sample.runs;
    e->sample.blocks=e->blocks; e->sample.unique_experts=e->experts;
    e->sample.weight_bytes=0;
    for (unsigned b=0;b<e->blocks;++b) e->sample.weight_bytes+=e->native[e->slot[b]]?
        (e->experts-1)*FR_NATIVE_SLICE_BYTES+FR_NATIVE_SHARED_BYTES:e->experts*FR_SLICE_BYTES;
    return FR_OK;
}
int fr_neon_sample_get(fr_neon_sample *out) { if (!out) return FR_INVALID; *out=state?state->sample:(fr_neon_sample){0}; return FR_OK; }
int fr_neon_close(void) { if (state && state->blocks) return FR_STATE; if (state) destroy(state); state=NULL; return FR_OK; }
static int self_test(const char *dir) {
    if (state && state->blocks) return FR_STATE;
    int rc=fr_vectors_check(&fr_neon,dir,&fr_backend_tolerance,NULL);
    return rc?rc:fr_native_self_test(&fr_neon);
}
static int bench(int burst,fr_costmodel *out) {
    return fr_bench_execute_backend(&fr_neon,burst,out);
}
const fr_backend fr_neon={"neon",probe,load,unload,run,bench,self_test};
