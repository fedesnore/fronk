/* Native Q4_K/MXFP4/Q8_0 uses the preserved phase6/panel4.h and
 * r4/native/src/q4_fixed.h kernels, with fp32 activation/FMA accumulators.
 * The legacy Q4_0 oracle path ports Shuttle q4panel.h
 * (source SHA e99513f8bb2f297fc076f5eacf2acd11e9b653d1e7881d8325b57c896dc0e20c).
 * Eight output lanes and four explicit position accumulators retain its AVX2
 * integer dots, FMA and shared unpacking. Only the canonical Q4_0 load/scale
 * boundary changes on that legacy path: two groups of 32 replace one of 64.
 * Borrowed files stay immutable; the adapter transposes only register values.
 * The persistent C11 pool supplies Fronk lifecycle and deterministic reduction.
 */
#include "../src/avx2.h"
#include "../src/vectors.h"
#include "../src/bench.h"
#include "../src/native.h"
#include <immintrin.h>
#include <fenv.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define AVX2 __attribute__((target("avx2,fma,f16c")))

#define TOKENS FR_MAX_BURST
#define WORKERS 32
#define EXPERTS (TOKENS*FR_ACTIVE_ROUTED+1)
typedef struct { int8_t q8[TOKENS][FR_HIDDEN]; float scale[TOKENS][FR_HIDDEN/64]; int sum[TOKENS][FR_HIDDEN/32]; } activation;
typedef struct { unsigned id,mask,rank; float gate[TOKENS];
#ifdef FR_AVX2_UNION
    unsigned route[TOKENS];
#endif
} expert;
typedef struct { int8_t q[TOKENS][FR_HIDDEN]; float d[TOKENS][FR_HIDDEN/256]; int16_t s[TOKENS][FR_HIDDEN/32]; } q8k;
typedef struct engine engine;
typedef struct { engine *engine; pthread_t thread; unsigned index; double phase[3]; float out[TOKENS][FR_HIDDEN]; } worker;
#ifdef FR_PHASES /* Probe-only timers: gate+up, silu*up, down per worker; quant, wake/join, reduce per run. */
#define PHASE(w,i,t) ((w)->phase[i]+=now()-(t),(t)=now())
#else
#define PHASE(w,i,t) ((void)0)
#endif
struct engine {
    const unsigned char *weight[FR_BLOCKS];
    unsigned native[FR_BLOCKS];
    const fr_req *request;
    unsigned slot[FR_BLOCKS],blocks,threads,burst,layer,experts;
    unsigned ranked;
#ifdef FR_AVX2_UNION
    unsigned union_enabled,union_active,reduce_only;
    float *union_y;
    float union_scale[FR_BLOCKS][TOKENS][FR_ACTIVE_ROUTED];
#ifdef FR_AVX2_UNION_PHASES
    unsigned union_profile;
    double union_compute_s,union_reduce_s;
#endif
#endif
#ifdef FR_AVX2_MTP_ROWS
    unsigned mtp_rows;
#endif
    atomic_uint sequence,remaining,started;
    atomic_int stop,error;
    pthread_mutex_t mutex;
    pthread_cond_t wake;
    expert expert[EXPERTS];
    activation input;
    q8k x8;
    worker worker[WORKERS];
    fr_avx2_sample sample;
    double phase[6];
};
static engine *state;
static unsigned configured_workers,configured_prefetch;
static struct { unsigned workers,prefetch; } tuning[FR_BLOCKS+1][FR_MAX_BURST+1];
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int probe(fr_hwinfo *out) {
    if (!out) return FR_INVALID;
    fr_hwinfo h;
    if (hw_probe(&h)) return FR_IO;
    if ((h.isa&FR_ISA_USABLE)!=FR_ISA_USABLE) return FR_UNSUPPORTED;
    *out=h; return FR_OK;
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
        out->sum[t][i/32]=0; out->sum[t][i/32+1]=0;
        for (unsigned j=0;j<count;++j) {
            float v=!scale?0:isfinite(inv)?x[i+j]*inv:(x[i+j]/maximum)*127.0f;
            v=nearbyintf(v); if (v>127) v=127; if (v< -127) v=-127;
            out->q8[t][i+j]=(int8_t)v; out->sum[t][(i+j)/32]+=(int)v;
        }
    }
    return FR_OK;
}
/* No file-wide ISA flags: probe/load remain executable on unsupported CPUs. */
typedef struct { __m256i packed[4]; __m256 scale; } group8;
typedef struct { __m256 t0,t1,t2,t3; } sum4;
AVX2 static inline int canonical_group(const unsigned char *p,size_t stride,group8 *out) {
    __m128 row[8]; uint16_t bits[8];
    for (unsigned r=0;r<8;++r) {
        const unsigned char *q=p+r*stride;
        bits[r]=(uint16_t)(q[0]|(uint16_t)q[1]<<8);
        if ((bits[r]&0x7c00)==0x7c00) return FR_NUMERIC;
        row[r]=_mm_castsi128_ps(_mm_loadu_si128((const __m128i *)(q+2)));
    }
    /* Four bytes per output lane, as in the retained interleaved image.
     * Low/high nibbles now address j and j+16 in a canonical 32-weight group. */
    _MM_TRANSPOSE4_PS(row[0],row[1],row[2],row[3]);
    _MM_TRANSPOSE4_PS(row[4],row[5],row[6],row[7]);
    for (unsigned j=0;j<4;++j) out->packed[j]=_mm256_castps_si256(_mm256_set_m128(row[j+4],row[j]));
    out->scale=_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)bits));
    return FR_OK;
}
AVX2 static inline __m256i input4(const activation *x,unsigned t,unsigned k) {
    int32_t bits; memcpy(&bits,x->q8[t]+k,sizeof(bits));
    return _mm256_set1_epi32(bits);
}
/* Port of q4_panel4's explicit accumulator core, with its non-unrolled loop.
 * Each call covers one canonical group; its Q8 correction is that half's sum. */
AVX2 static inline sum4 group_dot4(const group8 *w,const activation *x,unsigned k,
                                  const unsigned *pos,unsigned count,sum4 total) {
    __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;
    const __m256i mask=_mm256_set1_epi8(15),ones=_mm256_set1_epi16(1);
    #pragma GCC unroll 1
    for (unsigned j=0;j<4;++j) {
        __m256i packed=w->packed[j],lo=_mm256_and_si256(packed,mask);
        __m256i hi=_mm256_and_si256(_mm256_srli_epi16(packed,4),mask);
        a0=_mm256_add_epi32(a0,_mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(lo,input4(x,pos[0],k+j*4)),_mm256_maddubs_epi16(hi,input4(x,pos[0],k+j*4+16))),ones));
        if (count>1) a1=_mm256_add_epi32(a1,_mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(lo,input4(x,pos[1],k+j*4)),_mm256_maddubs_epi16(hi,input4(x,pos[1],k+j*4+16))),ones));
        if (count>2) a2=_mm256_add_epi32(a2,_mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(lo,input4(x,pos[2],k+j*4)),_mm256_maddubs_epi16(hi,input4(x,pos[2],k+j*4+16))),ones));
        if (count>3) a3=_mm256_add_epi32(a3,_mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(lo,input4(x,pos[3],k+j*4)),_mm256_maddubs_epi16(hi,input4(x,pos[3],k+j*4+16))),ones));
    }
    total.t0=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(a0,_mm256_set1_epi32(x->sum[pos[0]][k/32]*8))),_mm256_mul_ps(w->scale,_mm256_set1_ps(x->scale[pos[0]][k/64])),total.t0);
    if (count>1) total.t1=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(a1,_mm256_set1_epi32(x->sum[pos[1]][k/32]*8))),_mm256_mul_ps(w->scale,_mm256_set1_ps(x->scale[pos[1]][k/64])),total.t1);
    if (count>2) total.t2=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(a2,_mm256_set1_epi32(x->sum[pos[2]][k/32]*8))),_mm256_mul_ps(w->scale,_mm256_set1_ps(x->scale[pos[2]][k/64])),total.t2);
    if (count>3) total.t3=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(a3,_mm256_set1_epi32(x->sum[pos[3]][k/32]*8))),_mm256_mul_ps(w->scale,_mm256_set1_ps(x->scale[pos[3]][k/64])),total.t3);
    return total;
}
AVX2 static inline void store4(float out[TOKENS][8],const unsigned *pos,unsigned count,sum4 s) {
    _mm256_storeu_ps(out[pos[0]],s.t0);
    if (count>1) _mm256_storeu_ps(out[pos[1]],s.t1);
    if (count>2) _mm256_storeu_ps(out[pos[2]],s.t2);
    if (count>3) _mm256_storeu_ps(out[pos[3]],s.t3);
}
AVX2 __attribute__((noinline))
static int panel8(const unsigned char *p,const activation *x,unsigned n,const unsigned *pos,unsigned count,float out[TOKENS][8]) {
    __m256 z=_mm256_setzero_ps(); sum4 first={z,z,z,z}; size_t stride=(size_t)n/32*18;
    if (count<=4) {
        for (unsigned k=0;k<n;k+=32) {
            group8 g; if (canonical_group(p+(size_t)k/32*18,stride,&g)) return FR_NUMERIC;
            first=group_dot4(&g,x,k,pos,count,first);
        }
        store4(out,pos,count,first); return FR_OK;
    }
    /* Bursts 5..8 reuse the same loaded group across two four-position panels. */
    sum4 second={z,z,z,z};
    for (unsigned k=0;k<n;k+=32) {
        group8 g; if (canonical_group(p+(size_t)k/32*18,stride,&g)) return FR_NUMERIC;
        first=group_dot4(&g,x,k,pos,4,first);
        second=group_dot4(&g,x,k,pos+4,count-4,second);
    }
    store4(out,pos,4,first); store4(out,pos+4,count-4,second); return FR_OK;
}
/* Native Q4_K/MXFP4/Q8_0 in the Shuttle engine's integer style: activations are
 * quantized once per request (int8, one scale per 256, int16 sums per 32), so
 * Q4_K dots use maddubs/madd with the 6-bit group scales applied in integers.
 * MXFP4 decodes through pshufb; down rows run eight at a time with an hadd
 * transpose. Up to four token accumulators share each decoded vector. */
AVX2 static inline int hsum_i(__m256i v) {
    __m128i a=_mm_add_epi32(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1));
    a=_mm_hadd_epi32(a,a); a=_mm_hadd_epi32(a,a); return _mm_cvtsi128_si32(a);
}
AVX2 static float native_sum(__m256 v) {
    __m128 a=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
    a=_mm_hadd_ps(a,a); a=_mm_hadd_ps(a,a); return _mm_cvtss_f32(a);
}
/* int8 quantization of n values with one scale per `group` (256 or 32). */
AVX2 static void quantize8(const float *x,unsigned n,unsigned group,int8_t *q,float *d,int16_t *s) {
    const __m256 sign=_mm256_set1_ps(-0.0f);
    const __m256i order=_mm256_setr_epi32(0,4,1,5,2,6,3,7);
    for (unsigned i=0;i<n;i+=group) {
        __m256 m=_mm256_setzero_ps();
        for (unsigned j=0;j<group;j+=8) m=_mm256_max_ps(m,_mm256_andnot_ps(sign,_mm256_loadu_ps(x+i+j)));
        __m128 h=_mm_max_ps(_mm256_castps256_ps128(m),_mm256_extractf128_ps(m,1));
        h=_mm_max_ps(h,_mm_movehl_ps(h,h)); h=_mm_max_ss(h,_mm_shuffle_ps(h,h,1)); float amax=_mm_cvtss_f32(h);
        d[i/group]=amax/127.0f; __m256 inv=_mm256_set1_ps(amax?127.0f/amax:0);
        for (unsigned j=0;j<group;j+=32) {
            __m256i v0=_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x+i+j),inv));
            __m256i v1=_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x+i+j+8),inv));
            __m256i v2=_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x+i+j+16),inv));
            __m256i v3=_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x+i+j+24),inv));
            if (s) s[(i+j)/32]=(int16_t)hsum_i(_mm256_add_epi32(_mm256_add_epi32(v0,v1),_mm256_add_epi32(v2,v3)));
            __m256i b=_mm256_packs_epi16(_mm256_packs_epi32(v0,v1),_mm256_packs_epi32(v2,v3));
            _mm256_storeu_si256((__m256i *)(q+i+j),_mm256_permutevar8x32_epi32(b,order));
        }
    }
}
/* One gate/up row of FR_HIDDEN weights against up to four tokens. */
AVX2 static int row4(const unsigned char *p,unsigned type,const q8k *x,const unsigned *t,unsigned count,unsigned prefetch,float *out) {
    const __m256i mask=_mm256_set1_epi8(15),ones=_mm256_set1_epi16(1);
    __m256 acc[4]={_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
    float minimum[4]={0,0,0,0};
    for (unsigned sb=0;sb<FR_HIDDEN/256;++sb) {
        if (type==FR_Q4_K) {
            const unsigned char *b=p+sb*144,*s=b+4;
            if (prefetch) _mm_prefetch((const char *)(b+prefetch*144),_MM_HINT_T0);
            float d=_cvtsh_ss((uint16_t)(b[0]|(uint16_t)b[1]<<8)),dm=_cvtsh_ss((uint16_t)(b[2]|(uint16_t)b[3]<<8));
            if (!isfinite(d) || !isfinite(dm)) return FR_NUMERIC;
            /* 12 packed bytes -> scales u[0..1], mins u[2..3] (6 bits each). */
            uint32_t u[4]; memcpy(u,s,12);
            u[3]=((u[2]>>4)&0x0f0f0f0f)|(((u[1]>>6)&0x03030303)<<4);
            uint32_t m=u[1]&0x3f3f3f3f; u[1]=(u[2]&0x0f0f0f0f)|(((u[0]>>6)&0x03030303)<<4); u[2]=m; u[0]&=0x3f3f3f3f;
            const unsigned char *sc=(const unsigned char *)u;
            __m128i mi=_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(u+2)));
            __m256i is[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
            for (unsigned gp=0;gp<4;++gp) {
                __m256i q=_mm256_loadu_si256((const __m256i *)(b+16+gp*32));
                __m256i lo=_mm256_and_si256(q,mask),hi=_mm256_and_si256(_mm256_srli_epi16(q,4),mask);
                __m256i slo=_mm256_set1_epi16(sc[2*gp]),shi=_mm256_set1_epi16(sc[2*gp+1]);
                for (unsigned i=0;i<count;++i) {
                    const int8_t *xq=x->q[t[i]]+sb*256+gp*64;
                    is[i]=_mm256_add_epi32(is[i],_mm256_madd_epi16(_mm256_maddubs_epi16(lo,_mm256_loadu_si256((const __m256i *)xq)),slo));
                    is[i]=_mm256_add_epi32(is[i],_mm256_madd_epi16(_mm256_maddubs_epi16(hi,_mm256_loadu_si256((const __m256i *)(xq+32))),shi));
                }
            }
            for (unsigned i=0;i<count;++i) {
                __m128i mm=_mm_madd_epi16(mi,_mm_loadu_si128((const __m128i *)(x->s[t[i]]+sb*8)));
                mm=_mm_hadd_epi32(mm,mm); mm=_mm_hadd_epi32(mm,mm); int m=_mm_cvtsi128_si32(mm); float xd=x->d[t[i]][sb];
                acc[i]=_mm256_fmadd_ps(_mm256_cvtepi32_ps(is[i]),_mm256_set1_ps(d*xd),acc[i]); minimum[i]+=dm*xd*(float)m;
            }
        } else for (unsigned k=0;k<8;++k) {
            const unsigned char *b=p+(sb*8+k)*34; float d=_cvtsh_ss((uint16_t)(b[0]|(uint16_t)b[1]<<8));
            if (!isfinite(d)) return FR_NUMERIC;
            __m256i w=_mm256_loadu_si256((const __m256i *)(b+2)),aw=_mm256_sign_epi8(w,w);
            for (unsigned i=0;i<count;++i) {
                __m256i xq=_mm256_loadu_si256((const __m256i *)(x->q[t[i]]+sb*256+k*32));
                __m256i dot=_mm256_madd_epi16(_mm256_maddubs_epi16(aw,_mm256_sign_epi8(xq,w)),ones);
                acc[i]=_mm256_fmadd_ps(_mm256_cvtepi32_ps(dot),_mm256_set1_ps(d*x->d[t[i]][sb]),acc[i]);
            }
        }
    }
    for (unsigned i=0;i<count;++i) out[i]=native_sum(acc[i])-minimum[i];
    return FR_OK;
}
/* Q4_K gate and up rows of one neuron together, against `count` (1 or 2) tokens:
 * x loads are shared by both rows, scales and mins are unpacked in registers
 * (no store-to-load round trip) and broadcast per group with pshufb, and the
 * min term stays a vector until the row ends. Non-finite scales reach the
 * output and are rejected by the reduction. */
AVX2 static inline __attribute__((always_inline)) __m128i q4k_scales(const unsigned char *s) {
    uint32_t u0,u1,u2; memcpy(&u0,s,4); memcpy(&u1,s+4,4); memcpy(&u2,s+8,4);
    const uint32_t m3=((u2>>4)&0x0f0f0f0f)|(((u1>>6)&0x03030303)<<4),m0=u1&0x3f3f3f3f;
    const uint32_t s1=(u2&0x0f0f0f0f)|(((u0>>6)&0x03030303)<<4),s0=u0&0x3f3f3f3f;
    return _mm_set_epi32((int)m3,(int)m0,(int)s1,(int)s0); /* bytes 0-7 scales, 8-15 mins */
}
AVX2 static inline __attribute__((always_inline)) void q4k_pair(const unsigned char *pg,const unsigned char *pu,const q8k *x,
                                                               const unsigned *t,const unsigned count,float *og,float *ou) {
    const __m256i mask=_mm256_set1_epi8(15);
    static const int8_t pick[8][32]={
#define PICK(g) {2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1, \
                 2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1,2*g,2*g+1}
        PICK(0),PICK(1),PICK(2),PICK(3),PICK(4),PICK(5),PICK(6),PICK(7)
#undef PICK
    };
    __m256 acc[2][2]={{_mm256_setzero_ps(),_mm256_setzero_ps()},{_mm256_setzero_ps(),_mm256_setzero_ps()}};
    __m256 mins[2]={_mm256_setzero_ps(),_mm256_setzero_ps()}; /* lanes 0-3 gate, 4-7 up */
    for (unsigned sb=0;sb<FR_HIDDEN/256;++sb) {
        const unsigned char *bg=pg+sb*144,*bu=pu+sb*144;
#ifndef FR_Q4K_PREFETCH
#define FR_Q4K_PREFETCH 6 /* superblocks ahead; measured on the iMac: 2 none, 6 -2..-3%, 12 same */
#endif
#if FR_Q4K_PREFETCH
        _mm_prefetch((const char *)(bg+FR_Q4K_PREFETCH*144),_MM_HINT_T0); _mm_prefetch((const char *)(bg+FR_Q4K_PREFETCH*144+64),_MM_HINT_T0);
        _mm_prefetch((const char *)(bu+FR_Q4K_PREFETCH*144),_MM_HINT_T0); _mm_prefetch((const char *)(bu+FR_Q4K_PREFETCH*144+64),_MM_HINT_T0);
#endif
        uint32_t hg,hu; memcpy(&hg,bg,4); memcpy(&hu,bu,4);
        const __m128 fg=_mm_cvtph_ps(_mm_cvtsi32_si128((int)hg)),fu=_mm_cvtph_ps(_mm_cvtsi32_si128((int)hu)); /* d, dmin */
        const __m128i smg=q4k_scales(bg+4),smu=q4k_scales(bu+4);
        const __m256i scg=_mm256_broadcastsi128_si256(_mm_cvtepu8_epi16(smg)),scu=_mm256_broadcastsi128_si256(_mm_cvtepu8_epi16(smu));
        const __m128i ming=_mm_cvtepu8_epi16(_mm_srli_si128(smg,8)),minu=_mm_cvtepu8_epi16(_mm_srli_si128(smu,8));
        __m256i is[2][2]={{_mm256_setzero_si256(),_mm256_setzero_si256()},{_mm256_setzero_si256(),_mm256_setzero_si256()}};
        for (unsigned gp=0;gp<4;++gp) {
            const __m256i qg=_mm256_loadu_si256((const __m256i *)(bg+16+gp*32)),qu=_mm256_loadu_si256((const __m256i *)(bu+16+gp*32));
            const __m256i glo=_mm256_and_si256(qg,mask),ghi=_mm256_and_si256(_mm256_srli_epi16(qg,4),mask);
            const __m256i ulo=_mm256_and_si256(qu,mask),uhi=_mm256_and_si256(_mm256_srli_epi16(qu,4),mask);
            const __m256i plo=_mm256_loadu_si256((const __m256i *)pick[2*gp]),phi=_mm256_loadu_si256((const __m256i *)pick[2*gp+1]);
            const __m256i sglo=_mm256_shuffle_epi8(scg,plo),sghi=_mm256_shuffle_epi8(scg,phi);
            const __m256i sulo=_mm256_shuffle_epi8(scu,plo),suhi=_mm256_shuffle_epi8(scu,phi);
            for (unsigned i=0;i<count;++i) {
                const int8_t *xq=x->q[t[i]]+sb*256+gp*64;
                const __m256i xl=_mm256_loadu_si256((const __m256i *)xq),xh=_mm256_loadu_si256((const __m256i *)(xq+32));
                is[i][0]=_mm256_add_epi32(is[i][0],_mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(glo,xl),sglo),
                                                                     _mm256_madd_epi16(_mm256_maddubs_epi16(ghi,xh),sghi)));
                is[i][1]=_mm256_add_epi32(is[i][1],_mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ulo,xl),sulo),
                                                                     _mm256_madd_epi16(_mm256_maddubs_epi16(uhi,xh),suhi)));
            }
        }
        for (unsigned i=0;i<count;++i) {
            const float xd=x->d[t[i]][sb];
            const __m128i xs=_mm_loadu_si128((const __m128i *)(x->s[t[i]]+sb*8));
            acc[i][0]=_mm256_fmadd_ps(_mm256_cvtepi32_ps(is[i][0]),_mm256_set1_ps(_mm_cvtss_f32(fg)*xd),acc[i][0]);
            acc[i][1]=_mm256_fmadd_ps(_mm256_cvtepi32_ps(is[i][1]),_mm256_set1_ps(_mm_cvtss_f32(fu)*xd),acc[i][1]);
            const __m256i mm=_mm256_set_m128i(_mm_madd_epi16(minu,xs),_mm_madd_epi16(ming,xs));
            const __m256 dm=_mm256_set_m128(_mm_set1_ps(_mm_cvtss_f32(_mm_movehdup_ps(fu))*xd),_mm_set1_ps(_mm_cvtss_f32(_mm_movehdup_ps(fg))*xd));
            mins[i]=_mm256_fmadd_ps(_mm256_cvtepi32_ps(mm),dm,mins[i]);
        }
    }
    for (unsigned i=0;i<count;++i) {
        const __m128 m=_mm_hadd_ps(_mm256_castps256_ps128(mins[i]),_mm256_extractf128_ps(mins[i],1)); /* g01 g23 u01 u23 */
        const __m128 mm=_mm_hadd_ps(m,m);                                                          /* g u g u */
        og[i]=native_sum(acc[i][0])-_mm_cvtss_f32(mm); ou[i]=native_sum(acc[i][1])-_mm_cvtss_f32(_mm_movehdup_ps(mm));
    }
}
/* Specialized for one and two tokens, so the token loops unroll. */
AVX2 static void q4k_pair1(const unsigned char *pg,const unsigned char *pu,const q8k *x,const unsigned *t,float *og,float *ou) { q4k_pair(pg,pu,x,t,1,og,ou); }
AVX2 static void q4k_pair2(const unsigned char *pg,const unsigned char *pu,const q8k *x,const unsigned *t,float *og,float *ou) { q4k_pair(pg,pu,x,t,2,og,ou); }
#ifdef FR_AVX2_MTP_ROWS
#include "avx2_mtp.inc"
#endif
/* MXFP4 down: eight rows of 32 weights for `count` tokens. Rows r and r+4
 * share one vector (low and high lane), decoded once through a LUT biased by
 * +12 so maddubs needs no sign fix-up; the bias returns as 12 * sum(mid).
 * Three hadds give the eight row sums in order. Same integers and the same
 * float operations as down8, so results are bit-identical. */
AVX2 static void down8_mxfp4(const unsigned char *p,const int8_t (*m)[32],const int *msum,const float *md,const float *gate,
                             unsigned count,float *const *out
#ifdef FR_AVX2_UNION
                             ,unsigned raw
#endif
                             ) {
    const __m256i mask=_mm256_set1_epi8(15),ones=_mm256_set1_epi16(1);
    const __m256i lut=_mm256_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0,12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0);
    const __m256i at=_mm256_setr_epi32(0,17,34,51,68,85,102,119);
    __m256i e=_mm256_and_si256(_mm256_i32gather_epi32((const int *)p,at,1),_mm256_set1_epi32(255));
    __m256i bits=_mm256_blendv_epi8(_mm256_slli_epi32(e,23),_mm256_set1_epi32(0x00400000),_mm256_cmpeq_epi32(e,_mm256_setzero_si256()));
    const __m256 rs=_mm256_mul_ps(_mm256_castsi256_ps(bits),_mm256_set1_ps(.5f));
    __m256i lo[4],hi[4];
    for (unsigned r=0;r<4;++r) {
        const __m256i q=_mm256_set_m128i(_mm_loadu_si128((const __m128i *)(p+(r+4)*17+1)),_mm_loadu_si128((const __m128i *)(p+r*17+1)));
        lo[r]=_mm256_shuffle_epi8(lut,_mm256_and_si256(q,mask));
        hi[r]=_mm256_shuffle_epi8(lut,_mm256_and_si256(_mm256_srli_epi16(q,4),mask));
    }
    for (unsigned i=0;i<count;++i) {
        const __m256i xa=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)m[i]));
        const __m256i xb=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(m[i]+16)));
        __m256i d[4];
        for (unsigned r=0;r<4;++r) d[r]=_mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(lo[r],xa),_mm256_maddubs_epi16(hi[r],xb)),ones);
        __m256i s=_mm256_hadd_epi32(_mm256_hadd_epi32(d[0],d[1]),_mm256_hadd_epi32(d[2],d[3]));
        s=_mm256_sub_epi32(s,_mm256_set1_epi32(12*msum[i]));
        const __m256 y=_mm256_mul_ps(_mm256_cvtepi32_ps(s),rs);
#ifdef FR_AVX2_UNION
        if(raw) {_mm256_storeu_ps(out[i],y);continue;}
#endif
        _mm256_storeu_ps(out[i],_mm256_fmadd_ps(y,_mm256_set1_ps(md[i]*gate[i]),_mm256_loadu_ps(out[i])));
    }
}
/* Eight down rows of 32 weights: decode once, then one hadd transpose per token. */
AVX2 static int down8(const unsigned char *p,unsigned type,const int8_t (*m)[32],const float *md,const float *gate,unsigned count,float *const *out) {
    const __m128i mask=_mm_set1_epi8(15),lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i w[8],aw[8]; float scale[8];
    for (unsigned r=0;r<8;++r) {
        if (type==FR_MXFP4) {
            const unsigned char *b=p+r*17; if (b[0]==255) return FR_NUMERIC;
            uint32_t bits=b[0]?(uint32_t)b[0]<<23:0x00400000u; memcpy(&scale[r],&bits,4); scale[r]*=.5f;
            __m128i q=_mm_loadu_si128((const __m128i *)(b+1));
            w[r]=_mm256_set_m128i(_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(q,4),mask)),_mm_shuffle_epi8(lut,_mm_and_si128(q,mask)));
        } else {
            const unsigned char *b=p+r*34; scale[r]=_cvtsh_ss((uint16_t)(b[0]|(uint16_t)b[1]<<8));
            if (!isfinite(scale[r])) return FR_NUMERIC;
            w[r]=_mm256_loadu_si256((const __m256i *)(b+2));
        }
        aw[r]=_mm256_sign_epi8(w[r],w[r]);
    }
    __m256 rs=_mm256_loadu_ps(scale);
    for (unsigned i=0;i<count;++i) {
        __m256i x=_mm256_loadu_si256((const __m256i *)m[i]),d[8];
        for (unsigned r=0;r<8;++r) d[r]=_mm256_madd_epi16(_mm256_maddubs_epi16(aw[r],_mm256_sign_epi8(x,w[r])),ones);
        __m256i a=_mm256_hadd_epi32(_mm256_hadd_epi32(d[0],d[1]),_mm256_hadd_epi32(d[2],d[3]));
        __m256i b=_mm256_hadd_epi32(_mm256_hadd_epi32(d[4],d[5]),_mm256_hadd_epi32(d[6],d[7]));
        __m256i s=_mm256_add_epi32(_mm256_permute2x128_si256(a,b,0x20),_mm256_permute2x128_si256(a,b,0x31));
        __m256 y=_mm256_mul_ps(_mm256_cvtepi32_ps(s),rs);
        _mm256_storeu_ps(out[i],_mm256_fmadd_ps(y,_mm256_set1_ps(md[i]*gate[i]),_mm256_loadu_ps(out[i])));
    }
    return FR_OK;
}
#ifdef FR_AVX2_UNION
static inline __attribute__((always_inline)) int native_job_impl(worker *w,unsigned block,const expert *ex,unsigned raw) {
#else
static int native_job(worker *w,unsigned block,const expert *ex) {
#endif
    engine *e=w->engine;
    const unsigned char *p=e->weight[block]+fr_native_offset(e->layer,ex->id);
    unsigned type=ex->id==FR_SHARED_EXPERT?FR_Q8_0:FR_Q4_K,down=ex->id==FR_SHARED_EXPERT?FR_Q8_0:FR_MXFP4;
    size_t projection=ex->id==FR_SHARED_EXPERT?FR_NATIVE_SHARED_PROJECTION:FR_NATIVE_GATE_BYTES;
    unsigned pos[TOKENS],count=0; float hidden[TOKENS][32],md[TOKENS],gate[TOKENS]; int8_t mid[TOKENS][32];
    for (unsigned t=0;t<e->burst;++t) if (ex->mask&(1u<<t)) { gate[count]=ex->gate[t]; pos[count++]=t; }
    double clock=0; (void)clock;
#ifdef FR_PHASES
    clock=now();
#endif
    for (unsigned n=0;n<32;++n) {
        float a[TOKENS],u[TOKENS]; size_t row=n*fr_native_row(type,FR_HIDDEN);
        if (type==FR_Q4_K) for (unsigned c=0;c<count;) {
#ifdef FR_AVX2_MTP_ROWS
            if(e->mtp_rows && count-c>=e->mtp_rows && count-c>=3) {
                unsigned tile=count-c>=8?8:count-c>=4?4:3;
                void (*fn)(const unsigned char *,const q8k *,const unsigned *,float *)=tile==8?q4k_row8:tile==4?q4k_row4:q4k_row3;
                fn(p+row,&e->x8,pos+c,a+c);fn(p+projection+row,&e->x8,pos+c,u+c);
                c+=tile;continue;
            }
#endif
            if (count-c>1) q4k_pair2(p+row,p+projection+row,&e->x8,pos+c,a+c,u+c);
            else q4k_pair1(p+row,p+projection+row,&e->x8,pos+c,a+c,u+c);
            c+=count-c>1?2:1;
        } else for (unsigned c=0;c<count;c+=4) {
            unsigned k=count-c<4?count-c:4;
            if (row4(p+row,type,&e->x8,pos+c,k,e->sample.prefetch,a+c) || row4(p+projection+row,type,&e->x8,pos+c,k,e->sample.prefetch,u+c)) return FR_NUMERIC;
        }
        PHASE(w,0,clock);
        for (unsigned t=0;t<count;++t) {
            float v=a[t]; hidden[t][n]=v/(1+expf(-v))*u[t];
        }
        PHASE(w,1,clock);
    }
    for (unsigned t=0;t<count;++t) {
        for (unsigned n=0;n<32;++n) if (!isfinite(hidden[t][n])) return FR_NUMERIC;
        quantize8(hidden[t],32,32,mid[t],&md[t],NULL);
    }
    float *out[TOKENS]; size_t stride=fr_native_row(down,32); int msum[TOKENS];
    for (unsigned t=0;t<count;++t) { msum[t]=0; for (unsigned n=0;n<32;++n) msum[t]+=mid[t][n]; }
#ifdef FR_AVX2_UNION
    if(raw) for(unsigned t=0;t<count;t++)
        e->union_scale[block][pos[t]][ex->route[pos[t]]]=md[t]*gate[t];
#endif
    for (unsigned d=0;d<FR_HIDDEN;d+=8) {
        for (unsigned t=0;t<count;++t) out[t]=w->out[pos[t]]+d;
#ifdef FR_AVX2_UNION
        if(raw) for(unsigned t=0;t<count;t++)
            out[t]=e->union_y+((size_t)block*TOKENS*FR_ACTIVE_ROUTED+pos[t]*FR_ACTIVE_ROUTED+ex->route[pos[t]])*FR_HIDDEN+d;
#endif
        if (down==FR_MXFP4) down8_mxfp4(p+2*projection+d*stride,(const int8_t (*)[32])mid,msum,md,gate,count,out
#ifdef FR_AVX2_UNION
            ,raw
#endif
            );
        else if (down8(p+2*projection+d*stride,down,(const int8_t (*)[32])mid,md,gate,count,out)) return FR_NUMERIC;
    }
    PHASE(w,2,clock);
    return FR_OK;
}
#ifdef FR_AVX2_UNION
/* Specialize before entering the slice loops: no raw-output mode branch for
 * every eight dimensions on the ordinary or single-token path. */
static int native_job(worker *w,unsigned block,const expert *ex) {
    if(w->engine->union_active) return native_job_impl(w,block,ex,1);
    return native_job_impl(w,block,ex,0);
}
#endif
static int job(worker *w,unsigned index) {
    engine *e=w->engine; unsigned b=e->slot[index/e->experts]; const expert *ex=&e->expert[index%e->experts];
    if (e->native[b]) return native_job(w,b,ex);
    const unsigned char *p=e->weight[b]+fr_slice_offset(e->layer,ex->id);
    unsigned pos[TOKENS],count=0;
    for (unsigned t=0;t<e->burst;++t) if (ex->mask&(1u<<t)) pos[count++]=t;
    float hidden[TOKENS][FR_BLOCK_NEURONS]; activation h;
    for (unsigned n=0;n<FR_BLOCK_NEURONS;n+=8) {
        float a[TOKENS][8],u[TOKENS][8]; size_t row=(size_t)n*FR_HIDDEN/32*18;
        if (panel8(p+row,&e->input,FR_HIDDEN,pos,count,a) ||
            panel8(p+FR_PROJECTION_BYTES+row,&e->input,FR_HIDDEN,pos,count,u)) return FR_NUMERIC;
        for (unsigned i=0;i<count;++i) for (unsigned j=0;j<8;++j) {
            unsigned t=pos[i]; float v=a[t][j],z=v>=0?1/(1+expf(-v)):expf(v)/(1+expf(v));
            hidden[t][n+j]=(v*z)*u[t][j];
        }
    }
    for (unsigned i=0;i<count;++i) if (quantize(&h,pos[i],hidden[pos[i]],FR_BLOCK_NEURONS)) return FR_NUMERIC;
    for (unsigned d=0;d<FR_HIDDEN;d+=8) {
        float y[TOKENS][8]; if (panel8(p+2*FR_PROJECTION_BYTES+d*18,&h,32,pos,count,y)) return FR_NUMERIC;
        for (unsigned i=0;i<count;++i) for (unsigned j=0;j<8;++j) w->out[pos[i]][d+j]+=ex->gate[pos[i]]*y[pos[i]][j];
    }
    return FR_OK;
}
#ifdef FR_AVX2_UNION
/* Reproduce each original worker's FMA chain exactly, then retain the existing
 * cross-worker sum. Grouped compute owns (block,token,rank) output slots, so
 * no worker can overwrite another token or race an accumulation. */
AVX2 static void union_reduce(worker *w) {
    engine *e=w->engine;
    for(unsigned b=0;b<e->blocks;b++) for(unsigned rank=0;rank<FR_ACTIVE_ROUTED;rank++) {
        if((b*FR_ACTIVE_ROUTED+rank)%e->threads!=w->index) continue;
        unsigned block=e->slot[b];
        for(unsigned t=0;t<e->burst;t++) {
            const float *y=e->union_y+((size_t)block*TOKENS*FR_ACTIVE_ROUTED+t*FR_ACTIVE_ROUTED+rank)*FR_HIDDEN;
            const __m256 scale=_mm256_set1_ps(e->union_scale[block][t][rank]);
            for(unsigned d=0;d<FR_HIDDEN;d+=8)
                _mm256_storeu_ps(w->out[t]+d,_mm256_fmadd_ps(_mm256_loadu_ps(y+d),scale,_mm256_loadu_ps(w->out[t]+d)));
        }
    }
}
#endif
static void pass(worker *w) {
    engine *e=w->engine; memset(w->out,0,e->burst*sizeof(w->out[0])); memset(w->phase,0,sizeof(w->phase));
#ifdef FR_AVX2_UNION
    if(e->union_active) {
        if(e->reduce_only) {union_reduce(w);return;}
        for(unsigned index=w->index;index<e->experts*e->blocks;index+=e->threads)
            if(job(w,index)) atomic_store_explicit(&e->error,FR_NUMERIC,memory_order_relaxed);
        return;
    }
#endif
    /* A routed-only batch can share weights at the same routing rank while
     * retaining exactly the worker assignment and addition order of T=1.
     * The ordinary build keeps its existing fast assignment unchanged. */
#ifdef FR_EXPERT_RANKED
    if (e->ranked) {
        for (unsigned b=0;b<e->blocks;++b) for (unsigned j=0;j<e->experts;++j)
            if ((b*FR_ACTIVE_ROUTED+e->expert[j].rank)%e->threads==w->index)
                if (job(w,b*e->experts+j)) atomic_store_explicit(&e->error,FR_NUMERIC,memory_order_relaxed);
        return;
    }
#endif
    for (unsigned index=w->index;index<e->experts*e->blocks;index+=e->threads)
        if (job(w,index)) atomic_store_explicit(&e->error,FR_NUMERIC,memory_order_relaxed);
}
/* The caller is worker 0, so no thread spins idle while workers compute. */
static void *work(void *arg) {
    worker *w=arg; engine *e=w->engine; unsigned seen=0; double idle=now();
    if (fesetround(FE_TONEAREST)) atomic_store(&e->error,FR_IO);
#ifdef __APPLE__
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0)) atomic_store(&e->error,FR_IO);
#endif
    atomic_fetch_add_explicit(&e->started,1,memory_order_release);
    while (!atomic_load_explicit(&e->stop,memory_order_relaxed)) {
        unsigned seq=atomic_load_explicit(&e->sequence,memory_order_acquire);
        if (seq==seen) {
            if (now()-idle<0.050) { __asm__ volatile("pause"); continue; }
            pthread_mutex_lock(&e->mutex);
            while (!atomic_load(&e->stop) && atomic_load(&e->sequence)==seen) pthread_cond_wait(&e->wake,&e->mutex);
            pthread_mutex_unlock(&e->mutex); continue;
        }
        seen=seq; pass(w);
        atomic_fetch_sub_explicit(&e->remaining,1,memory_order_acq_rel); idle=now();
    }
    return NULL;
}
static void destroy(engine *e) {
    pthread_mutex_lock(&e->mutex); atomic_store(&e->stop,1); pthread_cond_broadcast(&e->wake); pthread_mutex_unlock(&e->mutex);
    for (unsigned i=1;i<e->threads;++i) pthread_join(e->worker[i].thread,NULL);
    pthread_cond_destroy(&e->wake); pthread_mutex_destroy(&e->mutex);
#ifdef FR_AVX2_UNION
    free(e->union_y);
#endif
    free(e);
}
static int prepare(void) {
    if (state) return FR_OK;
    fr_hwinfo hw; int rc=probe(&hw); if (rc) return rc;
    unsigned threads=configured_workers?configured_workers:(hw.cores?hw.cores:hw.logical_cores);
    if (!threads) return FR_UNSUPPORTED;
    if (threads>WORKERS) threads=WORKERS;
    engine *e=calloc(1,sizeof(*e)); if (!e) return FR_IO;
#ifdef FR_AVX2_MTP_ROWS
    const char *mtp_rows=getenv("FRONK_AVX2_MTP_ROWS");
    if(mtp_rows && !strcmp(mtp_rows,"1")) {
        /* Qualified full-backend crossover: the Apple clang/iMac path wins
         * only for eight shared rows; GCC/Shuttle also wins at three/four.
         * These are opt-in experiment policies, not universal CPU tuning. */
#ifdef __clang__
        e->mtp_rows=8;
#else
        e->mtp_rows=3;
#endif
    }
#endif
    atomic_init(&e->sequence,0); atomic_init(&e->remaining,0); atomic_init(&e->started,0); atomic_init(&e->stop,0); atomic_init(&e->error,0);
    if (pthread_mutex_init(&e->mutex,NULL)) { free(e); return FR_IO; }
    if (pthread_cond_init(&e->wake,NULL)) { pthread_mutex_destroy(&e->mutex); free(e); return FR_IO; }
    e->worker[0].engine=e; e->threads=1;
    for (unsigned i=1;i<threads;++i) {
        e->worker[i].engine=e; e->worker[i].index=i;
        if (pthread_create(&e->worker[i].thread,NULL,work,&e->worker[i])) { destroy(e); return FR_IO; }
        ++e->threads;
    }
    while (atomic_load_explicit(&e->started,memory_order_acquire)<threads-1) __asm__ volatile("pause");
    if (atomic_load(&e->error)) { destroy(e); return FR_IO; }
#ifdef FR_AVX2_UNION
    /* Fixed 15.625 MiB maximum, reserved at setup, never in run(). Only the
     * active block/token/rank slots are written and consumed. */
    e->union_y=calloc((size_t)FR_BLOCKS*TOKENS*FR_ACTIVE_ROUTED*FR_HIDDEN,sizeof(float));
    if(!e->union_y) {destroy(e);return FR_IO;}
    const char *un=getenv("FRONK_AVX2_UNION");
    e->union_enabled=un && !strcmp(un,"1");
#endif
    e->sample.workers=threads; e->sample.features=hw.isa; e->sample.prefetch=configured_prefetch; state=e; return FR_OK;
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
    double began=now(),mark=began; e->burst=r->burst; e->layer=r->layer; e->experts=0; e->request=r;
    unsigned legacy=0; for (unsigned b=0;b<e->blocks;++b) if (!e->native[e->slot[b]]) legacy=1;
    for (unsigned t=0;t<r->burst;++t) {
        if (legacy) { int rc=quantize(&e->input,t,r->x[t],FR_HIDDEN); if (rc) return rc; }
        else quantize8(r->x[t],FR_HIDDEN,256,e->x8.q[t],e->x8.d[t],e->x8.s[t]);
    }
    e->ranked=0;
#ifdef FR_AVX2_UNION
    e->union_active=e->reduce_only=0;
#endif
#ifdef FR_EXPERT_RANKED
    e->ranked=r->burst>1;
    for (unsigned t=0;t<r->burst;++t) if (r->gate[t][FR_ACTIVE_ROUTED]!=0) e->ranked=0;
#ifdef FR_AVX2_UNION
    e->union_active=e->union_enabled && e->ranked && !legacy;
#endif
    if (e->ranked
#ifdef FR_AVX2_UNION
        && !e->union_active
#endif
        ) {
        for (unsigned rank=0;rank<FR_ACTIVE_ROUTED;++rank) {
            unsigned begin=e->experts;
            for (unsigned t=0;t<r->burst;++t) {
                unsigned id=r->expert[t][rank],slot=begin;
                while (slot<e->experts && e->expert[slot].id!=id) ++slot;
                if (slot==e->experts) { e->expert[slot]=(expert){.id=id,.rank=rank}; ++e->experts; }
                e->expert[slot].mask|=1u<<t; e->expert[slot].gate[t]=r->gate[t][rank];
            }
        }
    } else
#endif
    for (unsigned t=0;t<r->burst;++t) {
        for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) {
            if (j==FR_ACTIVE_ROUTED && r->gate[t][j]==0) continue; /* shared expert computed by the head */
            unsigned id=j==FR_ACTIVE_ROUTED?FR_SHARED_EXPERT:r->expert[t][j],slot=0;
            for (;slot<e->experts && e->expert[slot].id!=id;++slot) {}
            if (slot==e->experts) { e->expert[slot]=(expert){.id=id}; ++e->experts; }
            e->expert[slot].mask|=1u<<t; e->expert[slot].gate[t]=r->gate[t][j];
#ifdef FR_AVX2_UNION
            e->expert[slot].route[t]=j;
#endif
        }
    }
    e->phase[0]=now()-mark; mark=now();
    atomic_store(&e->error,0); atomic_store(&e->remaining,e->threads-1);
    pthread_mutex_lock(&e->mutex); atomic_fetch_add_explicit(&e->sequence,1,memory_order_release);
    pthread_cond_broadcast(&e->wake); pthread_mutex_unlock(&e->mutex);
    pass(&e->worker[0]);
    while (atomic_load_explicit(&e->remaining,memory_order_acquire)) __asm__ volatile("pause");
    int rc=atomic_load(&e->error); if (rc) return rc;
#ifdef FR_AVX2_UNION_PHASES
    double union_done=e->union_profile?now():0;
    if(e->union_profile) e->union_compute_s=union_done-mark;
#endif
#ifdef FR_AVX2_UNION
    if(e->union_active) {
        e->reduce_only=1;
        atomic_store(&e->remaining,e->threads-1);
        pthread_mutex_lock(&e->mutex);atomic_fetch_add_explicit(&e->sequence,1,memory_order_release);
        pthread_cond_broadcast(&e->wake);pthread_mutex_unlock(&e->mutex);
        pass(&e->worker[0]);
        while(atomic_load_explicit(&e->remaining,memory_order_acquire)) __asm__ volatile("pause");
    }
#ifdef FR_AVX2_UNION_PHASES
    if(e->union_profile) e->union_reduce_s=now()-union_done;
#endif
#endif
    double busy=0; e->phase[1]=e->phase[2]=e->phase[3]=0;
    for (unsigned i=0;i<e->threads;++i) {
        double sum=0; for (unsigned j=0;j<3;++j) { e->phase[1+j]+=e->worker[i].phase[j]/e->threads; sum+=e->worker[i].phase[j]; }
        if (sum>busy) busy=sum;
    }
    e->phase[4]=now()-mark-busy; mark=now();
    fr_part result={.layer=r->layer,.burst=r->burst};
    for (unsigned t=0;t<r->burst;++t) for (unsigned d=0;d<FR_HIDDEN;++d) {
        float value=0; for (unsigned i=0;i<e->threads;++i) value+=e->worker[i].out[t][d];
        if (!isfinite(value)) return FR_NUMERIC;
        result.y[t][d]=value;
    }
    e->phase[5]=now()-mark;
    *out=result; e->sample.total_ns=(now()-began)*1e9; ++e->sample.runs;
    e->sample.blocks=e->blocks; e->sample.unique_experts=e->experts;
    e->sample.weight_bytes=0;
    for (unsigned b=0;b<e->blocks;++b) e->sample.weight_bytes+=e->native[e->slot[b]]?
        (e->experts-1)*FR_NATIVE_SLICE_BYTES+FR_NATIVE_SHARED_BYTES:e->experts*FR_SLICE_BYTES;
    return FR_OK;
}
int fr_avx2_sample_get(fr_avx2_sample *out) { if (!out) return FR_INVALID; *out=state?state->sample:(fr_avx2_sample){0}; return FR_OK; }
int fr_avx2_close(void) { if (state && state->blocks) return FR_STATE; if (state) destroy(state); state=NULL; return FR_OK; }
static int retune(unsigned workers,unsigned prefetch) {
    if (!state) return FR_STATE;
    const unsigned char *weight[FR_BLOCKS]; unsigned native[FR_BLOCKS];
    memcpy(weight,state->weight,sizeof(weight)); memcpy(native,state->native,sizeof(native));
    destroy(state); state=NULL; configured_workers=workers; configured_prefetch=prefetch;
    for (unsigned b=0;b<FR_BLOCKS;++b) if (weight[b]) {
        int rc=load((int)b,weight[b],native[b]?FR_NATIVE_BLOCK_BYTES:FR_WEIGHT_BLOCK_BYTES); if (rc) return rc;
    }
    return FR_OK;
}
void fr_avx2_set_tuning(unsigned blocks,unsigned burst,unsigned workers,unsigned prefetch) {
    if (blocks<=FR_BLOCKS && burst<=FR_MAX_BURST) { tuning[blocks][burst].workers=workers; tuning[blocks][burst].prefetch=prefetch; }
}
int fr_avx2_configure(unsigned blocks,unsigned burst) {
    if (!blocks || blocks>FR_BLOCKS || !burst || burst>FR_MAX_BURST) return FR_INVALID;
    int rc=fr_avx2_close(); if (rc) return rc;
    /* Shares above the measured sweep reuse the largest measured setting. */
    unsigned b=blocks; while (b>1 && !tuning[b][burst].workers) --b;
    configured_workers=tuning[b][burst].workers; configured_prefetch=tuning[b][burst].prefetch;
    /* In a session the node's RX thread busy-polls: leave it one logical CPU. */
    fr_hwinfo hw; unsigned cpus=!hw_probe(&hw)&&hw.logical_cores>1?hw.logical_cores-1:1;
    if (!configured_workers || configured_workers>cpus) configured_workers=cpus;
    return FR_OK;
}
int fr_avx2_tune(const fr_req *r,FILE *csv) {
    if (!r || !state || !state->blocks) return FR_STATE;
    fr_hwinfo hw; int rc=probe(&hw); if (rc) return rc;
    /* cores-1 leaves one core to the node's RX thread or a busy desktop. */
    unsigned choices[5]={1,hw.cores>1?hw.cores/2:1,hw.cores>1?hw.cores-1:1,hw.cores,hw.logical_cores};
    unsigned best_workers=state->threads,best_prefetch=0; double best=INFINITY;
    fr_part out; fr_req req=*r;
    for (unsigned c=0;c<5;++c) {
        unsigned workers=choices[c]; if (!workers || workers>WORKERS) continue;
        int duplicate=0; for (unsigned j=0;j<c;++j) if (choices[j]==workers) duplicate=1;
        if (duplicate) continue;
        for (unsigned prefetch=0;prefetch<=2;prefetch+=2) {
            rc=retune(workers,prefetch); if (rc) return rc;
            double total=0;
            for (unsigned i=0;i<8;++i) {
                req.layer=i%FR_LAYERS;
                double start=now(); rc=run(&req,&out); if (rc) return rc;
                double ns=(now()-start)*1e9;
                if (i>=2) {
                    total+=ns;
                    if (csv && fprintf(csv,"tuning,%u,%u,0,0,0,0,%.0f,%.0f,%u,%u\n",r->burst,i-2,ns,ns,workers,prefetch)<0) return FR_IO;
                }
            }
            if (total<best) { best=total; best_workers=workers; best_prefetch=prefetch; }
        }
    }
    tuning[state->blocks][r->burst].workers=best_workers;
    tuning[state->blocks][r->burst].prefetch=best_prefetch;
    return retune(best_workers,best_prefetch);
}
static int self_test(const char *dir) {
    if (state && state->blocks) return FR_STATE;
    int rc=fr_vectors_check(&fr_avx2,dir,&fr_backend_tolerance,NULL);
    return rc?rc:fr_native_self_test(&fr_avx2);
}
static int bench(int burst,fr_costmodel *out) {
    return fr_bench_execute_backend(&fr_avx2,burst,out);
}
const fr_backend fr_avx2={"avx2",probe,load,unload,run,bench,self_test};
