/* metal.m - macOS device reporting and Apple Silicon Q4_0 compute.
 * Runtime MSL, shared resources, grouped expert reads and synchronous bounded
 * submission. Intel builds retain hardware reporting only.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../src/hw.h"
#include <string.h>
int hw_metal_probe(fr_hwinfo *h) {
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        /* Intel Macs over SSH may have no default display device while the
         * Radeon remains available through enumeration. This is reporting,
         * not a GPU backend selection for x86 nodes. */
        if (!device) {
            NSArray<id<MTLDevice>> *devices=MTLCopyAllDevices();
            device=[[devices firstObject] retain];
            [devices release];
        }
        if (!device) return 0;
        const char *name=[[device name] UTF8String];
        uint64_t limit=[device recommendedMaxWorkingSetSize];
        int ok=name && strlen(name)<sizeof(h->gpu) && limit;
        if (ok) { strcpy(h->gpu,name); h->metal=1; h->metal_limit=limit; }
        [device release];
        return ok?0:-1;
    }
}

#if defined(FR_METAL_COMPUTE) && defined(__aarch64__)
#include "../src/metal.h"
#include "../src/vectors.h"
#include "../src/bench.h"
#include "../src/native.h"
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <mach/mach_time.h>

/* Binding submission decision, NOTES.md: only transient Metal command buffers
 * and compute encoders are exempt from the zero-allocation rule. All fronk
 * resources are created at load/setup; run has one macOS 15 submission path,
 * an autorelease pool, unretained resource references and waitUntilCompleted.
 */
#define MAX_EXPERTS (FR_MAX_BURST*FR_ACTIVE_ROUTED+1)
#define MSL(...) @#__VA_ARGS__
static NSString *const source=@"#include <metal_stdlib>\n" MSL(
using namespace metal;
struct Expert { uint id, mask; float gate[8]; };
struct Routing { uint burst, count, mask, slots[20]; Expert expert[81]; };
float scale16(device const uchar *q) {
    uint h=uint(q[0])|(uint(q[1])<<8), e=(h>>10)&31, m=h&1023;
    float f=e==0?float(m)*0x1p-24f:as_type<float>(((e==31?255:e+112)<<23)|(m<<13));
    return (h&32768)?-f:f;
}
float native_weight(device const uchar *p,uint type,uint j) {
    if (type==12) {
        device const uchar *b=p+(j/256)*144,*s=b+4;
        uint g=(j%256)/32,sc,mi;
        if (g<4) { sc=s[g]&63; mi=s[g+4]&63; }
        else { sc=(s[g+4]&15)|((s[g-4]&192)>>2); mi=(s[g+4]>>4)|((s[g]&192)>>2); }
        uint q=(b[16+(g/2)*32+j%32]>>((g%2)*4))&15;
        return (scale16(b)*float(sc))*float(q)-scale16(b+2)*float(mi);
    }
    if (type==39) {
        device const uchar *b=p+(j/32)*17;
        const float mx[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
        float scale=as_type<float>(b[0]?uint(b[0])<<23:0x00400000u);
        return scale*mx[(b[1+j%16]>>(((j%32)/16)*4))&15];
    }
    device const uchar *b=p+(j/32)*34;
    return scale16(b)*float(as_type<char>(b[2+j%32]));
}
kernel void gate_up(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                    constant Routing &r [[buffer(2)]], device float *h [[buffer(3)]],
                    constant uint &tag [[buffer(4)]], device atomic_uint *error [[buffer(5)]],
                    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint lane=tid&31, neuron=group.x*4+(tid>>5), job=group.y;
    uint block=tag&255; bool mixed=(tag&256)!=0;
    Expert expert=r.expert[job];
    bool shared=expert.id==512;
    device const uchar *g=w+ulong(expert.id)*(mixed?135680:138240)+neuron*(mixed&&shared?2720:1440);
    device const uchar *u=g+(mixed&&shared?87040:46080);
    float a[8]={0}, b[8]={0};
    if (mixed) {
        for (uint d=lane;d<2560;d+=32) {
            float gv=native_weight(g,shared?8:12,d),uv=native_weight(u,shared?8:12,d);
            if (!isfinite(gv)||!isfinite(uv)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            for (uint t=0;t<r.burst;++t) if (expert.mask&(1u<<t)) {
                float v=x[t*2560+d]; a[t]=fma(gv,v,a[t]); b[t]=fma(uv,v,b[t]);
            }
        }
    } else for (uint k=lane;k<1280;k+=32) {
        uint q=k/16, j=k%16, d=q*32+j;
        device const uchar *gq=g+q*18, *uq=u+q*18;
        float gs=scale16(gq), us=scale16(uq);
        if (!isfinite(gs)||!isfinite(us)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        uchar gv=gq[2+j], uv=uq[2+j];
        float gl=(int(gv&15)-8)*gs, gh=(int(gv>>4)-8)*gs;
        float ul=(int(uv&15)-8)*us, uh=(int(uv>>4)-8)*us;
        for (uint t=0;t<r.burst;++t) if (expert.mask&(1u<<t)) {
            float lo=x[t*2560+d], hi=x[t*2560+d+16];
            a[t]=fma(gl,lo,a[t]); a[t]=fma(gh,hi,a[t]);
            b[t]=fma(ul,lo,b[t]); b[t]=fma(uh,hi,b[t]);
        }
    }
    for (uint t=0;t<r.burst;++t) if (expert.mask&(1u<<t)) {
        float av=simd_sum(a[t]), bv=simd_sum(b[t]);
        if (!lane) {
            float z=av>=0?1/(1+exp(-av)):exp(av)/(1+exp(av));
            float value=(av*z)*bv;
            if (!isfinite(av)||!isfinite(bv)||!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            h[((block*81+job)*8+t)*32+neuron]=value;
        }
    }
}
kernel void down(device const uchar *w [[buffer(0)]], constant Routing &r [[buffer(1)]],
                 device const float *h [[buffer(2)]], device float *partial [[buffer(3)]],
                 constant uint &tag [[buffer(4)]], device atomic_uint *error [[buffer(5)]],
                 uint d [[thread_position_in_grid]]) {
    uint block=tag&255; bool mixed=(tag&256)!=0; float y[8]={0};
    for (uint job=0;job<r.count;++job) {
        Expert e=r.expert[job];
        bool shared=e.id==512;
        device const uchar *q=w+ulong(e.id)*(mixed?135680:138240)+(mixed&&shared?174080:92160)+d*(mixed?(shared?34:17):18);
        float dot[8]={0};
        if (mixed) {
            for (uint j=0;j<32;++j) {
                float v=native_weight(q,shared?8:39,j);
                if (!isfinite(v)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
                for (uint t=0;t<r.burst;++t) if (e.mask&(1u<<t)) dot[t]=fma(v,h[((block*81+job)*8+t)*32+j],dot[t]);
            }
        } else {
          float scale=scale16(q);
          if (!isfinite(scale)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
          for (uint j=0;j<16;++j) {
            uchar packed=q[2+j]; float lo=(int(packed&15)-8)*scale, hi=(int(packed>>4)-8)*scale;
            for (uint t=0;t<r.burst;++t) if (e.mask&(1u<<t)) {
                uint base=((block*81+job)*8+t)*32;
                dot[t]=fma(lo,h[base+j],dot[t]); dot[t]=fma(hi,h[base+j+16],dot[t]);
            }
          }
        }
        for (uint t=0;t<r.burst;++t) if (e.mask&(1u<<t)) y[t]=fma(e.gate[t],dot[t],y[t]);
    }
    for (uint t=0;t<r.burst;++t) {
        if (!isfinite(y[t])) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        partial[(block*8+t)*2560+d]=y[t];
    }
}
/* Native slices (Q4_K gate/up, MXFP4 down, Q8_0 shared), the format every
 * cluster ships. One simdgroup computes one neuron: each lane reads 4 bytes of
 * Q4_K (8 weights) per superblock, decodes its two sub-block scales once and
 * applies the same weights to every token of the call. */
kernel void gate_up_native(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                           constant Routing &r [[buffer(2)]], device float *h [[buffer(3)]],
                           constant uint &tag [[buffer(4)]], device atomic_uint *error [[buffer(5)]],
                           uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint lane=tid&31, neuron=group.x*4+(tid>>5), job=group.y, block=tag&255;
    const Expert e=r.expert[job]; const uint burst=r.burst;
    float a[8]={0,0,0,0,0,0,0,0}, b[8]={0,0,0,0,0,0,0,0};
    if (e.id!=512) {
        device const uchar *g=w+ulong(e.id)*135680+neuron*1440, *u=g+46080;
        const uint p=lane>>3, o=(lane&7)*4, lo=2*p, hi=2*p+1;
        for (uint sb=0;sb<10;++sb) {
            /* Decode this lane's 8 gate and 8 up weights once, then reuse them for every token. */
            float4 gql,gqh,uql,uqh; float gdl,gml,gdh,gmh,udl,uml,udh,umh;
            for (uint m=0;m<2;++m) {
                device const uchar *bb=(m?u:g)+sb*144, *sc=bb+4;
                const float d=float(*(device const half *)bb), dm=float(*(device const half *)(bb+2));
                uint s0,m0,s1,m1;
                if (lo<4) { s0=sc[lo]&63; m0=sc[lo+4]&63; } else { s0=(sc[lo+4]&15)|((sc[lo-4]&192)>>2); m0=(sc[lo+4]>>4)|((sc[lo]&192)>>2); }
                if (hi<4) { s1=sc[hi]&63; m1=sc[hi+4]&63; } else { s1=(sc[hi+4]&15)|((sc[hi-4]&192)>>2); m1=(sc[hi+4]>>4)|((sc[hi]&192)>>2); }
                const uint q=*(device const uint *)(bb+16+p*32+o);
                const float4 ql=float4(q&15,(q>>8)&15,(q>>16)&15,(q>>24)&15), qh=float4((q>>4)&15,(q>>12)&15,(q>>20)&15,(q>>28)&15);
                if (m) { uql=ql; uqh=qh; udl=d*float(s0); uml=dm*float(m0); udh=d*float(s1); umh=dm*float(m1); }
                else   { gql=ql; gqh=qh; gdl=d*float(s0); gml=dm*float(m0); gdh=d*float(s1); gmh=dm*float(m1); }
            }
            const uint base=sb*256+o;
            for (uint t=0;t<burst;++t) if (e.mask&(1u<<t)) {
                device const float *xt=x+t*2560+base;
                const float4 xl=*(device const float4 *)(xt+lo*32), xh=*(device const float4 *)(xt+hi*32);
                const float sl=xl.x+xl.y+xl.z+xl.w, sh=xh.x+xh.y+xh.z+xh.w;
                a[t]+=gdl*dot(gql,xl)-gml*sl+gdh*dot(gqh,xh)-gmh*sh;
                b[t]+=udl*dot(uql,xl)-uml*sl+udh*dot(uqh,xh)-umh*sh;
            }
        }
    } else {
        /* Shared expert: Q8_0 rows of 80 blocks (f16 scale + 32 int8). */
        device const uchar *g=w+ulong(512)*135680+neuron*2720, *u=g+87040;
        for (uint k=lane;k<80;k+=32) {
            device const uchar *gb=g+k*34, *ub=u+k*34;
            const float gs=float(*(device const half *)gb), us=float(*(device const half *)ub);
            for (uint j=0;j<32;j+=4) {
                const float4 gv=float4(char4(*(device const packed_char4 *)(gb+2+j)))*gs;
                const float4 uv=float4(char4(*(device const packed_char4 *)(ub+2+j)))*us;
                for (uint t=0;t<burst;++t) if (e.mask&(1u<<t)) {
                    const float4 xv=*(device const float4 *)(x+t*2560+k*32+j);
                    a[t]+=dot(gv,xv); b[t]+=dot(uv,xv);
                }
            }
        }
    }
    for (uint t=0;t<burst;++t) if (e.mask&(1u<<t)) {
        const float av=simd_sum(a[t]), bv=simd_sum(b[t]);
        if (!lane) {
            const float z=av>=0?1/(1+exp(-av)):exp(av)/(1+exp(av)), value=(av*z)*bv;
            if (!isfinite(value)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
            h[((block*81+job)*8+t)*32+neuron]=value;
        }
    }
}
/* One thread per output dimension: its MXFP4 row is 17 bytes (scale + 32
 * weights, one per neuron); weights decode once and serve every token. */
constant float fr_mx[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
kernel void down_native(device const uchar *w [[buffer(0)]], constant Routing &r [[buffer(1)]],
                        device const float *h [[buffer(2)]], device float *partial [[buffer(3)]],
                        constant uint &tag [[buffer(4)]], device atomic_uint *error [[buffer(5)]],
                        uint d [[thread_position_in_grid]]) {
    const uint block=tag&255, burst=r.burst; float y[8]={0,0,0,0,0,0,0,0};
    for (uint job=0;job<r.count;++job) {
        const Expert e=r.expert[job]; float wv[32];
        if (e.id!=512) {
            device const uchar *q=w+ulong(e.id)*135680+92160+d*17;
            const uint ex=q[0]; const float scale=as_type<float>(ex?ex<<23:0x00400000u);
            for (uint j=0;j<16;j+=4) {
                const uchar4 v=uchar4(*(device const packed_uchar4 *)(q+1+j));
                for (uint k=0;k<4;++k) { wv[j+k]=scale*fr_mx[v[k]&15]; wv[j+k+16]=scale*fr_mx[v[k]>>4]; }
            }
        } else {
            device const uchar *q=w+ulong(512)*135680+174080+d*34;
            const float scale=float(*(device const half *)q);
            for (uint j=0;j<32;++j) wv[j]=scale*float(as_type<char>(q[2+j]));
        }
        for (uint t=0;t<burst;++t) if (e.mask&(1u<<t)) {
            device const float *ht=h+((block*81+job)*8+t)*32; float acc=0;
            for (uint j=0;j<32;++j) acc=fma(wv[j],ht[j],acc);
            y[t]=fma(e.gate[t],acc,y[t]);
        }
    }
    for (uint t=0;t<burst;++t) {
        if (!isfinite(y[t])) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        partial[(block*8+t)*2560+d]=y[t];
    }
}
/* Load-time warm-up: one read per 16 KB page so the GPU maps every page of a
 * layer before the first request (cold pages cost tens of ms per call). */
/* Keep-alive: one threadgroup of FMAs on a side queue between calls, so the
 * GPU does not drop its clocks while the node waits for the next request. */
kernel void keepalive(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    float a=out[tid];
    for (uint i=0;i<iters;++i) { a=fma(a,1.000001f,0.000001f); a=fma(a,1.000001f,-0.000001f); }
    out[tid]=a;
}
kernel void touch(device const uint *w [[buffer(0)]], device atomic_uint *sink [[buffer(1)]],
                  constant ulong &words [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const ulong at=ulong(i)*4096;
    if (at<words && w[at]==0x9e3779b9u) atomic_fetch_add_explicit(sink,1u,memory_order_relaxed);
}
/* fronk demo: read a span of memory at full bandwidth, as a dense layer's
 * weights are read once per pass (coalesced uint4, folded into one word). */
kernel void stream(device const uint4 *w [[buffer(0)]], device atomic_uint *sink [[buffer(1)]],
                   constant ulong &n [[buffer(2)]], uint i [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    uint4 acc=uint4(0);
    for (ulong k=i;k<n;k+=threads) acc^=w[k];
    if ((acc.x^acc.y^acc.z^acc.w)==0x9e3779b9u) atomic_fetch_add_explicit(sink,1u,memory_order_relaxed);
}
kernel void reduce(constant Routing &r [[buffer(0)]], device const float *partial [[buffer(1)]],
                   device float *out [[buffer(2)]], device atomic_uint *error [[buffer(3)]],
                   uint d [[thread_position_in_grid]]) {
    for (uint t=0;t<8;++t) {
        float sum=0;
        if (t<r.burst) for (uint b=0;b<20;++b) if (r.mask&(1u<<b)) sum+=partial[(b*8+t)*2560+d];
        if (!isfinite(sum)) atomic_fetch_or_explicit(error,1u,memory_order_relaxed);
        out[t*2560+d]=sum;
    }
}
);
#undef MSL

typedef struct { uint32_t id,mask; float gate[FR_MAX_BURST]; } metal_expert;
typedef struct { uint32_t burst,count,mask,slots[FR_BLOCKS]; metal_expert expert[MAX_EXPERTS]; } metal_routing;
_Static_assert(sizeof(metal_expert)==40 && offsetof(metal_routing,expert)==92,"MSL routing layout");
static struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> gu,down,reduce,gu_native,down_native,touch,keepalive,stream;
    id<MTLCommandQueue> side;
    id<MTLBuffer> scratch;
    volatile double last_run;
    int keepalive_started;
    volatile int keepalive_stop;
    pthread_t keepalive_thread;
    id<MTLBuffer> weights[FR_BLOCKS][FR_LAYERS], input,routing,hidden,partial,output,error;
    id<MTLBuffer> dense; uint64_t dense_at; /* fronk demo: the head's synthetic dense weights */
    size_t bias[FR_BLOCKS][FR_LAYERS];
    uint32_t mask,native_mask;
    int in_run;
    uint64_t allocations,hot_allocations,runs;
    mach_timebase_info_data_t timebase;
    fr_metal_sample last;
    fr_ssd *ssd;
    uint64_t cache_bytes;
    uint32_t ssd_mask;
    int ssd_pipeline;
    id<MTLBuffer> staging[FR_BLOCKS];
} metal;
static void allocation(void) { ++metal.allocations; if (metal.in_run) ++metal.hot_allocations; }
static double stamp(void) { return (double)mach_absolute_time()*metal.timebase.numer/metal.timebase.denom; }
static id<MTLBuffer> buffer(size_t size) {
    allocation(); return [metal.device newBufferWithLength:size options:MTLResourceStorageModeShared];
}
static int finish_setup(void) {
    if (metal.device) return FR_OK;
    @autoreleasepool {
        allocation(); metal.device=MTLCreateSystemDefaultDevice();
        if (!metal.device || ![metal.device hasUnifiedMemory]) goto failed;
        if (mach_timebase_info(&metal.timebase)!=KERN_SUCCESS) goto failed;
        allocation(); metal.queue=[metal.device newCommandQueue];
        allocation(); MTLCompileOptions *options=[[MTLCompileOptions alloc] init];
        options.languageVersion=MTLLanguageVersion2_4;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled=NO;
#pragma clang diagnostic pop
        NSError *error=nil; allocation(); metal.library=[metal.device newLibraryWithSource:source options:options error:&error];
        [options release];
        if (!metal.library) { fprintf(stderr,"metal: shader compilation: %s\n",[[error localizedDescription] UTF8String]); goto failed; }
        const char *names[]={"gate_up","down","reduce","gate_up_native","down_native","touch","keepalive","stream"};
        id<MTLComputePipelineState> *pipelines[]={&metal.gu,&metal.down,&metal.reduce,&metal.gu_native,&metal.down_native,&metal.touch,&metal.keepalive,&metal.stream};
        for (int i=0;i<8;++i) {
            allocation(); id<MTLFunction> fn=[metal.library newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
            allocation(); *pipelines[i]=[metal.device newComputePipelineStateWithFunction:fn error:&error];
            [fn release];
            if (!*pipelines[i] || [*pipelines[i] threadExecutionWidth]!=32 || [*pipelines[i] maxTotalThreadsPerThreadgroup]<128) goto failed;
        }
        metal.input=buffer(sizeof(float)*FR_MAX_BURST*FR_HIDDEN);
        metal.routing=buffer(sizeof(metal_routing));
        metal.hidden=buffer((size_t)FR_BLOCKS*MAX_EXPERTS*FR_MAX_BURST*FR_BLOCK_NEURONS*sizeof(float));
        metal.partial=buffer((size_t)FR_BLOCKS*FR_MAX_BURST*FR_HIDDEN*sizeof(float));
        metal.output=buffer(sizeof(float)*FR_MAX_BURST*FR_HIDDEN); metal.error=buffer(sizeof(uint32_t));
        if (!metal.queue || !metal.input || !metal.routing || !metal.hidden || !metal.partial || !metal.output || !metal.error) goto failed;
        return FR_OK;
    failed:
        (void)fr_metal_close(); return FR_UNSUPPORTED;
    }
}
int fr_metal_close(void) {
    if (metal.mask || metal.in_run) return FR_STATE;
    if (metal.keepalive_started) { metal.keepalive_stop=1; pthread_join(metal.keepalive_thread,NULL); metal.keepalive_started=0; }
    fr_ssd_close(metal.ssd);
    [metal.side release]; [metal.scratch release];
    [metal.input release]; [metal.routing release]; [metal.hidden release]; [metal.partial release];
    [metal.output release]; [metal.error release]; [metal.gu release]; [metal.down release]; [metal.reduce release];
    [metal.gu_native release]; [metal.down_native release]; [metal.touch release]; [metal.keepalive release];
    [metal.stream release]; [metal.dense release];
    [metal.library release]; [metal.queue release]; [metal.device release];
    memset(&metal,0,sizeof(metal)); return FR_OK;
}
static int metal_probe(fr_hwinfo *out) {
    if (!out) return FR_INVALID;
    fr_hwinfo h; if (hw_probe(&h)) return FR_IO;
    if (h.translated || strcmp(h.arch,"arm64") || !h.metal) return FR_UNSUPPORTED;
    *out=h; return FR_OK;
}
static int metal_load(int block,const void *data,size_t len) {
    if (block<0 || block>=FR_BLOCKS || !data || (len!=FR_WEIGHT_BLOCK_BYTES && len!=FR_NATIVE_BLOCK_BYTES)) return FR_INVALID;
    if (metal.mask&(1u<<block)) return FR_STATE;
    int rc=finish_setup(); if (rc) return rc;
    long page_size=sysconf(_SC_PAGESIZE); if (page_size<=0) return FR_IO;
    size_t page=(size_t)page_size;
    /* A barrier references only one layer. Register per-layer views once so
     * GPU residency never includes 47 unused layers for every owned block. */
    for (unsigned layer=0;layer<FR_LAYERS;++layer) {
        int mixed=len==FR_NATIVE_BLOCK_BYTES;
        uintptr_t address=(uintptr_t)data+(mixed?fr_native_offset(layer,0):fr_slice_offset(layer,0));
        size_t bias=address%page, covered=((mixed?FR_NATIVE_LAYER_BYTES:FR_SLICE_BYTES*FR_EXPERTS)+bias+page-1)/page*page;
        if (covered>[metal.device maxBufferLength]) rc=FR_INVALID;
        else {
            allocation();
            metal.weights[block][layer]=[metal.device newBufferWithBytesNoCopy:(void *)(address-bias) length:covered
                                                                    options:MTLResourceStorageModeShared deallocator:nil];
            if (!metal.weights[block][layer]) rc=FR_IO;
        }
        if (rc) {
            for (unsigned j=0;j<=layer;++j) { [metal.weights[block][j] release]; metal.weights[block][j]=nil; }
            return rc;
        }
        metal.bias[block][layer]=bias;
    }
    /* Map every page of every layer on the GPU now, not during generation. */
    @autoreleasepool {
        id<MTLCommandBuffer> command=[metal.queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if (command && encoder) {
            [encoder setComputePipelineState:metal.touch]; [encoder setBuffer:metal.error offset:0 atIndex:1];
            for (unsigned layer=0;layer<FR_LAYERS;++layer) {
                const uint64_t words=([metal.weights[block][layer] length]-metal.bias[block][layer])/4;
                [encoder setBuffer:metal.weights[block][layer] offset:metal.bias[block][layer] atIndex:0];
                [encoder setBytes:&words length:sizeof(words) atIndex:2];
                const NSUInteger pages=(NSUInteger)((words+4095)/4096);
                [encoder dispatchThreads:MTLSizeMake(pages,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            }
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        }
        *(uint32_t *)[metal.error contents]=0;
    }
    metal.mask|=1u<<block;
    if (len==FR_NATIVE_BLOCK_BYTES) metal.native_mask|=1u<<block;
    return FR_OK;
}
static int load_ssd(unsigned block,int fd,uint64_t cache_bytes,int fixture) {
    if (block>=FR_BLOCKS || (metal.mask&(1u<<block))) return FR_STATE;
    int rc=finish_setup(); if (rc) return rc;
    if (metal.ssd && metal.cache_bytes!=cache_bytes) return FR_STATE;
    if (!metal.ssd) {
        metal.ssd=fr_ssd_open(cache_bytes,FR_SSD_WORKERS); metal.cache_bytes=cache_bytes;
        if (!metal.ssd) return FR_IO;
        const char *pipeline=getenv("FRONK_SSD_PIPELINE");
        /* Component wins have not yet translated into a bracketed chat win. */
        metal.ssd_pipeline=pipeline && !strcmp(pipeline,"1");
    }
    metal.staging[block]=buffer(FR_NATIVE_LAYER_BYTES);
    if (!metal.staging[block]) return FR_IO;
    rc=fixture?fr_ssd_attach_fixture(metal.ssd,block,fd):fr_ssd_attach(metal.ssd,block,fd);
    if (rc) { [metal.staging[block] release]; metal.staging[block]=nil; return rc; }
    metal.mask|=1u<<block; metal.native_mask|=1u<<block; metal.ssd_mask|=1u<<block;
    return FR_OK;
}
int fr_metal_load_ssd(unsigned block,int fd,uint64_t bytes) { return load_ssd(block,fd,bytes,0); }
int fr_metal_load_ssd_fixture(unsigned block,int fd,uint64_t bytes) { return load_ssd(block,fd,bytes,1); }
void fr_metal_ssd_clear(void) { if (metal.ssd) fr_ssd_clear(metal.ssd); }
int fr_metal_ssd_pipeline(int enabled) {
    if (metal.in_run || (enabled!=0 && enabled!=1)) return FR_STATE;
    metal.ssd_pipeline=enabled; return FR_OK;
}
void fr_metal_ssd_sample(fr_ssd_sample *out) {
    if (out) { memset(out,0,sizeof(*out)); if (metal.ssd) fr_ssd_sample_get(metal.ssd,out); }
}
static int metal_unload(int block) {
    if (block<0 || block>=FR_BLOCKS) return FR_INVALID;
    if (!(metal.mask&(1u<<block))) return FR_STATE;
    if (metal.ssd_mask&(1u<<block)) {
        int rc=fr_ssd_detach(metal.ssd,(unsigned)block); if (rc) return rc;
        [metal.staging[block] release]; metal.staging[block]=nil; metal.ssd_mask&=~(1u<<block);
        if (!metal.ssd_mask) { fr_ssd_close(metal.ssd); metal.ssd=NULL; metal.cache_bytes=0; }
    }
    for (unsigned layer=0;layer<FR_LAYERS;++layer) { [metal.weights[block][layer] release]; metal.weights[block][layer]=nil; }
    metal.mask&=~(1u<<block); metal.native_mask&=~(1u<<block); return FR_OK;
}
/* While calls keep arriving (last one under 50 ms ago) a side queue runs one
 * threadgroup of FMAs back to back, so the GPU keeps its clocks in the gaps
 * between layers. M4 Pro, 3 blocks, burst 2, calls 3 ms apart: p50 1085 ->
 * 303 us (back to back 312 -> 266). Short kernels (100k iterations) keep the
 * tail low; 1.2M blocked real work (p99 9.8 ms). FRONK_METAL_KEEPALIVE=N sets
 * the iterations, 0 turns it off. Parked when idle; close() joins it. */
static uint32_t keepalive_iters(void) {
    const char *v=getenv("FRONK_METAL_KEEPALIVE"); return v && *v?(uint32_t)strtoul(v,NULL,10):100000u;
}
static void *keepalive_thread(void *arg) {
    (void)arg; const uint32_t iters=keepalive_iters();
    while (!metal.keepalive_stop) {
        if (stamp()-metal.last_run>50e6) { usleep(1000); continue; }
        @autoreleasepool {
            id<MTLCommandBuffer> cb=[metal.side commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:metal.keepalive]; [enc setBuffer:metal.scratch offset:0 atIndex:0];
            [enc setBytes:&iters length:sizeof(iters) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
    }
    return NULL;
}
static void keepalive_start(void) {
    if (metal.keepalive_started || !keepalive_iters()) return;
    metal.keepalive_started=1;
    allocation(); metal.side=[metal.device newCommandQueue];
    allocation(); metal.scratch=[metal.device newBufferWithLength:256*sizeof(float) options:MTLResourceStorageModePrivate];
    metal.keepalive_stop=0;
    if (!metal.side || !metal.scratch || pthread_create(&metal.keepalive_thread,NULL,keepalive_thread,NULL)) metal.keepalive_started=0;
}
static int prepare(const fr_req *req) {
    if (!req || req->layer>=FR_LAYERS || req->burst<1 || req->burst>FR_MAX_BURST || fegetround()!=FE_TONEAREST) return FR_INVALID;
    for (unsigned t=0;t<req->burst;++t) {
        for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) {
            if (req->expert[t][e]>=FR_ROUTED_EXPERTS) return FR_INVALID;
            for (unsigned j=0;j<e;++j) if (req->expert[t][j]==req->expert[t][e]) return FR_INVALID;
        }
        for (unsigned e=0;e<FR_ACTIVE_EXPERTS;++e) {
            if (!isfinite(req->gate[t][e])) return FR_NUMERIC;
            if (req->gate[t][e]<0) return FR_INVALID;
        }
        for (unsigned d=0;d<FR_HIDDEN;++d) if (!isfinite(req->x[t][d])) return FR_NUMERIC;
    }
    if (!metal.mask) return FR_STATE;
    metal_routing *r=[metal.routing contents]; memset(r,0,sizeof(*r));
    r->burst=req->burst; r->mask=metal.mask;
    for (unsigned b=0;b<FR_BLOCKS;++b) r->slots[b]=b|((metal.native_mask&(1u<<b))?256:0);
#ifdef FR_EXPERT_RANKED
    unsigned ranked=req->burst>1;
    for (unsigned t=0;t<req->burst;++t) if (req->gate[t][FR_ACTIVE_ROUTED]!=0) ranked=0;
    if (ranked) {
        /* Rank-major grouping preserves each token's expert sum order. The
         * same expert at different ranks remains separate; no output scratch
         * allocation or second GPU reduction pass is needed. */
        for (unsigned rank=0;rank<FR_ACTIVE_ROUTED;++rank) {
            unsigned begin=r->count;
            for (unsigned t=0;t<req->burst;++t) {
                unsigned id=req->expert[t][rank],j=begin;
                while (j<r->count && r->expert[j].id!=id) ++j;
                if (j==r->count) { r->expert[j].id=id; ++r->count; }
                r->expert[j].mask|=1u<<t; r->expert[j].gate[t]=req->gate[t][rank];
            }
        }
    } else
#endif
    for (unsigned t=0;t<req->burst;++t) for (unsigned e=0;e<FR_ACTIVE_EXPERTS;++e) {
        if (e==FR_ACTIVE_ROUTED && req->gate[t][e]==0) continue; /* shared expert computed by the head */
        uint32_t id=e==FR_ACTIVE_ROUTED?FR_SHARED_EXPERT:req->expert[t][e]; unsigned j=0;
        while (j<r->count && r->expert[j].id!=id) ++j;
        if (j==r->count) { r->expert[j].id=id; ++r->count; }
        r->expert[j].mask|=1u<<t; r->expert[j].gate[t]=req->gate[t][e];
    }
    memcpy([metal.input contents],req->x,sizeof(req->x)); *(uint32_t *)[metal.error contents]=0;
    return FR_OK;
}
static int metal_ssd_run(const fr_req *req,fr_part *out);
static int metal_run(const fr_req *req,fr_part *out) {
    if (metal.ssd_mask) return metal_ssd_run(req,out);
    if (!out) return FR_INVALID;
    double start=metal.device?stamp():0;
    if (metal.device) { metal.last_run=start; keepalive_start(); }
    metal.in_run=1;
    int rc=prepare(req); if (rc) { metal.in_run=0; return rc; }
    double commit=0,gpu_start=0,gpu_end=0;
    unsigned blocks=0,experts=0;
    @autoreleasepool {
        id<MTLCommandBuffer> command=[metal.queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if (!command || !encoder) { metal.in_run=0; return FR_IO; }
        metal_routing *r=[metal.routing contents];
        experts=r->count;
        [encoder setBuffer:metal.input offset:0 atIndex:1]; [encoder setBuffer:metal.routing offset:0 atIndex:2];
        [encoder setBuffer:metal.hidden offset:0 atIndex:3]; [encoder setBuffer:metal.error offset:0 atIndex:5];
        for (unsigned b=0;b<FR_BLOCKS;++b) if (metal.mask&(1u<<b)) {
            ++blocks;
            [encoder setComputePipelineState:(metal.native_mask&(1u<<b))?metal.gu_native:metal.gu];
            [encoder setBuffer:metal.weights[b][req->layer] offset:metal.bias[b][req->layer] atIndex:0];
            [encoder setBuffer:metal.routing offset:offsetof(metal_routing,slots)+b*sizeof(uint32_t) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(8,r->count,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        }
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setBuffer:metal.routing offset:0 atIndex:1]; [encoder setBuffer:metal.hidden offset:0 atIndex:2];
        [encoder setBuffer:metal.partial offset:0 atIndex:3];
        for (unsigned b=0;b<FR_BLOCKS;++b) if (metal.mask&(1u<<b)) {
            [encoder setComputePipelineState:(metal.native_mask&(1u<<b))?metal.down_native:metal.down];
            [encoder setBuffer:metal.weights[b][req->layer] offset:metal.bias[b][req->layer] atIndex:0];
            [encoder setBuffer:metal.routing offset:offsetof(metal_routing,slots)+b*sizeof(uint32_t) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(FR_HIDDEN/128,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        }
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:metal.reduce];
        [encoder setBuffer:metal.routing offset:0 atIndex:0]; [encoder setBuffer:metal.partial offset:0 atIndex:1];
        [encoder setBuffer:metal.output offset:0 atIndex:2]; [encoder setBuffer:metal.error offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(FR_HIDDEN/128,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [encoder endEncoding];
        commit=stamp(); [command commit]; [command waitUntilCompleted];
        if (command.status!=MTLCommandBufferStatusCompleted) rc=FR_IO;
        else if (*(uint32_t *)[metal.error contents]) rc=FR_NUMERIC;
        gpu_start=command.GPUStartTime*1e9; gpu_end=command.GPUEndTime*1e9;
    }
    double end=stamp();
    if (!rc && (gpu_start<commit || gpu_end<gpu_start || end<gpu_end)) rc=FR_IO;
    if (!rc) {
        out->layer=req->layer; out->burst=req->burst; memcpy(out->y,[metal.output contents],sizeof(out->y));
        end=stamp(); ++metal.runs;
        metal.last=(fr_metal_sample){.ns={commit-start,gpu_start-commit,gpu_end-gpu_start,end-gpu_end,
                                         end-start-(gpu_end-gpu_start),end-start},
            .weight_bytes=(uint64_t)blocks*experts*FR_SLICE_BYTES,.runs=metal.runs,
            .hot_allocations=metal.hot_allocations,.resources=metal.allocations,.blocks=blocks,.unique_experts=experts};
        metal.last.weight_bytes=0;
        for (unsigned b=0;b<FR_BLOCKS;++b) if (metal.mask&(1u<<b)) metal.last.weight_bytes+=(metal.native_mask&(1u<<b))?
            (experts-1)*FR_NATIVE_SLICE_BYTES+FR_NATIVE_SHARED_BYTES:experts*FR_SLICE_BYTES;
    }
    metal.in_run=0; return rc;
}
/* Separate command buffers let resident GPU work run during selected SSD reads.
 * Routing/input/partial buffers are immutable or disjoint until the final
 * ordered reduction. Staging is not referenced until all reads have succeeded. */
static id<MTLCommandBuffer> hybrid_command(const fr_req *req,uint32_t mask,int final) {
    id<MTLCommandBuffer> cb=[metal.queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    if (!cb || !enc) return nil;
    metal_routing *r=[metal.routing contents];
    [enc setBuffer:metal.input offset:0 atIndex:1]; [enc setBuffer:metal.routing offset:0 atIndex:2];
    [enc setBuffer:metal.hidden offset:0 atIndex:3]; [enc setBuffer:metal.error offset:0 atIndex:5];
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask>>b&1) {
        [enc setComputePipelineState:(metal.native_mask>>b&1)?metal.gu_native:metal.gu];
        [enc setBuffer:(metal.ssd_mask>>b&1)?metal.staging[b]:metal.weights[b][req->layer]
                offset:(metal.ssd_mask>>b&1)?0:metal.bias[b][req->layer] atIndex:0];
        [enc setBuffer:metal.routing offset:offsetof(metal_routing,slots)+b*sizeof(uint32_t) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(8,r->count,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setBuffer:metal.routing offset:0 atIndex:1]; [enc setBuffer:metal.hidden offset:0 atIndex:2];
    [enc setBuffer:metal.partial offset:0 atIndex:3];
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask>>b&1) {
        [enc setComputePipelineState:(metal.native_mask>>b&1)?metal.down_native:metal.down];
        [enc setBuffer:(metal.ssd_mask>>b&1)?metal.staging[b]:metal.weights[b][req->layer]
                offset:(metal.ssd_mask>>b&1)?0:metal.bias[b][req->layer] atIndex:0];
        [enc setBuffer:metal.routing offset:offsetof(metal_routing,slots)+b*sizeof(uint32_t) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(FR_HIDDEN/128,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    if (final) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:metal.reduce]; [enc setBuffer:metal.routing offset:0 atIndex:0];
        [enc setBuffer:metal.partial offset:0 atIndex:1]; [enc setBuffer:metal.output offset:0 atIndex:2];
        [enc setBuffer:metal.error offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(FR_HIDDEN/128,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    [enc endEncoding]; return cb;
}
typedef struct {
    const fr_req *req;
    uint32_t submitted;
    unsigned count;
    id<MTLCommandBuffer> command[FR_SSD_MAX_BLOCKS];
} hybrid_stream;
static int hybrid_ready(void *context,uint32_t ready) {
    hybrid_stream *s=context;
    if (!ready || (ready&~metal.ssd_mask) || (ready&s->submitted) || s->count>=FR_SSD_MAX_BLOCKS) return FR_STATE;
    uint32_t submitted=s->submitted|ready;
    id<MTLCommandBuffer> cb=hybrid_command(s->req,ready,submitted==metal.ssd_mask);
    if (!cb) return FR_IO;
    s->command[s->count++]=cb; s->submitted=submitted; [cb commit];
    return FR_OK;
}
static int metal_ssd_run(const fr_req *req,fr_part *out) {
    if (!out) return FR_INVALID;
    double begin=stamp(); metal.in_run=1;
    int rc=prepare(req); if (rc) { metal.in_run=0; return rc; }
    double gpu=0;
    @autoreleasepool {
        uint32_t ram=metal.mask&~metal.ssd_mask;
        id<MTLCommandBuffer> resident=ram?hybrid_command(req,ram,0):nil;
        if (ram && !resident) { metal.in_run=0; return FR_IO; }
        if (resident) [resident commit];
        void *layers[FR_BLOCKS]={0};
        for (unsigned b=0;b<FR_BLOCKS;++b) if (metal.ssd_mask>>b&1) layers[b]=[metal.staging[b] contents];
        hybrid_stream stream={.req=req};
        if (metal.ssd_pipeline)
            rc=fr_ssd_load_ready(metal.ssd,req,metal.ssd_mask,layers,hybrid_ready,&stream);
        else {
            rc=fr_ssd_load(metal.ssd,req,metal.ssd_mask,layers);
            if (!rc) rc=hybrid_ready(&stream,metal.ssd_mask);
        }
        /* Drain both on every exit. A failed read must never leave GPU work
         * using buffers that the next call or unload may overwrite. */
        if (resident) {
            [resident waitUntilCompleted];
            if (resident.status!=MTLCommandBufferStatusCompleted) rc=FR_IO;
            gpu+=(resident.GPUEndTime-resident.GPUStartTime)*1e9;
        }
        for (unsigned i=0;i<stream.count;++i) {
            id<MTLCommandBuffer> cb=stream.command[i];
            [cb waitUntilCompleted];
            if (cb.status!=MTLCommandBufferStatusCompleted) rc=FR_IO;
            gpu+=(cb.GPUEndTime-cb.GPUStartTime)*1e9;
        }
        if (!rc && *(uint32_t *)[metal.error contents]) rc=FR_NUMERIC;
    }
    double elapsed=stamp()-begin;
    if (!rc) {
        out->layer=req->layer; out->burst=req->burst; memcpy(out->y,[metal.output contents],sizeof(out->y));
        ++metal.runs;
        metal.last=(fr_metal_sample){.ns={0,0,gpu,0,elapsed-gpu,elapsed},.runs=metal.runs,
            .hot_allocations=metal.hot_allocations,.resources=metal.allocations,
            .blocks=(unsigned)__builtin_popcount(metal.mask),.unique_experts=((metal_routing *)[metal.routing contents])->count};
    }
    metal.in_run=0; return rc;
}
/* fronk demo: the GPU time to read `bytes` of weights once, as the head's dense
 * part of a layer does per pass. Reads walk a 2 GB resident buffer, so no read
 * hits what an earlier one left in the GPU's caches. */
#define FR_DENSE_PROBE_BYTES (2ull<<30)
int fr_metal_stream(uint64_t bytes,double *gpu_ns) {
    if (!gpu_ns || !bytes || bytes>FR_DENSE_PROBE_BYTES/2) return FR_INVALID;
    int rc=finish_setup(); if (rc) return rc;
    @autoreleasepool {
        if (!metal.dense) {
            allocation(); metal.dense=[metal.device newBufferWithLength:FR_DENSE_PROBE_BYTES options:MTLResourceStorageModeShared];
            if (!metal.dense) return FR_IO;
            memset([metal.dense contents],0x11,FR_DENSE_PROBE_BYTES);
        }
        bytes&=~(uint64_t)15;
        if (metal.dense_at+bytes>FR_DENSE_PROBE_BYTES) metal.dense_at=0;
        const uint64_t n=bytes/16;
        id<MTLCommandBuffer> command=[metal.queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if (!command || !encoder) return FR_IO;
        [encoder setComputePipelineState:metal.stream];
        [encoder setBuffer:metal.dense offset:metal.dense_at atIndex:0]; [encoder setBuffer:metal.error offset:0 atIndex:1];
        [encoder setBytes:&n length:sizeof(n) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(65536,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        metal.dense_at+=bytes;
        if ([command status]!=MTLCommandBufferStatusCompleted) return FR_IO;
        *gpu_ns=([command GPUEndTime]-[command GPUStartTime])*1e9;
        *(uint32_t *)[metal.error contents]=0;
    }
    return FR_OK;
}
int fr_metal_sample_get(fr_metal_sample *out) {
    if (!out) return FR_INVALID;
    *out=metal.last; out->resources=metal.allocations; out->hot_allocations=metal.hot_allocations;
    return FR_OK;
}
static int metal_self_test(const char *dir) {
    if (metal.mask) return FR_STATE;
    int rc=fr_vectors_check(&fr_metal,dir,&fr_backend_tolerance,NULL);
    return rc?rc:fr_native_self_test(&fr_metal);
}
static int metal_bench(int burst_max,fr_costmodel *out) {
    return fr_bench_execute(burst_max,out);
}
const fr_backend fr_metal={"metal",metal_probe,metal_load,metal_unload,metal_run,metal_bench,metal_self_test};
const fr_backend fr_metal_ssd={"metal-ssd",metal_probe,metal_load,metal_unload,metal_run,metal_bench,metal_self_test};
#endif
