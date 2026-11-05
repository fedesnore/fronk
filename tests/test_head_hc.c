/* Real Metal arithmetic and guard checks, plus a small kernel-only benchmark.
 * No model, cluster or network is needed. The reference is independent T=1. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *f) { (void)f; return false; }
static void require(int ok,const char *what) {
    if(!ok) {fprintf(stderr,"FAIL %s\n",what);exit(1);}
}
static unsigned seed=7123981;
static float random_float(void) {
    seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
    return (float)(seed&0xffffff)/8388608.0f-1.0f;
}
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void compare(const float *a,const float *b,size_t n) {
    for(size_t i=0;i<n;i++) {
        unsigned x,y;memcpy(&x,a+i,4);memcpy(&y,b+i,4);
        if(x!=y || (x&0x7f800000u)==0x7f800000u) {
            fprintf(stderr,"FAIL index=%zu actual=%.9g expected=%.9g bits=%08x/%08x\n",i,a[i],b[i],x,y);exit(1);
        }
    }
}
static void shape(unsigned E,unsigned rank) {
    const unsigned guard=17,maxT=8;
    size_t bytes=((size_t)E*4*rank*2+16383)&~(size_t)16383;
    void *weights=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    require(weights!=MAP_FAILED,"weight allocation");
    for(size_t i=0;i<(size_t)E*4*rank;i++) ((__fp16 *)weights)[i]=random_float()*.3f;
    require(ds4_gpu_set_model_map(weights,bytes),"model map registration");
    float *x=malloc(maxT*4*E*4),*lo=malloc(maxT*rank*4);
    float *ref=malloc((maxT*E+guard)*4),*got=malloc((maxT*E+guard)*4);
    require(x&&lo&&ref&&got,"host allocation");
    for(unsigned i=0;i<maxT*4*E;i++) x[i]=random_float();
    for(unsigned i=0;i<maxT*rank;i++) lo[i]=random_float()*6;
    ds4_gpu_tensor *gx=ds4_gpu_tensor_alloc(maxT*4*E*4),*gl=ds4_gpu_tensor_alloc(maxT*rank*4);
    ds4_gpu_tensor *gr=ds4_gpu_tensor_alloc((maxT*E+guard)*4),*go=ds4_gpu_tensor_alloc((maxT*E+guard)*4);
    require(gx&&gl&&gr&&go,"device allocation");
    require(ds4_gpu_tensor_write(gx,0,x,maxT*4*E*4)&&ds4_gpu_tensor_write(gl,0,lo,maxT*rank*4),"upload");
    for(unsigned T=1;T<=2;T++) {
        require(ds4_gpu_tensor_fill_f32(gr,127.25f,maxT*E+guard)&&ds4_gpu_tensor_fill_f32(go,127.25f,maxT*E+guard),"guards");
        ds4_gpu_fronk_head_hc(false);
        require(ds4_gpu_begin_commands(),"reference begin");
        for(unsigned t=0;T==1 && t<T;t++) {
            ds4_gpu_tensor *xr=ds4_gpu_tensor_view(gx,(uint64_t)t*4*E*4,4*E*4);
            ds4_gpu_tensor *lr=ds4_gpu_tensor_view(gl,(uint64_t)t*rank*4,rank*4);
            ds4_gpu_tensor *out=ds4_gpu_tensor_view(gr,(uint64_t)t*E*4,E*4);
            require(xr&&lr&&out&&ds4_gpu_qwen4_hc_gate_mix_tensor(out,xr,lr,weights,bytes,0,1,1,E,4,rank),"reference mixer");
            ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(lr);ds4_gpu_tensor_free(xr);
        }
        if(T==2) require(ds4_gpu_qwen4_hc_gate_mix_tensor(gr,gx,gl,weights,bytes,0,1,T,E,4,rank),"unchanged pair reference");
        require(ds4_gpu_end_commands(),"reference end");
        ds4_gpu_fronk_head_hc(true);
        require(ds4_gpu_qwen4_hc_gate_mix_tensor(go,gx,gl,weights,bytes,0,1,T,E,4,rank),"native mixer");
        require(ds4_gpu_tensor_read(gr,0,ref,(maxT*E+guard)*4)&&ds4_gpu_tensor_read(go,0,got,(maxT*E+guard)*4),"readback");
        compare(got,ref,maxT*E+guard);
        if(E==2560 && T==1) for (unsigned repeat=0;repeat<5;++repeat) {
            double us[3];
            for(unsigned phase=0;phase<3;phase++) {
                ds4_gpu_fronk_head_hc(phase==1);
                for(unsigned i=0;i<4;i++) require(ds4_gpu_qwen4_hc_gate_mix_tensor(go,gx,gl,weights,bytes,0,1,T,E,4,rank),"benchmark warmup");
                const double start=now();
                require(ds4_gpu_begin_commands(),"benchmark begin");
                for(unsigned i=0;i<128;i++) require(ds4_gpu_qwen4_hc_gate_mix_tensor(go,gx,gl,weights,bytes,0,1,T,E,4,rank),"benchmark dispatch");
                require(ds4_gpu_end_commands(),"benchmark end");
                us[phase]=(now()-start)*1e6/128;
            }
            printf("HC T=%u repeat=%u baseline_before_us=%.2f native_us=%.2f baseline_after_us=%.2f\n",T,repeat,us[0],us[1],us[2]);
        }
    }
    printf("PASS exact scalar T=1 and unchanged existing T=2 E=%u rank=%u, output guards\n",E,rank);
    ds4_gpu_fronk_head_hc(false);
    ds4_gpu_tensor_free(go);ds4_gpu_tensor_free(gr);ds4_gpu_tensor_free(gl);ds4_gpu_tensor_free(gx);
    free(got);free(ref);free(lo);free(x);
    /* Metal may retain the wrapped weight mapping until shutdown. */
}

int main(void) {
 require(!chdir("build/head-cost-engine"),"staged Metal source directory");
 require(ds4_gpu_init(),"Metal device");
 shape(259,32); shape(263,68); shape(2560,320);
 puts("PASS head HC single-token exactness and guards"); return 0;
}
