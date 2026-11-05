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
static void history_prefix(void) {
    enum {C=259,GUARD=17};
    float initial[9*C+GUARD],rows[8*C],want[9*C+GUARD],got[9*C+GUARD];
    for(unsigned i=0;i<9*C+GUARD;i++) initial[i]=random_float();
    for(unsigned i=0;i<8*C;i++) rows[i]=random_float();
    ds4_gpu_tensor *h=ds4_gpu_tensor_alloc(sizeof(initial)),*x=ds4_gpu_tensor_alloc(sizeof(rows));
    require(h&&x&&ds4_gpu_tensor_write(x,0,rows,sizeof(rows)),"history buffers");
    for(unsigned length=1;length<=9;length++) for(unsigned count=0;count<=6;count++) {
        memcpy(want,initial,sizeof(want));
        for(unsigned t=0;t<count;t++) {
            memmove(want,want+C,(length-1)*C*sizeof(float));
            memcpy(want+(length-1)*C,rows+(2+t)*C,C*sizeof(float));
        }
        require(ds4_gpu_tensor_write(h,0,initial,sizeof(initial)) &&
            ds4_gpu_fronk_history_prefix(h,x,C,length,2,count) &&
            ds4_gpu_tensor_read(h,0,got,sizeof(got)),"history recovery");
        compare(got,want,9*C+GUARD);
    }
    require(!ds4_gpu_fronk_history_prefix(h,x,C,10,2,1) &&
        !ds4_gpu_fronk_history_prefix(h,x,C,2,2,7),"history invalid bounds");
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(h);
    puts("PASS history recovery: lengths 1..9, prefixes 0..6, exact state, guards and invalid bounds");
}
static void q8_shape(unsigned K,unsigned M) {
    const unsigned guard=17,maxT=8;
    const size_t row_bytes=(size_t)K/32*34;
    size_t bytes=(row_bytes*M+16383)&~(size_t)16383;
    unsigned char *weights=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    require(weights!=MAP_FAILED,"Q8 weights");
    for(unsigned row=0;row<M;row++) for(unsigned k=0;k<K/32;k++) {
        unsigned char *b=weights+row*row_bytes+k*34;
        __fp16 scale=(random_float()+1.01f)*.003f;memcpy(b,&scale,2);
        for(unsigned j=0;j<32;j++) ((signed char *)b)[2+j]=(signed char)(random_float()*127);
    }
    require(ds4_gpu_set_model_map(weights,bytes),"Q8 registration");
    float *x=malloc(maxT*K*4),*ref=malloc((maxT*M+guard)*4),*got=malloc((maxT*M+guard)*4);
    require(x&&ref&&got,"Q8 host buffers");
    for(unsigned i=0;i<maxT*K;i++) x[i]=random_float();
    ds4_gpu_tensor *gx=ds4_gpu_tensor_alloc(maxT*K*4),*gr=ds4_gpu_tensor_alloc((maxT*M+guard)*4),*go=ds4_gpu_tensor_alloc((maxT*M+guard)*4);
    require(gx&&gr&&go&&ds4_gpu_tensor_write(gx,0,x,maxT*K*4),"Q8 device buffers");
    for(unsigned T=1;T<=maxT;T++) {
        require(ds4_gpu_tensor_fill_f32(gr,127.25f,maxT*M+guard)&&ds4_gpu_tensor_fill_f32(go,127.25f,maxT*M+guard),"Q8 guards");
        ds4_gpu_fronk_native_rows(false);require(ds4_gpu_begin_commands(),"Q8 reference begin");
        for(unsigned t=0;t<T;t++) {
            ds4_gpu_tensor *xr=ds4_gpu_tensor_view(gx,(uint64_t)t*K*4,K*4),*out=ds4_gpu_tensor_view(gr,(uint64_t)t*M*4,M*4);
            require(xr&&out&&ds4_gpu_qwen4_matmul_q8_0_tensor(out,weights,bytes,0,K,M,xr,1),"Q8 reference");
            ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(xr);
        }
        require(ds4_gpu_end_commands(),"Q8 reference end");ds4_gpu_fronk_native_rows(true);
        require(ds4_gpu_qwen4_matmul_q8_0_tensor(go,weights,bytes,0,K,M,gx,T),"Q8 native");
        require(ds4_gpu_tensor_read(gr,0,ref,(maxT*M+guard)*4)&&ds4_gpu_tensor_read(go,0,got,(maxT*M+guard)*4),"Q8 readback");
        compare(got,ref,maxT*M+guard);
        if(K==2560 && (T==3 || T==4 || T==8)) {
            double us[3];
            for(unsigned phase=0;phase<3;phase++) {
                ds4_gpu_fronk_native_rows(phase==1);
                for(unsigned i=0;i<4;i++) require(ds4_gpu_qwen4_matmul_q8_0_tensor(go,weights,bytes,0,K,M,gx,T),"Q8 warmup");
                const double start=now();require(ds4_gpu_begin_commands(),"Q8 benchmark begin");
                for(unsigned i=0;i<64;i++) require(ds4_gpu_qwen4_matmul_q8_0_tensor(go,weights,bytes,0,K,M,gx,T),"Q8 benchmark");
                require(ds4_gpu_end_commands(),"Q8 benchmark end");us[phase]=(now()-start)*1e6/64;
            }
            printf("Q8 K=%u M=%u T=%u baseline_before_us=%.2f native_us=%.2f baseline_after_us=%.2f\n",K,M,T,us[0],us[1],us[2]);
            if(T==8) {
                ds4_gpu_fronk_native_rows(false);
                for(unsigned i=0;i<4;i++) require(ds4_gpu_qwen4_batch_mm_q8_tensor(go,gx,weights,bytes,0,T,K,M),"Q8 SIMD matrix warmup");
                require(ds4_gpu_tensor_read(go,0,got,(maxT*M+guard)*4),"matrix readback");
                double diff=0,norm=0;
                for(unsigned i=0;i<T*M;i++) {double e=(double)got[i]-ref[i];diff+=e*e;norm+=(double)ref[i]*ref[i];}
                double relative=sqrt(diff/fmax(norm,1e-30));
                require(isfinite(relative)&&relative<1e-5,"matrix numerical error");
                compare(got+T*M,ref+T*M,guard);
                /* A fixed-eight arithmetic policy must be invariant to a
                 * token's column and to the other seven scratch rows. */
                ds4_gpu_tensor *single=ds4_gpu_tensor_alloc(8ull*K*4);
                require(single&&ds4_gpu_tensor_fill_f32(single,0,8ull*K),"matrix single-row workspace");
                for(unsigned row=0;row<8;row++) {
                    require(ds4_gpu_begin_commands()&&
                        ds4_gpu_tensor_copy(single,0,gx,(uint64_t)row*K*4,K*4)&&
                        ds4_gpu_qwen4_batch_mm_q8_tensor(gr,single,weights,bytes,0,8,K,M)&&
                        ds4_gpu_end_commands()&&
                        ds4_gpu_tensor_read(gr,0,ref,M*4),"matrix independent row");
                    compare(got+(size_t)row*M,ref,M);
                }
                ds4_gpu_tensor_free(single);
                printf("PASS matrix8 column invariance K=%u M=%u, eight independent single-token calls\n",K,M);
                double matrix_us[3];
                /* Bracket this candidate too. The encoder timeline is a
                 * diagnostic run; repeat without it for timing decisions. */
                for(unsigned phase=0;phase<3;phase++) {
                    for(unsigned i=0;i<4;i++) require(phase==1 ?
                        ds4_gpu_qwen4_batch_mm_q8_tensor(go,gx,weights,bytes,0,T,K,M) :
                        ds4_gpu_qwen4_matmul_q8_0_tensor(go,weights,bytes,0,K,M,gx,T),"matrix control warmup");
                    const double start=now();require(ds4_gpu_begin_commands(),"matrix benchmark begin");
                    for(unsigned i=0;i<64;i++) require(phase==1 ?
                        ds4_gpu_qwen4_batch_mm_q8_tensor(go,gx,weights,bytes,0,T,K,M) :
                        ds4_gpu_qwen4_matmul_q8_0_tensor(go,weights,bytes,0,K,M,gx,T),"matrix benchmark");
                    require(ds4_gpu_end_commands(),"matrix benchmark end");
                    matrix_us[phase]=(now()-start)*1e6/64;
                }
                printf("Q8 SIMD matrix K=%u M=%u T=%u baseline_before_us=%.2f matrix_us=%.2f baseline_after_us=%.2f relative_l2=%.9g (different FP32 summation order)\n",K,M,T,matrix_us[0],matrix_us[1],matrix_us[2],relative);
            }
        }
    }
    printf("PASS exact Q8 T=1..8 K=%u M=%u and guards\n",K,M);
    ds4_gpu_fronk_native_rows(false);ds4_gpu_tensor_free(go);ds4_gpu_tensor_free(gr);ds4_gpu_tensor_free(gx);
    free(got);free(ref);free(x);
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
    for(unsigned T=1;T<=maxT;T++) {
        require(ds4_gpu_tensor_fill_f32(gr,127.25f,maxT*E+guard)&&ds4_gpu_tensor_fill_f32(go,127.25f,maxT*E+guard),"guards");
        ds4_gpu_fronk_native_rows(false);
        require(ds4_gpu_begin_commands(),"reference begin");
        for(unsigned t=0;t<T;t++) {
            ds4_gpu_tensor *xr=ds4_gpu_tensor_view(gx,(uint64_t)t*4*E*4,4*E*4);
            ds4_gpu_tensor *lr=ds4_gpu_tensor_view(gl,(uint64_t)t*rank*4,rank*4);
            ds4_gpu_tensor *out=ds4_gpu_tensor_view(gr,(uint64_t)t*E*4,E*4);
            require(xr&&lr&&out&&ds4_gpu_qwen4_hc_gate_mix_tensor(out,xr,lr,weights,bytes,0,1,1,E,4,rank),"reference mixer");
            ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(lr);ds4_gpu_tensor_free(xr);
        }
        require(ds4_gpu_end_commands(),"reference end");
        ds4_gpu_fronk_native_rows(true);
        require(ds4_gpu_qwen4_hc_gate_mix_tensor(go,gx,gl,weights,bytes,0,1,T,E,4,rank),"native mixer");
        require(ds4_gpu_tensor_read(gr,0,ref,(maxT*E+guard)*4)&&ds4_gpu_tensor_read(go,0,got,(maxT*E+guard)*4),"readback");
        compare(got,ref,maxT*E+guard);
        if(E==2560 && T>1) {
            double us[3];
            for(unsigned phase=0;phase<3;phase++) {
                ds4_gpu_fronk_native_rows(phase==1);
                for(unsigned i=0;i<4;i++) require(ds4_gpu_qwen4_hc_gate_mix_tensor(go,gx,gl,weights,bytes,0,1,T,E,4,rank),"benchmark warmup");
                const double start=now();
                require(ds4_gpu_begin_commands(),"benchmark begin");
                for(unsigned i=0;i<128;i++) require(ds4_gpu_qwen4_hc_gate_mix_tensor(go,gx,gl,weights,bytes,0,1,T,E,4,rank),"benchmark dispatch");
                require(ds4_gpu_end_commands(),"benchmark end");
                us[phase]=(now()-start)*1e6/128;
            }
            printf("HC T=%u baseline_before_us=%.2f native_us=%.2f baseline_after_us=%.2f\n",T,us[0],us[1],us[2]);
        }
    }
    printf("PASS exact T=1..8 E=%u rank=%u, output guards and token view offsets\n",E,rank);
    ds4_gpu_fronk_native_rows(false);
    ds4_gpu_tensor_free(go);ds4_gpu_tensor_free(gr);ds4_gpu_tensor_free(gl);ds4_gpu_tensor_free(gx);
    free(got);free(ref);free(lo);free(x);
    /* Metal may retain the wrapped weight mapping until shutdown. */
}
/* The full vocabulary projection was a recurring 4.5 ms encoder in the
 * physical-cluster trace. Qualify the existing matrix kernel at that actual
 * shape before changing the proposer. Only column zero is consumed here. */
static void vocabulary(void) {
    const unsigned K=2560,M=248320;
    const size_t bytes=((size_t)K/32*34*M+16383)&~(size_t)16383;
    unsigned char *weights=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    require(weights!=MAP_FAILED,"vocabulary weight allocation");
    for(size_t i=0;i<(size_t)K/32*M;i++) {
        __fp16 scale=.001f*(1.01f+random_float());memcpy(weights+i*34,&scale,2);
        for(unsigned j=0;j<32;j++) ((signed char *)weights)[i*34+2+j]=(signed char)(random_float()*127);
    }
    require(ds4_gpu_set_model_map(weights,bytes),"vocabulary registration");
    float *x=calloc(8*K,4),*ref=malloc((size_t)M*4),*got=malloc((size_t)M*4);
    require(x&&ref&&got,"vocabulary host buffers");
    for(unsigned j=0;j<K;j++) x[j]=random_float();
    ds4_gpu_tensor *gx=ds4_gpu_tensor_alloc(8ull*K*4),*out=ds4_gpu_tensor_alloc(8ull*M*4);
    require(gx&&out&&ds4_gpu_tensor_write(gx,0,x,8ull*K*4),"vocabulary device buffers");
    require(ds4_gpu_qwen4_matmul_q8_0_tensor(out,weights,bytes,0,K,M,gx,1)&&
        ds4_gpu_tensor_read(out,0,ref,(size_t)M*4),"vocabulary reference");
    require(ds4_gpu_qwen4_batch_mm_q8_tensor(out,gx,weights,bytes,0,8,K,M)&&
        ds4_gpu_tensor_read(out,0,got,(size_t)M*4),"vocabulary matrix");
    double diff=0,norm=0;
    for(unsigned i=0;i<M;i++) {double d=(double)got[i]-ref[i];diff+=d*d;norm+=(double)ref[i]*ref[i];}
    double relative=sqrt(diff/fmax(norm,1e-30));
    require(isfinite(relative)&&relative<1e-5,"vocabulary matrix numerical error");
    for(unsigned repeat=0;repeat<5;repeat++) {
        double us[3];
        for(unsigned phase=0;phase<3;phase++) {
            for(unsigned i=0;i<4;i++) require(phase==1 ?
                ds4_gpu_qwen4_batch_mm_q8_tensor(out,gx,weights,bytes,0,8,K,M) :
                ds4_gpu_qwen4_matmul_q8_0_tensor(out,weights,bytes,0,K,M,gx,1),"vocabulary warmup");
            const double start=now();
            /* A proposal consumes its argmax before starting the next one:
             * measure complete submissions, not only a long queued batch. */
            for(unsigned i=0;i<24;i++) require(ds4_gpu_begin_commands() && (phase==1 ?
                ds4_gpu_qwen4_batch_mm_q8_tensor(out,gx,weights,bytes,0,8,K,M) :
                ds4_gpu_qwen4_matmul_q8_0_tensor(out,weights,bytes,0,K,M,gx,1)) &&
                ds4_gpu_end_commands(),"vocabulary benchmark");
            us[phase]=(now()-start)*1e6/24;
        }
        printf("VOCAB K=%u M=%u repeat=%u before_us=%.2f matrix8_us=%.2f after_us=%.2f relative_l2=%.9g\n",
            K,M,repeat,us[0],us[1],us[2],relative);
    }
    ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(gx);free(got);free(ref);free(x);
}
int main(int argc,char **argv) {
    setbuf(stdout,NULL);
    if(argc==2 && !strcmp(argv[1],"--timeline")) {
        char cwd[2048],path[2300];require(getcwd(cwd,sizeof(cwd))!=NULL,"working directory");
        snprintf(path,sizeof(path),"%s/build/revision-qa/mtp-dense-timeline.tsv",cwd);
        setenv("DS4_METAL_ENCODER_TIMELINE",path,1);
    }
    require(chdir("build/mtp-native-engine")==0,"staged Metal source directory");
    require(ds4_gpu_init(),"Metal initialization");
    if(argc==2 && !strcmp(argv[1],"--vocab")) {vocabulary();return 0;}
    history_prefix();
    shape(97,8);shape(96,72);shape(2560,320);
    setenv("FRONK_NATIVE_Q8_REUSE","1",1);
    q8_shape(96,65);q8_shape(2560,4096);q8_shape(2560,10240);
    return 0;
}
