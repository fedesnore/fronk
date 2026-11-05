#include "native.h"
#include "q4.h"
#include <math.h>
#include <string.h>
#include <sys/mman.h>

int fr_native_dot(const unsigned char *p,unsigned type,const float *x,unsigned n,double *out) {
    if (!p || !x || !out || !n || (type!=FR_Q4_K && type!=FR_MXFP4 && type!=FR_Q8_0) ||
        n%(type==FR_Q4_K?256:32)) return FR_INVALID;
    static const float mx[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
    double sum=0;
    for (unsigned j=0;j<n;++j) {
        float w;
        if (type==FR_Q4_K) {
            const unsigned char *b=p+(size_t)(j/256)*144,*s=b+4;
            unsigned g=j%256/32,sc,mi;
            if (g<4) { sc=s[g]&63; mi=s[g+4]&63; }
            else { sc=(s[g+4]&15)|((s[g-4]>>6)<<4); mi=(s[g+4]>>4)|((s[g]>>6)<<4); }
            float d=fr_f16((uint16_t)(b[0]|(uint16_t)b[1]<<8));
            float m=fr_f16((uint16_t)(b[2]|(uint16_t)b[3]<<8));
            unsigned q=(b[16+(g/2)*32+j%32]>>((g%2)*4))&15;
            w=(d*(float)sc)*(float)q-m*(float)mi;
        } else if (type==FR_MXFP4) {
            const unsigned char *b=p+(size_t)(j/32)*17;
            /* E8M0 zero is 2^-127; 255 is a reserved nonfinite exponent. */
            if (b[0]==255) return FR_NUMERIC;
            w=ldexpf(1.0f,(int)b[0]-127)*mx[(b[1+j%16]>>((j%32/16)*4))&15];
        } else {
            const unsigned char *b=p+(size_t)(j/32)*34;
            w=fr_f16((uint16_t)(b[0]|(uint16_t)b[1]<<8))*(float)(int8_t)b[2+j%32];
        }
        if (!isfinite(w) || !isfinite(x[j])) return FR_NUMERIC;
        sum+=(double)w*x[j];
    }
    if (!isfinite(sum)) return FR_NUMERIC;
    *out=sum; return FR_OK;
}

static uint32_t random_word(uint32_t *s) { *s=*s*1664525u+1013904223u; return *s; }
static void fixture_row(unsigned char *p,unsigned type,unsigned n,uint32_t *seed) {
    unsigned count=n/(type==FR_Q4_K?256:32),bytes=type==FR_Q4_K?144:type==FR_Q8_0?34:17;
    for (unsigned i=0;i<count;++i,p+=bytes) {
        for (unsigned j=0;j<bytes;++j) p[j]=(unsigned char)(random_word(seed)>>24);
        if (type==FR_Q4_K) { p[0]=0; p[1]=0x08; p[2]=0; p[3]=0x14; }
        else if (type==FR_Q8_0) { p[0]=0; p[1]=0x1c; }
        else p[0]=(unsigned char)(120+random_word(seed)%5);
    }
}
static void fill_slice(unsigned char *p,unsigned id,uint32_t *seed) {
    unsigned type=id==FR_SHARED_EXPERT?FR_Q8_0:FR_Q4_K,down=id==FR_SHARED_EXPERT?FR_Q8_0:FR_MXFP4;
    size_t projection=id==FR_SHARED_EXPERT?FR_NATIVE_SHARED_PROJECTION:FR_NATIVE_GATE_BYTES;
    for (unsigned n=0;n<32;++n) {
        fixture_row(p+n*fr_native_row(type,FR_HIDDEN),type,FR_HIDDEN,seed);
        fixture_row(p+projection+n*fr_native_row(type,FR_HIDDEN),type,FR_HIDDEN,seed);
    }
    for (unsigned n=0;n<FR_HIDDEN;++n) fixture_row(p+2*projection+n*fr_native_row(down,32),down,32,seed);
}
void fr_native_fill(unsigned char *data,uint32_t seed) {
    for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_EXPERTS;++e)
        fill_slice(data+fr_native_offset(l,e),e,&seed);
}
void fr_native_fixture(unsigned char *data,unsigned layer,uint32_t seed) {
    for (unsigned e=0;e<80;++e) fill_slice(data+fr_native_offset(layer,e),e,&seed);
    fill_slice(data+fr_native_offset(layer,FR_SHARED_EXPERT),FR_SHARED_EXPERT,&seed);
}
int fr_native_self_test(const fr_backend *backend) {
    if (!backend || backend==&fr_ref) return FR_INVALID;
    unsigned char *data=mmap(NULL,FR_NATIVE_BLOCK_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (data==MAP_FAILED) return FR_IO;
    fr_req req={.layer=47,.burst=8}; fr_part expected,actual;
    uint32_t seed=7;
    /* Routed ids 0..79, including distinct and reused bursts, and shared 512. */
    for (unsigned e=0;e<=80;++e) {
        unsigned id=e==80?FR_SHARED_EXPERT:e;
        unsigned char *p=data+fr_native_offset(req.layer,id);
        fill_slice(p,id,&seed);
    }
    for (unsigned t=0;t<FR_MAX_BURST;++t) {
        for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) req.gate[t][j]=j==FR_ACTIVE_ROUTED?.5f:.0625f;
        for (unsigned j=0;j<FR_HIDDEN;++j) req.x[t][j]=(float)((int)(random_word(&seed)%257)-128)/128;
    }
    int rc=fr_ref.load(19,data,FR_NATIVE_BLOCK_BYTES),ref_loaded=!rc,candidate_loaded=0;
    if (!rc) { rc=backend->load(19,data,FR_NATIVE_BLOCK_BYTES); candidate_loaded=!rc; }
    if (!rc && mprotect(data,FR_NATIVE_BLOCK_BYTES,PROT_READ)) rc=FR_IO;
    for (unsigned mode=0;!rc && mode<2;++mode) for (unsigned burst=1;!rc && burst<=8;++burst) {
        req.burst=burst;
        for (unsigned t=0;t<8;++t) for (unsigned e=0;e<10;++e) req.expert[t][e]=(uint16_t)(e+(mode?0:t*10));
        rc=fr_ref.run(&req,&expected); if (!rc) rc=backend->run(&req,&actual);
        for (unsigned t=0;!rc && t<burst;++t) {
            fr_error error; rc=fr_compare(expected.y[t],actual.y[t],FR_HIDDEN,&error);
            if (!rc && !fr_conforms(&error,&fr_backend_tolerance)) rc=FR_MISMATCH;
        }
    }
#ifdef FR_EXPERT_RANKED
    /* Native weights, different activations/gates, permutations and partial
     * overlap: every batched row must equal its independent T=1 execution. */
    for (unsigned mode=0;!rc && mode<3;++mode) for (unsigned burst=2;!rc && burst<=8;++burst) {
        req.burst=burst;
        for (unsigned t=0;t<burst;++t) {
            req.gate[t][FR_ACTIVE_ROUTED]=0;
            for (unsigned j=0;j<FR_ACTIVE_ROUTED;++j) {
                req.expert[t][j]=(uint16_t)(mode==0?j:mode==1?(j+t)%10:j+2*(t%4));
                req.gate[t][j]=(float)(j+t+1)/128;
            }
        }
        rc=backend->run(&req,&actual);
        for (unsigned t=0;!rc && t<burst;++t) {
            fr_req row=req; row.burst=1;
            memcpy(row.x[0],req.x[t],sizeof(row.x[0]));
            memcpy(row.expert[0],req.expert[t],sizeof(row.expert[0]));
            memcpy(row.gate[0],req.gate[t],sizeof(row.gate[0]));
            rc=backend->run(&row,&expected);
            if (!rc && memcmp(actual.y[t],expected.y[0],sizeof(actual.y[t]))) rc=FR_MISMATCH;
        }
    }
    req.gate[0][FR_ACTIVE_ROUTED]=.5f;
#endif
    /* Invalid native scales must preserve the complete caller output. */
    if (!rc && mprotect(data,FR_NATIVE_BLOCK_BYTES,PROT_READ|PROT_WRITE)) rc=FR_IO;
    if (!rc) {
        size_t bad[]={fr_native_offset(47,0)+1,
            fr_native_offset(47,0)+2*FR_NATIVE_GATE_BYTES,
            fr_native_offset(47,FR_SHARED_EXPERT)+1};
        unsigned char value[]={0x7c,255,0x7c};
        for (unsigned i=0;!rc && i<3;++i) {
            unsigned char saved=data[bad[i]]; data[bad[i]]=value[i]; expected=actual;
            if (backend->run(&req,&actual)!=FR_NUMERIC || memcmp(&actual,&expected,sizeof(actual))) rc=FR_MISMATCH;
            data[bad[i]]=saved;
        }
    }
    if (candidate_loaded && backend->unload(19) && !rc) rc=FR_STATE;
    if (ref_loaded && fr_ref.unload(19) && !rc) rc=FR_STATE;
    if (munmap(data,FR_NATIVE_BLOCK_BYTES) && !rc) rc=FR_IO;
    return rc;
}
