#include "test.h"
#include "../src/plane.h"
#include "../src/gsq.h"
#include "../src/sha256.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
static fr_req req;
static _Alignas(float) unsigned char xq[FR_MAX_BURST][FR_Q8_TOKEN_BYTES];
static fr_assembly assembly;
static fr_plane_stats stats;
static unsigned char frame[FR_FRAME_MAX];
static unsigned starts[4]={0,3,6,8}, counts[4]={3,3,2,2};
int main(void) {
    test_begin("q8-wire");fr_gsq_init();fr_plane_stats_init(&stats);
    sha256_ctx digest;sha256_init(&digest);uint32_t seed=12345;
    int exact=1,atomic_reject=1,numeric=1,complete=1;unsigned frames=0;
    size_t cap=0;for(unsigned l=0;l<48;++l) if(fr_gsq_slice_bytes(l)>cap) cap=fr_gsq_slice_bytes(l);
    unsigned char *slice=malloc(cap);if(!slice)return 1;
    for(unsigned sample=0;sample<64;++sample) {
        req.layer=sample%48;req.burst=sample%8+1;
        memset(xq,0,sizeof(xq));
        for(unsigned t=0;t<req.burst;++t) {
            for(unsigned e=0;e<10;++e) {req.expert[t][e]=(uint16_t)(t*11+e);req.gate[t][e]=.125f;}
            req.gate[t][10]=0;
            for(unsigned d=0;d<FR_HIDDEN;++d) {
                seed=seed*1664525u+1013904223u;
                float x=sample==0?0:((int)(seed%8192)-4096)/2048.0f;
                if(sample==63) {uint16_t bits=(uint16_t)(seed&0xffff);if((bits&0x7c00)==0x7c00)bits=0x7bff;x=fr_unhalf(bits);}
                uint16_t half;if(fr_half(x,&half))return 2;
                req.x[t][d]=fr_unhalf(half);
            }
            fr_gsq_quantize_x(req.x[t],xq[t]);
        }
        memset(&assembly,0,sizeof(assembly));size_t total=0;
        /* Reverse ranges and tokens exercise assembly independent of arrival order. */
        for(int f=(int)(4*req.burst)-1;f>=0;--f) {
            size_t n=fr_req_q8_encode(frame,sizeof(frame),&req,xq,123,sample+1,(unsigned)f);
            fr_fragment decoded={0};
            if(!n || n>1472 || fr_frame_next(frame,n)!=n || fr_fragment_read(frame,n,&decoded))return 3;
            sha256_update(&digest,frame,n);total+=n;++frames;
            unsigned r=(unsigned)f%4,t=(unsigned)f/4;
            if(memcmp(decoded.q8,xq[t]+starts[r]*292,counts[r]*292))exact=0;
            int rc=fr_assemble(&assembly,&decoded,&stats);
            if(rc!=(f==0))complete=0;
            if(fr_assemble(&assembly,&decoded,&stats)!=0)complete=0;
            fr_fragment prior=decoded;
            frame[30]=1;
            if(!fr_fragment_read(frame,n,&decoded) || memcmp(&prior,&decoded,sizeof(prior)))atomic_reject=0;
            frame[30]=0;frame[32]=0x7f;frame[33]=0x80;frame[34]=frame[35]=0;
            if(!fr_fragment_read(frame,n,&decoded))atomic_reject=0;
            n=fr_req_q8_encode(frame,sizeof(frame),&req,xq,123,sample+1,(unsigned)f);frame[292]^=1;
            if(!fr_fragment_read(frame,n,&decoded))atomic_reject=0;
            n=fr_req_q8_encode(frame,sizeof(frame),&req,xq,123,sample+1,(unsigned)f);frame[36]^=1;
            if(!fr_fragment_read(frame,n,&decoded))atomic_reject=0;
            if(!fr_fragment_read(frame,n-1,&decoded))atomic_reject=0;
        }
        if(total!=3112*req.burst || memcmp(xq,assembly.xq,req.burst*FR_Q8_TOKEN_BYTES))exact=0;
        if(memcmp(req.expert,assembly.request.expert,req.burst*sizeof(req.expert[0])) ||
           memcmp(req.gate,assembly.request.gate,req.burst*sizeof(req.gate[0])))exact=0;
        if(sample<48) {
            float before[FR_HIDDEN]={0},after[FR_HIDDEN]={0};
            fr_gsq_fill_slice(req.layer,slice,1234+sample);
            if(fr_gsq_fast_slice(req.layer,slice,xq[0],.125f,before) ||
               fr_gsq_fast_slice(req.layer,slice,assembly.xq[0],.125f,after) ||
               memcmp(before,after,sizeof(before)))numeric=0;
            for(unsigned d=0;d<FR_HIDDEN;++d) if(!isfinite(before[d]))numeric=0;
        }
        fr_fragment mixed={.type=1,.burst=req.burst,.step=sample+1,.layer=req.layer};
        if(fr_assemble(&assembly,&mixed,&stats)!=-1)complete=0;
    }
    free(slice);
    check(exact,"64 fixtures, widths 1..8: Q8 bytes/routing exactly preserved; 3112 bytes/token");
    check(numeric,"all 48 layer types: legacy quantized vs transported activation gives bit-identical finite kernel outputs");
    check(complete,"reverse-order/duplicate ranges complete once; mixed codecs rejected");
    check(atomic_reject,"reserved fields, nonfinite scales, inconsistent sums, corrupt values and truncation rejected atomically");
    memset(frame,0xa5,sizeof(frame));
    check(!fr_req_q8_encode(frame,1,&req,xq,1,1,0) && frame[0]==0xa5,"short output capacity rejected before write");
    check(!fr_req_q8_encode(frame,sizeof(frame),&req,xq,0,1,0),"zero session rejected");
    size_t n=fr_req_q8_encode(frame,sizeof(frame),&req,xq,1,1,0);
    fr_fragment f={0};
    frame[8]=frame[9]=frame[10]=frame[11]=0;
    check(fr_fragment_read(frame,n,&f)!=0,"zero wire session rejected");
    n=fr_req_q8_encode(frame,sizeof(frame),&req,xq,1,1,0);frame[28]=0;frame[29]=1;
    check(fr_fragment_read(frame,n,&f)!=0,"wrong Q8 range geometry rejected");
    n=fr_req_q8_encode(frame,sizeof(frame),&req,xq,1,1,0);frame[910]=frame[908];frame[911]=frame[909];
    check(fr_fragment_read(frame,n,&f)!=0,"duplicate routing expert rejected");
    n=fr_req_q8_encode(frame,sizeof(frame),&req,xq,1,1,0);frame[928]=0x7f;frame[929]=0x80;frame[930]=frame[931]=0;
    check(fr_fragment_read(frame,n,&f)!=0,"nonfinite routing gate rejected");
    unsigned char hash[32];sha256_final(&digest,hash);printf("Q8_CANONICAL_SHA256 ");for(unsigned i=0;i<32;++i)printf("%02x",hash[i]);printf(" frames=%u\n",frames);
    return test_end();
}
