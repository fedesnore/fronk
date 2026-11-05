/* Sparse synthetic GGUF: markers at known tensor positions must land at the
 * native unit offsets; wrong types are rejected. No real model is read. */
#include "test.h"
#include "../src/prep.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static const char *suffix[6]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight",
    "ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"};
static uint64_t bytes(unsigned k) {
    uint64_t e=FR_ROUTED_EXPERTS,i=FR_INTERMEDIATE,h=FR_HIDDEN;
    return k<2?e*i*h/256*144:k==2?e*h*i/32*17:i*h/32*34;
}
static void put(FILE *f,const void *p,size_t n) { fwrite(p,1,n,f); }
static uint64_t alignment_at, tensor_offset_at[FR_LAYERS][6];
static uint64_t build(const char *path,uint32_t bad_type,uint64_t at[FR_LAYERS][6]) {
    FILE *f=fopen(path,"wb"); uint32_t u; uint64_t q;
    put(f,"GGUF",4); u=3; put(f,&u,4); q=6*FR_LAYERS; put(f,&q,8); q=1; put(f,&q,8);
    q=17; put(f,&q,8); put(f,"general.alignment",17); u=4; put(f,&u,4); u=32; alignment_at=(uint64_t)ftell(f); put(f,&u,4);
    uint64_t offset=0;
    for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned k=0;k<6;++k) {
        char name[64]; int n=snprintf(name,sizeof(name),"blk.%u.%s",l,suffix[k]); q=(uint64_t)n; put(f,&q,8); put(f,name,(size_t)n);
        uint32_t dims=k<3?3:2; put(f,&dims,4);
        uint64_t a=k==2||k==5?FR_INTERMEDIATE:FR_HIDDEN,b=k==2||k==5?FR_HIDDEN:FR_INTERMEDIATE,c=FR_ROUTED_EXPERTS;
        put(f,&a,8); put(f,&b,8); if (dims==3) put(f,&c,8);
        uint32_t type=k<2?FR_Q4_K:k==2?FR_MXFP4:FR_Q8_0; if (l==3 && k==1 && bad_type) type=bad_type;
        put(f,&type,4); tensor_offset_at[l][k]=(uint64_t)ftell(f); put(f,&offset,8); at[l][k]=offset; offset=(offset+bytes(k)+31)/32*32;
    }
    uint64_t data=((uint64_t)ftell(f)+31)/32*32;
    fclose(f); if (truncate(path,(off_t)(data+offset))) abort();
    return data;
}
typedef struct { uint64_t pos,want[5]; unsigned char got[5]; } probe;
static int sink(void *ctx,const void *data,size_t n) {
    probe *p=ctx;
    for (unsigned i=0;i<5;++i) if (p->want[i]>=p->pos && p->want[i]<p->pos+n) p->got[i]=((const unsigned char *)data)[p->want[i]-p->pos];
    p->pos+=n; return p->pos>fr_native_offset(2,0)?1:0;
}
/* Check the first/last neuron rows and output columns in every unit, both
 * boundary layers and boundary routed/shared experts. Markers are written
 * using full tensor coordinates, not the extractor's native offset helper. */
static void boundaries(const char *path,uint64_t data,uint64_t at[FR_LAYERS][6]) {
    const unsigned layers[]={0,FR_LAYERS-1},experts[]={0,FR_ROUTED_EXPERTS-1,FR_SHARED_EXPERT};
    unsigned char *out=malloc(FR_NATIVE_SHARED_BYTES);
    int fd=open(path,O_RDWR),written=fd>=0;
    for (unsigned b=0;b<FR_BLOCKS;++b) for (unsigned li=0;li<2;++li) for (unsigned ei=0;ei<3;++ei) {
        unsigned l=layers[li],e=experts[ei],shared=e==FR_SHARED_EXPERT;
        uint64_t row=shared?2720:1440,col=shared?34:17,expert=shared?0:e;
        unsigned k=shared?3:0;
        uint64_t source[6]={
            data+at[l][k]+(expert*640+b*32)*row,
            data+at[l][k]+(expert*640+b*32+32)*row-1,
            data+at[l][k+1]+(expert*640+b*32)*row,
            data+at[l][k+1]+(expert*640+b*32+32)*row-1,
            data+at[l][k+2]+expert*2560*20*col+b*col,
            data+at[l][k+2]+(expert*2560+2559)*20*col+(b+1)*col-1};
        for (unsigned j=0;j<6;++j) {
            unsigned char marker=(unsigned char)(1+b*7+li*11+ei*17+j*23);
            if (pwrite(fd,&marker,1,(off_t)source[j])!=1) written=0;
        }
    }
    if (fd>=0) close(fd);
    check(written,"write boundary markers for every neuron unit");
    fr_gguf g; int opened=!fr_gguf_open(path,&g); check(opened,"open complete boundary fixture");
    if (opened && out) {
        for (unsigned b=0;b<FR_BLOCKS;++b) {
            int correct=1;
            for (unsigned li=0;li<2;++li) for (unsigned ei=0;ei<3;++ei) {
                unsigned shared=experts[ei]==FR_SHARED_EXPERT;
                size_t row=shared?2720:1440,col=shared?34:17,projection=32*row;
                size_t size=2*projection+2560*col;
                size_t target[6]={0,projection-1,projection,2*projection-1,2*projection,size-1};
                if (fr_slice_copy(&g,b,layers[li],experts[ei],out,FR_NATIVE_SHARED_BYTES)!=size) correct=0;
                for (unsigned j=0;j<6;++j)
                    if (out[target[j]]!=(unsigned char)(1+b*7+li*11+ei*17+j*23)) correct=0;
            }
            check(correct,"unit preserves first/last neuron rows and down columns at both boundary layers");
        }
        memset(out,0x5a,FR_NATIVE_SHARED_BYTES);
        check(!fr_slice_copy(&g,FR_BLOCKS,0,0,out,FR_NATIVE_SHARED_BYTES) &&
              !fr_slice_copy(&g,0,FR_LAYERS,0,out,FR_NATIVE_SHARED_BYTES) &&
              !fr_slice_copy(&g,0,0,FR_EXPERTS,out,FR_NATIVE_SHARED_BYTES) &&
              !fr_slice_copy(&g,0,0,0,out,1) && out[0]==0x5a,"invalid slice coordinates/capacity publish nothing");
    }
    if (opened) fr_gguf_close(&g);
    free(out);
}
static void patch_value(const char *path,uint64_t where,const void *value,size_t size) {
    int fd=open(path,O_WRONLY);
    check(fd>=0 && pwrite(fd,value,size,(off_t)where)==(ssize_t)size,"write malformed header fixture");
    if (fd>=0) close(fd);
}
int main(void) {
    test_begin("prep");
    char path[]="/tmp/fronk-prep-XXXXXX"; int fd=mkstemp(path); close(fd);
    uint64_t at[FR_LAYERS][6],data=build(path,0,at);
    unsigned b=3,l=1,e=7; size_t row=FR_HIDDEN/256*144;
    uint64_t src[5]={
        data+at[l][0]+((uint64_t)e*FR_INTERMEDIATE+b*32+5)*row+100,
        data+at[l][1]+((uint64_t)e*FR_INTERMEDIATE+b*32+5)*row+100,
        data+at[l][2]+((uint64_t)e*FR_HIDDEN+200)*(FR_INTERMEDIATE/32*17)+b*17+4,
        data+at[l][3]+(uint64_t)(b*32+2)*(FR_HIDDEN/32*34)+50,
        data+at[l][5]+(uint64_t)200*(FR_INTERMEDIATE/32*34)+b*34+9};
    probe p={0,{fr_native_offset(l,e)+5*row+100,fr_native_offset(l,e)+FR_NATIVE_GATE_BYTES+5*row+100,
        fr_native_offset(l,e)+2*FR_NATIVE_GATE_BYTES+200*17+4,fr_native_offset(l,FR_SHARED_EXPERT)+2*(FR_HIDDEN/32*34)+50,
        fr_native_offset(l,FR_SHARED_EXPERT)+2*FR_NATIVE_SHARED_PROJECTION+200*34+9},{0}};
    fd=open(path,O_WRONLY);
    for (unsigned i=0;i<5;++i) { unsigned char v=(unsigned char)(0xa1+i); check(pwrite(fd,&v,1,(off_t)src[i])==1,"marker written"); }
    close(fd);
    fr_gguf g; check(!fr_gguf_open(path,&g),"synthetic GGUF parses");
    check(fr_unit_stream(&g,b,sink,&p)==1,"stream stops at sink request");
    for (unsigned i=0;i<5;++i) check(p.got[i]==0xa1+i,"marker at native offset (gate, up, down, shared gate, shared down)");
    fr_gguf_close(&g);
    boundaries(path,data,at);
    build(path,FR_Q8_0,at);
    check(fr_gguf_open(path,&g)<0,"wrong tensor type rejected");
    data=build(path,0,at);
    uint32_t alignment=0;
    patch_value(path,alignment_at,&alignment,sizeof(alignment));
    check(fr_gguf_open(path,&g)<0,"zero GGUF alignment rejected before division");
    alignment=3; patch_value(path,alignment_at,&alignment,sizeof(alignment));
    check(fr_gguf_open(path,&g)<0,"non-power-of-two alignment rejected");
    alignment=32; patch_value(path,alignment_at,&alignment,sizeof(alignment));
    uint64_t offset=UINT64_MAX;
    patch_value(path,tensor_offset_at[0][1],&offset,sizeof(offset));
    check(fr_gguf_open(path,&g)<0,"overflowing tensor offset rejected");
    offset=at[FR_LAYERS-1][5]+bytes(5)-32;
    patch_value(path,tensor_offset_at[0][1],&offset,sizeof(offset));
    check(fr_gguf_open(path,&g)<0,"complete up tensor extent checked, not only final down tensor");
    offset=at[0][0]; patch_value(path,tensor_offset_at[0][1],&offset,sizeof(offset));
    check(fr_gguf_open(path,&g)<0,"overlapping selected tensors rejected");
    unlink(path);
    return test_end();
}
