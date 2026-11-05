/* Synthetic conformance corpus. Sparse virtual mappings retain the real >3GB
 * addressing contract while materializing only the requested layer/experts.
 * This module is diagnostic/setup code, never part of the compute hot path.
 */
#include "vectors.h"
#include "sha256.h"
#include <math.h>
#include <string.h>
#include <sys/mman.h>
#include <ctype.h>

static const struct { const char *name; unsigned burst,layer; uint32_t mask; } cases[]={
    {"zero-input",1,0,1}, {"zero-gates",2,1,1u<<19},
    {"signed",1,0,1}, {"reused",4,17,1u<<7},
    {"mixed",8,47,1u|(1u<<19)}, {"outliers",2,31,1u<<19},
    {"tiny",1,23,1u<<11}, {"all-blocks",1,1,(1u<<FR_BLOCKS)-1}
};
_Static_assert(sizeof(cases)/sizeof(cases[0])==FR_VECTOR_COUNT,"vector count");
static uint32_t random32(uint32_t *s) { *s^=*s<<13; *s^=*s>>17; *s^=*s<<5; return *s; }
static void hex(const unsigned char digest[32], char out[65]) {
    static const char digits[]="0123456789abcdef";
    for (int i=0; i<32; ++i) { out[i*2]=digits[digest[i]>>4]; out[i*2+1]=digits[digest[i]&15]; }
    out[64]=0;
}
static void hash32(sha256_ctx *h, uint32_t v) {
    unsigned char bytes[4]; for (int i=0; i<4; ++i) bytes[i]=(unsigned char)(v>>(i*8));
    (void)sha256_update(h,bytes,sizeof(bytes));
}
static void hashfloat(sha256_ctx *h, float f) { uint32_t bits; memcpy(&bits,&f,4); hash32(h,bits); }
void fr_vector_close(fr_vector *v) {
    if (!v) return;
    for (int b=0; b<FR_BLOCKS; ++b) if (v->data[b]) {
        munmap(v->data[b],FR_WEIGHT_BLOCK_BYTES); v->data[b]=NULL;
    }
}
int fr_vector_open(unsigned index, fr_vector *v) {
    if (!v || index>=FR_VECTOR_COUNT) return FR_INVALID;
    memset(v,0,sizeof(*v)); v->name=cases[index].name; v->mask=cases[index].mask;
    fr_req *r=&v->req; r->layer=cases[index].layer; r->burst=cases[index].burst;
    uint32_t seed=0x6d2b79f5u+index; unsigned char used[FR_EXPERTS]={0}; used[FR_SHARED_EXPERT]=1;
    for (unsigned t=0; t<r->burst; ++t) {
        for (unsigned e=0; e<FR_ACTIVE_ROUTED; ++e) {
            unsigned id=index==4?t*10+e:e==0?0:e==1?511:e*47+13;
            if (index==4 && t==7 && e==9) id=511;
            r->expert[t][e]=(uint16_t)id; used[id]=1;
        }
        for (int e=0; e<FR_ACTIVE_EXPERTS; ++e)
            r->gate[t][e]=index==1?0:e==FR_ACTIVE_ROUTED?0.75f:(float)((e+(int)t)%5)*0.03125f;
        for (int d=0; d<FR_HIDDEN; ++d) {
            float x=(float)((int)(random32(&seed)%8193)-4096)*0x1p-12f;
            if (index==0) x=0;
            if (index==5) x=d%127==0?(d%2?16.0f:-16.0f):x*0.015625f;
            if (index==6) x*=0x1p-12f;
            r->x[t][d]=x;
        }
    }
    sha256_ctx h; sha256_init(&h);
    hash32(&h,index); hash32(&h,r->layer); hash32(&h,r->burst); hash32(&h,v->mask);
    for (unsigned t=0; t<r->burst; ++t) {
        for (int e=0; e<FR_ACTIVE_ROUTED; ++e) hash32(&h,r->expert[t][e]);
        for (int e=0; e<FR_ACTIVE_EXPERTS; ++e) hashfloat(&h,r->gate[t][e]);
        for (int d=0; d<FR_HIDDEN; ++d) hashfloat(&h,r->x[t][d]);
    }
    for (unsigned b=0; b<FR_BLOCKS; ++b) if (v->mask&(1u<<b)) {
        void *p=mmap(NULL,FR_WEIGHT_BLOCK_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (p==MAP_FAILED) { fr_vector_close(v); return FR_IO; } v->data[b]=p;
        for (unsigned e=0; e<FR_EXPERTS; ++e) if (used[e]) {
            seed=0x9e3779b9u^(b*0x85ebca6bu)^(e*0xc2b2ae35u)^(r->layer*0x27d4eb2du);
            unsigned char *slice=v->data[b]+fr_slice_offset(r->layer,e);
            for (size_t k=0; k<FR_SLICE_BYTES/FR_Q4_BYTES; ++k) {
                uint32_t bits=random32(&seed);
                uint16_t scale=(uint16_t)((k>=2*FR_PROJECTION_BYTES/18?0x2400:0x2000)|(bits&1023));
                if (bits&1024) scale|=0x8000;
                if (k%257==0) scale=(uint16_t)(bits&0x83ff); /* subnormal/zero */
                unsigned char *q=slice+k*18; q[0]=(unsigned char)scale; q[1]=(unsigned char)(scale>>8);
                for (int j=0; j<16; ++j) q[2+j]=(unsigned char)(random32(&seed)>>24);
            }
            (void)sha256_update(&h,slice,FR_SLICE_BYTES);
        }
    }
    unsigned char digest[32]; (void)sha256_final(&h,digest); hex(digest,v->digest); return FR_OK;
}
static int path(char out[1024], const char *dir, const char *name, const char *suffix) {
    if (!dir || !*dir) return FR_INVALID;
    int n=snprintf(out,1024,"%s/%s%s",dir,name,suffix);
    return n<0 || n>=1024?FR_INVALID:FR_OK;
}
static int file_hash(const char *filename, char digest[65]) {
    FILE *f=fopen(filename,"rb"); if (!f) return FR_IO;
    sha256_ctx h; sha256_init(&h); unsigned char buf[4096], sum[32]; size_t n;
    while ((n=fread(buf,1,sizeof(buf),f))) (void)sha256_update(&h,buf,n);
    int failed=ferror(f); if (fclose(f)) failed=1;
    if (failed) return FR_IO;
    (void)sha256_final(&h,sum); hex(sum,digest); return FR_OK;
}
static int trailing_space(FILE *f) {
    int c; while ((c=fgetc(f))!=EOF) if (!isspace((unsigned char)c)) return FR_MISMATCH;
    return ferror(f)?FR_IO:FR_OK;
}
static int unload_all(const fr_backend *b, uint32_t mask) {
    int rc=FR_OK;
    for (int i=0; i<FR_BLOCKS; ++i) if ((mask&(1u<<i)) && b->unload(i)) rc=FR_STATE;
    return rc;
}
static int compute(const fr_backend *b, const fr_vector *v, fr_part *out) {
    uint32_t loaded=0; int rc=FR_OK;
    for (int i=0; i<FR_BLOCKS; ++i) if (v->data[i]) {
        rc=b->load(i,v->data[i],FR_WEIGHT_BLOCK_BYTES); if (rc) break;
        loaded|=1u<<i;
    }
    if (!rc) rc=b->run(&v->req,out);
    int cleanup=unload_all(b,loaded); return rc?rc:cleanup;
}
int fr_vectors_generate(const char *dir) {
    char filename[1024]; int rc=path(filename,dir,"SHA256SUMS",""); if (rc) return rc;
    FILE *manifest=fopen(filename,"wx"); if (!manifest) return FR_IO;
    for (unsigned i=0; !rc && i<FR_VECTOR_COUNT; ++i) {
        fr_vector v; rc=fr_vector_open(i,&v); if (rc) break;
        fr_part out; rc=compute(&fr_ref,&v,&out);
        if (!rc) rc=path(filename,dir,v.name,".txt");
        FILE *f=rc?NULL:fopen(filename,"wx"); if (!rc && !f) rc=FR_IO;
        if (f) {
            fprintf(f,"FRV1 %s %u %u %u %s\n",v.name,v.req.layer,v.req.burst,v.mask,v.digest);
            for (unsigned t=0; t<out.burst; ++t) for (int d=0; d<FR_HIDDEN; ++d)
                fprintf(f,"%a%c",(double)out.y[t][d],d%16==15?'\n':' ');
            if (ferror(f)) rc=FR_IO;
            if (fclose(f)) rc=FR_IO;
        }
        char digest[65]; if (!rc) rc=file_hash(filename,digest);
        if (!rc && fprintf(manifest,"%s  %s.txt\n",digest,v.name)<0) rc=FR_IO;
        fr_vector_close(&v);
    }
    if (fclose(manifest)) rc=FR_IO;
    return rc;
}
int fr_vectors_check(const fr_backend *b, const char *dir, const fr_tolerance *limit, FILE *report) {
    if (!b || !b->load || !b->unload || !b->run || !limit) return FR_INVALID;
    char filename[1024]; int rc=path(filename,dir,"SHA256SUMS",""); if (rc) return rc;
    FILE *manifest=fopen(filename,"rb"); if (!manifest) return FR_IO;
    for (unsigned i=0; !rc && i<FR_VECTOR_COUNT; ++i) {
        char expected_hash[65], actual_hash[65], name[64], expected_name[64];
        snprintf(expected_name,sizeof(expected_name),"%s.txt",cases[i].name);
        if (fscanf(manifest,"%64s %63s",expected_hash,name)!=2 || strcmp(name,expected_name)) { rc=FR_MISMATCH; break; }
        rc=path(filename,dir,cases[i].name,".txt"); if (!rc) rc=file_hash(filename,actual_hash);
        if (!rc && strcmp(actual_hash,expected_hash)) rc=FR_MISMATCH;
        if (rc) break;
        fr_vector v; rc=fr_vector_open(i,&v); if (rc) break;
        FILE *f=fopen(filename,"rb"); if (!f) { fr_vector_close(&v); rc=FR_IO; break; }
        unsigned layer,burst,mask; char magic[8], digest[65]; fr_part golden={0}, actual;
        if (fscanf(f,"%7s %63s %u %u %u %64s",magic,name,&layer,&burst,&mask,digest)!=6 ||
            strcmp(magic,"FRV1") || strcmp(name,v.name) || layer!=v.req.layer || burst!=v.req.burst ||
            mask!=v.mask || strcmp(digest,v.digest)) rc=FR_MISMATCH;
        for (unsigned t=0; !rc && t<v.req.burst; ++t) for (int d=0; d<FR_HIDDEN; ++d) {
            if (fscanf(f,"%f",&golden.y[t][d])!=1 || !isfinite(golden.y[t][d])) { rc=FR_MISMATCH; break; }
        }
        if (!rc) rc=trailing_space(f);
        if (fclose(f)) rc=FR_IO;
        /* Sparse correctness fixtures are never timing or bandwidth inputs. */
        if (!rc) rc=compute(b,&v,&actual);
        if (!rc && (actual.layer!=v.req.layer || actual.burst!=v.req.burst)) rc=FR_MISMATCH;
        for (unsigned t=0; !rc && t<v.req.burst; ++t) {
            fr_error error; rc=fr_compare(golden.y[t],actual.y[t],FR_HIDDEN,&error);
            if (!rc && report) fprintf(report,"%s %s token=%u l2=%.12g max=%.12g cosine=%.12g\n",
                                       b->name,v.name,t,error.relative_l2,error.normalized_max,error.cosine);
            if (!rc && !fr_conforms(&error,limit)) rc=FR_MISMATCH;
        }
        if (!rc) for (unsigned t=v.req.burst; t<FR_MAX_BURST; ++t)
            for (int d=0; d<FR_HIDDEN; ++d) if (actual.y[t][d]!=0) rc=FR_MISMATCH;
        fr_vector_close(&v);
    }
    if (!rc) rc=trailing_space(manifest);
    if (fclose(manifest)) rc=FR_IO;
    return rc;
}
