/* GGUF -> native units. This is a byte gather, never a numerical conversion.
 * Gate/up contribute contiguous neuron rows; down contributes quantized
 * 32-neuron column blocks from every output row. Validate the complete tensor
 * extents before any gather, and publish files only after their digest matches.
 *
 * Read fr_gguf_open -> fr_slice_copy -> fr_unit_stream -> fr_unit_write.
 * For example, unit 2 owns neurons [64,96) in every expert of every layer;
 * it does not own the third expert. native.h defines the resulting byte layout.
 * A stream sink can hash, write or send those same bytes without another format.
 * File stamps cache a completed verification; they are not a replacement for
 * checking a changed file. docs/MODEL.md explains the user's model workflow;
 * tools/fronk-model-check.c supplies an independent coordinate-by-coordinate audit.
 */
#include "prep.h"
#include "sha256.h"
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
static int identity(const char *directory,unsigned block,char *text,size_t size) {
    char path[1024]; struct stat st;
    snprintf(path,sizeof(path),"%s/unit-%02u.bin",directory,block);
    if (stat(path,&st) || (uint64_t)st.st_size!=FR_NATIVE_BLOCK_BYTES) return -1;
#ifdef __APPLE__
    long long ns=(long long)st.st_mtimespec.tv_sec*1000000000LL+st.st_mtimespec.tv_nsec;
#else
    long long ns=(long long)st.st_mtim.tv_sec*1000000000LL+st.st_mtim.tv_nsec;
#endif
    snprintf(text,size,"%lld %llu",ns,(unsigned long long)st.st_ino); return 0;
}
int fr_units_stamp(const char *directory,unsigned block,const char *hex) {
    char path[1024],tmp[1040],id[64]; FILE *f;
    if (identity(directory,block,id,sizeof(id))) return -1;
    snprintf(path,sizeof(path),"%s/unit-%02u.sha256",directory,block);
    if (!(f=fopen(path,"w"))) return -1;
    int failed=fprintf(f,"%.64s\n",hex)<0;
    if (fclose(f)) failed=1;
    if (failed) return -1;
    snprintf(path,sizeof(path),"%s/unit-%02u.ok",directory,block); snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    if (!(f=fopen(tmp,"w"))) return -1;
    failed=fprintf(f,"%.64s %s\n",hex,id)<0;
    if (fclose(f)) failed=1;
    if (failed) { unlink(tmp); return -1; }
    return rename(tmp,path);
}
int fr_units_cached(const char *directory,unsigned block,char hex[65]) {
    char path[1024],line[160],id[64],want[160]; FILE *f;
    if (identity(directory,block,id,sizeof(id))) return -1;
    snprintf(path,sizeof(path),"%s/unit-%02u.ok",directory,block);
    if (!(f=fopen(path,"r"))) return -1;
    char *got=fgets(line,sizeof(line),f); fclose(f);
    if (!got || strlen(line)<66 || line[64]!=' ') return -1;
    snprintf(want,sizeof(want),"%.64s %s\n",line,id);
    if (strcmp(want,line)) return -1;
    memcpy(hex,line,64); hex[64]=0; return 0;
}
typedef struct { const unsigned char *p,*end; } cursor;
static int take(cursor *c,void *out,size_t n) { if ((size_t)(c->end-c->p)<n) return -1; memcpy(out,c->p,n); c->p+=n; return 0; }
static int skip_string(cursor *c,const char **s,uint64_t *len) {
    if (take(c,len,8) || (uint64_t)(c->end-c->p)<*len) return -1;
    *s=(const char *)c->p; c->p+=*len; return 0;
}
static int skip_value(cursor *c,uint32_t type,int depth) {
    static const unsigned char size[13]={1,1,2,2,4,4,4,1,0,0,8,8,8};
    const char *s; uint64_t n;
    if (type==8) return skip_string(c,&s,&n);
    if (type==9) {
        uint32_t inner; if (depth || take(c,&inner,4) || take(c,&n,8) || inner>12) return -1;
        if (inner==8 || inner==9) { for (uint64_t i=0;i<n;++i) if (skip_value(c,inner,1)) return -1; return 0; }
        if ((uint64_t)(c->end-c->p)/size[inner]<n) return -1;
        c->p+=n*size[inner]; return 0;
    }
    if (type>12 || !size[type] || (size_t)(c->end-c->p)<size[type]) return -1;
    c->p+=size[type]; return 0;
}
int fr_gguf_open(const char *path,fr_gguf *g) {
    if (!path || !g) return -1;
    memset(g,0,sizeof(*g)); g->fd=-1;
    int fd=open(path,O_RDONLY); struct stat st;
    if (fd<0 || fstat(fd,&st) || st.st_size<64) { if (fd>=0) close(fd); return -1; }
    void *map=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_SHARED,fd,0);
    if (map==MAP_FAILED) { close(fd); return -1; }
    g->fd=fd; g->map=map; g->size=(uint64_t)st.st_size;
#ifdef __APPLE__
    g->mtime=(uint64_t)st.st_mtimespec.tv_sec*1000000000u+(uint64_t)st.st_mtimespec.tv_nsec;
#else
    g->mtime=(uint64_t)st.st_mtim.tv_sec*1000000000u+(uint64_t)st.st_mtim.tv_nsec;
#endif
    cursor c={g->map,g->map+g->size}; uint32_t magic,version,type; uint64_t tensors,keys,align=32,len; const char *s;
    if (take(&c,&magic,4) || memcmp(&magic,"GGUF",4) || take(&c,&version,4) || version!=3 || take(&c,&tensors,8) || take(&c,&keys,8)) goto bad;
    for (uint64_t i=0;i<keys;++i) {
        if (skip_string(&c,&s,&len) || take(&c,&type,4)) goto bad;
        if (len==17 && !memcmp(s,"general.alignment",17)) {
            uint32_t a;
            if (type!=4 || take(&c,&a,4) || !a || (a&(a-1))) goto bad;
            align=a;
        }
        else if (skip_value(&c,type,0)) goto bad;
    }
    /* name suffix, GGUF type, ne0, ne1, ne2 (0 = 2-D) */
    static const struct { const char *name; uint32_t type; uint64_t ne[3]; } want[6]={
        {"ffn_gate_exps.weight",FR_Q4_K,{FR_HIDDEN,FR_INTERMEDIATE,FR_ROUTED_EXPERTS}},
        {"ffn_up_exps.weight",FR_Q4_K,{FR_HIDDEN,FR_INTERMEDIATE,FR_ROUTED_EXPERTS}},
        {"ffn_down_exps.weight",FR_MXFP4,{FR_INTERMEDIATE,FR_HIDDEN,FR_ROUTED_EXPERTS}},
        {"ffn_gate_shexp.weight",FR_Q8_0,{FR_HIDDEN,FR_INTERMEDIATE,0}},
        {"ffn_up_shexp.weight",FR_Q8_0,{FR_HIDDEN,FR_INTERMEDIATE,0}},
        {"ffn_down_shexp.weight",FR_Q8_0,{FR_INTERMEDIATE,FR_HIDDEN,0}}};
    uint64_t *slot[6]={g->gate,g->up,g->down,g->sgate,g->sup,g->sdown}; unsigned found=0;
    for (uint64_t i=0;i<tensors;++i) {
        uint32_t dims; uint64_t ne[4]={0,0,0,0},offset;
        if (skip_string(&c,&s,&len) || take(&c,&dims,4) || dims>4) goto bad;
        for (uint32_t d=0;d<dims;++d) if (take(&c,&ne[d],8)) goto bad;
        if (take(&c,&type,4) || take(&c,&offset,8)) goto bad;
        unsigned layer; int at=0; char name[96];
        if (len>=sizeof(name)) continue;
        memcpy(name,s,len); name[len]=0;
        if (sscanf(name,"blk.%u.%n",&layer,&at)!=1 || !at || layer>=FR_LAYERS) continue;
        for (unsigned k=0;k<6;++k) if (!strcmp(name+at,want[k].name)) {
            if (dims!=(k<3?3u:2u) || type!=want[k].type || ne[0]!=want[k].ne[0] ||
                ne[1]!=want[k].ne[1] || ne[2]!=want[k].ne[2] || slot[k][layer] ||
                offset==UINT64_MAX || offset%align || memchr(s,0,(size_t)len)) goto bad;
            slot[k][layer]=offset+1; ++found; /* data start added below */
        }
    }
    uint64_t data=((uint64_t)(c.p-g->map)+align-1)/align*align;
    if (found!=6*FR_LAYERS || data>g->size) goto bad;
    const uint64_t bytes[6]={
        (uint64_t)FR_ROUTED_EXPERTS*FR_INTERMEDIATE*(FR_HIDDEN/256*144),
        (uint64_t)FR_ROUTED_EXPERTS*FR_INTERMEDIATE*(FR_HIDDEN/256*144),
        (uint64_t)FR_ROUTED_EXPERTS*FR_HIDDEN*(FR_INTERMEDIATE/32*17),
        (uint64_t)FR_INTERMEDIATE*(FR_HIDDEN/32*34),
        (uint64_t)FR_INTERMEDIATE*(FR_HIDDEN/32*34),
        (uint64_t)FR_HIDDEN*(FR_INTERMEDIATE/32*34)};
    for (unsigned k=0;k<6;++k) for (unsigned l=0;l<FR_LAYERS;++l) {
        uint64_t offset=slot[k][l]-1;
        /* Subtraction checks avoid offset+length overflow on malformed files.
         * Checking just the last down tensor does not protect gate/up/shared. */
        if (offset>g->size-data || bytes[k]>g->size-data-offset) goto bad;
        slot[k][l]=data+offset;
    }
    for (unsigned a=0;a<6*FR_LAYERS;++a) for (unsigned b=0;b<a;++b) {
        unsigned ak=a/FR_LAYERS,al=a%FR_LAYERS,bk=b/FR_LAYERS,bl=b%FR_LAYERS;
        if (slot[ak][al]<slot[bk][bl]+bytes[bk] && slot[bk][bl]<slot[ak][al]+bytes[ak]) goto bad;
    }
    return 0;
bad:
    fr_gguf_close(g); return -1;
}
void fr_gguf_close(fr_gguf *g) {
    if (g->map) munmap((void *)g->map,g->size);
    if (g->fd>=0) close(g->fd);
    memset(g,0,sizeof(*g)); g->fd=-1;
}
/* Gathers one (layer, expert) slice into out; returns its byte count. */
size_t fr_slice_copy(const fr_gguf *g,unsigned block,unsigned l,unsigned e,void *output,size_t capacity) {
    size_t need=e<FR_ROUTED_EXPERTS?FR_NATIVE_SLICE_BYTES:FR_NATIVE_SHARED_BYTES;
    if (!g || !g->map || block>=FR_BLOCKS || l>=FR_LAYERS || e>=FR_EXPERTS || !output || capacity<need) return 0;
    unsigned char *out=output;
    const unsigned char *m=g->map; size_t n=0,b=block;
    if (e<FR_ROUTED_EXPERTS) {
        size_t row=FR_HIDDEN/256*144,expert=(size_t)FR_INTERMEDIATE*row;
        memcpy(out,m+g->gate[l]+e*expert+b*32*row,FR_NATIVE_GATE_BYTES); n+=FR_NATIVE_GATE_BYTES;
        memcpy(out+n,m+g->up[l]+e*expert+b*32*row,FR_NATIVE_GATE_BYTES); n+=FR_NATIVE_GATE_BYTES;
        const unsigned char *d=m+g->down[l]+(size_t)e*FR_HIDDEN*(FR_INTERMEDIATE/32*17)+b*17;
        for (unsigned r=0;r<FR_HIDDEN;++r,n+=17) memcpy(out+n,d+(size_t)r*(FR_INTERMEDIATE/32*17),17);
    } else {
        size_t row=FR_HIDDEN/32*34;
        memcpy(out,m+g->sgate[l]+b*32*row,FR_NATIVE_SHARED_PROJECTION); n+=FR_NATIVE_SHARED_PROJECTION;
        memcpy(out+n,m+g->sup[l]+b*32*row,FR_NATIVE_SHARED_PROJECTION); n+=FR_NATIVE_SHARED_PROJECTION;
        const unsigned char *d=m+g->sdown[l]+b*34;
        for (unsigned r=0;r<FR_HIDDEN;++r,n+=34) memcpy(out+n,d+(size_t)r*(FR_INTERMEDIATE/32*34),34);
    }
    return n;
}
int fr_unit_stream(const fr_gguf *g,unsigned block,fr_sink sink,void *ctx) {
    if (!g || !g->map || block>=FR_BLOCKS || !sink) return -1;
    size_t cap=(size_t)64*FR_NATIVE_SLICE_BYTES; unsigned char *buf=malloc(cap); int rc=buf?0:-1;
    for (unsigned l=0;!rc && l<FR_LAYERS;++l) {
        size_t used=0;
        for (unsigned e=0;!rc && e<FR_EXPERTS;++e) {
            size_t copied=fr_slice_copy(g,block,l,e,buf+used,cap-used);
            if (!copied) { rc=-1; break; }
            used+=copied;
            if (used+FR_NATIVE_SLICE_BYTES>cap || e==FR_EXPERTS-1) { rc=sink(ctx,buf,used); used=0; }
        }
    }
    free(buf); return rc;
}
static int hash_sink(void *ctx,const void *data,size_t n) { return sha256_update(ctx,data,n); }
static void hex32(const unsigned char *d,char *out) { for (unsigned i=0;i<32;++i) snprintf(out+2*i,3,"%02x",d[i]); }
typedef struct { const fr_gguf *g; unsigned next; pthread_mutex_t lock; char (*hex)[65]; atomic_int rc; } job;
static void *hash_worker(void *arg) {
    job *j=arg;
    for (;;) {
        pthread_mutex_lock(&j->lock); unsigned b=j->next++; pthread_mutex_unlock(&j->lock);
        if (b>=FR_BLOCKS) return NULL;
        sha256_ctx s; unsigned char d[32]; sha256_init(&s);
        if (fr_unit_stream(j->g,b,hash_sink,&s) || sha256_final(&s,d)) { atomic_store(&j->rc,-1); continue; }
        hex32(d,j->hex[b]);
    }
}
int fr_manifest(const fr_gguf *g,const char *dir,char hex[FR_BLOCKS][65]) {
    char path[512]; FILE *f; unsigned long long size,mtime; unsigned ok=0;
    if (snprintf(path,sizeof(path),"%s/manifest.txt",dir)>=(int)sizeof(path)) return -1;
    if ((f=fopen(path,"r"))) {
        if (fscanf(f,"gguf %llu %llu\n",&size,&mtime)==2 && size==g->size && mtime==g->mtime)
            for (unsigned b=0;b<FR_BLOCKS;++b) { unsigned n; if (fscanf(f,"unit-%u %64s\n",&n,hex[b])==2 && n==b && strlen(hex[b])==64) ++ok; }
        fclose(f);
        if (ok==FR_BLOCKS) return 0;
    }
    job j={g,0,PTHREAD_MUTEX_INITIALIZER,hex,ATOMIC_VAR_INIT(0)}; pthread_t t[8]; unsigned started=0;
    for (;started<8;++started) if (pthread_create(&t[started],NULL,hash_worker,&j)) break;
    /* Resource pressure may prevent a thread from starting. The caller can
     * complete the same queue, and only successfully created threads are joined. */
    if (started<8) hash_worker(&j);
    for (unsigned i=0;i<started;++i) pthread_join(t[i],NULL);
    pthread_mutex_destroy(&j.lock);
    if (atomic_load(&j.rc)) return -1;
    char tmp[520]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    if (!(f=fopen(tmp,"w"))) return -1;
    fprintf(f,"gguf %llu %llu\n",(unsigned long long)g->size,(unsigned long long)g->mtime);
    for (unsigned b=0;b<FR_BLOCKS;++b) fprintf(f,"unit-%02u %s\n",b,hex[b]);
    if (fclose(f) || rename(tmp,path)) return -1;
    return 0;
}
typedef struct { int fd; sha256_ctx sha; } file_sink;
static int write_sink(void *ctx,const void *data,size_t n) {
    file_sink *s=ctx; const unsigned char *p=data;
    if (sha256_update(&s->sha,data,n)) return -1;
    while (n) { ssize_t w=write(s->fd,p,n); if (w<=0) return -1; p+=w; n-=(size_t)w; }
    return 0;
}
/* Expert spill is served from state-dir, which may live on a different drive
 * from the source GGUF. Probe that filesystem before the user needs weights. */
double fr_cache_read_rate(const char *directory) {
    char path[512]; struct timespec a,b;
    if (snprintf(path,sizeof(path),"%s/.disk-probe-XXXXXX",directory)>=(int)sizeof(path)) return 0;
    int fd=mkstemp(path);
    if (fd<0) return 0;
    unlink(path);
    const size_t size=FR_NATIVE_SLICE_BYTES;
    uint32_t *buf=malloc(size); uint32_t seed=314159;
    double rate=0;
    if (!buf) goto done;
#ifdef F_NOCACHE
    if (fcntl(fd,F_NOCACHE,1)<0) goto done;
#endif
    for (unsigned chunk=0;chunk<128;++chunk) {
        for (size_t i=0;i<size/sizeof(*buf);++i) {
            seed^=seed<<13; seed^=seed>>17; seed^=seed<<5; buf[i]=seed;
        }
        if (write(fd,buf,size)!=(ssize_t)size) goto done;
    }
    if (fsync(fd)<0) goto done;
#ifdef __linux__
    if (posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED)) goto done;
#endif
    clock_gettime(CLOCK_MONOTONIC,&a);
    // Visit distinct locations so the same data cannot warm subsequent reads.
    for (unsigned i=0;i<64;++i)
        if (pread(fd,buf,size,(off_t)((i*73)%128)*size)!=(ssize_t)size) goto done;
    clock_gettime(CLOCK_MONOTONIC,&b);
    double seconds=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)*1e-9;
    if (seconds>0) rate=64.0*size/seconds;
done:
    free(buf); close(fd); return rate;
}
int fr_unit_write(const fr_gguf *g,unsigned block,const char *dir,const char *hex) {
    char path[512],tmp[520],have[65],got[65]; unsigned char d[32];
    if (!fr_units_cached(dir,block,have) && !memcmp(have,hex,64)) return 0;
    snprintf(path,sizeof(path),"%s/unit-%02u.bin",dir,block); snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    file_sink s; s.fd=open(tmp,O_WRONLY|O_CREAT|O_TRUNC,0600); if (s.fd<0) return -1;
    sha256_init(&s.sha);
    int rc=fr_unit_stream(g,block,write_sink,&s)||sha256_final(&s.sha,d)?-1:0;
    if (close(s.fd)) rc=-1;
    if (!rc) { hex32(d,got); rc=memcmp(got,hex,64)?-1:0; }
    if (!rc) rc=rename(tmp,path)||fr_units_stamp(dir,block,hex)?-1:0;
    if (rc) unlink(tmp);
    return rc;
}
