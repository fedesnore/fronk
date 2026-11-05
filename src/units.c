#include "units.h"
#include "prep.h"
#include "sha256.h"
#include "metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
int fr_units_close(fr_units *u) {
    int rc=0;
    for (unsigned b=0;b<FR_BLOCKS;++b) if (u->data[b] || (u->mask>>b&1)) {
        if ((u->mask&(1u<<b)) && u->backend->unload((int)b)) { rc=-1; continue; }
        if (u->data[b] && munmap(u->data[b],FR_NATIVE_BLOCK_BYTES)) rc=-1;
        u->data[b]=NULL; u->mask&=~(1u<<b);
    }
    u->paged&=u->mask; if (!u->paged) u->cache_bytes=0;
    return rc;
}
int fr_units_open(fr_units *u,const fr_backend *backend,const char *directory,uint32_t mask,uint32_t paged) {
    return fr_units_open_cache(u,backend,directory,mask,paged,FR_SSD_DEFAULT_CACHE);
}
int fr_units_open_cache(fr_units *u,const fr_backend *backend,const char *directory,uint32_t mask,uint32_t paged,uint64_t cache_bytes) {
    if (!u || !backend || !directory || !mask || mask>=(1u<<FR_BLOCKS) || (paged&~mask)) return -1;
    memset(u,0,sizeof(*u)); u->backend=backend; u->paged=paged; u->cache_bytes=cache_bytes;
    int fd=-1;
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask&(1u<<b)) {
        char path[1024],wanted[66],actual[65]; unsigned char digest[32];
        int n=snprintf(path,sizeof(path),"%s/unit-%02u.sha256",directory,b);
        if (n<0 || n>=(int)sizeof(path)) goto failed;
        FILE *f=fopen(path,"rb"); if (!f) goto failed;
        size_t got=fread(wanted,1,sizeof(wanted),f); int error=ferror(f); fclose(f);
        if (error || got!=65 || wanted[64]!='\n') goto failed;
        snprintf(path,sizeof(path),"%s/unit-%02u.bin",directory,b);
        fd=open(path,O_RDONLY); if (fd<0) goto failed;
        struct stat st;
        if (fstat(fd,&st) || !S_ISREG(st.st_mode) || (uint64_t)st.st_size!=FR_NATIVE_BLOCK_BYTES) goto failed;
        /* MAP_SHARED: Metal wraps these pages in Shared-mode buffers, and on a
         * MAP_PRIVATE mapping that made a copy-on-write anonymous copy of every
         * page, 3.45 GB per block, later compressed and outside the process
         * footprint. In a two-block M1 Pro trial that cost 6.7 GB; the shared
         * mapping added only about 0.09 GB. The weight pages remain read-only. */
        void *data=mmap(NULL,FR_NATIVE_BLOCK_BYTES,PROT_READ,MAP_SHARED,fd,0);
        if (data==MAP_FAILED) goto failed;
        u->data[b]=data;
        /* Resident before first use: the stamp skips hashing, so touch every
         * page explicitly. No mlock: wiring tens of GB starves the node's OS. */
        if (!(paged & (1u<<b))) {
            madvise(data,FR_NATIVE_BLOCK_BYTES,MADV_WILLNEED);
            long page=sysconf(_SC_PAGESIZE); volatile unsigned char sum=0;
            for (size_t o=0;o<FR_NATIVE_BLOCK_BYTES;o+=(size_t)(page>0?page:4096)) sum^=((const unsigned char *)data)[o];
            (void)sum;
        }
        if (fr_units_cached(directory,b,actual) || memcmp(actual,wanted,64)) {
            if (sha256(data,FR_NATIVE_BLOCK_BYTES,digest)) goto failed;
            for (unsigned i=0;i<32;++i) snprintf(actual+2*i,3,"%02x",digest[i]);
            if (memcmp(actual,wanted,64) || fr_units_stamp(directory,b,actual)) goto failed;
        }
        int rc;
#ifdef FR_METAL_COMPUTE
        if ((paged>>b&1) && (backend==&fr_metal || backend==&fr_metal_ssd)) rc=fr_metal_load_ssd(b,fd,cache_bytes);
        else
#endif
        rc=backend->load((int)b,data,FR_NATIVE_BLOCK_BYTES);
        if (rc) goto failed;
        u->mask|=1u<<b;
        close(fd); fd=-1;
    }
    return 0;
failed:
    if (fd>=0) close(fd);
    fr_units_close(u); return -1;
}
int fr_units_fixture(fr_units *u,const fr_backend *backend,unsigned block) {
    if (!u || !backend || block>=FR_BLOCKS) return -1;
    memset(u,0,sizeof(*u)); u->backend=backend;
    void *data=mmap(NULL,FR_NATIVE_BLOCK_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (data==MAP_FAILED) return -1;
    u->data[block]=data; fr_native_fixture(data,47,7+block);
    if (mprotect(data,FR_NATIVE_BLOCK_BYTES,PROT_READ) || backend->load((int)block,data,FR_NATIVE_BLOCK_BYTES)) { fr_units_close(u); return -1; }
    u->mask=1u<<block; return 0;
}
/* Synthetic units for `fronk demo`: the memory a real unit takes (resident:
 * written, not just reserved) and valid encodings everywhere. Every byte 0x11
 * gives finite Q4_K/MXFP4/Q8_0 scales and small weights, so a call reads the
 * same bytes and does the same work as on the model's units, without the model.
 * One region per block: Metal computed nothing usable when several blocks'
 * buffers mapped the same pages (BENCH run 91). */
int fr_units_synthetic(fr_units *u,const fr_backend *backend,uint32_t mask) {
    if (!u || !backend || !mask || mask>=(1u<<FR_BLOCKS)) return -1;
    memset(u,0,sizeof(*u)); u->backend=backend;
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask>>b&1) {
        void *data=mmap(NULL,FR_NATIVE_BLOCK_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (data==MAP_FAILED) { fr_units_close(u); return -1; }
        u->data[b]=data; memset(data,0x11,FR_NATIVE_BLOCK_BYTES);
        if (mprotect(data,FR_NATIVE_BLOCK_BYTES,PROT_READ) || backend->load((int)b,data,FR_NATIVE_BLOCK_BYTES)) { fr_units_close(u); return -1; }
        u->mask|=1u<<b;
    }
    return 0;
}
int fr_units_synthetic_cache(fr_units *u,const fr_backend *backend,uint32_t mask,uint32_t paged,uint64_t bytes,const char *dir) {
    if (!paged) return fr_units_synthetic(u,backend,mask);
#ifdef FR_METAL_COMPUTE
    if (!u || !backend || !mask || mask>=(1u<<FR_BLOCKS) || (paged&~mask) ||
        (backend!=&fr_metal && backend!=&fr_metal_ssd) || __builtin_popcount(paged)>FR_SSD_MAX_BLOCKS) return -1;
    memset(u,0,sizeof(*u)); u->backend=backend;
    if ((mask&~paged) && fr_units_synthetic(u,backend,mask&~paged)) return -1;
    u->paged=paged; u->cache_bytes=bytes;
    int fd=fr_ssd_fixture_file(dir); if (fd<0) { fr_units_close(u); return -1; }
    int rc=0;
    for (unsigned b=0;b<FR_BLOCKS;++b) if (paged>>b&1) {
        if (fr_metal_load_ssd_fixture(b,fd,bytes)) { rc=-1; break; }
        u->mask|=1u<<b;
    }
    close(fd); if (rc) fr_units_close(u); return rc;
#else
    (void)u; (void)backend; (void)mask; (void)bytes; (void)dir; return -1;
#endif
}
