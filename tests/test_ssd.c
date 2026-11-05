#include "test.h"
#include "../src/ssd.h"
#include "../src/metal.h"
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
typedef struct { const unsigned char *data; void **layers; uint32_t seen; unsigned count; int fail; } ready_check;
static int ready(void *context,uint32_t mask) {
    ready_check *c=context;
    if (!mask || (mask&c->seen)) return FR_STATE;
    c->seen|=mask; ++c->count;
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask>>b&1)
        for (unsigned e=0;e<30;++e)
            if (memcmp((unsigned char *)c->layers[b]+e*FR_NATIVE_SLICE_BYTES,c->data+fr_native_offset(47,e),FR_NATIVE_SLICE_BYTES)) return FR_NUMERIC;
    return c->fail?FR_STATE:FR_OK;
}
int main(void) {
    test_begin("ssd");
    char path[]="/tmp/fronk-ssd-test.XXXXXX"; int fd=mkstemp(path); if (fd>=0) unlink(path);
    check(fd>=0 && !ftruncate(fd,FR_NATIVE_BLOCK_BYTES),"sparse correctness fixture (never a timing fixture)");
    if (fd<0) return test_end();
    unsigned char *data=mmap(NULL,FR_NATIVE_BLOCK_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    check(data!=MAP_FAILED,"fixture mapping"); if (data==MAP_FAILED) { close(fd); return test_end(); }
    fr_native_fixture(data,47,21); msync(data+fr_native_offset(47,0)/16384*16384,FR_NATIVE_BLOCK_BYTES-fr_native_offset(47,0)/16384*16384,MS_SYNC);
    fr_req r={.layer=47,.burst=3};
    for (unsigned t=0;t<8;++t) {
        for (unsigned e=0;e<10;++e) { r.expert[t][e]=(uint16_t)(t*10+e); r.gate[t][e]=.0625f; }
        r.gate[t][10]=.5f;
        for (unsigned d=0;d<FR_HIDDEN;++d) r.x[t][d]=(float)((int)((d*17+t*31)%257)-128)/128;
    }
    fr_ssd *s=fr_ssd_open(64ull<<20,8); void *layers[FR_BLOCKS]={0};
    layers[1]=malloc(FR_NATIVE_LAYER_BYTES);
    check(s && layers[1] && !fr_ssd_attach(s,1,fd),"bounded cache and source opened");
    if (!s || !layers[1]) goto done;
    check(!fr_ssd_load(s,&r,2,layers),"parallel selected reads");
    fr_ssd_sample sample; fr_ssd_sample_get(s,&sample);
    check(sample.misses==30 && sample.hits==0 && sample.bytes==30*FR_NATIVE_SLICE_BYTES+FR_NATIVE_SHARED_BYTES,
          "only unique selected slices and shared weights read");
    int equal=1;
    for (unsigned e=0;e<30;++e) if (memcmp((unsigned char *)layers[1]+e*FR_NATIVE_SLICE_BYTES,data+fr_native_offset(47,e),FR_NATIVE_SLICE_BYTES)) equal=0;
    check(equal,"selected staging bytes exact");
    r.gate[0][10]=r.gate[1][10]=r.gate[2][10]=0;
    check(!fr_ssd_load(s,&r,2,layers),"warm selected load"); fr_ssd_sample_get(s,&sample);
    check(sample.hits==30 && !sample.misses && !sample.bytes,"fully cached burst performs no file reads");
    for (unsigned layer=0;layer<47;++layer) { r.layer=layer; if (fr_ssd_load(s,&r,2,layers)) equal=0; }
    r.layer=47; check(equal && !fr_ssd_load(s,&r,2,layers),"bounded eviction across all layers");
    fr_ssd_sample_get(s,&sample); check(sample.misses==30,"evicted entries reload");
    fr_ssd_clear(s); check(!ftruncate(fd,0) && fr_ssd_load(s,&r,2,layers)==FR_IO,"short reads fail closed");
    check(!ftruncate(fd,FR_NATIVE_BLOCK_BYTES),"restore failed source");
    fr_native_fixture(data,47,21); msync(data,FR_NATIVE_BLOCK_BYTES,MS_SYNC);
    check(!fr_ssd_load(s,&r,2,layers),"all failed slots can be retried");
    fr_ssd_sample_get(s,&sample); check(sample.misses==30 && !sample.hits,"failed reads were never admitted to cache");
    layers[2]=malloc(FR_NATIVE_LAYER_BYTES);
    check(layers[2] && !fr_ssd_attach(s,2,fd),"second streamed block attached");
    if (layers[2]) {
        ready_check seen={.data=data,.layers=layers};
        fr_ssd_clear(s);
        check(!fr_ssd_load_ready(s,&r,6,layers,ready,&seen) && seen.seen==6 && seen.count>=1 && seen.count<=2,
              "stream callback sees only complete immutable blocks and disjoint masks");
        seen=(ready_check){.data=data,.layers=layers};
        check(!fr_ssd_load_ready(s,&r,6,layers,ready,&seen) && seen.seen==6 && seen.count==1,
              "fully cached blocks coalesce into one GPU submission");
        fr_ssd_clear(s); seen=(ready_check){.data=data,.layers=layers,.fail=1};
        check(fr_ssd_load_ready(s,&r,6,layers,ready,&seen)==FR_STATE,"failed GPU callback drains all pending reads");
        check(!fr_ssd_load(s,&r,6,layers),"retry immediately after callback failure");
        fr_ssd_sample_get(s,&sample); check(sample.misses==60,"callback failure admits no partially completed cache entries");
        check(!fr_ssd_detach(s,2),"second source detached after drain");
    }
    check(!fr_ssd_detach(s,1),"detach source and invalidate entries");
#ifdef FR_METAL_COMPUTE
    fr_hwinfo hw;
    if (!fr_metal.probe(&hw)) {
        fr_part expected[8],actual;
        check(!fr_metal.load(1,data,FR_NATIVE_BLOCK_BYTES),"resident Metal oracle loaded");
        for (unsigned burst=1;burst<=8;++burst) {
            r.burst=burst;
            check(!fr_metal.run(&r,&expected[burst-1]),"resident oracle burst");
        }
        check(!fr_metal.unload(1) && !fr_metal_load_ssd(1,fd,64ull<<20),"same unit streamed through bounded staging");
        for (unsigned burst=1;burst<=8;++burst) {
            r.burst=burst;
            check(!fr_metal_ssd.run(&r,&actual) && !memcmp(&actual,&expected[burst-1],sizeof(actual)),
                  "streamed Metal output bit-identical for burst 1..8 including MTP");
        }
        check(!fr_metal_ssd_pipeline(1),"enable ready-block SSD pipeline");
        for (unsigned burst=1;burst<=8;++burst) {
            r.burst=burst; fr_metal_ssd_clear();
            check(!fr_metal_ssd.run(&r,&actual) && !memcmp(&actual,&expected[burst-1],sizeof(actual)),
                  "pipelined cold output remains bit-identical at widths 1..8");
        }
        fr_metal_ssd_clear(); check(!ftruncate(fd,0),"inject GPU-path read failure");
        memset(&actual,0x5a,sizeof(actual)); fr_part untouched=actual;
        check(fr_metal_ssd.run(&r,&actual)==FR_IO && !memcmp(&actual,&untouched,sizeof(actual)),"failed streaming preserves caller output");
        check(!ftruncate(fd,FR_NATIVE_BLOCK_BYTES),"restore GPU source size");
        fr_native_fixture(data,47,21); msync(data,FR_NATIVE_BLOCK_BYTES,MS_SYNC);
        check(!fr_metal_ssd.run(&r,&actual) && !memcmp(&actual,&expected[7],sizeof(actual)),"GPU retry after drained I/O failure remains exact");
        unsigned char *ram=mmap(NULL,FR_NATIVE_BLOCK_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        check(ram!=MAP_FAILED,"separate resident unit fixture");
        if (ram!=MAP_FAILED) {
            fr_native_fixture(ram,47,42); r.burst=3;
            check(!fr_metal_ssd.unload(1) && !fr_metal.load(1,data,FR_NATIVE_BLOCK_BYTES) &&
                  !fr_metal.load(2,ram,FR_NATIVE_BLOCK_BYTES) && !fr_metal.run(&r,&expected[2]),"two-unit resident oracle");
            check(!fr_metal.unload(1) && !fr_metal_load_ssd(1,fd,64ull<<20),"mixed resident and streamed units");
            check(!fr_metal_ssd.run(&r,&actual) && !memcmp(&actual,&expected[2],sizeof(actual)),
                  "RAM compute overlapped with SSD reads retains exact ordered sum");
            check(!fr_metal.unload(2),"resident part of hybrid unloads");
            munmap(ram,FR_NATIVE_BLOCK_BYTES);
        }
        check(!fr_metal_ssd.unload(1) && !fr_metal_close(),"streaming resources released");
    } else check(0,"Metal device unavailable");
#endif
done:
    fr_ssd_close(s); free(layers[1]); free(layers[2]); munmap(data,FR_NATIVE_BLOCK_BYTES); close(fd);
    return test_end();
}
