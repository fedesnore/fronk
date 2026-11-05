/* Install-time RAM+SSD calibration. Discovery reads saved tables only. No GGUF
 * is needed: resident native units and an actual, fully written SSD layer use
 * identical kernels, I/O workers, cache and staging to the runtime path. */
#include "selection.h"
#include "metal.h"
#include "sha256.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef FR_METAL_COMPUTE
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <time.h>
#include <unistd.h>
typedef struct { double cold[2],warm[2],tail[2]; } profile;
static int profile_path(const char *dir,unsigned ram,unsigned burst,char *path,size_t cap) {
    struct statfs fs; if (!dir || !dir[0] || statfs(dir,&fs) || !selection_cache_identity_get()[0]) return -1;
    char key[512],hex[65]; unsigned char digest[32];
    const char *pipeline=getenv("FRONK_SSD_PIPELINE");
    int n=snprintf(key,sizeof(key),"%s|%s|%s|%d:%d|ram%u|burst%u|cache%llu|workers%u",
        pipeline && !strcmp(pipeline,"1")?"hybrid-v3-ready-blocks":"hybrid-v2-uncached-fixture",selection_cache_identity_get(),
        fs.f_fstypename,fs.f_fsid.val[0],fs.f_fsid.val[1],ram,burst,(unsigned long long)FR_SSD_DEFAULT_CACHE,FR_SSD_WORKERS);
    if (n<0 || (size_t)n>=sizeof(key) || sha256(key,(size_t)n,digest)) return -1;
    for (unsigned i=0;i<32;++i) snprintf(hex+2*i,3,"%02x",digest[i]);
    n=snprintf(path,cap,"%s/bench/hybrid-%s.txt",dir,hex); return n<0 || (size_t)n>=cap?-1:0;
}
static int read_profile(const char *dir,unsigned ram,unsigned burst,profile *p) {
    char path[700],extra; unsigned version,r,b; unsigned long long bytes;
    if (profile_path(dir,ram,burst,path,sizeof(path))) return -1;
    FILE *f=fopen(path,"r"); if (!f) return -1;
    int n=fscanf(f,"FRHYBRID%u %u %u %llu %lf %lf %lf %lf %lf %lf %c",&version,&r,&b,&bytes,
        &p->cold[0],&p->warm[0],&p->tail[0],&p->cold[1],&p->warm[1],&p->tail[1],&extra);
    int error=ferror(f); fclose(f);
    if (error || n!=10 || version!=1 || r!=ram || b!=burst || bytes!=FR_SSD_DEFAULT_CACHE) return -1;
    for (unsigned i=0;i<2;++i) if (!isfinite(p->cold[i]) || !isfinite(p->warm[i]) || !isfinite(p->tail[i]) ||
        p->cold[i]<=0 || p->warm[i]<=0 || p->tail[i]<p->cold[i]) return -1;
    return 0;
}
static int write_profile(const char *dir,unsigned ram,unsigned burst,const profile *p) {
    char path[700],tmp[720]; if (profile_path(dir,ram,burst,path,sizeof(path))) return -1;
    snprintf(tmp,sizeof(tmp),"%s.tmp",path); FILE *f=fopen(tmp,"w"); if (!f) return -1;
    int n=fprintf(f,"FRHYBRID1 %u %u %llu %.17g %.17g %.17g %.17g %.17g %.17g\n",ram,burst,
        (unsigned long long)FR_SSD_DEFAULT_CACHE,p->cold[0],p->warm[0],p->tail[0],p->cold[1],p->warm[1],p->tail[1]);
    int error=n<0; if (fflush(f) || fsync(fileno(f))) error=1; if (fclose(f)) error=1;
    if (error || rename(tmp,path)) { unlink(tmp); return -1; } return 0;
}
static uint64_t now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
static int cmp(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
static int measure_point(unsigned burst,profile *p) {
    enum { SAMPLES=48 }; double cold[SAMPLES],warm[SAMPLES]; fr_req req={.burst=burst}; fr_part out;
        for (unsigned i=0;i<SAMPLES;++i) {
            req.layer=i%FR_LAYERS;
            for (unsigned t=0;t<burst;++t) {
                for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) { req.expert[t][e]=(uint16_t)((i*37+t*10+e)%FR_ROUTED_EXPERTS); req.gate[t][e]=.1f; }
                for (unsigned d=0;d<FR_HIDDEN;++d) req.x[t][d]=(float)((int)((d*17+t*31)%257)-128)/128;
            }
            fr_metal_ssd_clear(); uint64_t begin=now();
            if (fr_metal_ssd.run(&req,&out)) return -1; cold[i]=(double)(now()-begin);
            begin=now(); if (fr_metal_ssd.run(&req,&out)) return -1; warm[i]=(double)(now()-begin);
        }
    qsort(cold,SAMPLES,sizeof(double),cmp); qsort(warm,SAMPLES,sizeof(double),cmp);
    p->cold[0]=(cold[23]+cold[24])/2; p->tail[0]=cold[47]; p->warm[0]=(warm[23]+warm[24])/2;
    return 0;
}
typedef struct { void *data[FR_BLOCKS]; uint32_t loaded; } resident_fixture;
static int calibrate_ram(unsigned ram,int fd,profile p[FR_BENCH_BURST_MAX],unsigned missing,resident_fixture *resident) {
    uint32_t streamed=0; int rc=-1;
    for (unsigned b=0;b<ram;++b) {
        if (resident->loaded>>b&1) continue;
        resident->data[b]=mmap(NULL,FR_NATIVE_BLOCK_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (resident->data[b]==MAP_FAILED) { resident->data[b]=NULL; goto done; }
        memset(resident->data[b],0x11,FR_NATIVE_BLOCK_BYTES);
        if (fr_metal.load((int)b,resident->data[b],FR_NATIVE_BLOCK_BYTES)) goto done;
        resident->loaded|=1u<<b;
    }
    for (unsigned stream=1;stream<=2;++stream) {
        if (fr_metal_load_ssd_fixture(ram+stream-1,fd,FR_SSD_DEFAULT_CACHE)) goto done;
        streamed|=1u<<(ram+stream-1);
        for (unsigned burst=1;burst<=FR_BENCH_BURST_MAX;++burst) {
            if (!(missing>>(burst-1)&1)) continue;
            profile point={0}; if (measure_point(burst,&point)) goto done;
            p[burst-1].cold[stream-1]=point.cold[0]; p[burst-1].warm[stream-1]=point.warm[0]; p[burst-1].tail[stream-1]=point.tail[0];
        }
    }
    rc=0;
done:
    for (unsigned b=0;b<FR_BLOCKS;++b) if (streamed>>b&1) if (fr_metal.unload((int)b)) rc=-1;
    return rc;
}
#endif
void selection_hybrid_restore(const fr_hwinfo *hw,int head,fr_selection *s,const char *dir,FILE *progress) {
#ifdef FR_METAL_COMPUTE
    const char *enabled=getenv("FRONK_SSD_STREAMING");
    if (enabled && !strcmp(enabled,"0")) return;
    if (!hw || !s || !dir || s->count!=1 || strcmp(s->candidate[0].name,"metal")) return;
    unsigned cap=(unsigned)selection_capacity(hw,"metal-ssd",head);
    if (cap<2) return;
    unsigned ram=cap-2; if (ram>s->candidate[0].capacity) ram=s->candidate[0].capacity;
    profile p;
    if (!read_profile(dir,ram,s->burst,&p) && !selection_hybrid(s,ram,2,FR_SSD_DEFAULT_CACHE,p.cold,p.warm,p.tail)) {
        if (progress) fprintf(progress,"BENCH cached hybrid RAM=%u SSD=1..2 burst=%u cache=256 MiB (cold and warm; no remeasurement)\n",ram,s->burst);
    } else if (progress) fprintf(progress,"BENCH hybrid calibration unavailable for RAM=%u burst=%u; resident mode retained. Run fronk backends --measure during setup.\n",ram,s->burst);
#else
    (void)hw; (void)head; (void)s; (void)dir; (void)progress;
#endif
}
int selection_hybrid_calibrate(const fr_hwinfo *hw,int head,const fr_selection *resident,const char *dir,int refresh,FILE *progress) {
#ifdef FR_METAL_COMPUTE
    if (!hw || !resident || !resident->count || strcmp(resident->candidate[0].name,"metal")) return 0;
    unsigned cap=(unsigned)selection_capacity(hw,"metal-ssd",head); if (cap<2) return 0;
    unsigned max=cap-2; if (max>resident->candidate[0].capacity) max=resident->candidate[0].capacity;
    char path[700]; snprintf(path,sizeof(path),"%s/bench",dir); mkdir(path,0700);
    int fd=-1,rc=0; resident_fixture resident_data={0};
    for (unsigned ram=0;ram<=max;++ram) {
        unsigned missing=0; profile p[FR_BENCH_BURST_MAX]={0};
        for (unsigned burst=1;burst<=FR_BENCH_BURST_MAX;++burst) if (refresh || read_profile(dir,ram,burst,&p[burst-1])) missing|=1u<<(burst-1);
        if (!missing) { if (progress) fprintf(progress,"BENCH cached hybrid calibration RAM=%u bursts=1..4\n",ram); continue; }
        if (fd<0 && (fd=fr_ssd_fixture_file(dir))<0) { rc=-1; break; }
        if (progress) { fprintf(progress,"BENCH hybrid calibration RAM=%u SSD=1..2 bursts=1..4 (one-time; native GPU + parallel SSD reads)\n",ram); fflush(progress); }
        if (calibrate_ram(ram,fd,p,missing,&resident_data)) { rc=-1; break; }
        for (unsigned burst=1;burst<=FR_BENCH_BURST_MAX;++burst) {
            if (!(missing>>(burst-1)&1)) continue;
            if (write_profile(dir,ram,burst,&p[burst-1])) { rc=-1; break; }
            if (progress) fprintf(progress,"BENCH hybrid saved RAM=%u SSD=1/2 burst=%u cold=%.0f/%.0f us warm=%.0f/%.0f us\n",ram,burst,
                p[burst-1].cold[0]/1000,p[burst-1].cold[1]/1000,p[burst-1].warm[0]/1000,p[burst-1].warm[1]/1000);
        }
        if (rc) break;
    }
    for (unsigned b=0;b<FR_BLOCKS;++b) {
        if (resident_data.loaded>>b&1) if (fr_metal.unload((int)b)) rc=-1;
        if (resident_data.data[b]) munmap(resident_data.data[b],FR_NATIVE_BLOCK_BYTES);
    }
    if (fd>=0) close(fd); return rc;
#else
    (void)hw; (void)head; (void)resident; (void)dir; (void)refresh; (void)progress; return 0;
#endif
}
