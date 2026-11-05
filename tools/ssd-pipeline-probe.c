/* Bounded off/on/off test with verified native unit files; no model mutation.
 * Cold and warm timings remain separate. No resident fixture allocation. */
#include "../src/metal.h"
#include "../src/ssd.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
enum { N=48 };
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int cmp(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
int main(int argc,char **argv) {
    if (argc!=3) { fputs("usage: ssd-pipeline-probe UNIT0 UNIT1\n",stderr); return 2; }
    setenv("FRONK_METAL_KEEPALIVE","0",1);
    for (unsigned b=0;b<2;++b) {
        int fd=open(argv[b+1],O_RDONLY);
        if (fd<0 || fr_metal_load_ssd(b,fd,FR_SSD_DEFAULT_CACHE)) { fprintf(stderr,"load unit %u failed\n",b); return 1; }
        close(fd);
    }
    fr_part *reference=calloc(N,sizeof(*reference)); if (!reference) return 1;
    unsigned widths[]={1,3,4,8};
    for (unsigned w=0;w<4;++w) {
        for (unsigned phase=0;phase<3;++phase) {
            if (fr_metal_ssd_pipeline(phase==1)) return 1;
            double cold[N],warm[N]; uint64_t bytes=0,hits=0;
            for (unsigned i=0;i<N+4;++i) {
                fr_req req={.layer=i%FR_LAYERS,.burst=widths[w]}; fr_part got;
                for (unsigned t=0;t<req.burst;++t) {
                    for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) { req.expert[t][e]=(i*37+t*10+e)%FR_ROUTED_EXPERTS; req.gate[t][e]=.1f; }
                    for (unsigned d=0;d<FR_HIDDEN;++d) req.x[t][d]=(float)((int)((d*17+t*31)%257)-128)/128;
                }
                fr_metal_ssd_clear(); double start=now();
                if (fr_metal_ssd.run(&req,&got)) { fputs("cold run failed\n",stderr); return 1; }
                double elapsed=now()-start;
                fr_ssd_sample io; fr_metal_ssd_sample(&io);
                if (i>=4) {
                    unsigned at=i-4; cold[at]=elapsed; bytes+=io.bytes;
                    if (!phase) reference[at]=got;
                    else if (memcmp(&got,&reference[at],sizeof(got))) { fputs("cold output differs\n",stderr); return 1; }
                }
                start=now();
                if (fr_metal_ssd.run(&req,&got)) { fputs("warm run failed\n",stderr); return 1; }
                elapsed=now()-start; fr_metal_ssd_sample(&io);
                if (i>=4) {
                    warm[i-4]=elapsed; hits+=io.hits;
                    if (memcmp(&got,&reference[i-4],sizeof(got)) || io.bytes) { fputs("warm output/cache differs\n",stderr); return 1; }
                }
            }
            qsort(cold,N,sizeof(double),cmp); qsort(warm,N,sizeof(double),cmp);
            printf("SSD_PIPELINE width=%u phase=%u cold_us=%.3f warm_us=%.3f cold_p99_us=%.3f bytes=%llu hits=%llu exact=1\n",
                   widths[w],phase,(cold[23]+cold[24])/2,(warm[23]+warm[24])/2,cold[N-1],(unsigned long long)bytes,(unsigned long long)hits);
            fflush(stdout);
        }
    }
    for (unsigned b=0;b<2;++b) if (fr_metal_ssd.unload(b)) return 1;
    free(reference); return fr_metal_close()!=FR_OK;
}
