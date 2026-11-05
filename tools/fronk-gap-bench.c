/* fronk-gap-bench: one backend's call time when calls arrive with a gap, as on a
 * node in a cluster (the GPU or cores idle between layers). Native random
 * blocks, distinct routing per token.
 * Usage: fronk-gap-bench BACKEND BLOCKS BURST GAP_US CALLS */
#include "../src/backend.h"
#include "../src/native.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
static int cmp(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return x<y?-1:x>y; }
int main(int argc,char **argv) {
    if (argc!=6) { fputs("Usage: fronk-gap-bench BACKEND BLOCKS BURST GAP_US CALLS\n",stderr); return 2; }
    const fr_backend *be=fr_backend_find(argv[1]); unsigned blocks=(unsigned)atoi(argv[2]),burst=(unsigned)atoi(argv[3]),calls=(unsigned)atoi(argv[5]);
    double gap=atof(argv[4]);
    if (!be || !blocks || blocks>FR_BLOCKS || !burst || burst>FR_MAX_BURST || !calls) return 2;
    for (unsigned k=0;k<blocks;++k) {
        unsigned char *w=mmap(NULL,FR_NATIVE_BLOCK_BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (w==MAP_FAILED) { perror("mmap"); return 1; }
        fr_native_fill(w,0x9e3779b9u*(k+1));
        if (be->load((int)k,w,FR_NATIVE_BLOCK_BYTES)) { fputs("load failed\n",stderr); return 1; }
    }
    static fr_req r; static fr_part out; float x=0.01f;
    for (unsigned t=0;t<FR_MAX_BURST;++t) for (unsigned d=0;d<FR_HIDDEN;++d) r.x[t][d]=(x=-x*1.0007f);
    double *us=calloc(calls,sizeof(double));
    for (unsigned i=0;i<calls+20;++i) {
        r.burst=burst; r.layer=i%FR_LAYERS;
        for (unsigned t=0;t<burst;++t) for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) {
            if (j<FR_ACTIVE_ROUTED) r.expert[t][j]=(uint16_t)((i*41+t*FR_ACTIVE_ROUTED+j)%FR_ROUTED_EXPERTS);
            r.gate[t][j]=j<FR_ACTIVE_ROUTED?0.1f:0;
        }
        for (double t0=now_us();now_us()-t0<gap;) {}
        double t0=now_us(); if (be->run(&r,&out)) { fputs("run failed\n",stderr); return 1; }
        if (i>=20) us[i-20]=now_us()-t0;
    }
    qsort(us,calls,sizeof(double),cmp);
    printf("%s blocks=%u burst=%u gap=%.0f us: p50=%.1f p90=%.1f p99=%.1f us\n",argv[1],blocks,burst,gap,us[calls/2],us[calls*9/10],us[calls*99/100]);
    return 0;
}
