/* hot_fill.c - fill the hot-expert store as a cache node does, hold it, close it.
 *   hot_fill MODEL.gguf GIGABYTES HOLD_SECONDS      (FRONK_HOT_WIRE=after|none to compare) */
#include "../src/hot.h"
#include "../src/gsq.h"
#include "../src/prep.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
int main(int argc,char **argv) {
    static fr_gguf g;
    if (argc!=4 || fr_gguf_open(argv[1],&g) || g.layout!=FR_LAYOUT_GSQ) { fputs("usage: hot_fill MODEL.gguf GB SECONDS\n",stderr); return 2; }
    if (fr_hot_open(&g,(uint64_t)(atof(argv[2])*1e9))) { puts("open failed"); return 1; }
    struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
    unsigned n=0; int rc=0;
    /* scattered like hot experts: every third expert of every layer, then the next third */
    for (unsigned pass=0;pass<3 && !rc;++pass) for (unsigned e=pass;e<FR_ROUTED_EXPERTS && !rc;e+=3) for (unsigned l=0;l<FR_LAYERS && !rc;++l) { rc=fr_hot_admit(l,e); n+=!rc; }
    clock_gettime(CLOCK_MONOTONIC,&b);
    uint64_t bytes; int wired; fr_hot_usage(&bytes,&wired);
    printf("filled: %u experts, %.2f GB held, wired=%d, rc=%d, %.1f s\n",n,bytes/1e9,wired,rc,(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9); fflush(stdout);
    sleep((unsigned)atoi(argv[3]));
    fr_hot_close(); fr_gguf_close(&g); puts("closed"); return 0;
}
