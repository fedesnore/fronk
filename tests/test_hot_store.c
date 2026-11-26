/* test_hot_store.c - the hot-expert store against the real model file:
 * admit, evict, admit again; neighbours that share a page; the budget in pages.
 *   test_hot_store MODEL.gguf */
#include "../src/hot.h"
#include "../src/gsq.h"
#include "../src/prep.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static fr_gguf g; static size_t page; static unsigned char *want; static int failures;
#define CHECK(x) do { if (!(x)) { printf("FAIL line %d: %s\n",__LINE__,#x); ++failures; } } while (0)
static int same(unsigned layer,unsigned expert) {   /* the store's copy against a fresh one from the file */
    for (unsigned b=0;b<FR_BLOCKS;++b) {
        const unsigned char *have=fr_hot_peek(b,layer,expert);
        if (!have || fr_gsq_slice(&g.gsq,b,layer,expert,want) || memcmp(have,want,fr_gsq_slice_bytes(layer))) return 0;
    }
    return 1;
}
/* Memory the listed experts must hold: the union of their pages, in the 20 units. */
static uint64_t held(const unsigned (*list)[2],unsigned n) {
    uint64_t pages=0;
    for (unsigned i=0;i<n;++i) {
        const uint64_t at=fr_gsq_offset(list[i][0],list[i][1]); const size_t first=at/page,last=(at+fr_gsq_slice_bytes(list[i][0])-1)/page;
        for (size_t p=first;p<=last;++p) {
            int seen=0;
            for (unsigned k=0;k<i && !seen;++k) {
                const uint64_t o=fr_gsq_offset(list[k][0],list[k][1]); const size_t f=o/page,l=(o+fr_gsq_slice_bytes(list[k][0])-1)/page;
                seen=p>=f && p<=l;
            }
            pages+=!seen;
        }
    }
    return pages*page*FR_BLOCKS;
}
static uint64_t usage(void) { uint64_t bytes; int wired; fr_hot_usage(&bytes,&wired); return bytes; }
int main(int argc,char **argv) {
    if (argc!=2) { fputs("usage: test_hot_store MODEL.gguf\n",stderr); return 2; }
    if (fr_gguf_open(argv[1],&g) || g.layout!=FR_LAYOUT_GSQ) { puts("FAIL: not the GSQ model file"); return 1; }
    page=(size_t)sysconf(_SC_PAGESIZE); size_t cap=0;
    for (unsigned l=0;l<FR_LAYERS;++l) if (fr_gsq_slice_bytes(l)>cap) cap=fr_gsq_slice_bytes(l);
    want=malloc(cap);
    const uint64_t budget=96000000;
    CHECK(!fr_hot_open(&g,budget));
    /* three neighbours, the first and the last slice of a unit, one alone */
    static const unsigned set[][2]={{3,10},{3,11},{3,12},{0,0},{FR_LAYERS-1,FR_ROUTED_EXPERTS-1},{7,100}};
    const unsigned n=sizeof(set)/sizeof(set[0]);
    for (unsigned i=0;i<n;++i) CHECK(!fr_hot_admit(set[i][0],set[i][1]));
    for (unsigned i=0;i<n;++i) CHECK(same(set[i][0],set[i][1]));
    CHECK(usage()==held(set,n));
    CHECK(!fr_hot_admit(3,11) && usage()==held(set,n));            /* twice is once */
    { uint64_t bytes; int wired; fr_hot_usage(&bytes,&wired); printf("six experts hold %.2f MB (%.2f MB of weights), wired=%d, page=%zu\n",bytes/1e6,
        (double)FR_BLOCKS*(3*fr_gsq_slice_bytes(3)+fr_gsq_slice_bytes(0)+fr_gsq_slice_bytes(FR_LAYERS-1)+fr_gsq_slice_bytes(7))/1e6,wired,page); }
    /* the middle neighbour leaves: the two beside it keep every byte */
    CHECK(!fr_hot_evict(3,11) && !fr_hot_peek(0,3,11));
    CHECK(same(3,10) && same(3,12));
    { static const unsigned left[][2]={{3,10},{3,12},{0,0},{FR_LAYERS-1,FR_ROUTED_EXPERTS-1},{7,100}}; CHECK(usage()==held(left,5)); }
    CHECK(!fr_hot_evict(3,11));                                       /* not there: nothing to do */
    CHECK(!fr_hot_admit(3,11) && same(3,10) && same(3,11) && same(3,12) && usage()==held(set,n));
    /* fill a layer until the budget says no, then one out, one in */
    unsigned filled=0,e=0; int rc=0; const struct timespec t0={0}; (void)t0;
    struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
    for (e=0;e<FR_ROUTED_EXPERTS && !(rc=fr_hot_admit(20,e));++e) ++filled;
    clock_gettime(CLOCK_MONOTONIC,&b);
    CHECK(rc==1 && filled>8 && usage()<=budget && usage()+2*FR_BLOCKS*(cap+2*page)>budget);
    printf("budget %.0f MB: %u more experts fit, %.2f ms each; held %.2f MB\n",budget/1e6,filled,((b.tv_sec-a.tv_sec)*1e3+(b.tv_nsec-a.tv_nsec)/1e6)/(filled?filled:1),usage()/1e6);
    CHECK(fr_hot_admit(21,5)==1);
    CHECK(!fr_hot_evict(20,0) && !fr_hot_evict(20,1) && !fr_hot_admit(21,5) && same(21,5) && same(20,2) && same(20,filled-1));
    for (unsigned i=0;i<filled;++i) CHECK(!fr_hot_peek(0,20,i) || same(20,i));
    /* everything out: nothing held */
    for (unsigned i=0;i<n;++i) CHECK(!fr_hot_evict(set[i][0],set[i][1]));
    for (unsigned i=0;i<filled;++i) CHECK(!fr_hot_evict(20,i));
    CHECK(!fr_hot_evict(21,5) && usage()==0);
    /* and back in after the pages were given away */
    CHECK(!fr_hot_admit(3,11) && same(3,11) && !fr_hot_admit(20,3) && same(20,3));
    fr_hot_close(); fr_gguf_close(&g); free(want);
    puts(failures?"test_hot_store: FAILED":"test_hot_store: pass");
    return failures?1:0;
}
