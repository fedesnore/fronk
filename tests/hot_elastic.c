/* hot_elastic.c - the hot-expert store while its budget changes.
 *   hot_elastic MODEL.gguf FLOOR_GB CEILING_GB ROUNDS
 * Fills the store, then moves its budget up and down at random between the two
 * sizes, as an elastic cache does when the engine beside it grows and shrinks. It
 * must hold: the experts above a smaller budget leave, the memory really returns to
 * the system (the process shrinks), nothing is swapped, and every slice still in
 * the store is the model's. Experts come from a pool of 600, so the model file is
 * read once and then served from the file cache. */
#include "hot.h"
#include "gsq.h"
#include "prep.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <mach/mach.h>
static double footprint_gb(void) {
    task_vm_info_data_t info; mach_msg_type_number_t n=TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&n)==KERN_SUCCESS?(double)info.phys_footprint/1e9:-1;
}
static double swapped_gb(void) {
    vm_statistics64_data_t vm; mach_msg_type_number_t n=HOST_VM_INFO64_COUNT; vm_size_t page=0; mach_port_t host=mach_host_self();
    host_page_size(host,&page);
    return host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&vm,&n)==KERN_SUCCESS?(double)vm.swapouts*(double)page/1e9:-1;
}
#else
static double footprint_gb(void) { return 0; }
static double swapped_gb(void) { return 0; }
#endif
static uint32_t seed=20261111u;
static unsigned pick(unsigned n) { seed=seed*1664525u+1013904223u; return (unsigned)(((uint64_t)(seed>>8)*n)>>24); }
enum { ALL=FR_LAYERS*FR_ROUTED_EXPERTS, POOL=600 };
static unsigned pool[POOL],list[POOL],held; static int where[ALL];   /* where: index in list, or -1 */
static uint16_t gone[ALL][2];
static int fill(void) {   /* until the budget refuses POOL candidates in a row */
    for (unsigned refused=0;refused<POOL;) {
        const unsigned id=pool[pick(POOL)];
        if (where[id]>=0) { ++refused; continue; }
        const int rc=fr_hot_admit(id/FR_ROUTED_EXPERTS,id%FR_ROUTED_EXPERTS);
        if (rc<0) return -1;
        if (rc) { ++refused; continue; }
        where[id]=(int)held; list[held++]=id; refused=0;
    }
    return 0;
}
int main(int argc,char **argv) {
    static fr_gguf g;
    if (argc!=5 || fr_gguf_open(argv[1],&g) || g.layout!=FR_LAYOUT_GSQ) { fputs("usage: hot_elastic MODEL.gguf FLOOR_GB CEILING_GB ROUNDS\n",stderr); return 2; }
    const uint64_t floor_b=(uint64_t)(atof(argv[2])*1e9),ceiling=(uint64_t)(atof(argv[3])*1e9); const unsigned rounds=(unsigned)atoi(argv[4]);
    if (!floor_b || ceiling<=floor_b || fr_hot_open(&g,floor_b)) { puts("open failed"); return 1; }
    for (unsigned i=0;i<ALL;++i) where[i]=-1;
    for (unsigned i=0;i<POOL;) { const unsigned id=pick(ALL); unsigned k=0; while (k<i && pool[k]!=id) ++k; if (k==i) pool[i++]=id; }
    size_t largest=0; for (unsigned l=0;l<FR_LAYERS;++l) if (fr_gsq_slice_bytes(l)>largest) largest=fr_gsq_slice_bytes(l);
    unsigned char *fresh=malloc(largest); if (!fresh) return 1;
    const double swap0=swapped_gb(),base=footprint_gb(); unsigned long compared=0,wrong=0,fat=0; int failed=0;
    if (fill()) { puts("admit failed"); return 1; }
    uint64_t bytes; int wired; fr_hot_usage(&bytes,&wired);
    printf("filled at %.2f GB: %u experts, store %.2f GB, process %.2f GB (%.2f before)\n",(double)fr_hot_capacity()/1e9,held,(double)bytes/1e9,footprint_gb(),base);
    for (unsigned r=0;r<rounds && !failed;++r) {
        const uint64_t target=floor_b+(uint64_t)pick((unsigned)((ceiling-floor_b)>>24)+1)*(1u<<24),was=fr_hot_capacity(); unsigned left=0;
        if (target>was) { if (fr_hot_grow(target)) failed=1; }
        else {
            left=fr_hot_above(target,gone,ALL);
            for (unsigned i=0;i<left && i<ALL;++i) {
                const unsigned id=gone[i][0]*FR_ROUTED_EXPERTS+gone[i][1]; const int at=where[id];
                if (at<0 || fr_hot_evict(gone[i][0],gone[i][1])) { puts("evict failed"); failed=1; break; }
                list[at]=list[--held]; where[list[at]]=at; where[id]=-1;
            }
            if (!failed && fr_hot_shrink(target)) { puts("shrink refused: a slot above the budget is still in use"); failed=1; }
            fr_hot_usage(&bytes,&wired);
            if (bytes>fr_hot_capacity() || fr_hot_capacity()>target) { puts("the store is larger than its budget"); failed=1; }
            if (footprint_gb()>(double)bytes/1e9+base+0.25) ++fat;   /* the memory did not go back */
        }
        if (!failed && fill()) { puts("admit failed"); failed=1; }
        for (unsigned k=0;k<24 && held && !failed;++k) {
            const unsigned id=list[pick(held)],l=id/FR_ROUTED_EXPERTS,e=id%FR_ROUTED_EXPERTS;
            for (unsigned b=0;b<FR_BLOCKS;++b) {
                const unsigned char *have=fr_hot_peek(b,l,e);
                if (!have || fr_gsq_slice(&g.gsq,b,l,e,fresh) || memcmp(have,fresh,fr_gsq_slice_bytes(l))) ++wrong;
                ++compared;
            }
        }
        fr_hot_usage(&bytes,&wired);
        printf("  round %2u: budget %.2f -> %.2f GB%s; %u experts; store %.2f GB; process %.2f GB; swapped %.3f GB\n",r+1,(double)was/1e9,(double)fr_hot_capacity()/1e9,
               target>was?"":" (gave memory back)",held,(double)bytes/1e9,footprint_gb(),swapped_gb()-swap0);
        (void)left;
    }
    fr_hot_close(); free(fresh);
    const double swapped=swapped_gb()-swap0;
    printf("%lu slices compared with the model file, %lu differ; %lu times the process stayed larger than its store; swapped %.3f GB; process %.2f GB after closing\n",
           compared,wrong,fat,swapped,footprint_gb());
    fr_gguf_close(&g);
    const int bad=failed || wrong || fat || !compared || swapped>0.001;
    puts(bad?"hot_elastic: FAIL":"hot_elastic: pass");
    return bad;
}
