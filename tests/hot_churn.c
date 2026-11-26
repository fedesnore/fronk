/* hot_churn.c - the hot-expert store under the load of a busy session.
 *   hot_churn MODEL.gguf GIGABYTES REPLACEMENTS
 * Fills the store to its budget, then replaces experts at random that many times,
 * as the head's policy does when the chat changes subject. It must hold: memory
 * does not grow, nothing is swapped, and every slice in the store is still the
 * model's (compared with a fresh copy from the model file). */
#include "../src/hot.h"
#include "../src/gsq.h"
#include "../src/prep.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <mach/mach.h>
static double footprint_gb(void) {
    task_vm_info_data_t info; mach_msg_type_number_t n=TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&n)==KERN_SUCCESS?(double)info.phys_footprint/1e9:-1;
}
static double swapped_gb(void) {   /* the whole machine's, since boot */
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
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec+(double)t.tv_nsec*1e-9; }
enum { ALL=FR_LAYERS*FR_ROUTED_EXPERTS };
static unsigned char in[ALL]; static unsigned list[ALL];
int main(int argc,char **argv) {
    static fr_gguf g;
    if (argc!=4 || fr_gguf_open(argv[1],&g) || g.layout!=FR_LAYOUT_GSQ) { fputs("usage: hot_churn MODEL.gguf GB REPLACEMENTS\n",stderr); return 2; }
    const unsigned long swaps=strtoul(argv[3],NULL,10);
    if (fr_hot_open(&g,(uint64_t)(atof(argv[2])*1e9))) { puts("open failed"); return 1; }
    size_t sizes[FR_LAYERS]; unsigned kinds=0; size_t largest=0;
    for (unsigned l=0;l<FR_LAYERS;++l) { const size_t s=fr_gsq_slice_bytes(l); unsigned k=0; while (k<kinds && sizes[k]!=s) ++k; if (k==kinds) sizes[kinds++]=s; if (s>largest) largest=s; }
    printf("slice sizes:"); for (unsigned k=0;k<kinds;++k) printf(" %zu",sizes[k]); printf(" bytes (%u kinds)\n",kinds);
    const double swap0=swapped_gb(); double t0=now(); unsigned held=0; int rc=0;
    /* scattered like hot experts */
    for (unsigned tries=0;tries<4*ALL && rc<=0;++tries) { const unsigned id=pick(ALL); if (in[id]) continue; rc=fr_hot_admit(id/FR_ROUTED_EXPERTS,id%FR_ROUTED_EXPERTS); if (rc<0) break; if (!rc) { in[id]=1; list[held++]=id; } }
    if (rc<0) { puts("admit failed"); return 1; }
    uint64_t bytes; int wired; fr_hot_usage(&bytes,&wired);
    printf("filled: %u experts in %.1f s; store %.2f GB wired=%d; process %.2f GB; swapped since start %.3f GB\n",held,now()-t0,(double)bytes/1e9,wired,footprint_gb(),swapped_gb()-swap0); fflush(stdout);
    unsigned long done=0,refused=0; t0=now();
    for (unsigned long i=0;i<swaps;++i) {
        const unsigned slot=pick(held),out=list[slot];
        if (fr_hot_evict(out/FR_ROUTED_EXPERTS,out%FR_ROUTED_EXPERTS)) { puts("evict failed"); return 1; }
        in[out]=0; unsigned id=out; int ok=1;
        for (unsigned tries=0;tries<64;++tries) {
            const unsigned candidate=pick(ALL); if (in[candidate] || candidate==out) continue;
            ok=fr_hot_admit(candidate/FR_ROUTED_EXPERTS,candidate%FR_ROUTED_EXPERTS);
            if (ok<0) { puts("admit failed"); return 1; }
            if (!ok) { id=candidate; break; }
            ++refused;   /* no slot large enough was free: another candidate */
        }
        if (ok) { if (fr_hot_admit(out/FR_ROUTED_EXPERTS,out%FR_ROUTED_EXPERTS)) { puts("could not take the evicted expert back"); return 1; } }
        else ++done;
        in[id]=1; list[slot]=id;
        if ((i+1)%(swaps/5?swaps/5:1)==0) {
            fr_hot_usage(&bytes,&wired);
            printf("  %lu replacements (%.0f/s): store %.2f GB; process %.2f GB; swapped since start %.3f GB; %lu refused for size\n",i+1,(double)(i+1)/(now()-t0),(double)bytes/1e9,footprint_gb(),swapped_gb()-swap0,refused); fflush(stdout);
        }
    }
    unsigned char *fresh=malloc(largest); unsigned long compared=0,wrong=0;
    for (unsigned k=0;fresh && k<300 && k<held;++k) {
        const unsigned id=list[pick(held)],l=id/FR_ROUTED_EXPERTS,e=id%FR_ROUTED_EXPERTS;
        for (unsigned b=0;b<FR_BLOCKS;++b) {
            const unsigned char *have=fr_hot_peek(b,l,e);
            if (!have || fr_gsq_slice(&g.gsq,b,l,e,fresh) || memcmp(have,fresh,fr_gsq_slice_bytes(l))) ++wrong;
            ++compared;
        }
    }
    free(fresh);
    printf("after %lu replacements: %lu slices compared with the model file, %lu differ\n",done,compared,wrong);
    fr_hot_close();
    printf("closed: process %.2f GB; swapped since start %.3f GB\n",footprint_gb(),swapped_gb()-swap0);
    fr_gguf_close(&g);
    puts(wrong || !compared?"hot_churn: FAIL":"hot_churn: pass");
    return wrong || !compared;
}
