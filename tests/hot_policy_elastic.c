/* hot_policy_elastic.c - the policy and the store together while the cache's size
 * changes under work.
 *   hot_policy_elastic MODEL.gguf FLOOR_GB CEILING_GB SECONDS STATE_DIR
 * One thread plays the head's calls (routing drawn from a skewed pool, counted as
 * begun and done exactly as plane.c does); the test asks the cache for a new size
 * every second. It must hold at every call: an expert routed to the cache is in its
 * memory. And at the end: the process is as large as the store, nothing was swapped,
 * the slices in the store are the model's. */
#include "hot.h"
#include "gsq.h"
#include "prep.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
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
enum { POOL=900 };
static uint16_t pool[POOL][2];
static _Atomic uint64_t begun,done,calls,last;
static atomic_int stop; static atomic_ulong missing,served,asked_total;
static int admit(void *ctx,unsigned l,unsigned e) { (void)ctx; return fr_hot_admit(l,e); }
static int evict(void *ctx,unsigned l,unsigned e) { (void)ctx; return fr_hot_evict(l,e); }
static void *work(void *arg) {
    (void)arg; static fr_req req; uint32_t seed=99;
    while (!atomic_load(&stop)) {
        const unsigned layer=(seed=seed*1664525u+1013904223u)>>8&31; unsigned n=0;
        req.layer=layer; req.burst=4;
        for (unsigned t=0;t<4;++t) for (unsigned j=0;j<FR_ACTIVE_ROUTED;) {
            seed=seed*1664525u+1013904223u; unsigned r=(seed>>8)%POOL; r=r*r/POOL;   /* skewed: low numbers more often */
            if (pool[r][0]!=layer) { if (++n>4000) { req.expert[t][j]=(uint16_t)((j*37+t*11)%FR_ROUTED_EXPERTS); ++j; } continue; }
            int again=0; for (unsigned k=0;k<j;++k) again|=req.expert[t][k]==pool[r][1];
            if (again) { if (++n>4000) { req.expert[t][j]=(uint16_t)((j*37+t*11+400)%FR_ROUTED_EXPERTS); ++j; } continue; }
            req.expert[t][j++]=pool[r][1];
        }
        atomic_fetch_add(&begun,1);   /* before the routing is read, as fr_head_plane_run does */
        for (unsigned t=0;t<4;++t) for (unsigned j=0;j<FR_ACTIVE_ROUTED;++j) {
            atomic_fetch_add(&asked_total,1);
            if (fr_hot_where(layer,req.expert[t][j])) {
                if (!fr_hot_peek(0,layer,req.expert[t][j])) atomic_fetch_add(&missing,1);
                else atomic_fetch_add(&served,1);
            }
        }
        usleep(150);   /* the call's computation: the slot must stay for all of it */
        for (unsigned t=0;t<4;++t) for (unsigned j=0;j<FR_ACTIVE_ROUTED;++j)
            if (fr_hot_where(layer,req.expert[t][j])==0) continue;
        fr_hot_count(&req);
        atomic_fetch_add(&calls,1); atomic_fetch_add(&done,1);
    }
    return NULL;
}
int main(int argc,char **argv) {
    static fr_gguf g;
    if (argc!=6 || fr_gguf_open(argv[1],&g) || g.layout!=FR_LAYOUT_GSQ) { fputs("usage: hot_policy_elastic MODEL.gguf FLOOR_GB CEILING_GB SECONDS STATE_DIR\n",stderr); return 2; }
    const uint64_t floor_b=(uint64_t)(atof(argv[2])*1e9),ceiling=(uint64_t)(atof(argv[3])*1e9); const int seconds=atoi(argv[4]);
    uint32_t seed=7;
    for (unsigned i=0;i<POOL;++i) { seed=seed*1664525u+1013904223u; pool[i][0]=(uint16_t)((seed>>8)&31); seed=seed*1664525u+1013904223u; pool[i][1]=(uint16_t)((seed>>8)%FR_ROUTED_EXPERTS); }
    const double swap0=swapped_gb(),base=footprint_gb();
    if (fr_hot_open(&g,floor_b) || fr_hot_policy_start(argv[5],&begun,&done) || fr_hot_policy_add("head",1.0,admit,evict,NULL,&calls,&last)!=0 ||
        fr_hot_policy_elastic(0,floor_b,ceiling) || fr_hot_policy_go()) { puts("could not start"); return 1; }
    pthread_t th; pthread_create(&th,NULL,work,NULL);
    unsigned experts; double share; uint64_t bytes; int wired; unsigned long fat=0;
    for (int s=0;s<seconds;++s) {
        seed=seed*1664525u+1013904223u;
        const uint64_t want=floor_b+(uint64_t)(((seed>>8)%1000)/999.0*(double)(ceiling-floor_b));
        if (s>=3) fr_hot_policy_resize(0,want);
        sleep(1);
        fr_hot_report(0,&experts,&share); fr_hot_usage(&bytes,&wired);
        if (s>=4 && footprint_gb()>(double)bytes/1e9+base+0.35) ++fat;
        printf("  %2d s: asked %.2f GB, budget %.2f GB, store %.2f GB, process %.2f GB; %u experts routed, %.0f%% of uses; routed but missing: %lu\n",s+1,s>=3?(double)want/1e9:0.0,
               (double)fr_hot_capacity()/1e9,(double)bytes/1e9,footprint_gb(),experts,100*share,atomic_load(&missing));
        fflush(stdout);
    }
    atomic_store(&stop,1); pthread_join(th,NULL);
    fr_hot_policy_stop();
    size_t largest=0; for (unsigned l=0;l<FR_LAYERS;++l) if (fr_gsq_slice_bytes(l)>largest) largest=fr_gsq_slice_bytes(l);
    unsigned char *fresh=malloc(largest); unsigned long compared=0,wrong=0;
    for (unsigned i=0;fresh && i<POOL;++i) for (unsigned b=0;b<FR_BLOCKS;b+=7) {
        const unsigned char *have=fr_hot_peek(b,pool[i][0],pool[i][1]);
        if (!have) continue;
        if (fr_gsq_slice(&g.gsq,b,pool[i][0],pool[i][1],fresh) || memcmp(have,fresh,fr_gsq_slice_bytes(pool[i][0]))) ++wrong;
        ++compared;
    }
    free(fresh); fr_hot_close();
    const double swapped=swapped_gb()-swap0; const unsigned long miss=atomic_load(&missing);
    printf("%lu uses looked up, %lu served by the cache; routed but missing %lu; %lu slices compared, %lu differ; process larger than its store %lu times; swapped %.3f GB\n",
           atomic_load(&asked_total),atomic_load(&served),miss,compared,wrong,fat,swapped);
    fr_gguf_close(&g);
    const int bad=miss || wrong || fat || !compared || swapped>0.001;
    puts(bad?"hot_policy_elastic: FAIL":"hot_policy_elastic: pass");
    return bad;
}
