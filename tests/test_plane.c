#include "test.h"
#include "../src/plane.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
static int slow_run(const fr_req *r,fr_part *p) {
    struct timespec delay={.tv_nsec=10000000}; nanosleep(&delay,NULL);
    memset(p,0,sizeof(*p)); p->layer=r->layer; p->burst=r->burst;
    for (unsigned t=0;t<r->burst;++t) for (unsigned d=0;d<FR_HIDDEN;++d) p->y[t][d]=r->x[t][d]+2;
    return 0;
}
static const fr_backend slow={.name="deliberate-test-delay",.run=slow_run};
int main(void) {
    test_begin("plane"); fr_assembly a={0}; fr_plane_stats s; fr_plane_stats_init(&s);
    fr_fragment f={.type=1,.session=1,.step=4,.layer=47,.burst=8};
    for (unsigned e=0;e<10;++e) f.expert[e]=(uint16_t)e;
    int complete=0;
    for (int i=31;i>=0;--i) {
        f.fragment=(unsigned)i; f.token=f.fragment/4; f.start=f.fragment%4*640;
        for (unsigned d=0;d<640;++d) f.values[d]=(float)(f.start+d);
        complete+=fr_assemble(&a,&f,&s);
    }
    check(complete==1 && a.bitmap==UINT32_MAX && a.request.x[7][2559]==2559,"out-of-order burst eight REQ assembly");
    check(!fr_assemble(&a,&f,&s) && atomic_load(&s.duplicates)==1,"duplicate REQ ignored");
    a.submitted=1; f.step=3;
    check(!fr_assemble(&a,&f,&s) && atomic_load(&s.late)==1,"old request cannot replace latest assembly");
    f.step=7; fr_assemble(&a,&f,&s);
    check(atomic_load(&s.gaps)==2 && a.bitmap==1,"missing steps counted and bitmap reset");
    fr_accumulator sum={0}; fr_accumulator_begin(&sum,1,4,47,8,2);
    f.type=2; f.step=4;
    for (unsigned node=0;node<2;++node) for (int i=31;i>=0;--i) {
        f.node=node; f.fragment=(unsigned)i; f.token=f.fragment/4; f.start=f.fragment%4*640;
        for (unsigned d=0;d<640;++d) f.values[d]=(float)(node+1);
        fr_accumulator_add(&sum,&f);
    }
    check(!fr_accumulator_missing(&sum) && sum.sum.y[7][2559]==3,"PART ranges accumulated immediately for all nodes");
    check(!fr_accumulator_add(&sum,&f) && sum.duplicates==1 && sum.sum.y[0][0]==3,"duplicate PART never added twice");
    f.step=3; check(!fr_accumulator_add(&sum,&f) && sum.late==1,"late PART discarded");
    f.step=4; f.session=2; check(fr_accumulator_add(&sum,&f)<0 && sum.invalid==1,"cross-session PART rejected");
    fr_connection links={.session=1}; fr_head_plane *h=calloc(1,sizeof(*h));
    fr_req req={.layer=47,.burst=8}; fr_part part;
    for (unsigned t=0;t<8;++t) {
        for (unsigned e=0;e<10;++e) req.expert[t][e]=(uint16_t)e;
        for (unsigned d=0;d<2560;++d) req.x[t][d]=.33333333f;
    }
    check(h && !fr_head_plane_open(h,&links,"239.7.7.7",7702,&slow,1),"local-only session opens with same backend interface");
    int rc=fr_head_plane_run(h,&req,0,1000,&part);
    check(!rc && h->last_local_ns>=10000000 && h->last_total_ns>=h->last_local_ns && !h->last_missing && part.y[7][2559]==fr_unhalf(0x3555)+2,"head contribution is fp16-input aligned, timed, and never cut by deadline");
    struct timespec idle={.tv_nsec=80000000}; nanosleep(&idle,NULL);
    check(!fr_head_plane_run(h,&req,1,1000,&part),"parked compute worker wakes without losing request");
    check(!fr_head_plane_close(h),"compute worker joins cleanly");
    setenv("FRONK_REQUIRE_ALL","1",1);
    check(!fr_head_plane_open(h,&links,"239.7.7.7",7702,&slow,1) && h->exact_remote,
          "strict verification allocates deterministic remote sums");
    check(!fr_head_plane_run(h,&req,2,1000,&part) && part.y[7][2559]==fr_unhalf(0x3555)+2,
          "strict verification retains the local contribution for every row");
    req.burst=1;
    check(!fr_head_plane_run(h,&req,3,1000,&part) && part.y[0][0]==fr_unhalf(0x3555)+2,
          "strict verification resets sums when the burst changes");
    req.expert[0][1]=req.expert[0][0];
    check(fr_head_plane_run(h,&req,4,1000,&part)<0 && !atomic_load(&h->local.state),
          "invalid routing cannot launch early local work");
    req.expert[0][1]=1;
    check(!fr_head_plane_run(h,&req,5,1000,&part),"valid work still runs after rejected routing");
    /* No fanout was opened. Advertising a remote participant now injects a
     * send failure after submission; the local request must be fully drained. */
    links.count=1; part.y[0][0]=123;
    check(fr_head_plane_run(h,&req,6,1000,&part)<0 && !atomic_load(&h->local.state) &&
          part.y[0][0]==123,"failed fanout drains early local work without publishing a partial sum");
    links.count=0;
    check(!fr_head_plane_run(h,&req,7,1000,&part),"worker remains reusable after failed fanout");
    check(!fr_head_plane_close(h) && !h->exact_remote,"strict verification releases its workspace");
    unsetenv("FRONK_REQUIRE_ALL"); free(h);
    return test_end();
}
