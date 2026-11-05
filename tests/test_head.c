/* Placement logic for topology/device changes (C, D) and head-ssd overflow,
 * with fake inventories: cached units stay put, only missing units move. */
#include "test.h"
#include "../src/head.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
static unsigned sent(const fr_plan *p) { unsigned n=0; for (unsigned i=0;i<p->count;++i) n+=(unsigned)__builtin_popcount(p->p[i].mask&~p->p[i].cached); return n; }
static int covers(const fr_plan *p) {
    uint32_t all=p->ssd;
    for (unsigned i=0;i<p->count;++i) { if (all&p->p[i].mask) return 0; all|=p->p[i].mask; }
    return all==(1u<<FR_BLOCKS)-1;
}
static fr_selection curve(unsigned cap,double ns_per_block) {
    fr_selection s={.burst=1}; selection_add(&s,"avx2",cap);
    for (unsigned b=1;b<=FR_SWEEP && b<=cap;++b) selection_record(&s,0,b,ns_per_block*b,ns_per_block*b*1.1,b*FR_WEIGHT_BLOCK_BYTES);
    s.burst=1; return s;
}
int main(void) {
    test_begin("head");
    fr_plan p={.count=3}; /* imac 12, shuttle 3, m1pro 3, head 2 after first run */
    unsigned share[4]={12,3,3,2};
    p.count=4; for (unsigned i=0;i<4;++i) p.p[i].share=share[i];
    check(!head_assign(&p) && covers(&p) && !p.ssd && sent(&p)==20,"first run: every unit sent once, all 20 covered");
    for (unsigned i=0;i<4;++i) p.p[i].cached=p.p[i].mask;
    fr_plan c=p; check(!head_assign(&c) && sent(&c)==0 && covers(&c),"C: same devices on new links: no unit resent");
    fr_plan d=p; d.count=3; d.p[0].share=14; d.p[1].share=3; d.p[2].share=3; /* head removed */
    check(!head_assign(&d) && covers(&d) && sent(&d)==2,"D: participant removed: only its 2 units move");
    fr_plan a=p; a.count=5; a.p[0].share=10; memset(&a.p[4],0,sizeof(a.p[4])); a.p[4].share=2; /* new MacBook */
    check(!head_assign(&a) && covers(&a) && sent(&a)==2 && __builtin_popcount(a.p[4].mask)==2,"D: node added: only its new units are sent");
    for (unsigned i=0;i<4;++i) check((a.p[i].mask&~a.p[i].cached)==0,"D: existing nodes keep cached units");
    fr_plan o={.count=2}; o.p[0].share=12; o.p[1].share=5;
    check(!head_assign(&o) && __builtin_popcount(o.ssd)==3 && covers(&o),"capacity 17/20: 3 units become head-ssd");
    o.p[1].share=8; check(!head_assign(&o) && !o.ssd && covers(&o),"head-ssd disappears at 20/20");
    fr_plan spill={.count=2}; spill.p[0].share=3; spill.p[1].node=FR_PLAN_HEAD; spill.p[1].share=1;
    uint32_t missing=(1u<<0)|(1u<<10)|(1u<<13);
    spill.p[1].cached=((1u<<FR_BLOCKS)-1)&~missing;
    check(!head_assign(&spill) && covers(&spill) && spill.p[0].mask==missing && !(spill.ssd&missing),
          "fresh workers take blocks absent from the head SSD before displacing cached ones");
    fr_plan bad={.count=2}; bad.p[0].share=15; bad.p[1].share=6; check(head_assign(&bad)<0,"over-assignment rejected");
    fr_selection s[3]={curve(12,100000),curve(3,200000),curve(2,150000)}; unsigned out[3],ssd;
    check(!head_shares(s,NULL,3,out,&ssd) && ssd==3 && out[0]+out[1]+out[2]==17,"shares at capacity leave head-ssd remainder");
    s[1]=curve(6,200000); check(!head_shares(s,NULL,3,out,&ssd) && !ssd && out[0]+out[1]+out[2]==20 && out[0]<=12,"water-filling within capacity at 20/20");
    {   char dir[]="/tmp/fronk-netcal-XXXXXX"; double us=0;
        check(mkdtemp(dir)!=NULL,"netcal temp dir");
        check(!fr_netcal_get(dir,0x42,"10.0.0.2",2,&us),"no calibration before the first report");
        fr_netcal_put(dir,0x42,"10.0.0.2",2,810); fr_netcal_put(dir,0x43,"10.0.0.3",2,1150); fr_netcal_put(dir,0x42,"10.0.0.2",2,790);
        check(fr_netcal_get(dir,0x42,"10.0.0.2",2,&us) && us==800,"two measurements: their mean");
        fr_netcal_put(dir,0x42,"10.0.0.2",2,2230);
        check(fr_netcal_get(dir,0x42,"10.0.0.2",2,&us) && us==810,"one disturbed pass does not move the median");
        for (int k=0;k<5;++k) fr_netcal_put(dir,0x42,"10.0.0.2",2,1500);
        check(fr_netcal_get(dir,0x42,"10.0.0.2",2,&us) && us==1500,"five newer measurements replace the old ones");
        check(fr_netcal_get(dir,0x43,"10.0.0.3",2,&us) && us==1150,"other nodes kept");
        check(!fr_netcal_get(dir,0x42,"10.0.0.9",2,&us) && !fr_netcal_get(dir,0x42,"10.0.0.2",4,&us),"new address or burst ignores old measurement");
        char path[64]; snprintf(path,sizeof(path),"%s/netcal-v2.txt",dir); unlink(path); rmdir(dir);
    }
    return test_end();
}
