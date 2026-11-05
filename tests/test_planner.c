#include <math.h>
#include "test.h"
#include <stdint.h>
#include "../src/planner.h"
#include <string.h>
int main(void) {
    test_begin("planner-capacity");
    check(planner_burst(3,"",NULL,NULL)==1 && planner_burst(3,NULL,NULL,NULL)==3,"single-token engine selects its own saved width; ordinary mode retains configured width");
    check(planner_burst(3,"","4","1")==1,"explicitly disabled MTP dominates leftover speculative flags");
    check(planner_burst(3,NULL,"4","1")==2 && planner_burst(3,NULL,"8","1")==4 && planner_burst(3,NULL,"4","0")==4,
          "native wave planner measures the dispatched group, not full draft depth");
    check(planner_burst(3,NULL,"8","0")==3 && planner_burst(3,NULL,"4junk","1")==3,
          "unsupported or malformed widths never pretend to have a measured curve");
    fr_capacity_node nodes[]={{1,2},{2,1},{3,4},{4,10},{5,0}};
    fr_capacity c={0}; char text[512];
    check(planner_capacity(nodes,4,&c)==1 && c.total==17 && c.missing==((1u<<17)|(1u<<18)|(1u<<19)),
          "measured four-node snapshot refuses 17/20 and names all three unplaced blocks");
    check(planner_capacity_format(&c,text,sizeof(text))>0 && !strncmp(text,"CAPACITY total=17 required=20",29) &&
          strstr(text,"PLAN REFUSED unplaced_blocks=17,18,19\n"),"capacity is first, insufficient plan is explicitly refused");
    check(planner_capacity(nodes,5,&c)==1 && c.total==17,"fifth node with no measured capacity cannot fill the shortfall");
    nodes[4].blocks=1;
    check(planner_capacity(nodes,5,&c)==1 && c.total==18 && c.missing==((1u<<18)|(1u<<19)),"one measured fifth-node block still leaves two unplaced");
    nodes[1].blocks=3;
    check(!planner_capacity(nodes,5,&c) && c.total==20 && !c.missing,"M1 three plus fifth-node one fits exactly, only when measured");
    check(planner_capacity_format(&c,text,sizeof(text))>0 && !strstr(text,"REFUSED"),"feasible capacity does not emit a refusal");
    nodes[4].blocks=5;
    check(!planner_capacity(nodes,5,&c) && c.total==24 && !c.missing,"surplus does not demand more than twenty model blocks");
    fr_capacity saved=c; nodes[4].id=nodes[0].id;
    check(planner_capacity(nodes,5,&c)<0 && !memcmp(&c,&saved,sizeof(c)),"persistent identity cannot be counted twice");
    nodes[4].id=5; nodes[4].blocks=UINT64_MAX;
    check(planner_capacity(nodes,5,&c)<0 && !memcmp(&c,&saved,sizeof(c)),"capacity overflow rejected atomically");
    memset(text,'x',sizeof(text));
    check(planner_capacity_format(&c,text,2)<0 && text[0]=='x',"bounded report does not overwrite on failure");
    fr_selection curves[2]={{.burst=1},{.burst=1}};
    for (unsigned i=0;i<2;++i) {
        selection_add(&curves[i],"neon",12);
        for (unsigned b=1;b<=5;++b) selection_record(&curves[i],0,b,(i+1)*100*b,(i+1)*120*b,b*FR_WEIGHT_BLOCK_BYTES);
    }
    unsigned shares[2]={99,99};
    check(!planner_compute_shares(curves,2,shares) && shares[0]==12 && shares[1]==8,"water filling caps faster node and assigns all twenty");
    selection_add(&curves[1],"metal",10);
    for (unsigned b=1;b<=5;++b) selection_record(&curves[1],1,b,50*b,60*b,b*FR_WEIGHT_BLOCK_BYTES);
    check(!planner_compute_shares(curves,2,shares) && shares[0]==10 && shares[1]==10 && selection_pick(&curves[1],10)==1,"assigned share selects faster eligible backend");
    {   /* equal compute, one slow link: the slow link gets fewer blocks and the slowest node's total shrinks */
        fr_selection eq[2]={{.burst=1},{.burst=1}}; unsigned flat[2],aware[2]; double net[2]={0,2000};
        for (unsigned i=0;i<2;++i) { selection_add(&eq[i],"neon",20); for (unsigned b=1;b<=5;++b) selection_record(&eq[i],0,b,100*b,120*b,b*FR_WEIGHT_BLOCK_BYTES); }
        check(!planner_compute_shares(eq,2,flat) && flat[0]==10 && flat[1]==10,"compute-only split of equal nodes is even");
        check(!planner_shares(eq,net,2,aware) && aware[0]+aware[1]==20 && aware[0]>aware[1],"network-aware split favours the faster link");
        double before=fmax(10*100.0,2000+10*100.0),after=fmax(aware[0]*100.0,2000+aware[1]*100.0);
        check(after<before,"network-aware split shortens the slowest node's call");
        net[1]=NAN; check(planner_shares(eq,net,2,aware)<0,"non-finite link time rejected");
    }
    curves[0].candidate[0].capacity=5; curves[1].candidate[0].capacity=5; curves[1].candidate[1].capacity=5;
    shares[0]=shares[1]=99;
    check(planner_compute_shares(curves,2,shares)==1 && shares[0]==99,"insufficient measured capacity refuses atomic assignment");
    curves[0].candidate[0].capacity=20;
    for (unsigned b=0;b<5;++b) curves[0].candidate[0].ns[b]=1e-320;
    check(planner_compute_shares(curves,2,shares)<0 && shares[0]==99,"extreme costs cannot overflow water-fill arithmetic or replace output");
    {   /* exact: the solver's slowest node equals brute force over every split, 2-4 nodes */
        uint32_t seed=12345; int exact=1; unsigned evaluated=0;
        for (unsigned trial=0;trial<300 && exact;++trial) {
            unsigned count=2+trial%3; fr_selection n[4]; double net[4];
            for (unsigned i=0;i<count;++i) {
                seed=seed*1103515245u+12345u; unsigned cap=4+(seed>>16)%17;
                n[i]=(fr_selection){.burst=1}; selection_add(&n[i],"avx2",cap);
                seed=seed*1103515245u+12345u; double per=50+(seed>>16)%400,y=0;
                for (unsigned b=1;b<=5 && b<=cap;++b) { seed=seed*1103515245u+12345u; y+=per*(0.8+((seed>>16)%40)/100.0); selection_record(&n[i],0,b,y,y*1.2,b*FR_WEIGHT_BLOCK_BYTES); }
                seed=seed*1103515245u+12345u; net[i]=(seed>>16)%1500;
            }
            unsigned got[4]; if (planner_shares(n,net,count,got)) continue; /* capacity below 20 */
            ++evaluated;
            double t[4][FR_BLOCKS+1]={{0}};
            for (unsigned i=0;i<count;++i) for (unsigned b=1;b<=n[i].candidate[0].capacity;++b) t[i][b]=net[i]+selection_cost(&n[i].candidate[0],b);
            double mine=0,best=INFINITY; unsigned sum=0;
            for (unsigned i=0;i<count;++i) { if (got[i]) mine=fmax(mine,t[i][got[i]]); sum+=got[i]; }
            unsigned c[4]={0};
            for (c[0]=0;c[0]<=n[0].candidate[0].capacity;++c[0]) for (c[1]=0;c[1]<=n[1].candidate[0].capacity;++c[1])
            for (c[2]=0;c[2]<=(count>2?n[2].candidate[0].capacity:0);++c[2]) {
                int r=FR_BLOCKS-(int)(c[0]+c[1]+c[2]); if (count<4 && r) continue;
                if (count==4 && (r<0 || r>(int)n[3].candidate[0].capacity)) continue;
                if (count==4) c[3]=(unsigned)r;
                double worst=0; for (unsigned i=0;i<count;++i) if (c[i]) worst=fmax(worst,t[i][c[i]]);
                if (worst<best) best=worst;
            }
            if (sum!=FR_BLOCKS || fabs(mine-best)>1e-9*best) exact=0;
        }
        check(exact && evaluated>=100,"split equals the brute-force optimum on 300 random 2-4 node clusters");
        /* two nodes tie as slowest: one block moved at a time cannot improve, both moved can */
        fr_selection tie[3]; double tnet[3]={0,0,0}; unsigned ts[3];
        for (unsigned i=0;i<3;++i) { tie[i]=(fr_selection){.burst=1}; selection_add(&tie[i],"avx2",i<2?10:20);
            for (unsigned b=1;b<=5;++b) selection_record(&tie[i],0,b,(i<2?100.0:150.0)*b,(i<2?120.0:180.0)*b,b*FR_WEIGHT_BLOCK_BYTES); }
        check(!planner_shares(tie,tnet,3,ts) && ts[0]+ts[1]+ts[2]==20 && fmax(fmax(ts[0]*100.0,ts[1]*100.0),ts[2]*150.0)==800,"tied slowest nodes still reach the optimum (800 ns)");
    }
    {
        fr_selection modes[2]={{.burst=3},{.burst=3}}; double links[2]={400,0}; unsigned split[2];
        selection_add(&modes[0],"avx2",18); selection_add(&modes[1],"metal",2);
        for (unsigned b=1;b<=5;++b) selection_record(&modes[0],0,b,b*100,b*120,b*FR_NATIVE_BLOCK_BYTES);
        for (unsigned b=1;b<=2;++b) selection_record(&modes[1],0,b,b*50,b*60,b*FR_NATIVE_BLOCK_BYTES);
        double cold[2]={1700,2500},warm[2]={100,150},tail[2]={1900,3000};
        check(!selection_hybrid(&modes[1],2,2,FR_SSD_DEFAULT_CACHE,cold,warm,tail) &&
              !planner_shares(modes,links,2,split) && split[0]==17 && split[1]==3,
              "head SSD hidden beneath remote communication plus compute relieves slow node");
        links[1]=600;
        check(!planner_shares(modes,links,2,split) && split[0]==18 && split[1]==2,
              "same SSD path on remote Metal node includes its network round trip");
        modes[1].candidate[1].cold_ns[0]=3000; modes[1].candidate[1].cold_ns[1]=4000;
        modes[1].candidate[1].tail_ns[0]=3500; modes[1].candidate[1].tail_ns[1]=4500; links[1]=0;
        check(!planner_shares(modes,links,2,split) && split[1]==2,
              "excellent warm cache does not admit a slow cold SSD configuration");
    }
    return test_end();
}
