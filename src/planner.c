/* Pure capacity gates and communication-aware share water filling: each node's
 * time per call is its measured link time plus its measured compute time. */
#include "planner.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
unsigned planner_burst(unsigned configured,const char *mtp,const char *native,const char *wave) {
    if (mtp && !*mtp) return 1; /* --mtp is omitted, even if a draft flag remains set */
    if (native && native[0]>='3' && native[0]<='8' && !native[1]) {
        unsigned n=(unsigned)(native[0]-'0');
        if (n>=4 && wave && !strcmp(wave,"1")) n=(n+1)/2;
        return n<=FR_BENCH_BURST_MAX?n:configured;
    }
    return configured;
}
int planner_capacity(const fr_capacity_node *nodes,unsigned count,fr_capacity *out) {
    if (!nodes || !out || !count || count>FR_PLAN_NODES) return -1;
    fr_capacity result={.nodes=count};
    for (unsigned i=0;i<count;++i) {
        if (!nodes[i].id || nodes[i].blocks>UINT64_MAX-result.total) return -1;
        for (unsigned j=0;j<i;++j) if (nodes[i].id==nodes[j].id) return -1;
        result.total+=nodes[i].blocks;
    }
    for (unsigned b=0;b<FR_BLOCKS;++b) if (b>=result.total) result.missing|=1u<<b;
    *out=result; return result.missing?1:0;
}
int planner_capacity_format(const fr_capacity *c,char *out,size_t size) {
    if (!c || !out || !c->nodes || c->nodes>FR_PLAN_NODES) return -1;
    char text[512]; unsigned missing=c->total<FR_BLOCKS?(unsigned)(FR_BLOCKS-c->total):0;
    int n=snprintf(text,sizeof(text),"CAPACITY total=%llu required=%u nodes=%u missing=%u\n",
                   (unsigned long long)c->total,FR_BLOCKS,c->nodes,missing);
    if (n<0 || (size_t)n>=sizeof(text)) return -1;
    size_t used=(size_t)n;
    if (missing) {
        n=snprintf(text+used,sizeof(text)-used,"PLAN REFUSED unplaced_blocks=");
        if (n<0 || (size_t)n>=sizeof(text)-used) return -1;
        used+=(size_t)n;
        for (unsigned b=(unsigned)c->total;b<FR_BLOCKS;++b) {
            n=snprintf(text+used,sizeof(text)-used,"%s%u",b==c->total?"":",",b);
            if (n<0 || (size_t)n>=sizeof(text)-used) return -1;
            used+=(size_t)n;
        }
        text[used++]='\n';
    }
    if (used>=size) return -1;
    memcpy(out,text,used); out[used]=0; return (int)used;
}
/* Exact min-max split. A node with s blocks takes t[i][s] = its link time plus
 * its measured compute at s (estimated past the sweep), made nondecreasing in s:
 * a noisy sweep point never makes more blocks look cheaper, so any share up to
 * one that fits a time limit also fits it. The optimum is the smallest measured
 * value T at which the shares fitting under T add up to all twenty blocks; the
 * surplus is then taken from the slowest nodes first, which also lowers the
 * second-slowest time where it can. Ties among split options cannot hide the
 * optimum, as single-block moves could (two nodes equally slowest). */
static int compute_shares(const fr_selection *nodes,const double *link_ns,unsigned count,unsigned *out) {
    if (!nodes || !out || !count || count>FR_PLAN_NODES) return -1;
    if (link_ns) {
        for (unsigned i=0;i<count;++i) {
            if (!isfinite(link_ns[i]) || link_ns[i]<0) return -1;
        }
    }

    /* 1. Build a monotone completion-time curve for each participant. Backend
     * selection already describes resident and measured RAM+SSD candidates.
     * An unused participant costs zero: it sends no reply. All times are ns.
     * These small tables live only for this setup call, never in the layer loop. */
    double time[FR_PLAN_NODES][FR_BLOCKS+1];
    unsigned capacity[FR_PLAN_NODES]={0},total=0;
    for (unsigned i=0;i<count;++i) {
        if (!selection_complete(&nodes[i])) return -1;
        for (unsigned c=0;c<nodes[i].count;++c) {
            if (nodes[i].candidate[c].capacity>capacity[i])
                capacity[i]=nodes[i].candidate[c].capacity;
        }
        if (capacity[i]>FR_BLOCKS) capacity[i]=FR_BLOCKS;
        total+=capacity[i];
        time[i][0]=0;
        for (unsigned b=1;b<=capacity[i];++b) {
            int best=selection_pick(&nodes[i],b);
            if (best<0) return -1;
            double ns=selection_cost(&nodes[i].candidate[best],b)+(link_ns?link_ns[i]:0);
            if (!isfinite(ns) || ns<1 || ns>1e15) return -1; /* no real call takes under 1 ns */
            time[i][b]=b>1 && time[i][b-1]>ns?time[i][b-1]:ns;
        }
    }
    if (total<FR_BLOCKS) return 1;

    /* 2. Try the finite completion times. A limit is feasible if the largest
     * share each node can finish by then covers at least twenty units. The
     * optimum must be one of these times; no iterative hardware probe is needed. */
    double limit=INFINITY;
    for (unsigned j=0;j<count;++j) {
        for (unsigned b=1;b<=capacity[j];++b) {
            double candidate_limit=time[j][b];
            if (candidate_limit>=limit) continue;
            unsigned fit=0;
            for (unsigned i=0;i<count;++i) {
                unsigned share=0;
                while (share<capacity[i] && time[i][share+1]<=candidate_limit) ++share;
                fit+=share;
            }
            if (fit>=FR_BLOCKS) limit=candidate_limit;
        }
    }
    if (!isfinite(limit)) return -1;

    /* 3. Fill under that limit, then remove surplus from the slowest current
     * participant. Strict '>' keeps the existing first-node tie break. */
    unsigned shares[FR_PLAN_NODES]={0},used=0;
    for (unsigned i=0;i<count;++i) {
        while (shares[i]<capacity[i] && time[i][shares[i]+1]<=limit) ++shares[i];
        used+=shares[i];
    }
    for (;used>FR_BLOCKS;--used) {
        unsigned slow=count;
        for (unsigned i=0;i<count;++i) {
            if (shares[i] && (slow==count || time[i][shares[i]]>time[slow][shares[slow]]))
                slow=i;
        }
        --shares[slow];
    }

    /* Publish only a complete plan. Invalid/insufficient input leaves out alone. */
    memcpy(out,shares,count*sizeof(*out));
    return 0;
}

int planner_shares(const fr_selection *nodes,const double *net_ns,unsigned count,unsigned *out) {
    return compute_shares(nodes,net_ns,count,out);
}

int planner_compute_shares(const fr_selection *nodes,unsigned count,unsigned *out) {
    return compute_shares(nodes,NULL,count,out);
}
