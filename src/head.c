/* Head startup and session lifetime.
 *
 * Read head_main, then the phase loop it calls. The order is connection,
 * measurement/plan, preparation/transfer, session load, numerical check, serve.
 * head_demo_main follows the same placement and data path with synthetic units,
 * so users can test their machines before downloading the model.
 *
 * planner.c chooses share counts; head_assign turns counts into disjoint unit
 * identities while preserving cached files. push_units transfers missing bytes.
 * plane.c performs each layer call; serve.c joins it to the dense engine/chat.
 * Setup may allocate and do I/O. Neither belongs in the per-layer data path.
 * The cleanup path detaches serving and joins workers before unloading weights.
 * Read PROJECT.md for the design and docs/llm/DEBUG.md for failure diagnosis.
 */
#include "head.h"
#ifdef FR_AVX2_COMPUTE
#include "avx2.h"
#endif
#include "connection.h"
#include "discovery.h"
#include "control.h"
#include "identity.h"
#include "plane.h"
#include "prep.h"
#include "units.h"
#include "serve.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#define GB 1e9

/* Setup-only capability check: never add negotiation to an expert call.
 * A legacy/unknown node must not silently process a partially coalesced tree. */
static int require_ranked(const fr_connection *links,const fr_backend *local,uint32_t active) {
    const char *mode=getenv("FRONK_MTP_COALESCE");
    if (!mode || strcmp(mode,"2")) return 0;
#ifndef FR_EXPERT_RANKED
    (void)links; (void)local; (void)active;
    fprintf(stderr,"fronk: partial MTP reuse requires a ranked expert build on the head\n");
    return -1;
#else
    if (local && strcmp(local->name,"metal") && strcmp(local->name,"metal-ssd") && strcmp(local->name,"avx2")) return -1;
    for (unsigned i=0;i<links->count;++i) if (active&(1u<<i)) {
        char reply[RESP_MAX_MESSAGE]; resp_value values[RESP_MAX_VALUES];
        if (control_exchange(links->links[i].fd,"EXPERT-ORDER\r\n",14,reply,values,net_now_ns()/1000000+3000) ||
            values[0].type!=RESP_STRING || values[0].len!=9 || memcmp(values[0].data,"ranked-v1",9)) {
            fprintf(stderr,"fronk: %s does not confirm ranked-v1 expert reduction; partial MTP reuse refused\n",links->links[i].identity.name);
            return -1;
        }
    }
    return 0;
#endif
}

static void assign_available(fr_plan *plan,unsigned i,uint32_t available,uint32_t *taken) {
    available &= ~*taken;
    for (unsigned b=0;b<FR_BLOCKS && (unsigned)__builtin_popcount(plan->p[i].mask)<plan->p[i].share;++b) {
        uint32_t bit=1u<<b;
        if (available&bit) { plan->p[i].mask|=bit; *taken|=bit; }
    }
}
int head_assign(fr_plan *plan) {
    uint32_t taken=0,head_cache=0; unsigned total=0;
    for (unsigned i=0;i<plan->count;++i) {
        plan->p[i].mask=0; total+=plan->p[i].share;
        if (plan->p[i].node==FR_PLAN_HEAD) head_cache=plan->p[i].cached;
    }
    if (total>FR_BLOCKS) return -1;
    for (unsigned i=0;i<plan->count;++i) assign_available(plan,i,plan->p[i].cached,&taken);
    /* If the head must page, give unfilled remote slots the blocks missing
     * from its disk first. This avoids rebuilding already available SSD units. */
    if (total<FR_BLOCKS && head_cache) for (unsigned i=0;i<plan->count;++i)
        if (plan->p[i].node!=FR_PLAN_HEAD) assign_available(plan,i,~head_cache,&taken);
    for (unsigned i=0;i<plan->count;++i) assign_available(plan,i,UINT32_MAX,&taken);
    plan->ssd=((1u<<FR_BLOCKS)-1)&~taken; return 0;
}
static void netcal_path(const char *dir,char *path,size_t n) { snprintf(path,n,"%s/netcal-v2.txt",dir); }
/* One line per node address and burst: id ip burst v1..v5, newest last. The
 * planner uses the median, so one disturbed REPORT (head paging, a busy node)
 * cannot reshape the next plan on its own. */
#define NETCAL_KEEP 5
static int netcal_line(const char *line,unsigned long long *id,char ip[24],unsigned *burst,double *v) {
    int used=0,n=0; if (sscanf(line,"%llx %23s %u%n",id,ip,burst,&used)!=3) return 0;
    for (const char *p=line+used;n<NETCAL_KEEP;) { char *end; double x=strtod(p,&end); if (end==p) break; if (x>0 && x<1e6) v[n++]=x; p=end; }
    return n;
}
int fr_netcal_get(const char *dir,uint64_t id,const char *ip,unsigned burst,double *us) {
    char path[512],line[256],a[24]; unsigned long long i; unsigned b; double v[NETCAL_KEEP]; int found=0; netcal_path(dir,path,sizeof(path));
    FILE *f=fopen(path,"r"); if (!f) return 0;
    while (fgets(line,sizeof(line),f)) { int n=netcal_line(line,&i,a,&b,v);
        if (n && i==id && b==burst && !strcmp(a,ip)) {
            for (int x=1;x<n;++x) for (int y=x;y>0 && v[y-1]>v[y];--y) { double t=v[y]; v[y]=v[y-1]; v[y-1]=t; }
            *us=n%2?v[n/2]:(v[n/2-1]+v[n/2])/2; found=1; } }
    fclose(f); return found;
}
void fr_netcal_put(const char *dir,uint64_t id,const char *ip,unsigned burst,double us) {
    char path[512],tmp[520],line[256],a[24]; unsigned long long i; unsigned b; double v[NETCAL_KEEP],keep[NETCAL_KEEP]; int kept=0; netcal_path(dir,path,sizeof(path));
    snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE *in=fopen(path,"r"),*out=fopen(tmp,"w"); if (!out) { if (in) fclose(in); return; }
    if (in) { while (fgets(line,sizeof(line),in)) { int n=netcal_line(line,&i,a,&b,v); if (!n) continue;
        if (i==id && b==burst && !strcmp(a,ip)) { memcpy(keep,v,sizeof(double)*(size_t)n); kept=n; } else fputs(line,out); } fclose(in); }
    if (kept==NETCAL_KEEP) { memmove(keep,keep+1,sizeof(double)*(NETCAL_KEEP-1)); --kept; }
    keep[kept++]=us;
    fprintf(out,"%016llx %s %u",(unsigned long long)id,ip,burst);
    for (int k=0;k<kept;++k) fprintf(out," %.1f",keep[k]);
    fputc('\n',out);
    if (!fclose(out)) rename(tmp,path);
}
int head_shares(const fr_selection *curves,const double *net_ns,unsigned count,unsigned *shares,unsigned *ssd) {
    int rc=count?planner_shares(curves,net_ns,count,shares):1; *ssd=0;
    if (rc!=1) return rc;
    unsigned total=0;
    for (unsigned i=0;i<count;++i) {
        shares[i]=0;
        for (unsigned c=0;c<curves[i].count;++c) if (curves[i].candidate[c].capacity>shares[i]) shares[i]=curves[i].candidate[c].capacity;
        if (shares[i]>FR_BLOCKS-total) shares[i]=FR_BLOCKS-total;
        total+=shares[i];
    }
    *ssd=FR_BLOCKS-total; return 0;
}
static int exchange(int fd,const char *text,char *reply,resp_value *v,int ms) {
    const char *e=control_exchange(fd,text,strlen(text),reply,v,net_now_ns()/1000000+ms);
    if (e) fprintf(stderr,"fronk: %s\n",e);
    return e || v[0].type==RESP_ERROR ? -1:0;
}
static uint32_t remote_cached(int fd,char hex[FR_BLOCKS][65]) {
    char reply[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES]; uint32_t mask=0;
    if (exchange(fd,"UNITS\r\n",reply,v,10000) || v[0].type!=RESP_BULK) return 0;
    for (const char *p=v[0].data,*end=v[0].data+v[0].len;p+68<=end;p+=68) {
        unsigned b=(unsigned)((p[0]-'0')*10+(p[1]-'0'));
        if (b<FR_BLOCKS && !memcmp(p+3,hex[b],64)) mask|=1u<<b;
    }
    return mask;
}
static void ranges(uint32_t mask,char *out,size_t size) {
    size_t n=0; out[0]=0;
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask>>b&1) {
        unsigned e=b; while (e+1<FR_BLOCKS && (mask>>(e+1)&1)) ++e;
        n+=(size_t)snprintf(out+n,size-n,"%s%u-%u",n?",":"",b*FR_BLOCK_NEURONS,(e+1)*FR_BLOCK_NEURONS-1); b=e;
        if (n>=size) break;
    }
    if (!n) snprintf(out,size,"-");
}
/* Transfer: one thread per node streams missing units straight from the GGUF. */
typedef struct { int fd; const fr_gguf *g; uint32_t mask; char (*hex)[65]; const char *name; int rc; double seconds; uint64_t bytes; char error[200]; } push;
static int socket_sink(void *ctx,const void *data,size_t n) {
    push *p=ctx; const char *c=data;
    while (n) {
        if (net_wait(p->fd,1,30000)!=1) return -1;
        ssize_t w=send(p->fd,c,n,0);
        if (w<0 && (errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR)) continue;
        if (w<=0) return -1;
        c+=w; n-=(size_t)w; p->bytes+=(uint64_t)w;
    }
    return 0;
}
/* SESSION START loads a node's units; every node does it at the same time. */
typedef struct { int fd; char text[100]; const char *name; unsigned units; int rc; double seconds; char error[120]; } starter;
static void *start_session(void *arg) {
    starter *j=arg; char reply[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES]; int64_t t=net_now_ns();
    j->rc=exchange(j->fd,j->text,reply,v,600000); j->seconds=(net_now_ns()-t)/1e9;
    if (j->rc && v[0].type==RESP_ERROR) snprintf(j->error,sizeof(j->error),"%.*s",(int)(v[0].len<119?v[0].len:119),v[0].data);
    return NULL;
}
static void *push_units(void *arg) {
    push *p=arg; char reply[RESP_MAX_MESSAGE],text[100]; resp_value v[RESP_MAX_VALUES]={0}; int64_t start=net_now_ns();
    for (unsigned b=0;!p->rc && b<FR_BLOCKS;++b) if (p->mask>>b&1) {
        snprintf(text,sizeof(text),"RECV %u %s\r\n",b,p->hex[b]);
        if (exchange(p->fd,text,reply,v,10000) || fr_unit_stream(p->g,b,socket_sink,p) ||
            control_exchange(p->fd,"",0,reply,v,net_now_ns()/1000000+120000) || v[0].type==RESP_ERROR) {
            p->rc=-1;
            if (v[0].type==RESP_ERROR) snprintf(p->error,sizeof(p->error),"%.*s",(int)(v[0].len<199?v[0].len:199),v[0].data);
            else snprintf(p->error,sizeof(p->error),"unit %02u: connection, source read or transfer failed; check this node's log and network",b);
        }
        else { printf("FRONK unit %02u -> %s verified\n",b,p->name); fflush(stdout); }
    }
    p->seconds=(net_now_ns()-start)/1e9; return NULL;
}
/* Scalar oracle straight from GGUF bytes: the whole MoE for one token. */
static int oracle(const fr_gguf *g,const fr_req *r,unsigned t,float *y) {
    static float h[FR_INTERMEDIATE]; double sum;
    memset(y,0,FR_HIDDEN*sizeof(float));
    for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) {
        unsigned shared=j==FR_ACTIVE_ROUTED,e=shared?0:r->expert[t][j],type=shared?FR_Q8_0:FR_Q4_K,down=shared?FR_Q8_0:FR_MXFP4;
        size_t row=shared?FR_HIDDEN/32*34:FR_HIDDEN/256*144,drow=shared?FR_INTERMEDIATE/32*34:FR_INTERMEDIATE/32*17;
        const unsigned char *gate=g->map+(shared?g->sgate:g->gate)[r->layer]+(size_t)e*FR_INTERMEDIATE*row;
        const unsigned char *up=g->map+(shared?g->sup:g->up)[r->layer]+(size_t)e*FR_INTERMEDIATE*row;
        const unsigned char *dn=g->map+(shared?g->sdown:g->down)[r->layer]+(size_t)e*FR_HIDDEN*drow;
        for (unsigned n=0;n<FR_INTERMEDIATE;++n) {
            double a,u;
            if (fr_native_dot(gate+n*row,type,r->x[t],FR_HIDDEN,&a) || fr_native_dot(up+n*row,type,r->x[t],FR_HIDDEN,&u)) return -1;
            h[n]=(float)(a/(1+exp(-a))*u);
        }
        for (unsigned d=0;d<FR_HIDDEN;++d) { if (fr_native_dot(dn+d*drow,down,h,FR_INTERMEDIATE,&sum)) return -1; y[d]+=r->gate[t][j]*(float)sum; }
    }
    return 0;
}
static int head_selftest(fr_head_plane *h,const fr_gguf *g,FILE *out) {
    static fr_req req; static fr_part got; static float want[FR_HIDDEN]; double worst=0;
    req.layer=7; req.burst=2;
    for (unsigned t=0;t<2;++t) {
        for (unsigned j=0;j<FR_ACTIVE_ROUTED;++j) req.expert[t][j]=(uint16_t)((t*97+j*53+11)%FR_ROUTED_EXPERTS);
        for (unsigned j=0;j<FR_ACTIVE_EXPERTS;++j) req.gate[t][j]=j==FR_ACTIVE_ROUTED?.7f:.1f;
        for (unsigned d=0;d<FR_HIDDEN;++d) { uint16_t half; fr_half(sinf((float)(d*7+t*13))*.5f,&half); req.x[t][d]=fr_unhalf(half); }
    }
    /* Warm-up: one call per layer touches every node's units and Metal buffers
     * once, without a deadline; cold, the M1 Pro missed ~150 deadlines in the test pass. */
    for (unsigned layer=0;layer<FR_LAYERS;++layer) {
        req.layer=layer;
        if (fr_head_plane_run(h,&req,1+layer,60000000000ull,&got) || h->last_missing) {
            fprintf(out,"SELFTEST warm-up failed at layer %u; check compute and replies on",layer);
            for (unsigned n=0;n<h->links->count;++n) if (h->last_missing>>n&1) fprintf(out," %s",h->links->links[n].identity.name);
            fputc('\n',out); return -1;
        }
    }
    req.layer=7;
    for (unsigned attempt=1;;++attempt) {
        int rc=fr_head_plane_run(h,&req,FR_LAYERS+attempt,5000000000ull,&got);
        if (!rc && !h->last_missing) break;
        fprintf(out,"SELFTEST %s %u: no answer from",attempt<3?"retry":"failed after",attempt);
        for (unsigned n=0;n<h->links->count;++n) if (h->last_missing>>n&1) fprintf(out," %s",h->links->links[n].identity.name);
        fprintf(out,"%s\n",rc && !h->last_missing?" (head plane error)":"");
        if (attempt==3) return -1;
    }
    for (unsigned t=0;t<2;++t) {
        fr_error e; if (oracle(g,&req,t,want) || fr_compare(want,got.y[t],FR_HIDDEN,&e) || !fr_conforms(&e,&fr_backend_tolerance)) {
            fprintf(out,"SELFTEST mismatch token=%u l2=%.3g\n",t,e.relative_l2); return -1; }
        if (e.relative_l2>worst) worst=e.relative_l2;
    }
    fprintf(out,"SELFTEST all 20 blocks vs GGUF scalar oracle: relative_l2=%.3g pass\n",worst); return 0;
}
static char ask(const char *prompt) {
    const char *auto_answer=getenv("FRONK_ANSWER");
    printf("%s",prompt); fflush(stdout);
    if (auto_answer && *auto_answer) { printf("%c\n",*auto_answer); return *auto_answer; }
    int c; do c=getchar(); while (c==' ' || c=='\n');
    for (int d=c;d!='\n' && d!=EOF;d=getchar()) {}
    return c==EOF?'q':(char)c;
}
typedef struct {
    fr_connection *links; fr_discovery d; fr_plan plan; fr_head_plane *plane; fr_units local;
    const fr_backend *backend; int sessions,plane_open;
} cycle;
static void teardown(cycle *c) {
    char reply[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES];
    fr_serve_detach();
    if (c->plane_open) fr_head_plane_close(c->plane);
    c->plane_open=0;
    for (unsigned i=0;c->sessions && c->links && i<c->links->count;++i) (void)exchange(c->links->links[i].fd,"SESSION STOP\r\n",reply,v,10000);
    c->sessions=0; fr_units_close(&c->local);
    if (c->links) { connection_close(c->links); free(c->links); c->links=NULL; }
}
/* fronk demo (head_demo_main): the same cycle on synthetic units, without the
 * model, ds4 or any transfer; demo_passes is how many decode passes it runs. */
static unsigned demo_passes;
/* ds4's memory outside its mapped model at 32K context (KV 1.04 + expert cache
 * 1.29 + buffers 0.36 GiB, BENCH run 64): a demo head has no ds4 loaded, so its
 * free memory is taken that much lower to plan as the real head would. */
#define DEMO_DS4_RESIDENT 2.9e9
/* One CONNECTION → REPORT pass. 1 = re-plan requested, 0 = quit, -1 = error. */
static int fail(int line) { fprintf(stderr,"fronk: cycle step at head.c:%d failed\n",line); return -1; }
static int run_cycle(const fr_config *cfg,const fr_gguf *g,char hex[FR_BLOCKS][65],cycle *c) {
    unsigned configured_burst=cfg->burst;
    fr_config execution=*cfg;
    if (!demo_passes && !strcmp(cfg->ds4,"builtin")) {
        execution.burst=planner_burst(cfg->burst,getenv("FRONK_DS4_MTP"),getenv("FRONK_NATIVE_MTP"),getenv("FRONK_NATIVE_WAVE"));
        cfg=&execution;
    }
    memset(&c->d,0,sizeof(c->d)); memset(&c->plan,0,sizeof(c->plan));
    const char *r=getenv("FRONK_BENCH_REFRESH"); int refresh=r && *r=='1'; int64_t t0=net_now_ns();
    if (!(c->links=calloc(1,sizeof(*c->links))) || connection_run(cfg,c->links,stdout)) return fail(__LINE__);
    if (execution.burst!=configured_burst) {
        /* Changing width on a legacy worker can trigger minutes of full
         * sweeps. Negotiate once at setup, never on the inference path. */
        for (unsigned i=0;i<c->links->count;++i) {
            char reply[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES];
            static const char command[]="BENCH-PROFILE\r\n",version[]="quick-widths-v1";
            if (control_exchange(c->links->links[i].fd,command,sizeof(command)-1,reply,v,net_now_ns()/1000000+3000)) return fail(__LINE__);
            if (v[0].type!=RESP_STRING || v[0].len!=sizeof(version)-1 || memcmp(v[0].data,version,sizeof(version)-1)) {
                printf("PLAN keeps configured width=%u: %s uses legacy calibration; update its installer to enable quick width changes\n",configured_burst,c->links->links[i].identity.name);
                execution.burst=configured_burst; break;
            }
        }
        if (execution.burst!=configured_burst) printf("PLAN execution width=%u (configured %u); use saved matching compute/network profiles\n",execution.burst,configured_burst);
    }
    printf("TIMING CONNECTION %.1f s\n",(net_now_ns()-t0)/1e9); t0=net_now_ns();
    if (!demo_passes && cfg->ds4[0] && fr_serve_wait_ds4(1)) return fail(__LINE__);
    if (discovery_run(c->links,&c->d,stdout)) return fail(__LINE__);
    c->d.burst=cfg->burst;
    /* The nodes sweep while the head measures itself: DISCOVERY lasts as long
     * as the slowest machine, not the sum. */
    if (discovery_bench_start(c->links,&c->d,stdout,refresh)) return fail(__LINE__);
    fr_hwinfo head;
    /* Counted like a node's: the head's own blocks from the last run are file
     * cache, not used memory (free+inactive gave 2 blocks, then 0; BENCH run 58). */
    if (hw_probe(&head) || hw_node_available(&head) || identity_load(cfg->state_dir,&head.id)) return fail(__LINE__);
    if (demo_passes && strcmp(cfg->ds4,"tools/gpu-head/run.sh")) head.available=head.available>DEMO_DS4_RESIDENT?head.available-(uint64_t)DEMO_DS4_RESIDENT:0;
    head.margin=cfg->mem_margin; head.head_reserve=cfg->head_reserve; head.head_fixed=cfg->head_blocks;
    if (hw_finalize(&head) || selection_sweep_local(&head,cfg->burst,1,&c->d.head_selection,stdout,cfg->state_dir,refresh,NULL)) return fail(__LINE__);
    if (cfg->head_blocks && c->d.head_selection.count && c->d.head_selection.candidate[0].capacity<cfg->head_blocks)
        printf("PLAN this Mac: head-blocks %u asked, memory allows %u (%.1f GB not pinned with ds4 loaded, %.1f GB kept for ds4)\n",cfg->head_blocks,
               c->d.head_selection.candidate[0].capacity,head.available/1e9,head.head_reserve/1e9);
    if (discovery_bench_collect(c->links,&c->d,stdout)) return fail(__LINE__);
    if (discovery_save(cfg,c->links,&c->d)) return fail(__LINE__);
    printf("TIMING DISCOVERY %.1f s (BENCH %s)\n",(net_now_ns()-t0)/1e9,refresh?"refresh":"cached where keys match");
    /* Participants: eligible nodes, then the head. */
    fr_selection curves[FR_PLAN_NODES]; unsigned shares[FR_PLAN_NODES],ssd; fr_plan *p=&c->plan;
    /* Each node's link time per call: the network time REPORT measured last time on
     * this address and burst, else the round trip of a request-sized probe. */
    double net[FR_PLAN_NODES]={0}; char netsrc[FR_PLAN_NODES]={0};
    for (unsigned i=0;i<c->d.count;++i) if (c->d.nodes[i].id!=head.id && c->d.selection[i].count) {
        for (unsigned l=0;l<c->links->count;++l) if (c->links->links[l].identity.id==c->d.nodes[i].id) {
            double us; const fr_link *k=&c->links->links[l];
            if (fr_netcal_get(cfg->state_dir,k->identity.id,k->ip,cfg->burst,&us)) { net[p->count]=us*1e3; netsrc[p->count]='m'; }
            else { net[p->count]=(double)c->links->paths[k->path].percentile[FR_PATH_RTT_KIND][0]; netsrc[p->count]='p'; } }
        curves[p->count]=c->d.selection[i]; p->p[p->count++].node=i; }
    curves[p->count]=c->d.head_selection; p->p[p->count++].node=FR_PLAN_HEAD;
    if (head_shares(curves,net,p->count,shares,&ssd)) return fail(__LINE__);

    for (unsigned i=0;i<p->count;++i) {
        p->p[i].share=shares[i];
        for (unsigned k=0;k<curves[i].count;++k) if (curves[i].candidate[k].capacity>p->p[i].cap) p->p[i].cap=curves[i].candidate[k].capacity;
        if (demo_passes) continue; /* nothing cached matters: every unit is synthetic */
        if (p->p[i].node==FR_PLAN_HEAD) { for (unsigned b=0;b<FR_BLOCKS;++b) { char have[65]; if (!fr_units_cached(cfg->state_dir,b,have) && !memcmp(have,hex[b],64)) p->p[i].cached|=1u<<b; } }
        else for (unsigned l=0;l<c->links->count;++l) if (c->links->links[l].identity.id==c->d.nodes[p->p[i].node].id) p->p[i].cached=remote_cached(c->links->links[l].fd,hex);
    }
    if (head_assign(p)) return fail(__LINE__);
    unsigned hybrid_blocks=0;
    for (unsigned i=0;i<p->count;++i) if (p->p[i].share) {
        int best=selection_pick(&curves[i],p->p[i].share); if (best<0) return fail(__LINE__);
        const fr_candidate *candidate=&curves[i].candidate[best];
        p->p[i].paged=selection_paged(candidate,p->p[i].mask); p->p[i].cache_bytes=candidate->cache_bytes;
        hybrid_blocks+=(unsigned)__builtin_popcount(p->p[i].paged);
    }
    /* The local PLAN participant is the expert worker. It coexists with the
     * dense engine; on a discrete GPU these use separate memory pools. */
    if (!strcmp(cfg->ds4,"tools/gpu-head/run.sh"))
        puts("PLAN dense head: llama.cpp Vulkan GPU; the head row below is the local CPU worker using system RAM");
    else if (!strcmp(cfg->ds4,"builtin"))
        puts("PLAN dense head: ds4 Metal; the head row below is the local expert worker");
    /* PLAN */
    uint64_t total=head.total,usable=head.cap; unsigned capacity=p->p[p->count-1].cap;
    for (unsigned i=0;i+1<p->count;++i) { total+=c->d.nodes[p->p[i].node].total; usable+=c->d.nodes[p->p[i].node].cap; capacity+=p->p[i].cap; }
    printf("\nPLAN nodes=%u RAM total=%.1f GB usable=%.1f GB weights=%.1f GB (%u units of %.2f GB)\n",p->count,total/GB,usable/GB,
           FR_BLOCKS*(double)FR_NATIVE_BLOCK_BYTES/GB,FR_BLOCKS,FR_NATIVE_BLOCK_BYTES/GB);
    for (unsigned i=0;i<p->count;++i) {
        char r[160]; ranges(p->p[i].mask,r,sizeof(r));
        const char *name="head",*link="local",*via=""; unsigned node=p->p[i].node;
        if (node!=FR_PLAN_HEAD) for (unsigned l=0;l<c->links->count;++l) if (c->links->links[l].identity.id==c->d.nodes[node].id) {
            name=c->links->links[l].identity.name; link=interface_type(c->links->links[l].via.type); via=c->links->links[l].via.name; }
        int best=selection_pick(&curves[i],p->p[i].share?p->p[i].share:1);
        printf("  %-10s link=%-11s %-7s neurons %-22s blocks=%2u cap=%2u %.1f GB backend=%s cached=%u net=%.0f us%s\n",name,link,via,r,p->p[i].share,p->p[i].cap,
               p->p[i].share*(double)FR_NATIVE_BLOCK_BYTES/GB,best>=0?curves[i].candidate[best].name:"-",(unsigned)__builtin_popcount(p->p[i].mask&p->p[i].cached),net[i]/1e3,netsrc[i]=='m'?" (measured)":netsrc[i]=='p'?" (probe)":"");
        if (p->p[i].paged) {
            const fr_candidate *candidate=&curves[i].candidate[best]; unsigned n=(unsigned)__builtin_popcount(p->p[i].paged);
            printf("  HYBRID node=%s RAM=%u SSD=%u cache=%.0f MiB burst=%u compute+cold_io=%.0f us warm=%.0f us network=%.0f us predicted_reply=%.0f us ssd_mask=%u (saved calibration)\n",name,
                p->p[i].share-n,n,candidate->cache_bytes/1048576.0,curves[i].burst,candidate->cold_ns[n-1]/1e3,
                candidate->warm_ns[n-1]/1e3,net[i]/1e3,(candidate->cold_ns[n-1]+net[i])/1e3,p->p[i].paged);
        }
    }
    if (ssd) {
        char r[160]; ranges(p->ssd,r,sizeof(r));
        double rate=p->cache_read_bps=fr_cache_read_rate(cfg->state_dir);
        printf("  %-10s link=%-11s %-7s neurons %-22s blocks=%2u", "head-ssd","ssd","",r,ssd);
        if (rate>0) printf(" SSD %.2f GB/s uncached -> ~%.0f us/barrier at burst 1",rate/1e9,ssd*(double)FR_ACTIVE_ROUTED*FR_NATIVE_SLICE_BYTES/rate*1e6);
        puts(demo_passes?" (paging omitted from synthetic timing)":" (mmap, read-only)");
    }
    if (hybrid_blocks) printf("CAPACITY resident=%u/20 blocks — hybrid SSD=%u/20 planned=%u/20%s\n",FR_BLOCKS-hybrid_blocks-ssd,hybrid_blocks,FR_BLOCKS-ssd,ssd?"; legacy head-ssd covers the rest":" — fits");
    else printf("CAPACITY resident=%u/20 blocks%s\n",capacity>FR_BLOCKS?FR_BLOCKS:capacity,ssd?" — head-ssd covers the rest":" — fits");
    char answer=demo_passes?'p':ask("Proceed with this configuration? [p]roceed / [m]odified: ");
    if (answer=='m') return 1;
    if (answer!='p') return 0;
    t0=net_now_ns();
    /* FRONK: configure, transfer missing units, start sessions. */
    unsigned remote[FR_CONNECTION_NODES]={0},head_share=0;
    for (unsigned i=0;i<p->count;++i) { if (p->p[i].node==FR_PLAN_HEAD) head_share=p->p[i].share; else remote[p->p[i].node]=p->p[i].share; }
    if (discovery_configure(cfg,c->links,&c->d,head_share,remote,ssd,stdout)) return fail(__LINE__);
    push jobs[FR_CONNECTION_NODES]; pthread_t threads[FR_CONNECTION_NODES]; unsigned started=0; int failed=0;
    for (unsigned i=0;!demo_passes && i<p->count;++i) if (p->p[i].node!=FR_PLAN_HEAD && (p->p[i].mask&~p->p[i].cached)) {
        for (unsigned l=0;l<c->links->count;++l) if (c->links->links[l].identity.id==c->d.nodes[p->p[i].node].id) {
            jobs[started]=(push){.fd=c->links->links[l].fd,.g=g,.mask=p->p[i].mask&~p->p[i].cached,.hex=hex,.name=c->links->links[l].identity.name};
            if (!pthread_create(&threads[started],NULL,push_units,&jobs[started])) ++started; else failed=1;
        }
    }
    uint32_t local=p->p[p->count-1].mask|(demo_passes?0:p->ssd);
    for (unsigned b=0;!demo_passes && b<FR_BLOCKS;++b) if (local>>b&1) {
        printf("FRONK preparing local unit %02u (cached units are reused)\n",b);
        if (fr_unit_write(g,b,cfg->state_dir,hex[b])) { fprintf(stderr,"fronk: cannot write head unit %u\n",b); failed=1; break; }
    }
    for (unsigned i=0;i<started;++i) {
        pthread_join(threads[i],NULL); if (jobs[i].rc) { fprintf(stderr,"fronk: sending units to %s failed: %s\n",jobs[i].name,jobs[i].error); failed=1; }
        if (jobs[i].bytes) printf("FRONK sent %s %.2f GB in %.1f s (%.0f MB/s)\n",jobs[i].name,jobs[i].bytes/GB,jobs[i].seconds,jobs[i].bytes/1e6/jobs[i].seconds);
    }
    if (failed) return fail(__LINE__);
    puts(demo_passes?"DEMO synthetic units: every machine reads its blocks' bytes per call, as with the model":"FRONK cache: only missing units were sent; all units SHA-256 verified");
    unsigned local_blocks=(unsigned)__builtin_popcount(local);
    unsigned resident=p->p[p->count-1].share;
    uint32_t local_paged=p->p[p->count-1].paged|(demo_passes?0:p->ssd);
    uint64_t local_cache=p->p[p->count-1].cache_bytes?p->p[p->count-1].cache_bytes:FR_SSD_DEFAULT_CACHE;
    int best=selection_pick(&c->d.head_selection,resident?resident:1);
    c->backend=best>=0?fr_backend_find(c->d.head_selection.candidate[best].name):NULL;
    /* The resident sweep keeps only its winner. An SSD spill needs a CPU
     * reader even when Metal won or no resident block fits on the head. */
    if (!demo_passes && p->ssd && (!c->backend || !strncmp(c->backend->name,"metal",5)))
        c->backend=fr_backend_find(selection_recommend_cpu(&head));
    if (local && !c->backend) return fail(__LINE__);
#ifdef FR_AVX2_COMPUTE
    // Like a remote AVX2 node, give the receive loop a core and reuse the
    // measured worker count and prefetch settings for the assigned share.
    if (local && c->backend==&fr_avx2 && fr_avx2_configure(local_blocks,cfg->burst)) return fail(__LINE__);
#endif
    if (!demo_passes && p->ssd) printf("FRONK head-ssd: paging via %s\n",c->backend->name);
    static starter starts[FR_CONNECTION_NODES]; pthread_t starting[FR_CONNECTION_NODES]; unsigned nstart=0;
    for (unsigned i=0;i<p->count;++i) if (p->p[i].node!=FR_PLAN_HEAD && p->p[i].mask)
        for (unsigned l=0;l<c->links->count;++l) if (c->links->links[l].identity.id==c->d.nodes[p->p[i].node].id) {
            starter *j=&starts[nstart]; memset(j,0,sizeof(*j)); j->fd=c->links->links[l].fd; j->name=c->links->links[l].identity.name;
            j->units=(unsigned)__builtin_popcount(p->p[i].mask);
            snprintf(j->text,sizeof(j->text),"SESSION %s %u %u %u 50000\r\n",demo_passes?"SYNTHETIC":"START",c->links->session,p->p[i].mask,cfg->burst);
            c->sessions=1; /* teardown stops every session that may have started */
            if (pthread_create(&starting[nstart],NULL,start_session,j)) failed=1;
            else ++nstart;
        }
    int64_t t_local=net_now_ns(); /* this Mac loads its units while the nodes load theirs */
    int local_rc=local && (demo_passes?fr_units_synthetic_cache(&c->local,c->backend,local,local_paged,local_cache,cfg->state_dir):
                          fr_units_open_cache(&c->local,c->backend,cfg->state_dir,local,local_paged,local_cache));
    if (local && !local_rc) printf("FRONK this Mac: %u units ready in %.1f s\n",local_blocks,(net_now_ns()-t_local)/1e9);
    else if (local) fprintf(stderr,"fronk: this Mac could not load its %u units\n",local_blocks);
    int started_rc=0;
    for (unsigned k=0;k<nstart;++k) {
        pthread_join(starting[k],NULL);
        if (starts[k].rc) { fprintf(stderr,"fronk: %s could not load its units%s%s\n",starts[k].name,starts[k].error[0]?": ":"",starts[k].error); started_rc=-1; }
        else printf("FRONK %s: %u units ready in %.1f s\n",starts[k].name,starts[k].units,starts[k].seconds);
    }
    if (failed || local_rc || started_rc) return fail(__LINE__);
    if (!(c->plane=c->plane?c->plane:calloc(1,sizeof(*c->plane))) ||
        fr_head_plane_open(c->plane,c->links,cfg->multicast,cfg->discovery_port,local?c->backend:NULL,local_blocks)) return fail(__LINE__);
    c->plane_open=1; c->plane->active=0;
    for (unsigned i=0;i<p->count;++i) if (p->p[i].node!=FR_PLAN_HEAD && p->p[i].mask)
        for (unsigned l=0;l<c->links->count;++l) if (c->links->links[l].identity.id==c->d.nodes[p->p[i].node].id) c->plane->active|=1u<<l;
    if (require_ranked(c->links,local?c->backend:NULL,c->plane->active)) return fail(__LINE__);
    printf("TIMING FRONK %.1f s (units in place, sessions started)\n",(net_now_ns()-t0)/1e9); t0=net_now_ns();
    if (demo_passes) return fr_serve_demo(cfg,c->links,&c->plan,c->plane,demo_passes)?fail(__LINE__):0;
    if (head_selftest(c->plane,g,stdout)) return fail(__LINE__);
    printf("TIMING SELFTEST %.1f s\n",(net_now_ns()-t0)/1e9);
    if (fr_serve_attach(c->plane,c->links,&c->plan)) return fail(__LINE__);
    return fr_serve_session(cfg,c->links,&c->plan,c->plane);
}
int head_main(const fr_config *cfg) {
    fr_gguf g; static char hex[FR_BLOCKS][65];
    if (mkdir(cfg->state_dir,0700) && errno!=EEXIST) { perror("fronk: state-dir"); return 1; }
    if (!cfg->model[0] || fr_gguf_open(cfg->model,&g)) { fprintf(stderr,"fronk: model %s is not a valid Q4 GGUF\n",cfg->model[0]?cfg->model:"(unset)"); return 1; }
    puts("FRONK checking unit manifest (first run hashes all units once)"); fflush(stdout);
    if (fr_manifest(&g,cfg->state_dir,hex)) { fr_gguf_close(&g); fputs("fronk: cannot build unit manifest\n",stderr); return 1; }
    if (fr_serve_start(cfg)) { fr_gguf_close(&g); return 1; }
    cycle c; memset(&c,0,sizeof(c)); int rc;
    do { rc=run_cycle(cfg,&g,hex,&c); teardown(&c); if (rc<0) fputs("fronk: cycle failed; returning to CONNECTION on [m]\n",stderr); }
    while (rc>0 || (rc<0 && ask("[m]odified / [q]uit: ")=='m'));
    fr_serve_stop(); free(c.plane); fr_gguf_close(&g); return rc<0;
}
/* fronk demo: one cycle on synthetic units (no model file, no ds4): the plan,
 * then decode and prompt reading timed on the real machines and links. */
int head_demo_main(const fr_config *cfg,unsigned passes) {
    if (mkdir(cfg->state_dir,0700) && errno!=EEXIST) { perror("fronk: state-dir"); return 1; }
    demo_passes=passes?passes:40;
    cycle c; memset(&c,0,sizeof(c));
    int rc=run_cycle(cfg,NULL,NULL,&c);
    teardown(&c); free(c.plane); demo_passes=0;
    return rc<0;
}
