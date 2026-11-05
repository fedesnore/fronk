/* The boundary between the dense engine and distributed experts.
 *
 * Both the in-process Metal hook and the local socket adapter reach serve_call.
 * It serializes calls against the attached plane, requires complete expert sums
 * under ordinary policy, and records the report. Head setup attaches/detaches
 * that plane; a dense-engine call can wait here while a new plan is being loaded.
 *
 * Read serve_call, then moe_hook or the socket worker for the chosen engine.
 * fr_serve_demo exercises synthetic expert requests without loading the GGUF;
 * fr_serve_session drives real generation checks and the ready/report loop.
 * Neither is a second implementation of the expert kernels. Those live behind
 * backend.h, and remote traffic is handled by plane.c. See docs/PROTOCOL.md.
 */
#include "serve.h"
#include "control.h"
#include "net.h"
#include "native.h"
#ifdef FR_METAL_COMPUTE
#include "metal.h"
#endif
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <libproc.h>
#endif
#define SAMPLES (1u<<16)
/* Ordinary generation and its synthetic preview must wait for the same work.
 * This is a failure bound, not a target latency; measured time includes waits. */
#define COMPLETE_SUM_TIMEOUT_NS 500000000ull
static struct {
    int listener,stop,port,client; pthread_t thread; pid_t ds4; char path[300];
    pthread_mutex_t lock; pthread_cond_t cond;
    fr_head_plane *h; int require_all; uint64_t tau_ns,retries; uint32_t step;
    uint64_t ns[SAMPLES],calls,misses; unsigned count; volatile int pending;
    /* Per burst size: calls and distinct routed experts per call (10 per token
     * when no two tokens share one). Extra tokens cost the nodes per new expert. */
    uint64_t by_burst[FR_MAX_BURST+1],distinct[FR_MAX_BURST+1];
    int64_t started; /* head start, for TIMING READY */
} S={.listener=-1,.client=-1,.lock=PTHREAD_MUTEX_INITIALIZER,.cond=PTHREAD_COND_INITIALIZER};
static int full(int fd,void *p,size_t n,int writing) {
    for (char *c=p;n;) {
        ssize_t k=writing?send(fd,c,n,0):recv(fd,c,n,0);
        if (k<0 && errno==EINTR) continue;
        if (k<=0) return -1;
        c+=k; n-=(size_t)k;
    }
    return 0;
}
/* One expert call, from ds4's socket or (ds4 linked in) its hook: waits while
 * there is no plane (before FRONK, during a re-plan), runs one barrier, counts
 * it. Calls are serialized by S.lock, as ds4's fronk worker issues them. */
static int serve_call(const fr_req *req,fr_part *part,uint32_t *missing) {
    pthread_mutex_lock(&S.lock); S.pending=1;
    while (!S.h && !S.stop) pthread_cond_wait(&S.cond,&S.lock);
    if (S.stop) { pthread_mutex_unlock(&S.lock); return -1; }
    int status=fr_head_plane_run(S.h,req,++S.step,S.tau_ns,part);
    if (!status && S.require_all && S.h->last_missing) {
        // Expert sums have no side effects. A fresh step excludes late packets
        // from the original request and replaces the entire partial sum.
        ++S.retries;
        status=fr_head_plane_run(S.h,req,++S.step,1000000000,part);
        if (!status && S.h->last_missing) {
            fprintf(stderr,"fronk: incomplete expert sum after retry on layer %u (node mask 0x%x)\n",req->layer,S.h->last_missing);
            status=-1;
        }
    }
    if (status) { pthread_mutex_unlock(&S.lock); return status; }
    *missing=S.h->last_missing; S.ns[S.count++%SAMPLES]=S.h->last_total_ns; ++S.calls; S.misses+=*missing!=0;
    { uint64_t seen[FR_ROUTED_EXPERTS/64]={0}; unsigned n=0;
      for (unsigned t=0;t<req->burst;++t) for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) {
          unsigned id=req->expert[t][e]; if (!(seen[id/64]>>(id%64)&1)) { seen[id/64]|=1ull<<(id%64); ++n; } }
      ++S.by_burst[req->burst]; S.distinct[req->burst]+=n; }
    pthread_mutex_unlock(&S.lock);
    return status;
}
static void handle(int fd) {
    static fr_req req; static fr_part part;
    for (;;) {
        uint32_t head[3]; int32_t status=0; uint32_t missing=0;
        if (full(fd,head,sizeof(head),0) || head[0]!=FR_SERVE_MAGIC || head[1]>=FR_LAYERS || !head[2] || head[2]>FR_MAX_BURST) return;
        unsigned b=head[2]; req.layer=head[1]; req.burst=b;
        for (unsigned t=0;t<b;++t) if (full(fd,req.expert[t],sizeof(req.expert[t]),0)) return;
        for (unsigned t=0;t<b;++t) if (full(fd,req.gate[t],sizeof(req.gate[t]),0)) return;
        for (unsigned t=0;t<b;++t) if (full(fd,req.x[t],sizeof(req.x[t]),0)) return;
        if (S.stop) return;
        status=serve_call(&req,&part,&missing);
        if (S.stop) return;
        if (full(fd,&status,4,1) || full(fd,&missing,4,1)) return;
        for (unsigned t=0;t<b;++t) if (full(fd,part.y[t],sizeof(part.y[t]),1)) return;
    }
}
#ifdef FR_DS4_INPROCESS
/* ds4 linked into this program (libds4-fronk.a, `ds4 builtin` in the config):
 * its OpenAI server runs on a thread of the head and its expert calls come in
 * through fronk_moe_hook, the same request as the socket's without the copy
 * through a second process. ds4's own lines go to state-dir/ds4.log. */
extern int (*fronk_moe_hook)(unsigned layer,unsigned b,const uint16_t (*e)[FR_ACTIVE_ROUTED],
                             const float (*g)[FR_ACTIVE_EXPERTS],const float *x,float *y,uint32_t *missing);
int ds4_server_main(int argc,char **argv);
void ds4_server_stop(void);
static int moe_hook(unsigned layer,unsigned b,const uint16_t (*e)[FR_ACTIVE_ROUTED],
                    const float (*g)[FR_ACTIVE_EXPERTS],const float *x,float *y,uint32_t *missing) {
    static fr_req req; static fr_part part;  /* one call at a time: ds4's fronk worker */
    if (layer>=FR_LAYERS || !b || b>FR_MAX_BURST) return -1;
    req.layer=layer; req.burst=b;
    for (unsigned t=0;t<b;++t) {
        memcpy(req.expert[t],e[t],sizeof(req.expert[t])); memcpy(req.gate[t],g[t],sizeof(req.gate[t]));
        memcpy(req.x[t],x+(size_t)t*FR_HIDDEN,sizeof(req.x[t]));
    }
    int status=serve_call(&req,&part,missing);
    for (unsigned t=0;t<b;++t) memcpy(y+(size_t)t*FR_HIDDEN,part.y[t],sizeof(part.y[t]));
    return status;
}
static struct { pthread_t thread; int running; atomic_int exited; int argc; char *argv[24]; char text[8][PATH_MAX]; } D;
static void *ds4_thread(void *arg) {
    (void)arg;
    int rc=ds4_server_main(D.argc,D.argv);
    atomic_store(&D.exited,1);
    if (rc) fprintf(stderr,"fronk: ds4 stopped (%d); see ds4.log\n",rc);
    return NULL;
}
/* The engine settings tools/ds4-head.sh passes to ds4-server, here as arguments:
 * 32K context unless the chat page chose another (state-dir/ds4-ctx), 32-token
 * prefill chunks, the 512-expert cache for ds4's local MTP layer, MTP, and the
 * disk KV cache in state-dir/kv. */
static int ds4_start_builtin(const fr_config *cfg,const char *state) {
    unsigned ctx=32768; char path[PATH_MAX]; FILE *f;
    snprintf(path,sizeof(path),"%s/ds4-ctx",state);
    if ((f=fopen(path,"r"))) { if (fscanf(f,"%u",&ctx)!=1 || ctx<1024 || ctx>262144) ctx=32768; fclose(f); }
    const char *dir=getenv("FRONK_DS4_DIR"); if (!dir || !*dir) dir=FR_DS4_DIR;
    snprintf(D.text[0],PATH_MAX,"%d",S.port); snprintf(D.text[1],PATH_MAX,"%u",ctx);
    snprintf(D.text[2],PATH_MAX,"%s/kv",state); mkdir(D.text[2],0700);
    snprintf(D.text[3],PATH_MAX,"%s/ds4.log",state);
    setenv("DS4_QWEN4_STREAM_MAX_CTX",D.text[1],1); setenv("DS4_METAL_DIR",dir,1); setenv("DS4_SERVER_LOG",D.text[3],1);
    char *argv[]={"ds4-server","-m",(char *)cfg->model,"--host","127.0.0.1","--port",D.text[0],"--ctx",D.text[1],
                  "--prefill-chunk","32","--ssd-streaming","--ssd-streaming-cache-experts","512","--mtp",
                  "--kv-disk-dir",D.text[2],"--kv-disk-space-mb","8192",NULL};
    const char *mtp=getenv("FRONK_DS4_MTP");
    D.argc=0;
    for (char **a=argv;*a;++a) {
        if (!strcmp(*a,"--mtp") && mtp && !*mtp) continue;
        // Verification must not restore KV computed with a partial expert sum
        // by an earlier deadline-limited session.
        if (S.require_all && (!strcmp(*a,"--kv-disk-dir") || !strcmp(*a,"--kv-disk-space-mb"))) { ++a; continue; }
        D.argv[D.argc++]=*a;
    }
    const char *branches=getenv("FRONK_MTP_BRANCHES");
    if (branches && branches[0]>='2' && branches[0]<='4' && !branches[1])
        D.argv[D.argc++]="--mtp-exact-sampling";
    D.argv[D.argc]=NULL;
    atomic_store(&D.exited,0); fronk_moe_hook=moe_hook;
    if (pthread_create(&D.thread,NULL,ds4_thread,NULL)) { fronk_moe_hook=NULL; return -1; }
    D.running=1;
    printf("SERVE ds4 started in this process, log=%s (loads dense weights while CONNECTION runs)\n",D.text[3]);
    return 0;
}
static void ds4_stop_builtin(void) {
    if (!D.running) return;
    ds4_server_stop(); pthread_join(D.thread,NULL); D.running=0; fronk_moe_hook=NULL;
}
#endif
static void *service(void *arg) {
    (void)arg;
    while (!S.stop) {
        struct pollfd p={S.listener,POLLIN,0};
        if (poll(&p,1,200)<=0) continue;
        int fd=accept(S.listener,NULL,NULL); if (fd<0) continue;
        S.client=fd; handle(fd); S.client=-1; close(fd);
    }
    return NULL;
}
/* A head stopped by a signal (web console Stop, Ctrl-C, or SIGPIPE when the
 * console that reads its output is gone) must not orphan ds4. */
static char pidfile[PATH_MAX+16];
static void on_signal(int sig) { if (S.ds4>0) { kill(S.ds4,SIGTERM); unlink(pidfile); } signal(sig,SIG_DFL); raise(sig); }
static int is_ds4(pid_t pid) {
    char path[PATH_MAX+1];
#ifdef __APPLE__
    if (proc_pidpath(pid,path,sizeof(path))<=0) return 0;
#else
    char link[64]; snprintf(link,sizeof(link),"/proc/%d/exe",(int)pid);
    ssize_t n=readlink(link,path,sizeof(path)-1); if (n<=0) return 0; path[n]=0;
#endif
    const char *base=strrchr(path,'/'); return base && (!strcmp(base+1,"ds4-server") || !strcmp(base+1,"fronk-llama-server"));
}
/* A head killed outright (SIGKILL, a crash) leaves its ds4-server running with
 * the chat port and ~8 GB of this Mac's memory; its pid is in state-dir/ds4.pid. */
static void stop_stale_ds4(void) {
    FILE *f=fopen(pidfile,"r"); long pid=0; if (!f) return;
    if (fscanf(f,"%ld",&pid)!=1) pid=0;
    fclose(f); unlink(pidfile);
    if (pid<=1 || !is_ds4((pid_t)pid)) return;
    printf("SERVE stopping ds4-server pid=%ld left by an earlier head\n",pid); fflush(stdout);
    kill((pid_t)pid,SIGTERM);
    for (int i=0;i<50 && !kill((pid_t)pid,0);++i) usleep(100000);
    if (!kill((pid_t)pid,0)) kill((pid_t)pid,SIGKILL);
}
/* ---- Dense-engine lifecycle. The listener and expert service must exist before
 * the model can issue its first routed-layer request. ---- */
int fr_serve_start(const fr_config *cfg) {
    struct sockaddr_un a={0}; a.sun_family=AF_UNIX;
    char dir[PATH_MAX]; /* absolute: ds4 runs from its own directory */
    if (!realpath(cfg->state_dir,dir) || snprintf(S.path,sizeof(S.path),"%s/moe.sock",dir)>=(int)sizeof(a.sun_path)) return -1;
    strcpy(a.sun_path,S.path); unlink(S.path);
    if ((S.listener=socket(AF_UNIX,SOCK_STREAM,0))<0 || bind(S.listener,(struct sockaddr *)&a,sizeof(a)) || listen(S.listener,4)) return -1;
    const char *width=getenv("FRONK_MTP_BRANCHES"), *exact=getenv("FRONK_REQUIRE_ALL");
    S.require_all=!strcmp(cfg->ds4,"tools/gpu-head/run.sh") ||
                  (width && width[0]>='2' && width[0]<='4' && !width[1]) || (exact && !strcmp(exact,"1"));
    S.stop=0; S.tau_ns=S.require_all?COMPLETE_SUM_TIMEOUT_NS:50000000; S.port=(int)cfg->control_port+10; S.started=net_now_ns();
    if (pthread_create(&S.thread,NULL,service,NULL)) return -1;
    snprintf(pidfile,sizeof(pidfile),"%s/ds4.pid",dir); stop_stale_ds4();
    if (!strcmp(cfg->ds4,"builtin")) {
#ifdef FR_DS4_INPROCESS
        return ds4_start_builtin(cfg,dir);
#else
        fputs("fronk: `ds4 builtin` needs a fronk built with ds4 (make DS4=path/to/ds4-fronk)\n",stderr); return -1;
#endif
    }
    if (cfg->ds4[0]) {
        char log[300],port[16]; snprintf(log,sizeof(log),"%s/ds4.log",cfg->state_dir); snprintf(port,sizeof(port),"%d",S.port);
        if ((S.ds4=fork())<0) return -1;
        if (S.ds4) {
            signal(SIGTERM,on_signal); signal(SIGINT,on_signal); signal(SIGHUP,on_signal); signal(SIGPIPE,on_signal);
            FILE *f=fopen(pidfile,"w"); if (f) { fprintf(f,"%d\n",(int)S.ds4); fclose(f); }
        }
        if (!S.ds4) {
            int out=open(log,O_WRONLY|O_CREAT|O_APPEND,0600); if (out>=0) { dup2(out,1); dup2(out,2); }
            setenv("FRONK_MODEL",cfg->model,1);
            setenv("FRONK_MOE_SOCKET",S.path,1); setenv("FRONK_DS4_PORT",port,1);
            execl(cfg->ds4,cfg->ds4,(char *)NULL); _exit(127);
        }
        printf("SERVE %s started pid=%d log=%s (loads dense weights while CONNECTION runs)\n",
               !strcmp(cfg->ds4,"tools/gpu-head/run.sh")?"Vulkan head":"ds4",(int)S.ds4,log);
    }
    return 0;
}
int fr_serve_attach(fr_head_plane *h,const fr_connection *links,const fr_plan *plan) {
    (void)plan; uint32_t active=0;
    for (unsigned i=0;i<links->count;++i) active|=1u<<i; /* refined by head via h->active */
    pthread_mutex_lock(&S.lock); if (h->active==UINT32_MAX) h->active=active;
    S.h=h; S.step=1000; S.count=0; S.calls=S.misses=S.retries=0; memset(S.by_burst,0,sizeof(S.by_burst)); memset(S.distinct,0,sizeof(S.distinct)); pthread_cond_broadcast(&S.cond); pthread_mutex_unlock(&S.lock); return 0;
}
void fr_serve_detach(void) { pthread_mutex_lock(&S.lock); S.h=NULL; pthread_mutex_unlock(&S.lock); }
void fr_serve_stop(void) {
    pthread_mutex_lock(&S.lock); S.stop=1; pthread_cond_broadcast(&S.cond); pthread_mutex_unlock(&S.lock);
#ifdef FR_DS4_INPROCESS
    ds4_stop_builtin();
#endif
    if (S.ds4>0) { kill(S.ds4,SIGTERM); waitpid(S.ds4,NULL,0); S.ds4=0; unlink(pidfile); }
    if (S.client>=0) shutdown(S.client,SHUT_RDWR); /* unblock a reader waiting on ds4 */
    if (S.listener>=0) { pthread_join(S.thread,NULL); close(S.listener); unlink(S.path); S.listener=-1; }
    if (S.ds4>0) { kill(S.ds4,SIGTERM); waitpid(S.ds4,NULL,0); S.ds4=0; }
}
/* Minimal HTTP/1.0 to ds4-server on localhost; returns the body length or -1. */
static int http(const char *method,const char *path,const char *body,FILE *stream,double *first,double *last,unsigned *tokens) {
    int fd=net_tcp_connect("127.0.0.1",(unsigned)S.port,2000); if (fd<0) return -1;
    int fl=fcntl(fd,F_GETFL); fcntl(fd,F_SETFL,fl&~O_NONBLOCK);
    char head[512]; int n=snprintf(head,sizeof(head),"%s %s HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n",
        method,path,body?strlen(body):0);
    if (full(fd,head,(size_t)n,1) || (body && full(fd,(void *)body,strlen(body),1))) { close(fd); return -1; }
    char buf[8192],line[16384]; size_t used=0; int status=-1,body_started=0,done=0,failed=0;
    for (;;) {
        ssize_t k=recv(fd,buf,sizeof(buf),0); if (k<=0) break;
        for (ssize_t i=0;i<k;++i) {
            if (used<sizeof(line)-1) line[used++]=buf[i];
            if (buf[i]!='\n') continue;
            line[used]=0; used=0;
            if (status<0) { sscanf(line,"HTTP/%*s %d",&status); continue; }
            if (!body_started) { if (line[0]=='\r' || line[0]=='\n') body_started=1; continue; }
            if (!strncmp(line,"data: [DONE]",12)) done=1;
            if (strstr(line,"\"finish_reason\":\"error\"") || strstr(line,"\"finish_reason\": \"error\"") ||
                !strncmp(line,"event: error",12) || !strncmp(line,"data: {\"error\":",15)) failed=1;
            const char *c=strstr(line,"\"content\":\"");
            if (c) c+=11;
            if (!c || *c=='"') {
                c=strstr(line,"\"reasoning_content\":\"");
                if (c) c+=21;
            }
            if (stream && c && *c!='"') {
                double now=net_now_ns()/1e9; if (!*tokens) *first=now; *last=now; ++*tokens;
                for (;*c && *c!='"';++c) {
                    if (*c=='\\' && c[1]) { ++c; fputc(*c=='n'?'\n':*c=='t'?'\t':*c,stream); } else fputc(*c,stream);
                }
                fflush(stream);
            }
        }
    }
    close(fd); return stream && status==200 && (!done || failed)?-1:status;
}
/* Node counters are cumulative; REPORT shows each pass's delta. */
enum { REQS, COMPUTE, RX, WAIT, REPLY, COUNTERS };
static unsigned long long seen[FR_CONNECTION_NODES][COUNTERS];
static int node_counters(int fd,unsigned long long out[COUNTERS]) {
    char reply[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES];
    static const char *key[COUNTERS]={"requests:","compute_ns:","rx_ns:","wait_ns:","reply_ns:"};
    if (control_exchange(fd,"INFO dataplane\r\n",16,reply,v,net_now_ns()/1000000+5000) || v[0].type!=RESP_BULK) return -1;
    reply[v[0].data-reply+v[0].len]=0;
    for (unsigned k=0;k<COUNTERS;++k) { const char *c=strstr(v[0].data,key[k]); out[k]=c?strtoull(c+strlen(key[k]),NULL,10):0; }
    return 0;
}
static int compare(const void *a,const void *b) { uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b; return x<y?-1:x>y; }
int fr_serve_wait_ds4(int allow_pending_expert) {
    for (int i=0;i<1200;++i) {
        if (S.ds4>0 && waitpid(S.ds4,NULL,WNOHANG)==S.ds4) { S.ds4=0; unlink(pidfile); }
        int alive=S.ds4>0;
#ifdef FR_DS4_INPROCESS
        alive=alive || (D.running && !atomic_load(&D.exited));
#endif
        if (!alive) { fputs("fronk: model engine exited; see ds4.log in the state directory\n",stderr); return -1; }
        if ((allow_pending_expert && S.pending) || http("GET","/v1/models",NULL,NULL,NULL,NULL,NULL)==200) return 0;
        if (!i) { puts("SERVE waiting for the model engine to load"); fflush(stdout); }
        usleep(500000);
    }
    fputs("fronk: model engine did not become ready within ten minutes; see ds4.log\n",stderr);
    return -1;
}
static void pass_reset(const fr_connection *links,fr_head_plane *h) {
    pthread_mutex_lock(&S.lock); S.count=0; S.calls=0; S.misses=S.retries=0; memset(S.by_burst,0,sizeof(S.by_burst)); memset(S.distinct,0,sizeof(S.distinct)); pthread_mutex_unlock(&S.lock);
    h->barriers=0; h->local_ns=0; h->total_ns=0; h->send_ns=0; h->deadline_hits=0;
    memset(h->width_calls,0,sizeof(h->width_calls));
    memset(h->arrive_ns,0,sizeof(h->arrive_ns)); memset(h->last,0,sizeof(h->last)); memset(h->missing,0,sizeof(h->missing));
    for (unsigned i=0;i<links->count;++i) (void)node_counters(links->links[i].fd,seen[i]);
}
/* REPORT per participant since pass_reset: compute, network, arrival, who came
 * last, and the bottleneck. netcal: keep the measured network time for the next
 * plan. The installation demo also saves its real network measurement, with
 * the same burst size used by that plan. */
static void report_nodes(const fr_config *cfg,const fr_connection *links,const fr_plan *plan,fr_head_plane *h,int netcal) {
    double worst=0,mean=0; int bottleneck=-1; unsigned parts=0;
    for (unsigned i=0;i<=links->count;++i) {
        int head_row=i==links->count; if (!head_row && !(h->active>>i&1)) continue;
        if (head_row && !h->has_local) continue;
        double arrive=h->barriers?h->arrive_ns[i]/1e3/h->barriers:0,compute=0,stage[3]={0};
        if (head_row) compute=h->barriers?h->local_ns/1e3/h->barriers:0;
        else {
            unsigned long long now[COUNTERS];
            if (!node_counters(links->links[i].fd,now) && now[REQS]>seen[i][REQS]) {
                double r=(double)(now[REQS]-seen[i][REQS]);
                compute=(now[COMPUTE]-seen[i][COMPUTE])/r/1e3;
                for (unsigned k=0;k<3;++k) stage[k]=(now[RX+k]-seen[i][RX+k])/r/1e3;
            }
        }
        printf("REPORT node=%-10s link=%-11s compute=%.0f us network=%.0f us arrival=%.0f us last=%llu misses=%llu local_wait_us=%.0f%s\n",
               head_row?(plan->ssd?"head-ssd":"head"):links->links[i].identity.name,head_row?"local":interface_type(links->links[i].via.type),
               compute,!head_row && arrive>compute?arrive-compute:0,arrive,(unsigned long long)h->last[i],head_row?0ull:(unsigned long long)h->missing[i],
               head_row && arrive>compute?arrive-compute:0,
               !head_row && h->missing[i]?"  <- missed deadlines":"");
        if (!head_row) {
            double net=arrive>compute?arrive-compute:0,inside=stage[0]+stage[1]+stage[2];
            /* A pass where this node missed deadlines measured a disturbed link, not its usual cost. */
            /* A mixed-width generation must not overwrite a width-specific
             * link profile with its aggregate cost. The synthetic demo has a
             * fixed width and remains the fast source for this calibration. */
            if (netcal && h->barriers>=100 && h->width_calls[cfg->burst]==h->barriers && net>0 && !h->missing[i])
                fr_netcal_put(cfg->state_dir,links->links[i].identity.id,links->links[i].ip,cfg->burst,net);
            printf("REPORT split node=%-10s head_send=%.0f us node_rx=%.0f us node_wait=%.0f us node_reply=%.0f us wire_and_stack=%.0f us\n",
                   links->links[i].identity.name,h->barriers?h->send_ns/1e3/h->barriers:0,stage[0],stage[1],stage[2],net>inside?net-inside:0);
        }
        if (arrive>worst) { worst=arrive; bottleneck=(int)i; }
        mean+=arrive; ++parts;
    }
    if (parts) mean/=parts;
    if (bottleneck>=0) printf("REPORT bottleneck=%s %s\n",bottleneck==(int)links->count?(plan->ssd?"head-ssd":"head"):links->links[bottleneck].identity.name,
        worst>0 && (worst-mean)/worst<0.05?"balanced":"not balanced (re-plan with [m] after BENCH changes)");
}
/* fronk demo: the decode and prompt reading of the real cluster on synthetic
 * units (fr_units_synthetic on every participant), so a user learns what the
 * machines will do before downloading the model. Per layer the head's dense
 * part is a wait as long as the GPU takes to read that layer's dense weights
 * (DEMO_DENSE_KB, from the Qwen3.8-Flash-Next Q4 GGUF, plus the shared expert);
 * then the real barrier runs: activations through the head plane, the nodes'
 * backends on their assigned neurons, the parts back over the network.
 * Routing: 2 tokens per decode call sharing 4 of 10 experts (15.8 of 20 were
 * distinct in real traffic, BENCH run 80), 8 per prompt call sharing 5.
 * The dense times are scaled by DEMO_DENSE_REAL, the ratio of ds4's real dense
 * time per layer to this probe's on the M4 Pro (BENCH runs 85, 90). */
static const uint32_t DEMO_DENSE_KB[48]={91921,126201,91921,84343,91921,91921,91921,84343,91921,91921,91921,84343,91921,91921,91921,84343,
    91921,91921,91921,84343,91921,91921,91921,84343,91921,91921,91921,84343,91921,91921,91921,84343,91921,91921,91921,84343,
    91921,91921,91921,84343,91921,91921,91921,84343,91921,91921,91921,84343};
#define DEMO_SHARED_KB 5100           /* shared expert per layer, run on the head */
#define DEMO_PASS_KB (659570+108258)  /* per pass: output head and the MTP layer */
#define DEMO_DENSE_REAL 1.80          /* ds4 725 us per layer (run 85) / this probe 402 us (run 90), M4 Pro */
#define DEMO_PASS_FIXED_NS 4.7e6      /* per pass besides the reads (10.5 ms real, run 85): sampling, MTP draft */
static void spin_until(int64_t until) { while (net_now_ns()<until) {} }
static uint32_t demo_rand(uint32_t *x) { *x^=*x<<13; *x^=*x>>17; *x^=*x<<5; return *x; }
static void demo_route(fr_req *r,unsigned layer,unsigned burst,unsigned shared_with_previous,uint32_t *seed) {
    memset(r,0,sizeof(*r)); r->layer=layer; r->burst=burst;
    for (unsigned t=0;t<burst;++t) {
        unsigned n=0;
        if (t) for (unsigned k=0;k<shared_with_previous;++k) r->expert[t][n++]=r->expert[t-1][k];
        while (n<FR_ACTIVE_ROUTED) {
            uint16_t id=(uint16_t)(demo_rand(seed)%FR_ROUTED_EXPERTS); int dup=0;
            for (unsigned k=0;k<n;++k) dup|=r->expert[t][k]==id;
            if (!dup) r->expert[t][n++]=id;
        }
        for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) r->gate[t][e]=.1f;
        r->gate[t][FR_ACTIVE_ROUTED]=0; /* the head computes the shared expert: in its dense time */
        for (unsigned d=0;d<FR_HIDDEN;++d) r->x[t][d]=(float)((int)(demo_rand(seed)%2001)-1000)/1000.f;
    }
}
static int cmp_double(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return x<y?-1:x>y; }
int fr_serve_demo(const fr_config *cfg,const fr_connection *links,const fr_plan *plan,fr_head_plane *h,unsigned passes) {
    double dense_ns[FR_LAYERS],pass_extra=0,dense_sum=0;
    int vulkan=!strcmp(cfg->ds4,"tools/gpu-head/run.sh");
    if (vulkan) {
        FILE *probe=popen("MVK_CONFIG_LOG_LEVEL=1 ./gpu/fronk-gpu-probe --timing","r");
        if (!probe) return -1;
        double ns_per_byte=0;
        int got=fscanf(probe,"%lf",&ns_per_byte);
        if (pclose(probe) || got!=1 || !isfinite(ns_per_byte) || ns_per_byte<=0) return -1;
        for (unsigned l=0;l<FR_LAYERS;++l) {
            dense_ns[l]=(DEMO_DENSE_KB[l]+DEMO_SHARED_KB)*1024.0*ns_per_byte;
            dense_sum+=dense_ns[l];
        }
        pass_extra=659570.0*1024*ns_per_byte; /* vocabulary projection; no MTP */
        puts("DEMO Vulkan dense estimate uses a verified Q8 projection on this GPU. Host fallback, attention, sampling and layer offload can add time.");
    } else {
#ifdef FR_METAL_COMPUTE
    for (unsigned l=0;l<FR_LAYERS;++l) {
        double best=0;
        for (unsigned r=0;r<3;++r) { double ns; if (fr_metal_stream((uint64_t)(DEMO_DENSE_KB[l]+DEMO_SHARED_KB)*1024,&ns)) return -1; if (!r || ns<best) best=ns; }
        dense_ns[l]=best*DEMO_DENSE_REAL; dense_sum+=dense_ns[l];
    }
    for (unsigned r=0;r<3;++r) { double ns; if (fr_metal_stream((uint64_t)DEMO_PASS_KB*1024/2,&ns)) return -1; if (!r || 2*ns<pass_extra) pass_extra=2*ns; }
    pass_extra=pass_extra*DEMO_DENSE_REAL+DEMO_PASS_FIXED_NS;
#else
        fputs("fronk: configure a tested Vulkan head or Apple Silicon Metal head first\n",stderr); return -1;
#endif
    }
    printf("DEMO head dense part (synthetic, GPU reads of each layer's weights): %.0f us per layer, %.1f ms per pass + %.1f ms between passes\n",
           dense_sum/FR_LAYERS/1e3,dense_sum/1e6,pass_extra/1e6);
    fflush(stdout);
    static fr_req req; static fr_part part; uint32_t seed=0x2545f491u; double *pass_ms=calloc(passes,sizeof(double));
    if (!pass_ms) return -1;
    fr_plan placed=*plan; placed.ssd=0;  /* the demo runs only the blocks that found memory */
    /* Like the real head's self-test, touch every layer before timing decode.
     * A newly loaded Metal/SSD worker can spend more than 50 ms on its first
     * request. That is setup, not a lost data path. Warm both request widths;
     * require every reply. Timed calls use the ordinary chat's failure bound,
     * so a slow SSD read contributes its real cost instead of a partial sum.
     * A separate seed leaves the measured routing sequence reproducible. */
    uint32_t warm_seed=seed;
    for (unsigned width=0;width<(vulkan?1u:2u);++width) {
        unsigned burst=vulkan?1:width?8:cfg->burst;
        for (unsigned l=0;l<FR_LAYERS;++l) {
            demo_route(&req,l,burst,width?5:4,&warm_seed);
            if (fr_head_plane_run(h,&req,++S.step,60000000000ull,&part) || h->last_missing) {
                fprintf(stderr,"fronk: synthetic warm-up failed at layer %u width %u; check node compute and replies\n",l,burst);
                free(pass_ms); return -1;
            }
        }
    }
    puts("DEMO warm-up complete: every node answered every layer; timing starts now");
    pass_reset(links,h);
    for (unsigned p=0;p<passes;++p) {           /* decode: 48 layers of dense then barrier */
        int64_t t0=net_now_ns();
        for (unsigned l=0;l<FR_LAYERS;++l) {
            spin_until(net_now_ns()+(int64_t)dense_ns[l]);
            demo_route(&req,l,vulkan?1:cfg->burst,4,&seed);
            if (fr_head_plane_run(h,&req,++S.step,COMPLETE_SUM_TIMEOUT_NS,&part)) { free(pass_ms); return -1; }
        }
        spin_until(net_now_ns()+(int64_t)pass_extra);
        pass_ms[p]=(net_now_ns()-t0)/1e6;
    }
    const double barrier_us=h->barriers?h->total_ns/1e3/h->barriers:0;
    uint64_t misses=0; for (unsigned n=0;n<links->count;++n) misses+=h->missing[n];
    printf("DEMO decode: %u passes of 48 layers, barrier %.0f us per layer (%s)\n",passes,barrier_us,
           misses?"some nodes did not answer in time: see REPORT; the estimate is not meaningful":"every node answered every layer");
    report_nodes(cfg,links,&placed,h,1);
    qsort(pass_ms,passes,sizeof(double),cmp_double);
    const double p10=pass_ms[passes/10],p50=pass_ms[passes/2],p90=pass_ms[(size_t)passes*9/10];
    /* Prompt reading: chunks of 32 tokens, 4 calls of 8 per layer (ds4's fronk client). */
    pass_reset(links,h); int64_t t0=net_now_ns(); const unsigned chunks=3,chunk_tokens=vulkan?1:32;
    for (unsigned c=0;c<chunks;++c) for (unsigned l=0;l<FR_LAYERS;++l) {
        spin_until(net_now_ns()+(int64_t)dense_ns[l]);
        for (unsigned k=0;k<(vulkan?1u:4u);++k) { demo_route(&req,l,vulkan?1:8,5,&seed); if (fr_head_plane_run(h,&req,++S.step,COMPLETE_SUM_TIMEOUT_NS,&part)) { free(pass_ms); return -1; } }
    }
    for (unsigned n=0;n<links->count;++n) misses+=h->missing[n];
    const double prefill=chunks*chunk_tokens/((net_now_ns()-t0)/1e9);
    /* Tokens per decode pass: 1.56 measured with MTP on prose (BENCH run 79); text
     * the answer copies (quotes, code) verifies up to 8 (context drafts). */
    const double lo=(vulkan?1:1.3)/(p90/1e3),mid=(vulkan?1:1.56)/(p50/1e3),hi=(vulkan?1:1.8)/(p10/1e3);
    const double cluster=barrier_us*FR_LAYERS/1e3,dense=dense_sum/1e6,between=pass_extra/1e6,total=cluster+dense+between;
    if (misses) {
        fprintf(stderr,"fronk: synthetic demo missed %llu node replies; check the node logs and local network permission, then retry\n",(unsigned long long)misses);
        free(pass_ms); return -1;
    }
    printf("DEMO estimate%s: %.1f tok/s (range %.1f-%.1f), pass %.0f ms (p10 %.0f, p90 %.0f); prompt reading %.0f tokens/s\n",plan->ssd?" (placed blocks only)":"",mid,lo,hi,p50,p10,p90,prefill);
    printf("DEMO where a pass goes: cluster %.0f ms (%.0f%%), head dense %.0f ms (%.0f%%), between passes %.0f ms (%.0f%%)\n",
           cluster,100*cluster/total,dense,100*dense/total,between,100*between/total);
    if (plan->ssd) {
        const unsigned missing=(unsigned)__builtin_popcount(plan->ssd);
        printf("DEMO verdict: the model does NOT fit: %u of %u blocks have no memory (%.1f GB more needed across the machines);"
               " the numbers above time only the %u placed blocks, so they are not the speed you would get\n",
               missing,FR_BLOCKS,missing*(double)FR_NATIVE_BLOCK_BYTES/1e9,FR_BLOCKS-missing);
        if (plan->cache_read_bps>0) {
            // Cold reads for distinct experts in one decode pass. Resident
            // compute and link time are already included in the measured p50.
            double disk_ms=1000.0*FR_LAYERS*missing*(vulkan?10.0:15.8)*FR_NATIVE_SLICE_BYTES/plan->cache_read_bps;
            printf("DEMO full model with cold spill: ~%.2f tok/s if disk reads add %.0f ms to the %.0f ms resident pass "
                   "(cache drive %.2f GB/s). Cache hits, overlap and extra CPU work change the real rate.\n",
                   (vulkan?1.0:1.56)*1000.0/(p50+disk_ms),disk_ms,p50,plan->cache_read_bps/1e9);
        } else puts("DEMO full-model rate unavailable: could not measure the unit cache drive.");
    } else puts("DEMO verdict: the model fits: all 20 blocks covered by the measured RAM/SSD plan");
    if (vulkan) puts("DEMO range: one token per decode pass, 10th to 90th percentile. Confirm speed with the real model.");
    else puts("DEMO range: 1.3-1.8 tokens per pass (MTP acceptance varies with the text) and the 10th-90th percentile pass time; copied text goes faster, long contexts slower (attention grows with the context)");
    free(pass_ms); fflush(stdout); return 0;
}
/* One TEST generation at the current tau, then its REPORT. The first pass runs
 * with tau=50 ms (every node awaited); later passes use the p99 just measured. */
static int test_pass(const fr_config *cfg,const fr_connection *links,const fr_plan *plan,fr_head_plane *h,unsigned number) {
    double first=0,last=0,start=net_now_ns()/1e9; unsigned tokens=0; int status=-1;
    pass_reset(links,h);
    uint64_t tau=S.tau_ns;
    if (cfg->ds4[0]) {
        printf("TEST pass %u generation (temperature 0, tau=%.0f us%s):\n",number,tau/1e3,number==1?", no deadline in practice":", p99 of the previous pass");
        start=net_now_ns()/1e9;
        status=http("POST","/v1/chat/completions","{\"model\":\"ds4\",\"stream\":true,\"think\":false,\"chat_template_kwargs\":{\"enable_thinking\":false},\"temperature\":0,\"max_tokens\":64,"
            "\"messages\":[{\"role\":\"user\",\"content\":\"Explain in three sentences why the sky is blue.\"}]}",stdout,&first,&last,&tokens);
        putchar('\n');
    }
    pthread_mutex_lock(&S.lock);
    unsigned n=S.count<SAMPLES?S.count:SAMPLES; uint64_t *sorted=malloc((n?n:1)*sizeof(uint64_t));
    if (sorted) { memcpy(sorted,S.ns,n*sizeof(uint64_t)); qsort(sorted,n,sizeof(uint64_t),compare); }
    double p50=n&&sorted?sorted[n/2]/1e3:0,p99=n&&sorted?sorted[(size_t)(n*0.99)]/1e3:0;
    uint64_t calls=S.calls,misses=S.misses,retries=S.retries; free(sorted);
    printf("REPORT routing");
    for (unsigned b=1;b<=FR_MAX_BURST;++b) if (S.by_burst[b])
        printf(" burst%u: calls=%llu distinct_experts=%.1f of %u",b,(unsigned long long)S.by_burst[b],(double)S.distinct[b]/S.by_burst[b],10*b);
    putchar('\n');
    if (p99>0 && !S.require_all) S.tau_ns=(uint64_t)(p99*1e3);
    pthread_mutex_unlock(&S.lock);
    printf("\nREPORT pass=%u http=%d tokens=%u",number,status,tokens);
    if (tokens>1 && last>first) printf(" accepted=%.2f tok/s (decode) first_token=%.2f s",(tokens-1)/(last-first),first-start);
    printf("\nREPORT barriers=%llu p50=%.0f us p99=%.0f us tau=%.0f us (next tau=%.0f us) deadline_misses=%llu (%.2f%%)\n",(unsigned long long)calls,p50,p99,
           tau/1e3,S.tau_ns/1e3,(unsigned long long)misses,calls?100.0*misses/calls:0);
    if (cfg->ds4[0] && calls) {
        double wall=net_now_ns()/1e9-start,experts=h->total_ns/1e9;
        printf("REPORT generation time wall_s=%.3f experts_s=%.3f outside_experts_s=%.3f\n",
               wall,experts,fmax(0,wall-experts));
    }
    if (S.require_all) printf("REPORT complete sums: %llu, retried requests: %llu\n",(unsigned long long)calls,(unsigned long long)retries);
    report_nodes(cfg,links,plan,h,1);
    if (cfg->ds4[0] && (status!=200 || !tokens || !calls)) {
        fprintf(stderr,"fronk: generation check failed (HTTP %d, %u tokens, %llu cluster calls); see ds4.log\n",
                status,tokens,(unsigned long long)calls);
        return -1;
    }
    return 0;
}
int fr_serve_session(const fr_config *cfg,const fr_connection *links,const fr_plan *plan,fr_head_plane *h) {
    if (!cfg->ds4[0]) puts("SERVE no ds4 configured: expert service only");
    else if (fr_serve_wait_ds4(0)) return -1;
    unsigned number=1; if (test_pass(cfg,links,plan,h,number)) return -1;
    printf("TIMING READY %.1f s after the head started\n",(net_now_ns()-S.started)/1e9);
    if (cfg->ds4[0]) printf("\nChat: %s on http://127.0.0.1:%d (OpenAI API) through the cluster.\n",
                           !strcmp(cfg->ds4,"tools/gpu-head/run.sh")?"llama.cpp Vulkan":"ds4-server",S.port);
    /* FRONK_ANSWER_AFTER is a key sequence, e.g. "rq": one pass at p99, then quit. */
    const char *answers=getenv("FRONK_ANSWER_AFTER");
    for (;;) {
        printf("[m]odified (re-run CONNECTION) / [r]eport (test again at tau=p99) / [q]uit: "); fflush(stdout);
        int c;
        if (answers && *answers) { c=*answers++; putchar(c); putchar('\n'); }
        else if (answers) c='q';
        else { do c=getchar(); while (c==' ' || c=='\n'); for (int d=c;d!='\n' && d!=EOF;d=getchar()) {} }
        if (c=='m') return 1;
        if (c=='q' || c==EOF) return 0;
        if (c=='r' && test_pass(cfg,links,plan,h,++number)) return -1;
    }
}
