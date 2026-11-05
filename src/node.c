/* node.c - bounded control server and CONNECTION setup probes.
 *
 * PING/NONCE/AUTH are the only commands allowed before authentication.
 * Each connection owns its challenge and authorization. Connection storage is
 * allocated once; slow readers disable input until reply space is available.
 *
 * Start at node_run and dispatch. The event loop handles control connections;
 * BENCH has a separate job, and SESSION attaches a receiver/compute worker from
 * plane.c. RECV temporarily treats input as raw unit bytes, then returns to RESP.
 * Control authentication is not evidence that UDP expert replies work: the
 * head's demo and real generation test that separate boundary.
 *
 * A session belongs to its authenticated client. Stop/join its receiver before
 * changing the backend's borrowed unit mappings. Keep reusable real weights
 * loaded between compatible sessions; synthetic demo units must not survive as
 * real model weights. See docs/PROTOCOL.md and docs/llm/RUNTIME.md.
 */
#include "wire.h"
#include "node.h"
#include "ae.h"
#include "net.h"
#include "resp.h"
#include "auth.h"
#include "sha256.h"
#include "connection.h"
#include "hw.h"
#include "identity.h"
#include "backend.h"
#include "bench.h"
#include "neon.h"
#include "selection.h"
#include "plane.h"
#include "units.h"
#include "prep.h"
#include "avx2.h"
#include <pthread.h>
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#define NODE_CLIENTS 64
/* A pipelined control command can return every compiled backend's cost table. */
#define REPLY_RESERVE (FR_SELECTION_TEXT+1024)
typedef struct {
    int fd, mask, eof, closing, bench_pending;
    uint64_t tag;
    size_t in_len, out_len;
    auth_state auth;
    uint32_t link_session;
    unsigned link_node;
    struct sockaddr_in udp_peer;
    unsigned udp_index; /* interface the head's probes arrived on */
    int rx_fd; unsigned rx_block; uint64_t rx_left; sha256_ctx rx_sha; char rx_hex[65];
    char input[RESP_MAX_MESSAGE], output[RESP_MAX_MESSAGE];
} client;
typedef struct { uint64_t p[3]; unsigned count, losses; char hist[192]; } link_report;
typedef struct {
    pthread_t thread;
    atomic_int done,cancel;
    int active,result;
    unsigned blocks,candidate,burst;
    const fr_backend *backend;
    uint64_t owner;
    fr_bench_report report;
    int sweep,refresh; fr_hwinfo hw; fr_selection selection; const char *cache; /* BENCH SWEEP */
} bench_job;
typedef struct {
    const fr_config *config;
    int loop, listener, port, debug;
    unsigned connected;
    uint64_t accepted, rejected, commands, protocol_errors, received, sent, generation;
    uint64_t nonces, auth_successes, auth_failures, noauth_commands, entropy_errors;
    int discovery, multicast, data, discovery_port, data_port;
    int64_t multicast_refreshed; int refresh_warned;
    uint64_t hellos, probes, multicast_probes, udp_ignored, udp_dropped;
    unsigned probe_index; /* arrival interface of the last authenticated probe */
    uint32_t report_session;
    link_report reports[3];
    fr_interface interfaces[FR_INTERFACES],head_path;
    unsigned interface_count;
    int head_path_valid;
    unsigned char discovery_cookie[16];
    int discovery_cookie_valid;
    uint64_t id;
    fr_hwinfo hw;
    int hw_valid;
    uint64_t hw_samples, hw_errors;
    bench_job bench;
    fr_bench_report measured;
    const fr_backend *measured_backend;
    fr_selection selection;
    int bench_valid;
    uint64_t bench_errors;
    fr_receiver *receiver;
    fr_units units; unsigned units_burst; /* burst the kept units were started with */
    int units_synthetic; /* the kept units are SESSION SYNTHETIC ones: reload for a real session */
    uint64_t session_owner;
    struct timespec started;
    client *clients;
} node;
/* Units stay loaded after a session: the next head usually plans the same
 * blocks, and reloading tens of GB took most of a minute (BENCH run 38). They
 * are dropped when a session needs other blocks or a backend, or before a
 * benchmark that loads its own. */
/* ---- Session lifetime: detach the receiver before changing loaded units. ---- */
static int session_stop(node *s) {
    int rc=0;
    if (s->receiver) {
        rc=fr_receiver_close(s->receiver); free(s->receiver); s->receiver=NULL; s->session_owner=0;
        if (ae_set(s->loop,s->data,0,AE_READ,UINT64_MAX-2) ||
            ae_set(s->loop,s->multicast,0,AE_READ,UINT64_MAX-1)) rc=-1;
    }
    /* Real units stay loaded for the next session (they are file cache); a demo's
     * synthetic ones are anonymous memory the next plan must find free again. */
    if (s->units_synthetic) { if (fr_units_close(&s->units)) rc=-1; s->units_synthetic=0; }
    return rc;
}

static volatile sig_atomic_t stopping;
static void stop_signal(int signal) { (void)signal; stopping=1; }

static void log_line(const char *level, const char *format, ...) {
    time_t now=time(NULL); struct tm tm; char stamp[32]="unknown-time";
    if (gmtime_r(&now,&tm)) (void)strftime(stamp,sizeof(stamp),"%Y-%m-%dT%H:%M:%SZ",&tm);
    fprintf(stderr,"%s %s node ",stamp,level);
    va_list args; va_start(args,format); vfprintf(stderr,format,args); va_end(args);
    fputc('\n',stderr);
}
static int same(const resp_value *v, const char *text) {
    size_t n=strlen(text);
    if (v->len!=n) return 0;
    for (size_t i=0; i<n; ++i)
        if (tolower((unsigned char)v->data[i]) != tolower((unsigned char)text[i])) return 0;
    return 1;
}
static int add_text(char *text, size_t cap, size_t *used, const char *format, ...) {
    va_list args; va_start(args,format);
    int n=vsnprintf(text+*used,cap-*used,format,args);
    va_end(args);
    if (n<0 || (size_t)n>=cap-*used) return -1;
    *used+=(size_t)n; return 0;
}
#ifdef FR_BENCH_RUNTIME
/* ---- Calibration: one cancellable job owns its backend and replies to its client. ---- */
static void *bench_worker(void *arg) {
    bench_job *job=arg; fr_costmodel cost;
    if (job->sweep) {
        job->result=selection_sweep_local(&job->hw,job->burst,0,&job->selection,stderr,job->cache,job->refresh,&job->cancel)?FR_STATE:FR_OK;
        atomic_store(&job->done,1); return NULL;
    }
    job->result=job->burst?fr_bench_configure_point(job->blocks,job->burst,&job->cancel):fr_bench_configure(job->blocks,&job->cancel);
    if (!job->result) job->result=job->backend->bench(FR_BENCH_BURST_MAX,&cost);
    if (!job->result) job->result=fr_bench_last(&job->report);
    (void)fr_bench_configure(3,NULL);
    atomic_store(&job->done,1); return NULL;
}
#endif
static void bench_reply(const fr_bench_report *report,resp_buf *out) {
    char text[FR_BENCH_TEXT]; int n=fr_bench_format(report,text,sizeof(text));
    if (n<0) resp_error(out,"ERR invalid benchmark result");
    else resp_bulk(out,text,(size_t)n);
}
static int argument_unsigned(const resp_value *v,unsigned *out) {
    if (!v->len || v->len>3) return -1;
    unsigned n=0;
    for (size_t i=0;i<v->len;++i) { unsigned d=(unsigned char)v->data[i]-'0'; if (d>9) return -1; n=n*10+d; }
    *out=n; return 0;
}
/* Cached unit pages are memory our own session reuses: hw_node_available
 * counts file cache once, so kept units neither shrink nor inflate the cap. */
static int node_finalize(node *s,fr_hwinfo *hw) {
    if (hw_node_available(hw)) return -1;
    /* Owned SSD cache/staging is reusable by the next session, unlike other
     * applications' anonymous RAM. Do not subtract the same budget twice. */
    if (s->units.mask && s->units.paged && s->units.backend && !strncmp(s->units.backend->name,"metal",5)) {
        uint64_t reusable=s->units.cache_bytes+(unsigned)__builtin_popcount(s->units.paged)*FR_NATIVE_LAYER_BYTES+(4ull<<20);
        hw->available=reusable>hw->total-hw->available?hw->total:hw->available+reusable;
    }
    return hw_finalize(hw);
}
static int candidate_list(node *s,unsigned burst) {
    fr_hwinfo hw;
    if (hw_probe(&hw)) return -1;
    hw.margin=s->config->mem_margin; hw.head_reserve=s->config->head_reserve;
    return node_finalize(s,&hw) || selection_discover(&hw,burst,0,&s->selection);
}
static void backend_command(node *s,resp_value *v,resp_buf *out) {
    unsigned blocks;
    if (v[0].len!=2 || argument_unsigned(&v[2],&blocks) || !blocks || blocks>FR_BLOCKS) { resp_error(out,"ERR BACKEND requires assigned blocks"); return; }
    if (s->bench.active || s->receiver) { resp_error(out,"BUSY benchmark or session running"); return; }
    int best=selection_pick(&s->selection,blocks); fr_hwinfo hw;
    if (best<0 || hw_probe(&hw)) { resp_error(out,"ERR complete backend sweep required"); return; }
    hw.margin=s->config->mem_margin; hw.head_reserve=s->config->head_reserve;
    if (node_finalize(s,&hw) || !selection_fits(&s->selection.candidate[best],&hw,0,blocks)) { resp_error(out,"ERR selected backend capacity changed; rerun DISCOVERY"); return; }
    char reason[1024];
    int n=selection_reason(&s->selection,blocks,reason,sizeof(reason));
    if (n<0) { resp_error(out,"ERR backend choice unavailable"); return; }
    s->selection.assigned=blocks; resp_bulk(out,reason,(size_t)n);
}
static void bench_command(node *s,client *c,resp_value *v,resp_buf *out) {
    if (s->receiver) { resp_error(out,"BUSY session running"); return; }
    if (v[0].len==2 && same(&v[2],"get")) {
        if (s->bench_valid) bench_reply(&s->measured,out);
        else resp_error(out,"ERR BENCH has no measurement");
        return;
    }
    if (s->bench.active) {
        /* A sweep whose head went away keeps running to the end; the next head
         * takes it over and gets its result instead of BUSY. */
        unsigned want=0; int owner_here=0;
        for (int i=0;i<NODE_CLIENTS;++i) if (s->clients[i].fd>=0 && s->clients[i].tag==s->bench.owner) owner_here=1;
        if (s->bench.sweep && !owner_here && (v[0].len==3 || v[0].len==4) && same(&v[2],"sweep") && !argument_unsigned(&v[3],&want) && want==s->bench.burst) {
            s->bench.owner=c->tag; c->bench_pending=1; log_line("notice","BENCH sweep taken over by a new head"); return;
        }
        resp_error(out,"BUSY benchmark running"); return;
    }
    if (v[0].len==3 && same(&v[2],"list")) {
        unsigned burst; char text[FR_SELECTION_TEXT];
        if (argument_unsigned(&v[3],&burst) || candidate_list(s,burst)) { resp_error(out,"ERR backend eligibility unavailable"); return; }
        int n=selection_format(&s->selection,text,sizeof(text));
        if (n<0) resp_error(out,"ERR invalid backend list"); else resp_bulk(out,text,(size_t)n);
        return;
    }
    unsigned blocks=3,burst=s->config->burst,candidate=FR_CANDIDATES;
    s->bench.sweep=0;
    if ((v[0].len==3 || v[0].len==4) && same(&v[2],"sweep")) {
        /* Node-side cached sweep: oracle rule, short probe, winner-only statistics. */
        if (argument_unsigned(&v[3],&burst) || !burst || burst>FR_BENCH_BURST_MAX || hw_probe(&s->bench.hw)) { resp_error(out,"ERR BENCH SWEEP burst [refresh]"); return; }
        s->bench.sweep=1; s->bench.refresh=v[0].len==4 && same(&v[4],"refresh");
        /* A refresh measures for real and loads its own blocks: free the kept units.
         * So does a sweep whose SELFTEST passes are not saved yet (a loaded backend
         * refuses SELFTEST). Otherwise kept units already count as free
         * (node_finalize adds resident unit pages). */
        if (s->bench.refresh || (s->units.mask && s->units_burst!=burst) ||
            !selection_self_tests_cached(&s->bench.hw,burst,0,s->config->state_dir)) (void)fr_units_close(&s->units);
        s->bench.hw.margin=s->config->mem_margin; s->bench.hw.head_reserve=s->config->head_reserve;
        if (node_finalize(s,&s->bench.hw)) { resp_error(out,"ERR hardware probe unavailable"); return; } s->bench.cache=s->config->state_dir; s->bench.burst=burst;
        atomic_store(&s->bench.done,0); atomic_store(&s->bench.cancel,0); s->bench.owner=c->tag; s->selection.assigned=0;
        if (pthread_create(&s->bench.thread,NULL,bench_worker,&s->bench)) { ++s->bench_errors; resp_error(out,"ERR cannot start benchmark worker"); return; }
        s->bench.active=1; c->bench_pending=1; return;
    }
    (void)fr_units_close(&s->units); /* the benchmarks below load their own blocks */
    if (v[0].len==1) {
        /* Legacy standalone Metal diagnostic. It does not select a backend. */
        if (candidate_list(s,burst)) { resp_error(out,"ERR backend benchmark unavailable"); return; }
        for (unsigned i=0;i<s->selection.count;++i) if (!strcmp(s->selection.candidate[i].name,"metal")) candidate=i;
    } else if (v[0].len==4) {
        if (argument_unsigned(&v[3],&blocks) || argument_unsigned(&v[4],&burst) || burst!=s->selection.burst) { resp_error(out,"ERR BENCH burst differs from LIST"); return; }
        for (unsigned i=0;i<s->selection.count;++i) if (same(&v[2],s->selection.candidate[i].name)) candidate=i;
    } else { resp_error(out,"ERR wrong number of arguments for BENCH [LIST burst | backend blocks burst | GET]"); return; }
    if (candidate>=s->selection.count || !blocks || blocks>FR_SWEEP || blocks>s->selection.candidate[candidate].capacity) {
        resp_error(out,"ERR BENCH backend/share ineligible"); return;
    }
    s->bench.backend=fr_backend_find(s->selection.candidate[candidate].name);
    if (!s->bench.backend) { resp_error(out,"ERR backend not compiled"); return; }
    atomic_store(&s->bench.done,0); atomic_store(&s->bench.cancel,0);
    s->bench.blocks=blocks; s->bench.burst=v[0].len==1?0:burst; s->bench.candidate=candidate; s->bench.owner=c->tag; s->bench_valid=0; s->selection.assigned=0;
    int error=pthread_create(&s->bench.thread,NULL,bench_worker,&s->bench);
    if (error) { ++s->bench_errors; resp_error(out,"ERR cannot start benchmark worker"); return; }
    s->bench.active=1; c->bench_pending=1;
}
static int bench_complete(node *s) {
    if (!s->bench.active || !atomic_load(&s->bench.done)) return 0;
    if (pthread_join(s->bench.thread,NULL)) return -1;
    s->bench.active=0;
    if (s->bench.sweep && !s->bench.result) s->selection=s->bench.selection;
    else if (!s->bench.result) {
        unsigned b=s->selection.burst-1;
        if (selection_record(&s->selection,s->bench.candidate,s->bench.blocks,
            s->bench.report.percentile[0][b][FR_METAL_TOTAL][0],s->bench.report.percentile[0][b][FR_METAL_TOTAL][1],s->bench.report.resident_bytes) ||
            selection_settings(&s->selection,s->bench.candidate,s->bench.blocks,s->bench.report.workers,s->bench.report.prefetch)) s->bench.result=FR_STATE;
    }
    if (s->bench.result) { ++s->bench_errors; log_line("warning","BENCH failed status=%d",s->bench.result); }
    else if (!s->bench.sweep) { s->measured=s->bench.report; s->measured_backend=s->bench.backend; s->bench_valid=1; }
    for (int i=0;i<NODE_CLIENTS;++i) {
        client *c=&s->clients[i]; if (c->fd<0 || c->tag!=s->bench.owner) continue;
        c->bench_pending=0;
        resp_buf out; resp_init(&out,c->output,sizeof(c->output)); out.len=c->out_len;
        if (s->bench.result) {
            char error[80]; snprintf(error,sizeof(error),"ERR backend benchmark failed status=%d",s->bench.result); resp_error(&out,error);
        }
        else if (s->bench.sweep) {
            char text[FR_SELECTION_TEXT]; int n=selection_format(&s->selection,text,sizeof(text));
            if (n<0) resp_error(&out,"ERR invalid backend list"); else resp_bulk(&out,text,(size_t)n);
        } else bench_reply(&s->measured,&out);
        if (out.error) return -1;
        c->out_len=out.len;
        int mask=c->mask|AE_WRITE;
        if (ae_set(s->loop,c->fd,c->mask,mask,c->tag)) return -1;
        c->mask=mask; break;
    }
    return 0;
}
/* ---- Diagnostics: bounded text snapshots; no kernel execution here. ---- */
static void info(node *s, const resp_value *section, resp_buf *out) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC,&now)<0) { resp_error(out,"ERR cannot read uptime"); return; }
    long uptime=(long)(now.tv_sec-s->started.tv_sec)-(now.tv_nsec<s->started.tv_nsec);
#ifdef FRONK_DEBUG
    const char *build="debug";
#else
    const char *build="release";
#endif
    const char *names[]={"server","backend","memory","session","dataplane","system"};
    int selected=-1;
    if (section && !same(section,"all")) {
        for (int i=0; i<6; ++i) if (same(section,names[i])) selected=i;
        if (selected<0) { resp_error(out,"ERR unknown INFO section"); return; }
    }
    char text[6144]; size_t used=0, input_bytes=0, output_bytes=0; unsigned authenticated=0, links=0;
    for (int i=0; i<NODE_CLIENTS; ++i) if (s->clients[i].fd>=0) {
        input_bytes+=s->clients[i].in_len; output_bytes+=s->clients[i].out_len;
        authenticated+=(unsigned)s->clients[i].auth.authenticated;
        links+=s->clients[i].link_session!=0;
    }
    for (int i=0; i<6; ++i) {
        if (selected>=0 && i!=selected) continue;
        int rc=add_text(text,sizeof(text),&used,"# %s\r\n",names[i]);
        if (rc<0) { out->error=1; return; }
        switch (i) {
        case 0:
            rc=add_text(text,sizeof(text),&used,
                "fronk_version:%s\r\nmode:node\r\nbuild:%s\r\nmilestone:10\r\n"
                "name:%s\r\nid:%016" PRIx64 "\r\npid:%ld\r\nuptime_seconds:%ld\r\ncontrol_port:%d\r\n"
                "auth:hmac-sha256\r\nconnected_clients:%u\r\ntotal_connections:%" PRIu64 "\r\n"
                "rejected_connections:%" PRIu64 "\r\ncommands:%" PRIu64 "\r\nprotocol_errors:%" PRIu64 "\r\n"
                "authenticated_clients:%u\r\nnonces_issued:%" PRIu64 "\r\nauth_successes:%" PRIu64 "\r\n"
                "auth_failures:%" PRIu64 "\r\nnoauth_commands:%" PRIu64 "\r\nentropy_errors:%" PRIu64 "\r\n",
                FR_VERSION,build,s->config->name,s->id,(long)getpid(),uptime,s->port,
                s->connected,s->accepted,s->rejected,s->commands,s->protocol_errors,
                authenticated,s->nonces,s->auth_successes,s->auth_failures,s->noauth_commands,s->entropy_errors);
            break;
        case 1: {
            int best=selection_pick(&s->selection,s->selection.assigned);
            const char *chosen=s->units.mask?s->units.backend->name:best>=0?s->selection.candidate[best].name:"unmeasured";
            rc=add_text(text,sizeof(text),&used,
                    "backend:%s\r\nbackend_ready:%d\r\nbackends_compiled:%s\r\nreference_ready:1\r\n"
                    "golden_vectors_version:1\r\ngolden_format:q4_0\r\nformat:q4_k,mxfp4,q8_0\r\nblocks_loaded:%u\r\n",
                    chosen,best>=0 || s->units.mask!=0,fr_backends_compiled(),(unsigned)__builtin_popcount(s->units.mask));
            if (!rc && best>=0) {
                char reason[1024];
                if (selection_reason(&s->selection,s->selection.assigned,reason,sizeof(reason))<0) rc=-1;
                else rc=add_text(text,sizeof(text),&used,"backend_assigned_blocks:%u\r\nbackend_reason:%s\r\n",s->selection.assigned,reason);
            }
            for (unsigned k=0;!rc && k<s->selection.count;++k) {
                fr_candidate *candidate=&s->selection.candidate[k];
                rc=add_text(text,sizeof(text),&used,"candidate_%s_capacity:%u\r\ncandidate_%s_measured_mask:%u\r\n",candidate->name,candidate->capacity,candidate->name,candidate->measured);
                for (unsigned b=0;!rc && b<FR_SWEEP;++b) if (candidate->measured&(1u<<b))
                    rc=add_text(text,sizeof(text),&used,"candidate_%s_%u_p50_ns:%.0f\r\ncandidate_%s_%u_p99_ns:%.0f\r\n",candidate->name,b+1,candidate->ns[b],candidate->name,b+1,candidate->p99[b]);
            }
#ifdef FR_NEON_COMPUTE
            if (!rc) {
                unsigned features=fr_neon_features();
                rc=add_text(text,sizeof(text),&used,"neon_dotprod:%u\r\nneon_i8mm:%u\r\nneon_kernel:dotprod_q8\r\n",
                    !!(features&FR_ARM_DOTPROD),!!(features&FR_ARM_I8MM));
            }
#endif
#ifdef FR_AVX2_COMPUTE
            if (!rc) rc=add_text(text,sizeof(text),&used,
                "avx2_kernel:shuttle_q4k_mxfp4_panel\r\navx2_weight_formats:q4_k,mxfp4,q8_0,q4_0\r\n"
                "avx2_isa_required:avx2,fma,f16c,os_ymm\r\n");
#endif
            if (!rc) rc=add_text(text,sizeof(text),&used,"bench_status:%s\r\nbench_errors:%" PRIu64 "\r\n",
                s->bench.active?"running":s->bench_valid?"ready":s->bench_errors?"failed":"unmeasured",s->bench_errors);
            if (!rc && s->bench_valid) {
                rc=add_text(text,sizeof(text),&used,"bench_backend:%s\r\nbench_blocks:%u\r\nbench_samples:%u\r\nbench_resident_bytes:%" PRIu64
                    "\r\nhot_path_allocations:%" PRIu64 "\r\nbw_eff_gbs:%.9g\r\nt0_us:%.9g\r\nburst_slope:%.9g\r\nsustained_ratio:%.9g\r\n",
                    s->measured_backend->name,s->measured.blocks,s->measured.samples,s->measured.resident_bytes,s->measured.hot_allocations,
                    s->measured.cost.bw_eff/1e9,s->measured.cost.t0_ns/1000,s->measured.burst_slope,s->measured.sustained_ratio);
                if (!rc && s->measured.workers) rc=add_text(text,sizeof(text),&used,"bench_workers:%u\r\nbench_prefetch_groups:%u\r\n",s->measured.workers,s->measured.prefetch);
                for (unsigned m=0;!rc && m<2;++m) for (unsigned b=0;!rc && b<4;++b)
                    rc=add_text(text,sizeof(text),&used,"submission_%s_%u_overhead_p50_ns:%.0f\r\nsubmission_%s_%u_overhead_p99_ns:%.0f\r\n",
                        m?"reused":"distinct",b+1,s->measured.percentile[m][b][FR_METAL_OVERHEAD][0],
                        m?"reused":"distinct",b+1,s->measured.percentile[m][b][FR_METAL_OVERHEAD][1]);
            }
            break;
        }
        case 2: rc=add_text(text,sizeof(text),&used,
                    "weights_bytes:%" PRIu64 "\r\ncontrol_buffer_bytes:%zu\r\ncontrol_input_bytes:%zu\r\ncontrol_output_bytes:%zu\r\n"
                    "mem_total_bytes:%" PRIu64 "\r\nmem_available_bytes:%" PRIu64 "\r\nmem_margin_bytes:%" PRIu64 "\r\nmem_cap_bytes:%" PRIu64 "\r\n",
                    (uint64_t)__builtin_popcount(s->units.mask)*FR_NATIVE_BLOCK_BYTES,sizeof(client)*NODE_CLIENTS,input_bytes,output_bytes,s->hw_valid?s->hw.total:0,s->hw_valid?s->hw.available:0,
                    s->hw_valid?s->hw.margin:0,s->hw_valid?s->hw.cap:0); break;
        case 3: rc=add_text(text,sizeof(text),&used,"session:%s\r\nsession_id:%u\r\nunit_mask:%u\r\n",
                    s->receiver?"active":"inactive",s->receiver?s->receiver->session:0,s->units.mask); break;
        case 4:
            rc=add_text(text,sizeof(text),&used,"dataplane:%s\r\ndata_port:%d\r\n",s->receiver?"active":"idle",s->data_port);
            if (!rc && s->receiver) {
                fr_plane_stats *p=&s->receiver->stats;
                rc=add_text(text,sizeof(text),&used,
                    "requests:%llu\r\nparts:%llu\r\nduplicates:%llu\r\nlate:%llu\r\ninvalid:%llu\r\nrequest_gaps:%llu\r\noverwritten:%llu\r\ncompute_errors:%llu\r\ncompute_ns:%llu\r\nrx_ns:%llu\r\nwait_ns:%llu\r\nreply_ns:%llu\r\n",
                    (unsigned long long)atomic_load(&p->requests),(unsigned long long)atomic_load(&p->parts),
                    (unsigned long long)atomic_load(&p->duplicates),(unsigned long long)atomic_load(&p->late),
                    (unsigned long long)atomic_load(&p->invalid),(unsigned long long)atomic_load(&p->gaps),
                    (unsigned long long)atomic_load(&p->overwritten),(unsigned long long)atomic_load(&p->errors),
                    (unsigned long long)atomic_load(&p->compute_ns),(unsigned long long)atomic_load(&p->rx_ns),
                    (unsigned long long)atomic_load(&p->wait_ns),(unsigned long long)atomic_load(&p->reply_ns));
            }
            break;
        default: rc=add_text(text,sizeof(text),&used,
                    "event_loop:%s\r\nos:%s\r\narch:%s\r\ndebug_enabled:%d\r\nmax_clients:%d\r\n"
                    "bytes_received:%" PRIu64 "\r\nbytes_sent:%" PRIu64 "\r\n"
                    "discovery_port:%d\r\nlink_active:%u\r\nlink_report_session:%u\r\n"
                    "discovery_hellos:%" PRIu64 "\r\nlink_probes:%" PRIu64 "\r\nmulticast_probes:%" PRIu64 "\r\n"
                    "udp_ignored:%" PRIu64 "\r\nudp_send_drops:%" PRIu64 "\r\n",
                    ae_name(),FR_BUILD_OS,FR_BUILD_ARCH,s->debug,NODE_CLIENTS,s->received,s->sent,
                    s->discovery_port,links,s->report_session,s->hellos,s->probes,s->multicast_probes,s->udp_ignored,s->udp_dropped);
            for (int k=0; !rc && k<3; ++k) {
                const char *size=k==0?"64":k==1?"5k":"20k"; link_report *r=&s->reports[k];
                rc=add_text(text,sizeof(text),&used,
                    "link_rtt_%s_samples:%u\r\nlink_rtt_%s_losses:%u\r\nlink_rtt_%s_p50_ns:%" PRIu64 "\r\n"
                    "link_rtt_%s_p99_ns:%" PRIu64 "\r\nlink_rtt_%s_p999_ns:%" PRIu64 "\r\nlink_rtt_%s_hist:%s\r\n",
                    size,r->count,size,r->losses,size,r->p[0],size,r->p[1],size,r->p[2],size,r->hist);
            }
            if (!rc) rc=add_text(text,sizeof(text),&used,"hw_status:%s\r\nhw_samples:%" PRIu64 "\r\nhw_errors:%" PRIu64 "\r\n",
                s->hw_valid?"ok":s->hw_errors?"error":"unqueried",s->hw_samples,s->hw_errors);
            if (!rc && s->head_path_valid) rc=add_text(text,sizeof(text),&used,
                "head_interface:%s\r\nhead_interface_address:%s\r\nhead_link_type:%s\r\nhead_link_speed_bps:%" PRIu64 "\r\n",
                s->head_path.name,s->head_path.address,interface_type(s->head_path.type),s->head_path.speed_bps);
            if (!rc && s->hw_valid) {
                char hardware[FR_HW_TEXT];
                if (hw_format(&s->hw,hardware,sizeof(hardware))<0) rc=-1;
                else {
                    /* HW describes pre-selection eligibility. INFO's backend
                     * section owns the selected state; do not shadow it here. */
                    char *policy=strstr(hardware,"backend:"); if (policy) *policy=0;
                    rc=add_text(text,sizeof(text),&used,"%s",hardware);
                }
            }
            break;
        }
        if (rc<0 || add_text(text,sizeof(text),&used,"\r\n")<0) { out->error=1; return; }
    }
    resp_bulk(out,text,used);
}
static int decimal(const resp_value *v, uint64_t max, uint64_t *result) {
    if (!v->len) return -1;
    uint64_t n=0;
    for (size_t i=0; i<v->len; ++i) {
        unsigned char c=(unsigned char)v->data[i];
        if (c<'0' || c>'9' || (unsigned)(c-'0')>max || n>(max-(unsigned)(c-'0'))/10) return -1;
        n=n*10+(unsigned)(c-'0');
    }
    *result=n; return 0;
}
static int histogram(const resp_value *v, unsigned count, char out[192]) {
    if (!v->len || v->len>=192) return -1;
    unsigned sum=0, value=0, bins=0, digits=0;
    for (size_t i=0; i<=v->len; ++i) {
        char c=i==v->len?',':v->data[i];
        if (c==',') {
            if (!digits || ++bins>32 || value>count-sum) return -1;
            sum+=value; value=digits=0;
        } else {
            if (c<'0' || c>'9' || ++digits>4) return -1;
            value=value*10+(unsigned)(c-'0');
        }
    }
    if (bins!=32 || sum!=count) return -1;
    memcpy(out,v->data,v->len); out[v->len]='\0'; return 0;
}
/* RECV block sha256: the next FR_NATIVE_BLOCK_BYTES on this connection are
 * the unit, hashed while written to a temporary file and renamed on a match. */
/* ---- Weight transfer: stream into temporary storage; publish after SHA-256. ---- */
static void recv_command(node *s,client *c,resp_value *v,resp_buf *out) {
    uint64_t block; struct statvfs fs; char path[300];
    if (v[0].len!=3 || decimal(&v[2],FR_BLOCKS-1,&block) || v[3].len!=64) { resp_error(out,"ERR RECV block sha256"); return; }
    for (size_t i=0;i<64;++i) if (!isxdigit((unsigned char)v[3].data[i]) || isupper((unsigned char)v[3].data[i])) { resp_error(out,"ERR RECV block sha256"); return; }
    if (s->receiver) { resp_error(out,"BUSY unit in use by a session"); return; }
    if (s->units.mask&(1u<<block)) (void)fr_units_close(&s->units); /* kept from the last session; about to be replaced */
    if (statvfs(s->config->state_dir,&fs)) { resp_error(out,"ERR cannot inspect unit-cache disk; check state-dir on this node"); return; }
    uint64_t available=(uint64_t)fs.f_bavail*fs.f_frsize,needed=FR_NATIVE_BLOCK_BYTES+((uint64_t)1<<30);
    if (available<needed) {
        char error[200];
        snprintf(error,sizeof(error),"ERR unit %02u needs %llu bytes free including reserve; %llu available. Free disk space on this node or use a larger state-dir",
                 (unsigned)block,(unsigned long long)needed,(unsigned long long)available);
        resp_error(out,error); return;
    }
    snprintf(path,sizeof(path),"%s/unit-%02u.tmp",s->config->state_dir,(unsigned)block);
    if ((c->rx_fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0600))<0) { resp_error(out,"ERR cannot create unit"); return; }
    c->rx_block=(unsigned)block; c->rx_left=FR_NATIVE_BLOCK_BYTES; memcpy(c->rx_hex,v[3].data,64); c->rx_hex[64]=0;
    sha256_init(&c->rx_sha); resp_simple(out,"OK");
}
static void recv_data(node *s,client *c,resp_buf *out) {
    size_t n=c->in_len<c->rx_left?c->in_len:(size_t)c->rx_left; int bad=sha256_update(&c->rx_sha,c->input,n);
    for (size_t done=0;!bad && done<n;) { ssize_t w=write(c->rx_fd,c->input+done,n-done); if (w<=0) bad=1; else done+=(size_t)w; }
    c->in_len-=n; memmove(c->input,c->input+n,c->in_len); c->rx_left-=n;
    if (bad) c->rx_left=0;
    if (c->rx_left) return;
    unsigned char d[32]; char hex[65],tmp[300],path[300];
    if (sha256_final(&c->rx_sha,d)) bad=1;
    for (unsigned i=0;i<32;++i) snprintf(hex+2*i,3,"%02x",d[i]);
    if (close(c->rx_fd)) bad=1;
    c->rx_fd=-1;
    snprintf(tmp,sizeof(tmp),"%s/unit-%02u.tmp",s->config->state_dir,c->rx_block);
    snprintf(path,sizeof(path),"%s/unit-%02u.bin",s->config->state_dir,c->rx_block);
    if (bad || memcmp(hex,c->rx_hex,64) || rename(tmp,path) || fr_units_stamp(s->config->state_dir,c->rx_block,hex)) {
        unlink(tmp); resp_error(out,"ERR unit SHA-256 mismatch or write failure");
    } else resp_simple(out,"OK");
}
/* ---- Session control: validate ownership, choose measured backend, then load. ---- */
static void session_command(node *s,client *c,resp_value *v,resp_buf *out) {
    if (v[0].len==2 && same(&v[2],"stop")) {
        if (s->receiver && s->session_owner!=c->tag) { resp_error(out,"ERR session owned by another connection"); return; }
        if (session_stop(s)) resp_error(out,"ERR session stop failed"); else resp_simple(out,"OK");
        return;
    }
    uint64_t id,blocks,burst,deadline=0; int fixture=0;
#ifdef FRONK_DEBUG
    fixture=v[0].len==6 && same(&v[2],"test");
    /* SESSION SYNTHETIC: as START, on synthetic units of the same mask (fronk demo). */
    int synthetic=v[0].len==6 && same(&v[2],"synthetic");
#endif
    if (v[0].len!=6 || (!fixture && !synthetic && !same(&v[2],"start")) ||
        decimal(&v[3],UINT32_MAX,&id) || !id || id!=c->link_session ||
        decimal(&v[4],(1u<<FR_BLOCKS)-1,&blocks) || decimal(&v[5],8,&burst) || !burst ||
        (fixture?blocks>=FR_BLOCKS:!blocks || decimal(&v[6],UINT32_MAX,&deadline) || !deadline)) {
        resp_error(out,"ERR SESSION START id unit-mask burst deadline-us (authenticated LINK required)"); return;
    }
    if (s->receiver || s->bench.active) { resp_error(out,"BUSY session or benchmark running"); return; }
    const fr_backend *backend=NULL;
    uint32_t paged=0; uint64_t cache_bytes=FR_SSD_DEFAULT_CACHE;
    if (fixture) {
        char name[16]={0};
        if (!v[6].len || v[6].len>=sizeof(name)) { resp_error(out,"ERR test backend"); return; }
        memcpy(name,v[6].data,v[6].len); backend=fr_backend_find(name);
    } else {
        unsigned count=(unsigned)__builtin_popcount((unsigned)blocks);
        int best=selection_pick(&s->selection,count);
        if (best>=0 && s->selection.burst==burst && s->selection.assigned==count) {
            const fr_candidate *candidate=&s->selection.candidate[best];
            backend=fr_backend_find(candidate->name); paged=selection_paged(candidate,(uint32_t)blocks);
            if (candidate->ssd_blocks) cache_bytes=candidate->cache_bytes;
        }
#ifdef FR_AVX2_COMPUTE
        /* Kept units of this exact share and burst already run with its tuning.
         * Retuning needs the AVX2 state empty (fr_avx2_close refuses loaded
         * blocks), so any other share unloads first; keeping them made every
         * SESSION START after a head fail with "backend unavailable" (run 44). */
        if (backend==&fr_avx2 && !(s->units.mask==(uint32_t)blocks && s->units.backend==backend && s->units_burst==(unsigned)burst)) {
            (void)fr_units_close(&s->units);
            if (fr_avx2_configure(count,(unsigned)burst)) backend=NULL;
        }
#endif
    }
    fr_hwinfo hardware;
    if (!backend || backend->probe(&hardware)) { resp_error(out,"ERR BENCH-selected backend unavailable"); return; }
    /* Auto inventory deduplicates physical interfaces, but the established
     * control path may use a secondary IPv4 alias (the real iMac does).
     * Pin PART to that exact local address instead of guessing by subnet. */
    fr_udp_info local={0}; struct sockaddr_in endpoint; socklen_t endpoint_len=sizeof(endpoint);
    fr_interface resolved[FR_INTERFACES]; unsigned resolved_count=0; char address[16];
    if (getsockname(c->fd,(struct sockaddr *)&endpoint,&endpoint_len) || endpoint.sin_family!=AF_INET ||
        !inet_ntop(AF_INET,&endpoint.sin_addr,address,sizeof(address)) ||
        interfaces_read(address,NULL,resolved,&resolved_count) || resolved_count!=1) {
        resp_error(out,"ERR no interface for linked head"); return;
    }
    /* The probes' arrival interface beats address lookup when an address is
     * also left on an inactive bridge member (the iMac's en2). */
    local.index=c->udp_index?c->udp_index:s->probe_index?s->probe_index:resolved[0].index; /* one head at a time */
    local.address=endpoint.sin_addr;
    int loaded=0;
    if (fixture || synthetic || s->units_synthetic || s->units.mask!=(uint32_t)blocks || s->units.backend!=backend || s->units.paged!=paged || (paged && s->units.cache_bytes!=cache_bytes)) {
        (void)fr_units_close(&s->units);
        loaded=fixture?fr_units_fixture(&s->units,backend,(unsigned)blocks):
               synthetic?fr_units_synthetic_cache(&s->units,backend,(uint32_t)blocks,paged,cache_bytes,s->config->state_dir):
               fr_units_open_cache(&s->units,backend,s->config->state_dir,(uint32_t)blocks,paged,cache_bytes);
        s->units_synthetic=synthetic; /* never kept for a real session */
    }
    if (loaded) { resp_error(out,"ERR unit load or SHA-256 validation failed"); return; }
    fr_receiver *r=calloc(1,sizeof(*r));
    if (!r) { fr_units_close(&s->units); resp_error(out,"ERR session allocation failed"); return; }
    if (ae_set(s->loop,s->data,AE_READ,0,UINT64_MAX-2) || ae_set(s->loop,s->multicast,AE_READ,0,UINT64_MAX-1) ||
        fr_receiver_open(r,s->data,s->multicast,(uint32_t)id,c->link_node,(unsigned)burst,&c->udp_peer,&local,backend)) {
        free(r); fr_units_close(&s->units);
        (void)ae_set(s->loop,s->data,0,AE_READ,UINT64_MAX-2); (void)ae_set(s->loop,s->multicast,0,AE_READ,UINT64_MAX-1);
        resp_error(out,"ERR session worker failed"); return;
    }
    s->receiver=r; s->session_owner=c->tag; s->units_burst=(unsigned)burst; resp_simple(out,"OK");
}
static void link_command(node *s, client *c, resp_value *v, resp_buf *out) {
    uint64_t n[6];
    if (v[0].len==2 && same(&v[2],"release")) {
        if (s->session_owner==c->tag) (void)session_stop(s);
        c->link_session=0; resp_simple(out,"OK"); return;
    }
    if (s->receiver) { resp_error(out,"BUSY session pins linked path"); return; }
    if (v[0].len==6 && same(&v[2],"path") && c->link_session) {
        fr_interface path={0}; struct in_addr ip;
        if (!v[3].len || v[3].len>=sizeof(path.name) || !v[6].len || v[6].len>=sizeof(path.address) ||
            decimal(&v[5],UINT64_C(1000000000000000),&path.speed_bps)) goto invalid;
        for (size_t i=0;i<v[3].len;++i) {
            unsigned char ch=(unsigned char)v[3].data[i];
            if (!isalnum(ch) && ch!='_' && ch!='-' && ch!='.') goto invalid;
        }
        if (same(&v[4],"ethernet")) path.type=FR_LINK_ETHERNET;
        else if (same(&v[4],"thunderbolt")) path.type=FR_LINK_THUNDERBOLT;
        else if (same(&v[4],"wifi")) path.type=FR_LINK_WIFI;
        else if (!same(&v[4],"other")) goto invalid;
        memcpy(path.name,v[3].data,v[3].len); memcpy(path.address,v[6].data,v[6].len);
        if (inet_pton(AF_INET,path.address,&ip)!=1 || ip.s_addr!=c->udp_peer.sin_addr.s_addr) goto invalid;
        s->head_path=path; s->head_path_valid=1; resp_simple(out,"OK"); return;
    }
    if (v[0].len==9 && same(&v[2],"report") && c->link_session) {
        uint64_t limits[]={2,1000,10,UINT64_MAX,UINT64_MAX,UINT64_MAX};
        for (int i=0; i<6; ++i) if (decimal(&v[3+i],limits[i],&n[i])) goto invalid;
        link_report report={.count=(unsigned)n[1],.losses=(unsigned)n[2],.p={n[3],n[4],n[5]}};
        if (!report.count || !n[3] || n[3]>n[4] || n[4]>n[5] || histogram(&v[9],report.count,report.hist)) goto invalid;
        s->reports[n[0]]=report; resp_simple(out,"OK"); return;
    }
    if (v[0].len!=4 || decimal(&v[2],UINT32_MAX,&n[0]) || !n[0] ||
        decimal(&v[3],65535,&n[1]) || !n[1] || decimal(&v[4],FR_CONNECTION_NODES-1,&n[2])) goto invalid;
    for (int i=0; i<NODE_CLIENTS; ++i) if (&s->clients[i]!=c && s->clients[i].fd>=0 && s->clients[i].link_session) {
        resp_error(out,"ERR link already owned by another connection"); return;
    }
    struct sockaddr_in peer; socklen_t len=sizeof(peer);
    if (getpeername(c->fd,(struct sockaddr *)&peer,&len)<0 || peer.sin_family!=AF_INET) {
        resp_error(out,"ERR cannot bind probe peer"); return;
    }
    peer.sin_port=htons((uint16_t)n[1]); c->udp_peer=peer;
    c->link_session=(uint32_t)n[0]; c->link_node=(unsigned)n[2];
    memset(s->reports,0,sizeof(s->reports)); s->report_session=c->link_session;
    s->head_path_valid=0;
    char id[17]; snprintf(id,sizeof(id),"%016" PRIx64,s->id);
    resp_array(out,4); resp_bulk(out,s->config->name,strlen(s->config->name));
    resp_integer(out,s->port); resp_integer(out,s->data_port); resp_bulk(out,id,16); return;
invalid:
    resp_error(out,"ERR invalid LINK arguments or inactive link");
}
/* ---- Control plane: AUTH replaces link authority; NONCE replaces the challenge. ---- */
static void dispatch(node *s, client *c, resp_value *v, resp_buf *out) {
    ++s->commands;
    if (v[0].type!=RESP_ARRAY || v[0].is_null || v[0].len<1 ||
        (v[1].type!=RESP_BULK && v[1].type!=RESP_STRING) || v[1].is_null) {
        resp_error(out,"ERR expected a command array of 1..16 strings"); return;
    }
    /* Recognize AUTH before validating its arguments: even an invalid attempt
     * consumes the challenge and replaces the connection's authorization. */
    if (same(&v[1],"auth")) {
        if (s->session_owner==c->tag) (void)session_stop(s);
        c->link_session=0;
        int valid=v[0].len==2 && (v[2].type==RESP_BULK || v[2].type==RESP_STRING) && !v[2].is_null;
        if (auth_verify(&c->auth,s->config->password,valid?v[2].data:NULL,valid?v[2].len:0)) {
            ++s->auth_successes; resp_simple(out,"OK");
        } else { ++s->auth_failures; resp_error(out,"WRONGPASS invalid or consumed challenge"); }
        return;
    }
    if (!c->auth.authenticated && !same(&v[1],"ping") && !same(&v[1],"nonce")) {
        ++s->noauth_commands; resp_error(out,"NOAUTH authentication required"); return;
    }
    if (v[0].len>16) { resp_error(out,"ERR too many command arguments"); return; }
    for (size_t i=1; i<=v[0].len; ++i) {
        if ((v[i].type!=RESP_BULK && v[i].type!=RESP_STRING) || v[i].is_null) {
            resp_error(out,"ERR command arguments must be strings"); return;
        }
    }
    if (same(&v[1],"ping")) {
        if (v[0].len!=1) resp_error(out,"ERR wrong number of arguments for PING");
        else resp_simple(out,"PONG");
    } else if (same(&v[1],"bench-profile")) {
        if (v[0].len!=1) resp_error(out,"ERR wrong number of arguments for BENCH-PROFILE");
        else resp_simple(out,"quick-widths-v1");
    } else if (same(&v[1],"expert-order")) {
        if (v[0].len!=1) resp_error(out,"ERR wrong number of arguments for EXPERT-ORDER");
#ifdef FR_EXPERT_RANKED
        else if (s->units.mask && s->units.backend && (!strcmp(s->units.backend->name,"avx2") || !strcmp(s->units.backend->name,"metal") || !strcmp(s->units.backend->name,"metal-ssd")))
            resp_simple(out,"ranked-v1");
#endif
        else resp_simple(out,"legacy");
    } else if (same(&v[1],"nonce")) {
        char hex[AUTH_NONCE_HEX+1];
        if (v[0].len!=1) resp_error(out,"ERR wrong number of arguments for NONCE");
        else if (auth_nonce(&c->auth,hex)<0) {
            if (s->session_owner==c->tag) (void)session_stop(s);
            c->link_session=0;
            ++s->entropy_errors; log_line("warning","cannot obtain nonce entropy");
            resp_error(out,"ERR nonce unavailable");
        } else { ++s->nonces; resp_bulk(out,hex,AUTH_NONCE_HEX); }
        secure_clear(hex,sizeof(hex));
    } else if (same(&v[1],"link")) {
        link_command(s,c,v,out);
    } else if (same(&v[1],"hw")) {
        if (v[0].len!=1) resp_error(out,"ERR wrong number of arguments for HW");
        else {
            fr_hwinfo hw; char text[FR_HW_TEXT]; int n;
            int rc=hw_probe(&hw);
            if (!rc) {
                hw.id=s->id; strcpy(hw.name,s->config->name);
                hw.margin=s->config->mem_margin; hw.head_reserve=s->config->head_reserve;
                rc=node_finalize(s,&hw);
            }
            if (rc || (n=hw_format(&hw,text,sizeof(text)))<0) {
                s->hw_valid=0; ++s->hw_errors; resp_error(out,"ERR hardware probe unavailable");
            } else {
                s->hw=hw; s->hw_valid=1; ++s->hw_samples;
                resp_bulk(out,text,(size_t)n);
            }
        }

    } else if (same(&v[1],"backend")) {
        backend_command(s,v,out);
    } else if (same(&v[1],"bench")) {
        bench_command(s,c,v,out);
    } else if (same(&v[1],"session")) {
        session_command(s,c,v,out);
    } else if (same(&v[1],"units")) {
        char text[FR_BLOCKS*68+1]; size_t n=0;
        for (unsigned b=0;b<FR_BLOCKS;++b) { char hex[65]; if (!fr_units_cached(s->config->state_dir,b,hex)) n+=(size_t)snprintf(text+n,sizeof(text)-n,"%02u %s\n",b,hex); }
        resp_bulk(out,text,n);
    } else if (same(&v[1],"recv")) {
        recv_command(s,c,v,out);
    } else if (same(&v[1],"blocks")) {
        if (v[0].len!=1) resp_error(out,"ERR BLOCKS takes no arguments");
        else resp_integer(out,s->units.mask);
    } else if (same(&v[1],"info")) {
        if (v[0].len>2) resp_error(out,"ERR wrong number of arguments for INFO");
        else info(s,v[0].len==2?&v[2]:NULL,out);
    } else resp_error(out,"ERR unknown command");
}
/* ---- TCP framing and backpressure. Partial reads stay in the owning client. ---- */
static int process(node *s, client *c) {
    if (c->rx_left && c->in_len) {
        resp_buf out; resp_init(&out,c->output,sizeof(c->output)); out.len=c->out_len;
        recv_data(s,c,&out); if (out.error) return -1;
        c->out_len=out.len;
    }
    if (c->rx_left && c->eof) { close(c->rx_fd); c->rx_fd=-1; c->rx_left=0; c->closing=1; }
    while (!c->rx_left && c->in_len && !c->closing && !c->bench_pending && sizeof(c->output)-c->out_len>=REPLY_RESERVE) {
        resp_value values[RESP_MAX_VALUES];
        ssize_t n=resp_parse(c->input,c->in_len,values,RESP_MAX_VALUES);
        if (!n && !c->eof && c->in_len<sizeof(c->input)) break;
        resp_buf out; resp_init(&out,c->output,sizeof(c->output)); out.len=c->out_len;
        if (n<=0) {
            ++s->protocol_errors;
            resp_error(&out,"ERR Protocol error or request exceeds limit");
            c->closing=1; c->in_len=0;
        } else {
            dispatch(s,c,values,&out);
            c->in_len-=(size_t)n; memmove(c->input,c->input+n,c->in_len);
            secure_clear(c->input+c->in_len,(size_t)n);
        }
        if (out.error) return -1;
        c->out_len=out.len;
    }
    if (c->eof && !c->in_len) c->closing=1;
    return 0;
}
static void disconnect(node *s, client *c) {
    if (s->session_owner==c->tag) (void)session_stop(s);
    /* A sweep outlives its head: the next head takes it over (BENCH SWEEP). */
    if (s->bench.active && s->bench.owner==c->tag && !s->bench.sweep) atomic_store(&s->bench.cancel,1);
    if (ae_set(s->loop,c->fd,c->mask,0,c->tag)<0)
        log_line("warning","unregister fd=%d: %s",c->fd,strerror(errno));
    if (close(c->fd)<0) log_line("warning","close fd=%d: %s",c->fd,strerror(errno));
    auth_reset(&c->auth);
    if (c->rx_fd>=0) { char tmp[300]; close(c->rx_fd); c->rx_fd=-1; c->rx_left=0;
        snprintf(tmp,sizeof(tmp),"%s/unit-%02u.tmp",s->config->state_dir,c->rx_block); unlink(tmp); }
    c->link_session=0;
    secure_clear(c->input,sizeof(c->input)); secure_clear(c->output,sizeof(c->output));
    c->fd=-1; --s->connected;
}
static int serve(node *s, client *c, int ready) {
    if ((ready&AE_READ) && !c->closing && !c->eof && c->in_len<sizeof(c->input)) {
        ssize_t n=recv(c->fd,c->input+c->in_len,sizeof(c->input)-c->in_len,0);
        if (n>0) { c->in_len+=(size_t)n; s->received+=(uint64_t)n; }
        else if (!n) c->eof=1;
        else if (errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) return -1;
    }
    if (process(s,c)<0) return -1;
    while (c->out_len) {
        ssize_t n=net_send(c->fd,c->output,c->out_len);
        if (n<0 && (errno==EAGAIN || errno==EWOULDBLOCK)) break;
        if (n<=0) return -1;
        s->sent+=(uint64_t)n; c->out_len-=(size_t)n;
        memmove(c->output,c->output+n,c->out_len);
    }
    /* Resume buffered pipelines after a writer makes room, even without a new
     * read event. The next write event flushes any newly queued replies. */
    if (process(s,c)<0 || (c->closing && !c->out_len)) return -1;
    int mask=c->out_len?AE_WRITE:0;
    if (!c->eof && !c->closing && c->in_len<sizeof(c->input) && sizeof(c->output)-c->out_len>=REPLY_RESERVE) mask|=AE_READ;
    if (mask!=c->mask) {
        if (ae_set(s->loop,c->fd,c->mask,mask,c->tag)<0) return -1;
        c->mask=mask;
    }
    return 0;
}
static int accept_clients(node *s) {
    /* Bound accepts per turn so a connection flood cannot starve live clients. */
    for (int accepted=0; accepted<NODE_CLIENTS; ++accepted) {
        int fd=net_tcp_accept(s->listener);
        if (fd<0) {
            if (errno==EAGAIN || errno==EWOULDBLOCK) return 0;
            if (errno==ECONNABORTED) continue;
            return -1;
        }
        int slot;
        for (slot=0; slot<NODE_CLIENTS && s->clients[slot].fd>=0; ++slot) {}
        if (slot==NODE_CLIENTS) { ++s->rejected; close(fd); continue; }
        client *c=&s->clients[slot];
        auth_reset(&c->auth);
        c->link_session=0;
        c->fd=fd; c->mask=AE_READ; c->eof=c->closing=c->bench_pending=0; c->in_len=c->out_len=0; c->rx_fd=-1; c->rx_left=0;
        c->tag=(++s->generation<<8)|(unsigned)(slot+1);
        if (ae_set(s->loop,fd,0,AE_READ,c->tag)<0) { close(fd); c->fd=-1; return -1; }
        ++s->connected; ++s->accepted;
    }
    return 0;
}
/* At each new head's HELLO and every 60 s, also during a session: an IGMP-
 * snooping switch with no querier stops forwarding the group ~4 min after the
 * last report (shuttle and hp behind one switch, 2026-11-02). */
/* ---- Discovery/probe sockets. Session datagrams use the dedicated receiver. ---- */
static void refresh_multicast(node *s) {
    for (unsigned i=0;i<s->interface_count;++i)
        if (!s->interfaces[i].loopback && s->interfaces[i].type!=FR_LINK_WIFI &&
            net_multicast_refresh(s->multicast,s->config->multicast,s->interfaces[i].address) && !s->refresh_warned) {
            s->refresh_warned=1;
            log_line("notice","multicast refresh failed on %s: %s; a snooping switch may stop forwarding the group",s->interfaces[i].name,strerror(errno));
        }
    s->multicast_refreshed=net_now_ns();
}
static int refresh_interfaces(node *s) {
    fr_interface current[FR_INTERFACES]; unsigned count=0;
    if (interfaces_read(s->config->interfaces,s->config->interface,current,&count)) return -1;
    for (unsigned i=0;i<count;++i) {
        unsigned old;
        for (old=0;old<s->interface_count && s->interfaces[old].index!=current[i].index;++old) {}
        if (old<s->interface_count) continue;
        if (current[i].loopback && net_multicast(s->discovery,FR_DISCOVERY_GROUP,current[i].address,1) && errno!=EADDRINUSE) return -1;
        if (current[i].type!=FR_LINK_WIFI && net_multicast(s->multicast,s->config->multicast,current[i].address,1) && errno!=EADDRINUSE)
            log_line("notice","multicast unavailable on %s; unicast remains available",current[i].name);
    }
    for (unsigned old=0;old<s->interface_count;++old) {
        unsigned i;
        for (i=0;i<count && current[i].index!=s->interfaces[old].index;++i) {}
        if (i<count) continue;
        (void)net_multicast_leave(s->discovery,FR_DISCOVERY_GROUP,s->interfaces[old].address);
        (void)net_multicast_leave(s->multicast,s->config->multicast,s->interfaces[old].address);
    }
    memcpy(s->interfaces,current,sizeof(current)); s->interface_count=count;
    refresh_multicast(s); return 0;
}
static int udp_ready(node *s, int fd, int discovery, int multicast) {
    for (int batch=0; batch<64; ++batch) {
        static unsigned char bytes[FR_FRAME_MAX],out[FR_FRAME_MAX]; unsigned char cookie[16]; struct sockaddr_in peer; fr_udp_info received;
        ssize_t n=net_udp_recv_info(fd,bytes,sizeof(bytes),&peer,&received);
        if (n<0) {
            if (errno==EAGAIN || errno==EWOULDBLOCK) return 0;
            if (errno==EMSGSIZE) { ++s->udp_ignored; continue; }
            return -1;
        }
        /* HELLO is an explicit setup-phase restart, not live path monitoring.
         * Auto listeners refresh only here; future SESSION handling must keep
         * the established session paths fixed until that session stops. */
        if (discovery && !s->receiver && !strcmp(s->config->interfaces,"auto")) {
            fr_identity ignored;
            if (probe_discovery_read(bytes,(size_t)n,cookie,&ignored)!=1) { ++s->udp_ignored; continue; }
            if (!s->discovery_cookie_valid || memcmp(cookie,s->discovery_cookie,sizeof(cookie))) {
                if (refresh_interfaces(s)) { ++s->udp_ignored; continue; }
                memcpy(s->discovery_cookie,cookie,sizeof(cookie)); s->discovery_cookie_valid=1;
            }
        }
        /* The arrival interface index wins: macOS can leave a bridge's address
         * also on an inactive member port, and replying there loses the packet. */
        unsigned iface=s->interface_count;
        for (unsigned i=0;i<s->interface_count && iface==s->interface_count;++i) if (s->interfaces[i].index==received.index) iface=i;
        for (unsigned i=0;i<s->interface_count && iface==s->interface_count;++i) {
            struct in_addr addr; inet_pton(AF_INET,s->interfaces[i].address,&addr);
            if (addr.s_addr==received.address.s_addr) iface=i;
        }
        if (iface==s->interface_count || (s->interfaces[iface].loopback && (ntohl(peer.sin_addr.s_addr)>>24)!=127)) { ++s->udp_ignored; continue; }
        size_t size=0;
        if (discovery) {
            fr_identity id={.id=s->id};
            if (probe_discovery_read(bytes,(size_t)n,cookie,&id)!=1) { ++s->udp_ignored; continue; }
            memset(&id,0,sizeof(id)); id.id=s->id; strcpy(id.name,s->config->name);
            id.control_port=(unsigned)s->port; id.data_port=(unsigned)s->data_port;
            size=probe_hello(bytes,cookie,&id); ++s->hellos;
        } else {
            /* A probe train arrives as one datagram and is answered by one. */
            fr_probe p; client *owner=NULL;
            for (size_t at=0,len;at<(size_t)n;at+=len) {
                if (!(len=fr_frame_next(bytes+at,(size_t)n-at)) || probe_read(bytes+at,len,&p) || p.reply || (multicast && p.kind)) { size=0; break; }
                if (!owner) for (int i=0; i<NODE_CLIENTS; ++i) {
                    client *c=&s->clients[i];
                    if (c->fd>=0 && c->auth.authenticated && c->link_session==p.session &&
                        c->udp_peer.sin_addr.s_addr==peer.sin_addr.s_addr && c->udp_peer.sin_port==peer.sin_port) { owner=c; break; }
                }
                if (!owner) { size=0; break; }
                p.reply=1; p.node=owner->link_node; owner->udp_index=received.index; s->probe_index=received.index; size+=probe_make(out+size,&p);
            }
            if (!size) { ++s->udp_ignored; continue; }
            memcpy(bytes,out,size);
            ++s->probes; if (multicast) ++s->multicast_probes;
        }
        fr_udp_info reply={.index=s->interfaces[iface].index};
        inet_pton(AF_INET,s->interfaces[iface].address,&reply.address);
        if (net_udp_send_from(discovery?s->discovery:s->data,bytes,size,&peer,&reply)!=(ssize_t)size) ++s->udp_dropped;
    }
    return 0;
}
/* ---- Process lifetime: initialize once, drive events, then drain and close. ---- */
int node_run(const fr_config *config, int debug) {
    if (!config->password[0]) {
        log_line("warning","a nonempty password is required in the node configuration"); return 1;
    }
    /* A head that goes away mid-reply must cost the node one EPIPE, not its life:
     * macOS gives accepted sockets no SO_NOSIGPIPE and has no MSG_NOSIGNAL, and a
     * node killed by SIGPIPE (exit 141) dropped out of the M1 Pro's demo (run 91). */
    signal(SIGPIPE,SIG_IGN);
    node s={.config=config,.loop=-1,.listener=-1,.debug=debug,.discovery=-1,.multicast=-1,.data=-1};
    atomic_init(&s.bench.done,0); atomic_init(&s.bench.cancel,0);
    if (identity_load(config->state_dir,&s.id)) { log_line("warning","cannot load persistent node identity"); return 1; }
    s.clients=calloc(NODE_CLIENTS,sizeof(client));
    if (!s.clients) { log_line("warning","cannot allocate control buffers"); return 1; }
    for (int i=0; i<NODE_CLIENTS; ++i) s.clients[i].fd=-1;
    int result=1, handlers=0;
    if (clock_gettime(CLOCK_MONOTONIC,&s.started)<0) goto done;
    struct sigaction action={0}, old_int, old_term;
    action.sa_handler=stop_signal;
    if (sigemptyset(&action.sa_mask)<0) goto done;
    stopping=0;
    if (sigaction(SIGINT,&action,&old_int)<0) goto done;
    handlers=1;
    if (sigaction(SIGTERM,&action,&old_term)<0) goto done;
    handlers=2;
    s.loop=ae_open();
    if (s.loop<0) goto done;
    s.listener=net_tcp_listen(config->bind,config->control_port);
    if (s.listener<0) goto done;
    s.port=net_port(s.listener);
    if (s.port<0 || ae_set(s.loop,s.listener,0,AE_READ,0)<0) goto done;
    if (interfaces_read(config->interfaces,config->interface,s.interfaces,&s.interface_count)) goto done;
    s.discovery=net_udp_bind("0.0.0.0",config->discovery_port,1);
    if (s.discovery<0 || net_udp_info_enable(s.discovery)) goto done;
    s.discovery_port=net_port(s.discovery);
    if (s.discovery_port<0) goto done;
    s.multicast=net_udp_bind(config->multicast,(unsigned)s.discovery_port,1);
    s.data=net_udp_bind(config->bind,config->data_port,0);
    if (s.multicast<0 || s.data<0 || net_udp_info_enable(s.multicast) || net_udp_info_enable(s.data)) goto done;
    for (unsigned i=0;i<s.interface_count;++i) {
        if (s.interfaces[i].loopback && net_multicast(s.discovery,FR_DISCOVERY_GROUP,s.interfaces[i].address,1)) goto done;
        if (s.interfaces[i].type!=FR_LINK_WIFI && net_multicast(s.multicast,config->multicast,s.interfaces[i].address,1))
            log_line("notice","multicast unavailable on %s; unicast remains available",s.interfaces[i].name);
    }
    s.data_port=net_port(s.data);
    if (s.data_port<0 || ae_set(s.loop,s.discovery,0,AE_READ,UINT64_MAX)<0 ||
        ae_set(s.loop,s.multicast,0,AE_READ,UINT64_MAX-1)<0 || ae_set(s.loop,s.data,0,AE_READ,UINT64_MAX-2)<0) goto done;
    log_line("notice","listening name=%s bind=%s control_port=%d data_port=%d discovery_port=%d event_loop=%s auth=hmac-sha256",
             config->name,config->bind,s.port,s.data_port,s.discovery_port,ae_name());
    while (!stopping) {
        if (bench_complete(&s)) goto done;
        ae_event events[AE_MAX_EVENTS];
        int n=ae_wait(s.loop,events,AE_MAX_EVENTS,100);
        if (n<0) goto done;
        int64_t now=net_now_ns();
        if (now>0 && now-s.multicast_refreshed>INT64_C(60000000000)) refresh_multicast(&s);
        for (int i=0; i<n; ++i) {
            uint64_t tag=events[i].tag;
            if (tag>=UINT64_MAX-2) {
                if (s.receiver && tag!=UINT64_MAX) continue;
                int fd=tag==UINT64_MAX?s.discovery:tag==UINT64_MAX-1?s.multicast:s.data;
                if (udp_ready(&s,fd,tag==UINT64_MAX,tag==UINT64_MAX-1)<0) goto done;
                continue;
            }
            if (!tag) { if (accept_clients(&s)<0) goto done; continue; }
            unsigned slot=(unsigned)(tag&255)-1;
            if (slot>=NODE_CLIENTS) continue;
            client *c=&s.clients[slot];
            if (c->fd<0 || c->tag!=tag) continue;
            if (serve(&s,c,events[i].mask)<0) disconnect(&s,c);
        }
    }
    result=0;
done:
    if (s.bench.active) { atomic_store(&s.bench.cancel,1); if (pthread_join(s.bench.thread,NULL)) result=1; s.bench.active=0; }
    (void)session_stop(&s); if (fr_units_close(&s.units)) result=1;
    if (result) log_line("warning","control server failed: %s",strerror(errno));
    for (int i=0; i<NODE_CLIENTS; ++i) if (s.clients[i].fd>=0) disconnect(&s,&s.clients[i]);
    if (s.listener>=0 && close(s.listener)<0) result=1;
    if (s.discovery>=0 && close(s.discovery)<0) result=1;
    if (s.multicast>=0 && close(s.multicast)<0) result=1;
    if (s.data>=0 && close(s.data)<0) result=1;
    if (s.loop>=0 && close(s.loop)<0) result=1;
    if (handlers>=1 && sigaction(SIGINT,&old_int,NULL)<0) result=1;
    if (handlers>=2 && sigaction(SIGTERM,&old_term,NULL)<0) result=1;
    free(s.clients);
    if (!result) log_line("notice","stopped");
    return result;
}
