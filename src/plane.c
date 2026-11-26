/* One distributed expert call, from request to complete sum.
 *
 * Read fr_head_plane_run first: round the activation, start local work, send
 * remote work, collect tagged partials, join the local worker, publish output.
 * On a node, receive_loop assembles requests and hands them to compute_loop.
 * The same compute worker is used for the head's local expert slices.
 *
 * plane.h owns the session buffers; wire.c owns their encoding. This file owns
 * ordering and completion, not model placement or kernel math. A partial must
 * match the current session/step/layer and contribute only once. For the ordinary
 * complete-sum policy, see serve.c; a timeout here must remain visible to it.
 * Protocol and failure walkthrough: docs/PROTOCOL.md, docs/llm/RUNTIME.md.
 */
#include "plane.h"
#include "hot.h"
_Static_assert(FR_PLANE_CACHES==FR_HOT_CACHES,"one plane view per cache of the policy");
#ifdef FR_GSQ_COMPUTE
#include "gsq.h"
_Static_assert(FR_GSQ_Q8K_BYTES==FR_Q8_TOKEN_BYTES,"native/wire Q8_K shape");
#endif
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/socket.h>
#include <poll.h>
#include <stdlib.h>

#ifdef FR_GSQ_COMPUTE
int (*fr_plane_q8_run)(const fr_req *req,const void *xq,fr_part *out)=fr_gsq_run_quantized;
#else
int (*fr_plane_q8_run)(const fr_req *req,const void *xq,fr_part *out)=NULL;
#endif
int fr_plane_patient;
static void relax(void) {
#if defined(__aarch64__) || defined(__arm64__)
    __asm__ volatile("yield");
#elif defined(__x86_64__)
    __asm__ volatile("pause");
#endif
}
void fr_plane_stats_init(fr_plane_stats *s) {
    atomic_init(&s->requests,0); atomic_init(&s->parts,0); atomic_init(&s->duplicates,0);
    atomic_init(&s->late,0); atomic_init(&s->invalid,0); atomic_init(&s->gaps,0);
    atomic_init(&s->overwritten,0); atomic_init(&s->errors,0); atomic_init(&s->compute_ns,0);
    atomic_init(&s->rx_ns,0); atomic_init(&s->wait_ns,0); atomic_init(&s->reply_ns,0);
}
static void *compute_loop(void *arg) {
    /* One mailbox, one producer, one consumer. Submit copies the entire request
     * before release-publishing state 1. This acquire makes that copy visible;
     * release-publishing state 3 makes the result visible to the joining owner.
     * The owner must consume completion before submitting or freeing storage.
     * Spin during a chat; park after 50 ms idle to avoid burning a quiet CPU. */
    fr_compute *w=arg; int64_t active=net_now_ns();
    while (!atomic_load(&w->stop)) {
        if (atomic_load_explicit(&w->state,memory_order_acquire)==1) {
            atomic_store(&w->state,2); int64_t start=net_now_ns(); w->started=start;
#ifdef FR_GSQ_COMPUTE
            w->result=w->q8_k?fr_plane_q8_run(&w->request,w->xq,&w->part):w->backend->run(&w->request,&w->part);
#else
            w->result=w->q8_k?FR_UNSUPPORTED:w->backend->run(&w->request,&w->part);
#endif
            w->elapsed=(uint64_t)(net_now_ns()-start); active=net_now_ns();
            atomic_store_explicit(&w->state,3,memory_order_release);
        } else if (fr_plane_patient || net_now_ns()-active>=50000000) {
            pthread_mutex_lock(&w->mutex); atomic_store(&w->parked,1);
            while (!atomic_load(&w->stop) && atomic_load(&w->state)!=1) pthread_cond_wait(&w->condition,&w->mutex);
            atomic_store(&w->parked,0); pthread_mutex_unlock(&w->mutex); active=net_now_ns();
        } else relax();
    }
    return NULL;
}
int fr_compute_open(fr_compute *w,const fr_backend *backend) {
    if (!w || !backend) return -1;
    memset(w,0,sizeof(*w)); w->backend=backend;
    atomic_init(&w->state,0); atomic_init(&w->stop,0); atomic_init(&w->parked,0);
    if (pthread_mutex_init(&w->mutex,NULL)) return -1;
    if (pthread_cond_init(&w->condition,NULL)) { pthread_mutex_destroy(&w->mutex); return -1; }
    if (!fr_thread_create(&w->thread,compute_loop,w)) return 0;
    pthread_cond_destroy(&w->condition); pthread_mutex_destroy(&w->mutex); return -1;
}
static void wake(fr_compute *w) {
    /* Lock only after parking. Publishing state first prevents a lost wakeup. */
    if (atomic_load(&w->parked)) {
        pthread_mutex_lock(&w->mutex); pthread_cond_signal(&w->condition); pthread_mutex_unlock(&w->mutex);
    }
}
int fr_compute_submit(fr_compute *w,const fr_req *request,uint32_t step) {
    if (atomic_load_explicit(&w->state,memory_order_acquire)) return -1;
    w->q8_k=0; w->request=*request; w->step=step; w->submitted=net_now_ns();
    atomic_store(&w->state,1); wake(w); return 0;
}
int fr_compute_submit_q8(fr_compute *w,const fr_req *request,
                        const unsigned char xq[FR_MAX_BURST][FR_Q8_TOKEN_BYTES],uint32_t step) {
#ifndef FR_GSQ_COMPUTE
    (void)w;(void)request;(void)xq;(void)step;return FR_UNSUPPORTED;
#else
    if (!w || !request || !xq || w->backend!=&fr_gsq || !request->burst || request->burst>FR_MAX_BURST ||
        atomic_load_explicit(&w->state,memory_order_acquire)) return FR_INVALID;
    w->request.layer=request->layer;w->request.burst=request->burst;
    memcpy(w->request.expert,request->expert,sizeof(w->request.expert));
    memcpy(w->request.gate,request->gate,sizeof(w->request.gate));
    memcpy(w->xq,xq,request->burst*FR_Q8_TOKEN_BYTES);
    w->q8_k=1;w->step=step;w->submitted=net_now_ns();
    atomic_store_explicit(&w->state,1,memory_order_release);wake(w);return 0;
#endif
}
int fr_compute_close(fr_compute *w) {
    atomic_store(&w->stop,1);
    pthread_mutex_lock(&w->mutex); pthread_cond_signal(&w->condition); pthread_mutex_unlock(&w->mutex);
    int rc=pthread_join(w->thread,NULL);
    pthread_cond_destroy(&w->condition); pthread_mutex_destroy(&w->mutex); return rc?-1:0;
}
static fr_bits expected(unsigned burst) { fr_bits b; fr_bits_first(&b,4*burst); return b; }
static int assembled(const fr_assembly *a) { fr_bits want=expected(a->request.burst); return fr_bits_equal(&a->bitmap,&want); }
int fr_assemble(fr_assembly *a,const fr_fragment *f,fr_plane_stats *s) {
    if (f->type!=1 && f->type!=5) return -1;
    uint64_t sequence=(uint64_t)f->step*FR_LAYERS+f->layer+1;
    if (sequence<a->sequence) { atomic_fetch_add(&s->late,1); return 0; }
    if (sequence>a->sequence) {
        if (a->sequence && !a->submitted) atomic_fetch_add(&s->overwritten,1);
        if (a->sequence && (uint64_t)f->step>(uint64_t)a->step+1) atomic_fetch_add(&s->gaps,(uint64_t)f->step-a->step-1);
        a->sequence=sequence; a->step=f->step; fr_bits_clear(&a->bitmap); a->submitted=0;
        a->q8_k=f->type==5;
        if (!a->q8_k) memset(&a->request,0,sizeof(a->request));
        a->request.layer=f->layer; a->request.burst=f->burst;
    }
    if (f->burst!=a->request.burst || a->q8_k!=(f->type==5)) { atomic_fetch_add(&s->invalid,1); return -1; }
    if (fr_bits_test(&a->bitmap,f->fragment)) { atomic_fetch_add(&s->duplicates,1); return 0; }
    if (a->q8_k) {
        static const unsigned starts[4]={0,3,6,8},counts[4]={3,3,2,2};
        unsigned range=f->fragment%4;
        memcpy(a->xq[f->token]+starts[range]*FR_Q8_BLOCK_BYTES,f->q8,counts[range]*FR_Q8_BLOCK_BYTES);
    } else memcpy(a->request.x[f->token]+f->start,f->values,sizeof(f->values));
    if (!(f->fragment%4)) {
        memcpy(a->request.expert[f->token],f->expert,sizeof(f->expert));
        memcpy(a->request.gate[f->token],f->gate,sizeof(f->gate));
    }
    fr_bits_set(&a->bitmap,f->fragment); return assembled(a);
}
void fr_accumulator_begin(fr_accumulator *a,uint32_t session,uint32_t step,unsigned layer,unsigned burst,unsigned nodes) {
    /* Only the call's rows are summed and read: a whole fr_part is FR_MAX_BURST rows. */
    memset(a->sum.y,0,burst*sizeof(a->sum.y[0])); memset(a->bitmap,0,sizeof(a->bitmap));
    a->sum.layer=layer; a->sum.burst=burst; a->session=session; a->step=step; a->nodes=nodes; a->expected=expected(burst); a->active=UINT32_MAX;
}
static int accumulator_add(fr_accumulator *a,const fr_fragment *f,double (*remote)[FR_HIDDEN]) {
    if (f->type!=2 || f->session!=a->session || f->node>=a->nodes || f->burst!=a->sum.burst) { ++a->invalid; return -1; }
    if (f->step!=a->step || f->layer!=a->sum.layer) { ++a->late; return 0; }
    if (fr_bits_test(&a->bitmap[f->node],f->fragment)) { ++a->duplicates; return 0; }
    if (remote) {
        /* At most 20 finite binary16 values. Their sum is exactly representable
         * in binary64, independently of packet arrival and node enumeration. */
        for (unsigned d=0;d<FR_FRAGMENT_DIMS;++d) remote[f->token][f->start+d]+=f->values[d];
    } else for (unsigned d=0;d<FR_FRAGMENT_DIMS;++d) a->sum.y[f->token][f->start+d]+=f->values[d];
    fr_bits_set(&a->bitmap[f->node],f->fragment); return 1;
}
int fr_accumulator_add(fr_accumulator *a,const fr_fragment *f) { return accumulator_add(a,f,NULL); }
uint32_t fr_accumulator_missing(const fr_accumulator *a) {
    uint32_t missing=0;
    for (unsigned n=0;n<a->nodes;++n) if ((a->active>>n&1) && !fr_bits_equal(&a->bitmap[n],&a->expected)) missing|=1u<<n;
    return missing;
}
/* Send a frame of whole fragments as consecutive datagrams of at most
 * FR_DATAGRAM_MAX bytes, cut only at fragment boundaries. Returns bytes sent or -1. */
static ssize_t send_batches(int fd,int connected,const unsigned char *p,size_t len,
                            const struct sockaddr_in *peer,const fr_udp_info *local) {
    size_t done=0;
    while (done<len) {
        size_t batch=0,next;
        while (done+batch<len && (next=fr_frame_next(p+done+batch,len-done-batch)) && (!batch || batch+next<=FR_DATAGRAM_MAX)) batch+=next;
        if (!batch) return -1;
        ssize_t sent=connected?send(fd,p+done,batch,0):net_udp_send_from(fd,p+done,batch,peer,local);
        if (sent!=(ssize_t)batch) return sent<0?-1:(ssize_t)done;
        done+=batch;
    }
    return (ssize_t)done;
}
/* The head repeats a request when its reply did not arrive whole (a lost datagram
 * in either direction). The result is still in the worker until a newer request is
 * submitted: encode and send it again. */
static void reply_again(fr_receiver *r,unsigned char *packet,size_t capacity) {
    const fr_compute *w=&r->compute; size_t frame=0,n=1;
    for (unsigned f=0;n && f<4*w->part.burst;++f) frame+=n=fr_part_encode(packet+frame,capacity-frame,&w->part,r->session,w->step,r->node,f);
    if (!n) return;
    if (r->reply_fd>=0) (void)send_batches(r->reply_fd,1,packet,frame,NULL,NULL);
    else (void)send_batches(r->fd,0,packet,frame,&r->head,&r->local);
}
static void *receive_loop(void *arg) {
    fr_receiver *r=arg; int64_t active=net_now_ns();
    static _Thread_local unsigned char packet[FR_FRAME_MAX];
    while (!atomic_load(&r->stop)) {
        int drained=1,again=0;
        /* Bounded drain keeps only the newest request without starving replies. */
        for (unsigned socket=0;socket<2;++socket) for (unsigned i=0;i<256;++i) {
            int fd=socket?r->multicast_fd:r->fd; if (fd<0) break;
            struct sockaddr_in peer; fr_udp_info local; fr_fragment f;
            ssize_t n=net_udp_recv_info(fd,packet,sizeof(packet),&peer,&local);
            if (n<0) {
                if (errno==EAGAIN || errno==EWOULDBLOCK) break;
                atomic_fetch_add(&r->stats.invalid,1); break;
            }
            if (i==255) drained=0;
            active=net_now_ns();
            /* The session's address, on whichever interface carries it: a bridged Mac can
             * hold the same address on bridge0 and on its Thunderbolt port at once. */
            if (peer.sin_addr.s_addr!=r->head.sin_addr.s_addr || (local.index!=r->local.index && local.address.s_addr!=r->local.address.s_addr)) {
                if (atomic_fetch_add(&r->stats.invalid,1)<3) fprintf(stderr,"dataplane drop: from head=%d arrival_if=%u session_if=%u\n",
                    peer.sin_addr.s_addr==r->head.sin_addr.s_addr,local.index,r->local.index);
                continue; }
            for (size_t at=0,len;at<(size_t)n;at+=len) {
                if (!(len=fr_frame_next(packet+at,(size_t)n-at))) { atomic_fetch_add(&r->stats.invalid,1); break; }
                if (fr_fragment_read(packet+at,len,&f) || f.session!=r->session || f.burst>FR_MAX_BURST || (f.type!=1 && !(f.type==5 && r->q8_allowed))) {
                    atomic_fetch_add(&r->stats.invalid,1); continue;
                }
                if (!f.fragment) {   /* a request's first fragment seen again: the head asks for its reply */
                    const uint64_t asked=(uint64_t)f.step*FR_LAYERS+f.layer+1;
                    if (r->replied==asked) again=1;   /* computed: send it (again, or now if it was kept) */
                    else if (r->assembly.sequence==asked && fr_bits_test(&r->assembly.bitmap,0)) r->polled=asked; /* still computing */
                }
                int before=r->assembly.sequence==(uint64_t)f.step*FR_LAYERS+f.layer+1 && fr_bits_any(&r->assembly.bitmap);
                (void)fr_assemble(&r->assembly,&f,&r->stats);
                if (!before) r->first_rx=active;
            }
        }
        if (atomic_load_explicit(&r->compute.state,memory_order_acquire)==3) {
            fr_compute *w=&r->compute;
            atomic_fetch_add(&r->stats.compute_ns,w->elapsed);
            if (w->result) {
                if (!atomic_fetch_add(&r->stats.errors,1)) fprintf(stderr,"dataplane: layer %u compute failed (backend rc %d); later failures are counted in INFO\n",w->request.layer,w->result);
            } else {
                size_t frame=0,n=1;
                for (unsigned f=0;n && f<4*w->part.burst;++f) frame+=n=fr_part_encode(packet+frame,sizeof(packet)-frame,&w->part,r->session,w->step,r->node,f);
                /* Paced: keep a reply of several datagrams until the head asks (reply_again). */
                const int kept=n && frame>FR_DATAGRAM_MAX && atomic_load(&r->paced) && r->polled!=(uint64_t)w->step*FR_LAYERS+w->request.layer+1;
                ssize_t sent=!n?-1:kept?(ssize_t)frame:r->reply_fd>=0?send_batches(r->reply_fd,1,packet,frame,NULL,NULL):send_batches(r->fd,0,packet,frame,&r->head,&r->local);
                if (n && sent<0 && r->reply_fd>=0 && errno==EPIPE) {
                    /* macOS closed the flow: a program started in the background with no
                     * Local Network permission (nohup from an ssh that then closed: run 92),
                     * or a VPN/filter extension. Try the unconnected data socket once more. */
                    fprintf(stderr,"dataplane: macOS closed the reply socket (EPIPE); replying from the data socket. If replies still fail, "
                                   "this node has no Local Network permission: start it as the installed service (fronk-install.sh) or from Terminal\n");
                    close(r->reply_fd); r->reply_fd=-1;
                    sent=send_batches(r->fd,0,packet,frame,&r->head,&r->local);
                }
                if (sent==(ssize_t)frame) atomic_fetch_add(&r->stats.parts,4*w->part.burst);
                else if (!atomic_fetch_add(&r->stats.errors,1)) {
                    const int e=errno; char to[INET_ADDRSTRLEN]="?"; inet_ntop(AF_INET,&r->head.sin_addr,to,sizeof(to));
                    if (!n) { /* fr_half refused a value: the backend's output is not finite or exceeds fp16 */
                        unsigned t=(unsigned)(frame/1312/4),d=0; while (d<FR_HIDDEN && isfinite(w->part.y[t][d]) && fabsf(w->part.y[t][d])<65504.f) ++d;
                        fprintf(stderr,"dataplane: layer %u burst %u: token %u output y[%u]=%g does not fit fp16; reply dropped (later failures are counted in INFO)\n",
                                w->part.layer,w->part.burst,t,d,d<FR_HIDDEN?w->part.y[t][d]:0.f);
                    } else fprintf(stderr,"dataplane: reply of %zu bytes to %s:%u not sent (%s); later failures are counted in INFO\n",
                                   frame,to,ntohs(r->head.sin_port),sent<0?strerror(e):"short send");
                }
            }
            r->replied=(uint64_t)w->step*FR_LAYERS+w->request.layer+1; again=0;
            atomic_fetch_add(&r->stats.wait_ns,(uint64_t)(w->started-w->submitted));
            atomic_fetch_add(&r->stats.reply_ns,(uint64_t)(net_now_ns()-w->started-(int64_t)w->elapsed));
            atomic_store_explicit(&w->state,0,memory_order_release);
        }
        fr_assembly *a=&r->assembly;
        if (again && r->replied==a->sequence && !atomic_load(&r->compute.state)) reply_again(r,packet,sizeof(packet));
        if (drained && a->sequence && !a->submitted && assembled(a) && !atomic_load(&r->compute.state)) {
            int rc=a->q8_k?fr_compute_submit_q8(&r->compute,&a->request,a->xq,a->step):fr_compute_submit(&r->compute,&a->request,a->step);
            if (!rc) {
                a->submitted=1; atomic_fetch_add(&r->stats.requests,1);
                atomic_fetch_add(&r->stats.rx_ns,(uint64_t)(r->compute.submitted-r->first_rx));
            }
        }
        if ((fr_plane_patient || net_now_ns()-active>=50000000) && !atomic_load(&r->compute.state)) {
            struct pollfd fds[2]={{.fd=r->fd,.events=POLLIN},{.fd=r->multicast_fd,.events=POLLIN}};
            (void)poll(fds,2,20);
        }
        else if (fr_plane_patient) usleep(80);   /* its GPU is at work: nothing to do on a core */
        else relax();
    }
    return NULL;
}
int fr_receiver_open(fr_receiver *r,int fd,int multicast_fd,uint32_t session,unsigned node,unsigned burst,
                     const struct sockaddr_in *head,const fr_udp_info *local,const fr_backend *backend) {
    if (!r || fd<0 || !session || node>=FR_CONNECTION_NODES || !burst || burst>8 || !head || !local) return -1;
    memset(r,0,sizeof(*r)); r->fd=fd; r->multicast_fd=multicast_fd; r->session=session; r->node=node; r->burst=burst; r->head=*head; r->local=*local;
    atomic_init(&r->stop,0); fr_plane_stats_init(&r->stats);
#ifdef FR_GSQ_COMPUTE
    r->q8_allowed=backend==&fr_gsq;
#endif
    /* A connected socket skips the per-send route/policy lookup (~3x cheaper on macOS). */
    char ip[INET_ADDRSTRLEN]; r->reply_fd=-1;
    if (inet_ntop(AF_INET,&local->address,ip,sizeof(ip)) && (r->reply_fd=net_udp_bind(ip,0,0))>=0 &&
        connect(r->reply_fd,(const struct sockaddr *)head,sizeof(*head))) {
        close(r->reply_fd); r->reply_fd=-1;
    }
    if (fr_compute_open(&r->compute,backend)) { if (r->reply_fd>=0) close(r->reply_fd); return -1; }
    if (!fr_thread_create(&r->thread,receive_loop,r)) return 0;
    fr_compute_close(&r->compute); if (r->reply_fd>=0) close(r->reply_fd); return -1;
}
int fr_receiver_close(fr_receiver *r) {
    atomic_store(&r->stop,1); int rc=pthread_join(r->thread,NULL);
    if (r->reply_fd>=0) close(r->reply_fd);
    return fr_compute_close(&r->compute) || rc?-1:0;
}
int fr_head_plane_open(fr_head_plane *h,const fr_connection *links,const char *group,unsigned port,const fr_backend *local,unsigned blocks) {
    if (!h || !links || links->count>FR_CONNECTION_NODES || !links->session || (!local && !links->count) || (!!local!=!!blocks)) return -1;
    memset(h,0,sizeof(*h)); h->links=links; h->local_blocks=blocks; h->active=UINT32_MAX; h->local_owner=-1;
    snprintf(h->group,sizeof(h->group),"%s",group); h->port=port;
    if (links->count && fanout_open(links,group,port,&h->fanout)) return -1;
    if (local && fr_compute_open(&h->local,local)) { fanout_close(&h->fanout); return -1; }
    h->has_local=local!=NULL;
    const char *width=getenv("FRONK_MTP_BRANCHES"), *exact=getenv("FRONK_REQUIRE_ALL");
    if ((width && width[0]>='2' && width[0]<='4' && !width[1]) || (exact && !strcmp(exact,"1"))) {
        h->exact_remote=calloc(FR_MAX_BURST,sizeof(*h->exact_remote));
        if (!h->exact_remote) { fr_head_plane_close(h); return -1; }
    }
#ifdef FRONK_PATH_PROFILE
    const char *capture=getenv("FRONK_EXPERT_CAPTURE"),*capture_width=getenv("FRONK_EXPERT_CAPTURE_WIDTH");
    if(capture && *capture) {
        h->capture_width=capture_width?(unsigned)strtoul(capture_width,NULL,10):4;
        struct {char magic[8];uint32_t size,endian;} header={{'F','R','N','C','A','P','0','1'},sizeof(fr_req),0x01020304};
        if(!h->capture_width || h->capture_width>8 || !(h->capture=fopen(capture,"wb")) ||
            fwrite(&header,sizeof(header),1,h->capture)!=1) {fr_head_plane_close(h);return -1;}
    }
#endif
    return 0;
}
/* A reply that has not arrived whole is asked for again after this long, from one
 * node at a time: the head's USB Ethernet adapter drops a whole datagram when two
 * nodes send large replies at the same instant, and two repeats would collide
 * again. Far above a normal call (3 ms narrow, 10-15 ms wide), far below the
 * complete-sum timeout. */
#define FR_RESEND_NARROW_NS 20000000ll
#define FR_RESEND_WIDE_NS 40000000ll
/* One view of a call (cold or warm) to the links of mask, whole fragments per datagram. */
static int send_view(fr_head_plane *h,uint32_t mask,const unsigned char *frame,const size_t *lengths,unsigned fragments,size_t bytes) {
    for (size_t at=0,f=0;at<bytes;) {
        size_t batch=0;
        while (f<fragments && (!batch || batch+lengths[f]<=FR_DATAGRAM_MAX)) batch+=lengths[f++];
        if (!batch) return -1;
        if (h->unicast?fanout_send_mask(&h->fanout,mask,frame+at,batch):fanout_send(&h->fanout,frame+at,batch)) return -1;
        at+=batch;
    }
    return 0;
}
/* ---- Expert calls. begin: validate -> stage every view -> publish local work -> fan out.
 * poll: drain replies for every call in flight. end: hand out the sum. Do not reorder
 * publication or buffer reuse without checking both error paths. ---- */
static void call_ended(fr_head_plane *h,uint64_t order) {
    /* hot.c frees a cache slot when every call that began before is over: count calls as
     * done only in the order they began, so a newer call cannot hide an older one. */
    if (order==h->ended_upto) {
        ++h->ended_upto; atomic_fetch_add(&h->runs_done,1);
        if (h->ended_early==h->ended_upto+1) { ++h->ended_upto; atomic_fetch_add(&h->runs_done,1); h->ended_early=0; }
    } else h->ended_early=order+1;
}
static int local_submit(fr_head_plane *h,fr_call *c) {
    const int rc=h->q8_k?fr_compute_submit_q8(&h->local,&c->rounded,c->xq,c->step):fr_compute_submit(&h->local,&c->rounded,c->step);
    if (!rc) { c->local_state=2; h->local_owner=(int)c->slot; }
    return rc;
}
int fr_head_plane_begin(fr_head_plane *h,unsigned slot,const fr_req *req,uint32_t step,uint64_t deadline_ns) {
    if (!h || slot>1 || !req || !deadline_ns || !req->burst || req->burst>FR_MAX_BURST || req->layer>=FR_LAYERS || h->call[slot].active) return -1;
    fr_call *c=&h->call[slot]; const fr_call *o=h->call[slot^1u].active?&h->call[slot^1u]:NULL;
    const int wide=(size_t)req->burst*FR_TOKEN_FRAGMENTS*1312u>FR_DATAGRAM_MAX;
    /* Paced replies come one node at a time, and exact sums use one buffer: such a call
     * has the wire to itself. */
    if (o && (wide || o->wide || h->exact_remote)) return FR_PLANE_BUSY;
    const int profile=h->profile;
    const int64_t path_start=profile?net_now_ns():0;
    atomic_fetch_add(&h->runs_begun,1);   /* before the routing is read (hot.c settle) */
    c->order=h->order_next++; c->slot=slot;
    unsigned uses_slices=0,uses_cache[FR_PLANE_CACHES]={0};
    /* Stage all packets first: numeric/routing failure must send nothing. The
     * local slice sees exactly the same fp16-rounded activations as the nodes. */
    c->rounded.layer=req->layer; c->rounded.burst=req->burst;
    memcpy(c->rounded.expert,req->expert,req->burst*sizeof(req->expert[0]));
    memcpy(c->rounded.gate,req->gate,req->burst*sizeof(req->gate[0]));
    for (unsigned t=0;t<req->burst;++t) for (unsigned d=0;d<FR_HIDDEN;++d) {
        if (fr_half(req->x[t][d],&c->half[t][d])) { call_ended(h,c->order); return -1; }
        c->rounded.x[t][d]=fr_unhalf(c->half[t][d]);
    }
    /* With caches, split the call by where each expert lives (hot.h): one view for the
     * slice nodes and one for each cache that holds some of this call's experts. The
     * slice view (cold) is also what any node is asked again with; without caches it
     * is the call itself. */
    uint32_t slices=h->active,cache=0,warm=0; int warm_local=0;
    c->cold.layer=req->layer; c->cold.burst=req->burst;
    memcpy(c->cold.expert,req->expert,req->burst*sizeof(req->expert[0])); memcpy(c->cold.gate,req->gate,req->burst*sizeof(req->gate[0]));
    if (h->caches) {
        int cold=0;
        for (unsigned k=0;k<h->caches;++k) {
            c->warm_req[k].layer=req->layer; c->warm_req[k].burst=req->burst;
            memcpy(c->warm_req[k].expert,req->expert,req->burst*sizeof(req->expert[0])); memset(c->warm_req[k].gate,0,req->burst*sizeof(req->gate[0]));
        }
        for (unsigned t=0;t<req->burst;++t) for (unsigned j=0;j<FR_ACTIVE_ROUTED;++j) {
            if (req->gate[t][j]==0.0f) continue;
            const unsigned where=fr_hot_where(req->layer,req->expert[t][j]);
            if (where && where<=h->caches) { c->cold.gate[t][j]=0; c->warm_req[where-1].gate[t][j]=req->gate[t][j]; warm|=1u<<(where-1); ++uses_cache[where-1]; }
            else { cold=1; ++uses_slices; }
        }
        for (unsigned k=0;k<h->caches;++k) if (warm>>k&1) {
            if (h->cache_link[k]<0) {   /* the local worker computes this view */
                warm_local=1; memcpy(c->rounded.gate,c->warm_req[k].gate,req->burst*sizeof(req->gate[0]));
            } else cache|=1u<<h->cache_link[k];
        }
        if (!cold) slices=0;   /* every expert of this call is cached: the slice nodes rest */
    }
    const int local_work=h->cache_local?warm_local:h->has_local;
#ifdef FRONK_PATH_PROFILE
    /* Bounded replay fixture, never a performance run. Capture decode only
     * after the first single-token verifier, excluding initial prefill. */
    if(h->capture && req->layer==0 && req->burst==1) h->capture_decode=1;
    if(h->capture && h->capture_decode && req->burst==h->capture_width && h->capture_count<192) {
        if(fwrite(&c->rounded,sizeof(c->rounded),1,h->capture)!=1) { call_ended(h,c->order); return -1; }
        ++h->capture_count;
        if(h->capture_count==192) {
            if(fflush(h->capture)) { call_ended(h,c->order); return -1; }
            fprintf(stderr,"FRONK_CAPTURE complete calls=%u T=%u\n",h->capture_count,h->capture_width);
        }
    }
#endif
#ifdef FR_GSQ_COMPUTE
    if (h->q8_k) for (unsigned t=0;t<req->burst;++t) {
        /* ggml leaves unused bsums unwritten for a zero block. Never transmit
         * stale bytes: initialize them before the unchanged quantizer runs. */
        memset(c->xq[t],0,FR_Q8_TOKEN_BYTES);
        fr_gsq_quantize_x(c->rounded.x[t],c->xq[t]);
    }
#else
    if (h->q8_k) { call_ended(h,c->order); return FR_UNSUPPORTED; }
#endif
    if (step<=h->step_hi) step=h->step_hi+1;   /* steps only grow (two calls, fresh asks) */
    h->step_hi=step; c->step=step;
    size_t frame=0;
    for (unsigned f=0;f<4*req->burst;++f) {
        c->lengths[f]=h->q8_k?fr_req_q8_encode(c->frame+frame,sizeof(c->frame)-frame,&c->cold,c->xq,h->links->session,step,f):
            fr_req_encode(c->frame+frame,sizeof(c->frame)-frame,&c->cold,c->half,h->links->session,step,f);
        if (!c->lengths[f]) { call_ended(h,c->order); return -1; }
        frame+=c->lengths[f];
    }
    c->frame_len=frame;
    for (unsigned k=0;k<h->caches;++k) {
        c->frame_warm_len[k]=0;
        if (h->cache_link[k]<0 || !(warm>>k&1)) continue;
        for (unsigned f=0;f<4*req->burst;++f) {
            c->lengths_warm[k][f]=h->q8_k?fr_req_q8_encode(c->frame_warm[k]+c->frame_warm_len[k],sizeof(c->frame_warm[k])-c->frame_warm_len[k],&c->warm_req[k],c->xq,h->links->session,step,f):
                fr_req_encode(c->frame_warm[k]+c->frame_warm_len[k],sizeof(c->frame_warm[k])-c->frame_warm_len[k],&c->warm_req[k],c->half,h->links->session,step,f);
            if (!c->lengths_warm[k][f]) { call_ended(h,c->order); return -1; }
            c->frame_warm_len[k]+=c->lengths_warm[k][f];
        }
    }
    if (h->exact_remote) memset(h->exact_remote,0,req->burst*sizeof(*h->exact_remote));
    fr_accumulator_begin(&c->acc,h->links->session,step,req->layer,req->burst,h->links->count); c->acc.active=h->caches?slices|cache:h->active;
    memset(c->alt,0,sizeof(c->alt));
    c->start=net_now_ns(); c->path_start=profile?path_start:c->start; c->deadline_ns=deadline_ns;
    c->rc=0; c->last=-1; c->missing=0; c->local_ns=0; c->rx_ns=0; c->datagrams=0;
    c->slices=slices; c->cache=cache; c->warm=warm; c->warm_local=warm_local; c->local_work=local_work;
    c->uses_slices=uses_slices; memcpy(c->uses_cache,uses_cache,sizeof(uses_cache));
    /* Publish local work before fanout: its compute (or SSD faults) can overlap
     * send/receive latency. Encoding/validation above must still finish first. The
     * worker takes one view at a time: the second call's waits for the first's. */
    c->local_state=0;
    if (local_work) {
        if (h->local_owner<0 && !atomic_load_explicit(&h->local.state,memory_order_acquire)) {
            if (local_submit(h,c)) { call_ended(h,c->order); return -1; }
        } else c->local_state=1;
    }
    c->send_start=net_now_ns();
    /* Whole fragments per datagram, within FR_DATAGRAM_MAX: a 64-token request is ~350 KB.
     * A failed send is drained by poll together with the already running worker. */
    if (h->links->count && slices && send_view(h,slices,c->frame,c->lengths,4*req->burst,c->frame_len)) c->rc=-1;
    for (unsigned k=0;k<h->caches && !c->rc;++k)
        if (h->cache_link[k]>=0 && (warm>>k&1) && send_view(h,1u<<h->cache_link[k],c->frame_warm[k],c->lengths_warm[k],4*req->burst,c->frame_warm_len[k])) c->rc=-1;
    /* A reply of several datagrams from a paced node comes when asked: see poll. */
    c->wide=wide; c->asked=wide?0:UINT32_MAX;
    c->resend_ns=wide?FR_RESEND_WIDE_NS:FR_RESEND_NARROW_NS;
    c->resend_at=net_now_ns()+c->resend_ns; c->resend_turn=0;
    c->sent=net_now_ns();
    h->send_ns+=(uint64_t)(c->sent-c->send_start);
    c->remote_done=0; c->active=1;
    return 0;
}
/* Ask node `one` again for call c. Alone on the wire, the same step: the node sends the
 * result it kept, or computes it now. With another call in flight, a fresh step: the node
 * keeps one result and the other call may have replaced it. */
static void ask_again(fr_head_plane *h,fr_call *c,uint32_t one,int view) {
    const fr_call *o=h->call[c->slot^1u].active?&h->call[c->slot^1u]:NULL;
    if (!o) {
        if (view>=0) (void)send_view(h,one,c->frame_warm[view],c->lengths_warm[view],4*c->acc.sum.burst,c->frame_warm_len[view]);
        else (void)send_view(h,one,c->frame,c->lengths,4*c->acc.sum.burst,c->frame_len);
        return;
    }
    const fr_req *wire=view>=0?&c->warm_req[view]:&c->cold;
    const uint32_t step=++h->step_hi; size_t frame=0;
    for (unsigned f=0;f<4*c->acc.sum.burst;++f) {
        h->resend_lengths[f]=h->q8_k?fr_req_q8_encode(h->resend_frame+frame,sizeof(h->resend_frame)-frame,wire,c->xq,h->links->session,step,f):
            fr_req_encode(h->resend_frame+frame,sizeof(h->resend_frame)-frame,wire,c->half,h->links->session,step,f);
        if (!h->resend_lengths[f]) return;
        frame+=h->resend_lengths[f];
    }
    c->alt[__builtin_ctz(one)]=step;
    (void)send_view(h,one,h->resend_frame,h->resend_lengths,4*c->acc.sum.burst,frame);
}
unsigned fr_head_plane_poll(fr_head_plane *h) {
    unsigned done=0; int receive=0;
    const int64_t now=net_now_ns();
    for (unsigned s=0;s<2;++s) {
        fr_call *c=&h->call[s];
        if (!c->active || c->remote_done) continue;
        if ((uint64_t)(now-c->start)>=c->deadline_ns) {
            c->missing=fr_accumulator_missing(&c->acc); c->remote_done=1;
            if (c->missing) ++h->deadline_hits;
        } else receive=1;
    }
    if (receive) for (unsigned i=0;i<h->links->interface_count;++i) for (unsigned batch=0;batch<32;++batch) {
        struct sockaddr_in peer; fr_fragment f;
        ssize_t n=net_udp_recv(h->links->sockets[i],h->rx,sizeof(h->rx),&peer);
        if (n<0) { if (errno!=EAGAIN && errno!=EWOULDBLOCK) ++h->call[0].acc.invalid; break; }
        const int64_t rx_start=h->profile?net_now_ns():0; fr_call *owner=NULL;
        for (size_t at=0,len;at<(size_t)n;at+=len) {
            if (!(len=fr_frame_next(h->rx+at,(size_t)n-at))) { ++h->call[0].acc.invalid; break; }
            if (fr_fragment_read(h->rx+at,len,&f) || f.node>=h->links->count) { ++h->call[0].acc.invalid; continue; }
            const fr_link *link=&h->links->links[f.node]; struct sockaddr_in source;
            if (net_address(link->ip,link->identity.data_port,&source) || i!=link->interface_slot ||
                peer.sin_addr.s_addr!=source.sin_addr.s_addr) { ++h->call[0].acc.invalid; continue; }
            fr_call *c=NULL;
            for (unsigned s=0;s<2 && !c;++s) {
                fr_call *k=&h->call[s];
                if (!k->active || k->remote_done) continue;
                if (f.step==k->step) c=k;
                else if (k->alt[f.node] && f.step==k->alt[f.node]) { c=k; f.step=k->step; }   /* asked again under a fresh step */
            }
            if (!c) { ++h->call[0].acc.late; continue; }
            owner=c;
            if (accumulator_add(&c->acc,&f,h->exact_remote)==1 && fr_bits_equal(&c->acc.bitmap[f.node],&c->acc.expected)) {
                uint64_t arrive=(uint64_t)(net_now_ns()-c->start);
                h->arrive_ns[f.node]+=arrive; c->last=(int)f.node;
                if (h->profile) { h->path[c->acc.sum.burst].arrive[f.node]+=arrive; ++h->path[c->acc.sum.burst].answered[f.node]; }
            }
        }
        if (h->profile && owner) { owner->rx_ns+=(uint64_t)(net_now_ns()-rx_start); ++owner->datagrams; }
    }
    for (unsigned s=0;s<2;++s) {
        fr_call *c=&h->call[s];
        if (!c->active || c->remote_done) continue;
        /* On an interface with paced nodes, large replies come one node at a time:
         * the first node of the interface sends when it is ready, each paced one when
         * the head asks, which it does as soon as the node before it is complete. The
         * nodes compute at the same time; only their bursts are kept apart. */
        if ((h->paced&c->slices&~c->asked) && !c->rc) for (unsigned i=0;i<h->fanout.count;++i) {
            const uint32_t members=h->fanout.groups[i].members&c->slices;
            if (!(members&h->paced)) continue;
            for (unsigned link=0;link<h->links->count;++link) {
                if (!(members>>link&1) || fr_bits_equal(&c->acc.bitmap[link],&c->acc.expected)) continue;
                if ((h->paced>>link&1) && !(c->asked>>link&1)) {
                    (void)fanout_send_mask(&h->fanout,1u<<link,c->frame,c->lengths[0]);   /* the first fragment again */
                    c->asked|=1u<<link;
                }
                break;   /* this node's reply is on its way: the next one waits for it */
            }
        }
        const uint32_t missing=fr_accumulator_missing(&c->acc);
        if (!missing) c->remote_done=1;
        else if (h->unicast && !c->rc && net_now_ns()>=c->resend_at) {
            /* The request again, to the lowest missing node only. A node that never
             * got it computes now; one that answered sends the same result again
             * (plane.c reply_again); parts that did arrive are not counted twice.
             * In turn, so that one silent node does not keep the others waiting. */
            uint32_t after=missing&~((c->resend_turn<32?1u<<c->resend_turn:0)-1u); if (!after) after=missing;
            const uint32_t one=after&(~after+1);
            c->resend_turn=(unsigned)__builtin_ctz(one)+1;
            int view=-1;
            for (unsigned k=0;k<h->caches;++k) if (h->cache_link[k]>=0 && one==1u<<h->cache_link[k]) view=(int)k;
            ask_again(h,c,one,view);
            ++h->resends;   /* its own statement: read and increment in one expression are unsequenced */
            if (!(h->resends&(h->resends-1))) fprintf(stderr,"fronk: %llu replies asked again so far (lost datagrams recovered without waiting)\n",(unsigned long long)h->resends);
            c->resend_at=net_now_ns()+c->resend_ns;
        }
    }
    if (h->local_owner>=0 && atomic_load_explicit(&h->local.state,memory_order_acquire)==3) {
        fr_call *c=&h->call[h->local_owner];
        if (h->local.result) c->rc=-1;
        else if (!h->exact_remote) for (unsigned t=0;t<c->acc.sum.burst;++t) for (unsigned d=0;d<FR_HIDDEN;++d) c->acc.sum.y[t][d]+=h->local.part.y[t][d];
        c->local_ns=h->local.elapsed; h->local_ns+=h->local.elapsed;
        uint64_t arrive=(uint64_t)(net_now_ns()-c->start);
        h->arrive_ns[h->links->count]+=arrive; c->last=(int)h->links->count;
        if (h->profile) { h->path[c->acc.sum.burst].arrive[h->links->count]+=arrive; ++h->path[c->acc.sum.burst].answered[h->links->count]; }
        if (!h->exact_remote) atomic_store(&h->local.state,0);   /* exact sums read the part at end */
        c->local_state=3; h->local_owner=-1;
    }
    if (h->local_owner<0 && !atomic_load_explicit(&h->local.state,memory_order_acquire)) {
        fr_call *next=NULL;   /* the older call first */
        for (unsigned s=0;s<2;++s) { fr_call *k=&h->call[s]; if (k->active && k->local_state==1 && (!next || k->order<next->order)) next=k; }
        if (next && local_submit(h,next)) { next->rc=-1; next->local_state=3; }
    }
    for (unsigned s=0;s<2;++s) {
        const fr_call *c=&h->call[s];
        if (c->active && c->remote_done && (c->local_state==0 || c->local_state==3)) done|=1u<<s;
    }
    return done;
}
int fr_head_plane_end(fr_head_plane *h,unsigned slot,fr_part *out) {
    if (!h || slot>1 || !out || !h->call[slot].active) return -1;
    fr_call *c=&h->call[slot]; const unsigned burst=c->acc.sum.burst;
    c->total_ns=(uint64_t)(net_now_ns()-c->start); h->total_ns+=c->total_ns; ++h->barriers;
    ++h->width_calls[burst];
    if (c->last>=0) ++h->last[c->last];
    if (c->slices) for (unsigned k=0;k<h->caches;++k) {
        atomic_fetch_add(&h->cache_calls[k],1);
        if ((c->warm>>k&1) && c->last==(h->cache_link[k]<0?(int)h->links->count:h->cache_link[k])) atomic_fetch_add(&h->cache_last[k],1);
    }
    for (unsigned n=0;n<h->links->count;++n) if (c->missing&(1u<<n)) ++h->missing[n];
    if (!c->rc && h->exact_remote) {
        for (unsigned t=0;t<burst;++t) for (unsigned d=0;d<FR_HIDDEN;++d)
            c->acc.sum.y[t][d]=(float)(h->exact_remote[t][d]+(c->local_work?h->local.part.y[t][d]:0.0));
    }
    if (h->exact_remote && c->local_work) atomic_store(&h->local.state,0);
    if (!c->rc) {
        out->layer=c->acc.sum.layer; out->burst=c->acc.sum.burst;
        memcpy(out->y,c->acc.sum.y,burst*sizeof(out->y[0]));
        memset(out->y[burst],0,(FR_MAX_BURST-burst)*sizeof(out->y[0]));
    }
    h->last_missing=c->missing; h->last_local_ns=c->local_ns; h->last_total_ns=c->total_ns; h->last_acc=&c->acc;
    if (h->profile) {
        const int64_t finished=net_now_ns();
        ++h->path[burst].calls;
        h->path[burst].prepare+=(uint64_t)(c->start-c->path_start);
        h->path[burst].submit+=(uint64_t)(c->send_start-c->start);
        h->path[burst].send+=(uint64_t)(c->sent-c->send_start);
        h->path[burst].wait+=c->total_ns-(uint64_t)(c->sent-c->start);
        h->path[burst].finish+=(uint64_t)(finished-c->start)-c->total_ns;
        h->path[burst].local+=c->local_ns;
        h->path[burst].rx+=c->rx_ns; h->path[burst].datagrams+=c->datagrams;
        h->path[burst].uses_slices+=c->uses_slices;
        for (unsigned k=0;k<h->caches;++k) h->path[burst].uses_cache[k]+=c->uses_cache[k];
        if (c->last>=0) ++h->path[burst].last[c->last];
        if (++h->path_calls==480) {
            for (unsigned t=1;t<=FR_MAX_BURST;t++) if (h->path[t].calls) {
                const double n=(double)h->path[t].calls,scale=1e-3/n;
                fprintf(stderr,"FRONK_PATH T=%u calls=%llu prepare_us=%.1f send_us=%.1f wait_us=%.1f finish_us=%.1f rx_us=%.1f datagrams=%.1f local_compute_us=%.1f uses: slices=%.1f",
                    t,(unsigned long long)h->path[t].calls,h->path[t].prepare*scale,(h->path[t].submit+h->path[t].send)*scale,
                    h->path[t].wait*scale,h->path[t].finish*scale,h->path[t].rx*scale,h->path[t].datagrams/n,h->path[t].local*scale,h->path[t].uses_slices/n);
                for (unsigned k=0;k<h->caches;++k) fprintf(stderr," cache%u=%.1f",k,h->path[t].uses_cache[k]/n);
                fputc('\n',stderr);
                for (unsigned i=0;i<=h->links->count;i++) {
                    if (!h->path[t].answered[i]) continue;
                    fprintf(stderr,"FRONK_PATH_NODE T=%u node=%s answered=%llu arrival_us=%.1f last=%llu\n",t,
                        i==h->links->count?"head":h->links->links[i].identity.name,(unsigned long long)h->path[t].answered[i],
                        h->path[t].arrive[i]*1e-3/(double)h->path[t].answered[i],(unsigned long long)h->path[t].last[i]);
                }
            }
            memset(h->path,0,sizeof(h->path)); h->path_calls=0;
        }
    }
    c->active=0;
    call_ended(h,c->order);
    return c->rc;
}
int fr_head_plane_run(fr_head_plane *h,const fr_req *req,uint32_t step,uint64_t deadline_ns,fr_part *out) {
    if (!h) return -1;
    const int rc=fr_head_plane_begin(h,0,req,step,deadline_ns);
    if (rc) return rc==FR_PLANE_BUSY?-1:rc;
    while (!(fr_head_plane_poll(h)&1u)) relax();
    return fr_head_plane_end(h,0,out);
}
void fr_head_plane_missing_text(const fr_head_plane *h,char *text,size_t capacity) {
    size_t at=0; if (capacity) text[0]=0;
    const fr_accumulator *a=h->last_acc; if (!a) return;
    const unsigned want=4*a->sum.burst;
    for (unsigned n=0;n<h->links->count && at+1<capacity;++n) if (h->last_missing>>n&1) {
        unsigned got=0; for (unsigned f=0;f<want;++f) got+=(unsigned)fr_bits_test(&a->bitmap[n],f);
        int k=snprintf(text+at,capacity-at," %s %u/%u",h->links->links[n].identity.name,got,want);
        if (k<0) break;
        at+=(size_t)k;
    }
    /* Cumulative: a count that grows with every miss says what the head threw away
     * (late = a reply to another step, invalid = wrong sender or shape). */
    if (at+1<capacity) snprintf(text+at,capacity-at,"; head rejected so far: late %llu invalid %llu duplicate %llu",
        (unsigned long long)(h->call[0].acc.late+h->call[1].acc.late),(unsigned long long)(h->call[0].acc.invalid+h->call[1].acc.invalid),
        (unsigned long long)(h->call[0].acc.duplicates+h->call[1].acc.duplicates));
}
int fr_head_plane_unicast(fr_head_plane *h) {
    if (!h || !h->links || !h->links->count || h->unicast) return -1;
    fr_fanout split;
    if (fanout_open_split(h->links,h->group,h->port,UINT32_MAX,&split)) return -1;
    (void)fanout_close(&h->fanout); h->fanout=split; h->unicast=1;
    return 0;
}
int fr_head_plane_cache_local(fr_head_plane *h) {
#ifdef FR_GSQ_COMPUTE
    if (!h || h->has_local || h->cache_local || h->caches>=FR_PLANE_CACHES || fr_compute_open(&h->local,&fr_gsq)) return -1;
    h->has_local=1; h->cache_link[h->caches]=-1; h->cache_local=(int)h->caches+1;
    return (int)h->caches++;
#else
    (void)h; return -1;
#endif
}
int fr_head_plane_cache(fr_head_plane *h,unsigned link) {
    if (!h || !h->links || link>=h->links->count || h->caches>=FR_PLANE_CACHES || (h->active>>link&1)) return -1;
    for (unsigned c=0;c<h->caches;++c) if (h->cache_link[c]==(int)link) return -1;
    if (!h->unicast && fr_head_plane_unicast(h)) return -1;   /* the views go to different nodes */
    h->cache_link[h->caches]=(int)link;
    return (int)h->caches++;
}
int fr_head_plane_close(fr_head_plane *h) {
    free(h->exact_remote); h->exact_remote=NULL;
    int rc=h->has_local?fr_compute_close(&h->local):0; h->has_local=0; h->cache_local=0;
#ifdef FRONK_PATH_PROFILE
    if(h->capture && fclose(h->capture)) rc=-1;
    h->capture=NULL;
#endif
    return fanout_close(&h->fanout) || rc?-1:0;
}
int fr_head_plane_report(const fr_head_plane *h,FILE *output) {
    if (!h || !output) return -1;
    if (fprintf(output,"INFO node=head link=local backend=%s blocks=%u deadline_exempt=1\n"
        "STATS node=head link=local barriers=%llu compute_ns=%llu total_ns=%llu deadline_hits=%llu duplicates=%llu late=%llu invalid=%llu\n",
        h->has_local?h->local.backend->name:"none",h->local_blocks,
        (unsigned long long)h->barriers,(unsigned long long)h->local_ns,(unsigned long long)h->total_ns,
        (unsigned long long)h->deadline_hits,(unsigned long long)(h->call[0].acc.duplicates+h->call[1].acc.duplicates),
        (unsigned long long)(h->call[0].acc.late+h->call[1].acc.late),(unsigned long long)(h->call[0].acc.invalid+h->call[1].acc.invalid))<0) return -1;
    for (unsigned n=0;n<h->links->count;++n) if (fprintf(output,"STATS node=%s link=%s deadline_misses=%llu\n",
        h->links->links[n].identity.name,interface_type(h->links->links[n].via.type),(unsigned long long)h->missing[n])<0) return -1;
    return 0;
}
