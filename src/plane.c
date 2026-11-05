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
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/socket.h>
#include <poll.h>
#include <stdlib.h>

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
            w->result=w->backend->run(&w->request,&w->part);
            w->elapsed=(uint64_t)(net_now_ns()-start); active=net_now_ns();
            atomic_store_explicit(&w->state,3,memory_order_release);
        } else if (net_now_ns()-active>=50000000) {
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
    if (!pthread_create(&w->thread,NULL,compute_loop,w)) return 0;
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
    w->request=*request; w->step=step; w->submitted=net_now_ns();
    atomic_store(&w->state,1); wake(w); return 0;
}
int fr_compute_close(fr_compute *w) {
    atomic_store(&w->stop,1);
    pthread_mutex_lock(&w->mutex); pthread_cond_signal(&w->condition); pthread_mutex_unlock(&w->mutex);
    int rc=pthread_join(w->thread,NULL);
    pthread_cond_destroy(&w->condition); pthread_mutex_destroy(&w->mutex); return rc?-1:0;
}
static uint32_t expected(unsigned burst) { return UINT32_MAX>>(32-4*burst); }
int fr_assemble(fr_assembly *a,const fr_fragment *f,fr_plane_stats *s) {
    if (f->type!=1) return -1;
    uint64_t sequence=(uint64_t)f->step*FR_LAYERS+f->layer+1;
    if (sequence<a->sequence) { atomic_fetch_add(&s->late,1); return 0; }
    if (sequence>a->sequence) {
        if (a->sequence && !a->submitted) atomic_fetch_add(&s->overwritten,1);
        if (a->sequence && (uint64_t)f->step>(uint64_t)a->step+1) atomic_fetch_add(&s->gaps,(uint64_t)f->step-a->step-1);
        a->sequence=sequence; a->step=f->step; a->bitmap=0; a->submitted=0;
        memset(&a->request,0,sizeof(a->request)); a->request.layer=f->layer; a->request.burst=f->burst;
    }
    if (f->burst!=a->request.burst) { atomic_fetch_add(&s->invalid,1); return -1; }
    uint32_t bit=1u<<f->fragment;
    if (a->bitmap&bit) { atomic_fetch_add(&s->duplicates,1); return 0; }
    memcpy(a->request.x[f->token]+f->start,f->values,sizeof(f->values));
    if (!(f->fragment%4)) {
        memcpy(a->request.expert[f->token],f->expert,sizeof(f->expert));
        memcpy(a->request.gate[f->token],f->gate,sizeof(f->gate));
    }
    a->bitmap|=bit; return a->bitmap==expected(f->burst);
}
void fr_accumulator_begin(fr_accumulator *a,uint32_t session,uint32_t step,unsigned layer,unsigned burst,unsigned nodes) {
    memset(&a->sum,0,sizeof(a->sum)); memset(a->bitmap,0,sizeof(a->bitmap));
    a->sum.layer=layer; a->sum.burst=burst; a->session=session; a->step=step; a->nodes=nodes; a->expected=expected(burst); a->active=UINT32_MAX;
}
static int accumulator_add(fr_accumulator *a,const fr_fragment *f,double (*remote)[FR_HIDDEN]) {
    if (f->type!=2 || f->session!=a->session || f->node>=a->nodes || f->burst!=a->sum.burst) { ++a->invalid; return -1; }
    if (f->step!=a->step || f->layer!=a->sum.layer) { ++a->late; return 0; }
    uint32_t bit=1u<<f->fragment;
    if (a->bitmap[f->node]&bit) { ++a->duplicates; return 0; }
    if (remote) {
        /* At most 20 finite binary16 values. Their sum is exactly representable
         * in binary64, independently of packet arrival and node enumeration. */
        for (unsigned d=0;d<FR_FRAGMENT_DIMS;++d) remote[f->token][f->start+d]+=f->values[d];
    } else for (unsigned d=0;d<FR_FRAGMENT_DIMS;++d) a->sum.y[f->token][f->start+d]+=f->values[d];
    a->bitmap[f->node]|=bit; return 1;
}
int fr_accumulator_add(fr_accumulator *a,const fr_fragment *f) { return accumulator_add(a,f,NULL); }
uint32_t fr_accumulator_missing(const fr_accumulator *a) {
    uint32_t missing=0;
    for (unsigned n=0;n<a->nodes;++n) if ((a->active>>n&1) && a->bitmap[n]!=a->expected) missing|=1u<<n;
    return missing;
}
static void *receive_loop(void *arg) {
    fr_receiver *r=arg; int64_t active=net_now_ns();
    static _Thread_local unsigned char packet[FR_FRAME_MAX];
    while (!atomic_load(&r->stop)) {
        int drained=1;
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
                if (fr_fragment_read(packet+at,len,&f) || f.session!=r->session || f.burst>FR_MAX_BURST || f.type!=1) {
                    atomic_fetch_add(&r->stats.invalid,1); continue;
                }
                uint32_t before=r->assembly.sequence==(uint64_t)f.step*FR_LAYERS+f.layer+1?r->assembly.bitmap:0;
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
                ssize_t sent=!n?-1:r->reply_fd>=0?send(r->reply_fd,packet,frame,0):net_udp_send_from(r->fd,packet,frame,&r->head,&r->local);
                if (n && sent<0 && r->reply_fd>=0 && errno==EPIPE) {
                    /* macOS closed the flow: a program started in the background with no
                     * Local Network permission (nohup from an ssh that then closed: run 92),
                     * or a VPN/filter extension. Try the unconnected data socket once more. */
                    fprintf(stderr,"dataplane: macOS closed the reply socket (EPIPE); replying from the data socket. If replies still fail, "
                                   "this node has no Local Network permission: start it as the installed service (fronk-install.sh) or from Terminal\n");
                    close(r->reply_fd); r->reply_fd=-1;
                    sent=net_udp_send_from(r->fd,packet,frame,&r->head,&r->local);
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
            atomic_fetch_add(&r->stats.wait_ns,(uint64_t)(w->started-w->submitted));
            atomic_fetch_add(&r->stats.reply_ns,(uint64_t)(net_now_ns()-w->started-(int64_t)w->elapsed));
            atomic_store_explicit(&w->state,0,memory_order_release);
        }
        fr_assembly *a=&r->assembly;
        if (drained && a->sequence && !a->submitted && a->bitmap==expected(a->request.burst) && !atomic_load(&r->compute.state)) {
            if (!fr_compute_submit(&r->compute,&a->request,a->step)) {
                a->submitted=1; atomic_fetch_add(&r->stats.requests,1);
                atomic_fetch_add(&r->stats.rx_ns,(uint64_t)(r->compute.submitted-r->first_rx));
            }
        }
        if (net_now_ns()-active>=50000000 && !atomic_load(&r->compute.state)) {
            struct pollfd fds[2]={{.fd=r->fd,.events=POLLIN},{.fd=r->multicast_fd,.events=POLLIN}};
            (void)poll(fds,2,20);
        }
        else relax();
    }
    return NULL;
}
int fr_receiver_open(fr_receiver *r,int fd,int multicast_fd,uint32_t session,unsigned node,unsigned burst,
                     const struct sockaddr_in *head,const fr_udp_info *local,const fr_backend *backend) {
    if (!r || fd<0 || !session || node>=FR_CONNECTION_NODES || !burst || burst>8 || !head || !local) return -1;
    memset(r,0,sizeof(*r)); r->fd=fd; r->multicast_fd=multicast_fd; r->session=session; r->node=node; r->burst=burst; r->head=*head; r->local=*local;
    atomic_init(&r->stop,0); fr_plane_stats_init(&r->stats);
    /* A connected socket skips the per-send route/policy lookup (~3x cheaper on macOS). */
    char ip[INET_ADDRSTRLEN]; r->reply_fd=-1;
    if (inet_ntop(AF_INET,&local->address,ip,sizeof(ip)) && (r->reply_fd=net_udp_bind(ip,0,0))>=0 &&
        connect(r->reply_fd,(const struct sockaddr *)head,sizeof(*head))) {
        close(r->reply_fd); r->reply_fd=-1;
    }
    if (fr_compute_open(&r->compute,backend)) { if (r->reply_fd>=0) close(r->reply_fd); return -1; }
    if (!pthread_create(&r->thread,NULL,receive_loop,r)) return 0;
    fr_compute_close(&r->compute); if (r->reply_fd>=0) close(r->reply_fd); return -1;
}
int fr_receiver_close(fr_receiver *r) {
    atomic_store(&r->stop,1); int rc=pthread_join(r->thread,NULL);
    if (r->reply_fd>=0) close(r->reply_fd);
    return fr_compute_close(&r->compute) || rc?-1:0;
}
int fr_head_plane_open(fr_head_plane *h,const fr_connection *links,const char *group,unsigned port,const fr_backend *local,unsigned blocks) {
    if (!h || !links || links->count>FR_CONNECTION_NODES || !links->session || (!local && !links->count) || (!!local!=!!blocks)) return -1;
    memset(h,0,sizeof(*h)); h->links=links; h->local_blocks=blocks; h->active=UINT32_MAX;
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
/* ---- One layer barrier. Validate -> publish local work -> fan out -> drain replies.
 * Do not reorder publication or buffer reuse without checking both error paths. ---- */
int fr_head_plane_run(fr_head_plane *h,const fr_req *req,uint32_t step,uint64_t deadline_ns,fr_part *out) {
    if (!h || !req || !out || !deadline_ns || !req->burst || req->burst>8 || req->layer>=48) return -1;
#ifdef FRONK_PATH_PROFILE
    const int64_t path_start=net_now_ns();
#endif
    /* Stage all packets first: numeric/routing failure must send nothing. The
     * local slice sees exactly the same fp16-rounded activations as the nodes. */
    h->rounded=*req;
    for (unsigned t=0;t<req->burst;++t) for (unsigned d=0;d<FR_HIDDEN;++d) {
        if (fr_half(req->x[t][d],&h->half[t][d])) return -1;
        h->rounded.x[t][d]=fr_unhalf(h->half[t][d]);
    }
#ifdef FRONK_PATH_PROFILE
    /* Bounded replay fixture, never a performance run. Capture decode only
     * after the first single-token verifier, excluding initial prefill. */
    if(h->capture && req->layer==0 && req->burst==1) h->capture_decode=1;
    if(h->capture && h->capture_decode && req->burst==h->capture_width && h->capture_count<192) {
        if(fwrite(&h->rounded,sizeof(h->rounded),1,h->capture)!=1) return -1;
        ++h->capture_count;
        if(h->capture_count==192) {
            if(fflush(h->capture)) return -1;
            fprintf(stderr,"FRONK_CAPTURE complete calls=%u T=%u\n",h->capture_count,h->capture_width);
        }
    }
#endif
    size_t frame=0;
    for (unsigned f=0;f<4*req->burst;++f) {
        h->lengths[f]=fr_req_encode(h->frame+frame,sizeof(h->frame)-frame,req,h->half,h->links->session,step,f);
        if (!h->lengths[f]) return -1;
        frame+=h->lengths[f];
    }
    if (h->exact_remote) memset(h->exact_remote,0,req->burst*sizeof(*h->exact_remote));
    fr_accumulator *a=&h->accumulator;
    fr_accumulator_begin(a,h->links->session,step,req->layer,req->burst,h->links->count); a->active=h->active;
    int64_t start=net_now_ns(); int rc=0,local_done=!h->has_local,remote_done=0; int last=-1;
    /* Publish local work before fanout: its compute (or SSD faults) can overlap
     * send/receive latency. Encoding/validation above must still finish first.
     * A failed send is drained below together with the already running worker. */
    if (h->has_local && fr_compute_submit(&h->local,&h->rounded,step)) return -1;
    int64_t send_start=net_now_ns();
    if (h->links->count && fanout_send(&h->fanout,h->frame,frame)) rc=-1;
    const int64_t sent=net_now_ns();
    h->send_ns+=(uint64_t)(sent-send_start);
    h->last_missing=0; h->last_local_ns=0;
    do {
        if (!remote_done) {
            if ((uint64_t)(net_now_ns()-start)>=deadline_ns) {
                h->last_missing=fr_accumulator_missing(a); remote_done=1;
                if (h->last_missing) ++h->deadline_hits;
            } else {
                for (unsigned i=0;i<h->links->interface_count;++i) for (unsigned batch=0;batch<32;++batch) {
                    struct sockaddr_in peer; fr_fragment f;
                    ssize_t n=net_udp_recv(h->links->sockets[i],h->rx,sizeof(h->rx),&peer);
                    if (n<0) { if (errno!=EAGAIN && errno!=EWOULDBLOCK) ++a->invalid; break; }
                    if ((uint64_t)(net_now_ns()-start)>=deadline_ns) { ++a->late; break; }
                    for (size_t at=0,len;at<(size_t)n;at+=len) {
                        if (!(len=fr_frame_next(h->rx+at,(size_t)n-at))) { ++a->invalid; break; }
                        if (fr_fragment_read(h->rx+at,len,&f) || f.node>=h->links->count) { ++a->invalid; continue; }
                        const fr_link *link=&h->links->links[f.node]; struct sockaddr_in source;
                        if (net_address(link->ip,link->identity.data_port,&source) || i!=link->interface_slot ||
                            peer.sin_addr.s_addr!=source.sin_addr.s_addr) { ++a->invalid; continue; }
                        if (accumulator_add(a,&f,h->exact_remote)==1 && a->bitmap[f.node]==a->expected) {
                            uint64_t arrive=(uint64_t)(net_now_ns()-start);
                            h->arrive_ns[f.node]+=arrive; last=(int)f.node;
#ifdef FRONK_PATH_PROFILE
                            h->path[req->burst].arrive[f.node]+=arrive;
#endif
                        }
                    }
                }
                if (!fr_accumulator_missing(a)) remote_done=1;
            }
        }
        if (!local_done && atomic_load_explicit(&h->local.state,memory_order_acquire)==3) {
            if (h->local.result) rc=-1;
            else if (!h->exact_remote) for (unsigned t=0;t<req->burst;++t) for (unsigned d=0;d<FR_HIDDEN;++d) a->sum.y[t][d]+=h->local.part.y[t][d];
            h->last_local_ns=h->local.elapsed; h->local_ns+=h->local.elapsed;
            uint64_t arrive=(uint64_t)(net_now_ns()-start);
            h->arrive_ns[h->links->count]+=arrive; last=(int)h->links->count;
#ifdef FRONK_PATH_PROFILE
            h->path[req->burst].arrive[h->links->count]+=arrive;
#endif
            atomic_store(&h->local.state,0); local_done=1;
        }
        if (!local_done || !remote_done) relax();
    } while (!local_done || !remote_done);
    h->last_total_ns=(uint64_t)(net_now_ns()-start); h->total_ns+=h->last_total_ns; ++h->barriers;
    ++h->width_calls[req->burst];
    if (last>=0) ++h->last[last];
    for (unsigned n=0;n<h->links->count;++n) if (h->last_missing&(1u<<n)) ++h->missing[n];
    if (!rc && h->exact_remote) {
        for (unsigned t=0;t<req->burst;++t) for (unsigned d=0;d<FR_HIDDEN;++d)
            a->sum.y[t][d]=(float)(h->exact_remote[t][d]+(h->has_local?h->local.part.y[t][d]:0.0));
    }
    if (!rc) *out=a->sum;
#ifdef FRONK_PATH_PROFILE
    const int64_t finished=net_now_ns();
    ++h->path[req->burst].calls;
    h->path[req->burst].prepare+=(uint64_t)(start-path_start);
    h->path[req->burst].submit+=(uint64_t)(send_start-start);
    h->path[req->burst].send+=(uint64_t)(sent-send_start);
    h->path[req->burst].wait+=h->last_total_ns-(uint64_t)(sent-start);
    h->path[req->burst].finish+=(uint64_t)(finished-start)-h->last_total_ns;
    h->path[req->burst].local+=h->last_local_ns;
    if(last>=0) ++h->path[req->burst].last[last];
    if(++h->path_calls==480) {
        for(unsigned t=1;t<=FR_MAX_BURST;t++) if(h->path[t].calls) {
            const double scale=1e-3/h->path[t].calls;
            fprintf(stderr,"FRONK_PATH T=%u calls=%llu prepare_us=%.3f submit_us=%.3f send_us=%.3f wait_us=%.3f finish_us=%.3f local_compute_us=%.3f\n",
                t,(unsigned long long)h->path[t].calls,h->path[t].prepare*scale,h->path[t].submit*scale,
                h->path[t].send*scale,h->path[t].wait*scale,h->path[t].finish*scale,h->path[t].local*scale);
            for(unsigned i=0;i<=h->links->count;i++) {
                if(i==h->links->count ? !h->has_local : !(h->active>>i&1)) continue;
                fprintf(stderr,"FRONK_PATH_NODE T=%u calls=%llu node=%s arrival_us=%.3f last=%llu\n",t,
                    (unsigned long long)h->path[t].calls,i==h->links->count?"head":h->links->links[i].identity.name,
                    h->path[t].arrive[i]*scale,(unsigned long long)h->path[t].last[i]);
            }
        }
        memset(h->path,0,sizeof(h->path));h->path_calls=0;
    }
#endif
    return rc;
}
int fr_head_plane_close(fr_head_plane *h) {
    free(h->exact_remote); h->exact_remote=NULL;
    int rc=h->has_local?fr_compute_close(&h->local):0; h->has_local=0;
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
        (unsigned long long)h->deadline_hits,(unsigned long long)h->accumulator.duplicates,
        (unsigned long long)h->accumulator.late,(unsigned long long)h->accumulator.invalid)<0) return -1;
    for (unsigned n=0;n<h->links->count;++n) if (fprintf(output,"STATS node=%s link=%s deadline_misses=%llu\n",
        h->links->links[n].identity.name,interface_type(h->links->links[n].via.type),(unsigned long long)h->missing[n])<0) return -1;
    return 0;
}
