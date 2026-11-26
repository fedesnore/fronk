/* Session-owned storage. Backend calls stay serialized; no hot-path allocation.
 * Node RX and compute are separate threads. The head uses the same compute
 * worker in-process while draining PART on every active head interface. */
#ifndef FR_PLANE_H
#define FR_PLANE_H
#include "fanout.h"
#include "wire.h"
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
/* One bit per logical fragment (4 per token): FR_MAX_BURST 64 needs 256 bits,
 * more than the 32-bit masks that served 8-token calls. */
#define FR_FRAGMENT_WORDS ((4*FR_MAX_BURST+63)/64)
typedef struct { uint64_t w[FR_FRAGMENT_WORDS]; } fr_bits;
static inline void fr_bits_clear(fr_bits *b) { memset(b,0,sizeof(*b)); }
static inline int fr_bits_test(const fr_bits *b,unsigned i) { return (int)(b->w[i/64]>>(i%64)&1); }
static inline void fr_bits_set(fr_bits *b,unsigned i) { b->w[i/64]|=UINT64_C(1)<<(i%64); }
static inline int fr_bits_any(const fr_bits *b) { for (unsigned k=0;k<FR_FRAGMENT_WORDS;++k) if (b->w[k]) return 1; return 0; }
static inline int fr_bits_equal(const fr_bits *a,const fr_bits *b) { return !memcmp(a,b,sizeof(*a)); }
/* The first n bits set: every fragment of an n/4-token call. */
static inline void fr_bits_first(fr_bits *b,unsigned n) {
    fr_bits_clear(b);
    for (unsigned k=0;k<FR_FRAGMENT_WORDS && n;++k) { unsigned take=n<64?n:64; b->w[k]=take==64?UINT64_MAX:(UINT64_C(1)<<take)-1; n-=take; }
}
/* UDP datagram budget for one batch of whole fragments; IPv4 caps a datagram at 65507.
 * Measured with all four nodes replying at once (2026-11-09): 32-token replies lost
 * 0.2% of bursts in 60 KB datagrams but 3.3% in 16 KB ones; 64-token replies lost 17.5%. */
#define FR_DATAGRAM_MAX 61440
/* Threads that may hold an fr_req/fr_part on their stack (~650 KB each at 64
 * tokens): backends keep one locally, and macOS gives new threads only 512 KB. */
#define FR_THREAD_STACK (16u<<20)
static inline int fr_thread_create(pthread_t *t,void *(*fn)(void *),void *arg) {
    pthread_attr_t attr; if (pthread_attr_init(&attr)) return -1;
    int rc=pthread_attr_setstacksize(&attr,FR_THREAD_STACK);
    if (!rc) rc=pthread_create(t,&attr,fn,arg);
    pthread_attr_destroy(&attr); return rc;
}
typedef struct {
    atomic_uint_fast64_t requests,parts,duplicates,late,invalid,gaps,overwritten,errors,compute_ns;
    /* Latency split per request: first fragment -> submit, submit -> compute
     * start, compute end -> last PART sent. */
    atomic_uint_fast64_t rx_ns,wait_ns,reply_ns;
} fr_plane_stats;
typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    atomic_int state,stop,parked; /* 0 idle, 1 submitted, 2 running, 3 complete */
    const fr_backend *backend;
    fr_req request;
    _Alignas(float) unsigned char xq[FR_MAX_BURST][FR_Q8_TOKEN_BYTES];
    int q8_k;
    fr_part part;
    uint32_t step;
    int result;
    uint64_t elapsed;
    int64_t submitted,started;
} fr_compute;
typedef struct {
    fr_req request;
    uint64_t sequence;
    fr_bits bitmap;
    uint32_t step;
    int submitted,q8_k;
    _Alignas(float) unsigned char xq[FR_MAX_BURST][FR_Q8_TOKEN_BYTES];
} fr_assembly;
typedef struct {
    fr_part sum;
    fr_bits expected,bitmap[FR_CONNECTION_NODES];
    uint32_t session,step,active;
    unsigned nodes;
    uint64_t duplicates,late,invalid;
} fr_accumulator;
typedef struct {
    pthread_t thread;
    atomic_int stop;
    int fd,multicast_fd,reply_fd; /* reply_fd: connected to the head, ephemeral port */
    uint32_t session;
    unsigned node,burst;
    int q8_allowed;
    struct sockaddr_in head;
    fr_udp_info local;
    fr_compute compute;
    fr_assembly assembly;
    fr_plane_stats stats;
    int64_t first_rx;
    uint64_t replied; /* sequence of the last request answered: a repeat of it is answered again */
    atomic_int paced; /* control PACE: a reply of several datagrams waits until the head asks for it */
    uint64_t polled;  /* sequence the head has asked for while it was still being computed */
} fr_receiver;
#define FR_PLANE_CACHES 3   /* hot.h FR_HOT_CACHES */
/* One expert call while it is in flight: its staged views, its partial sum and its clock.
 * The head keeps two of them, so that the engine's two half-batches can each have a call
 * at the nodes while the other computes on the GPU. */
typedef struct {
    int active,wide,rc,last,warm_local,local_work,remote_done;
    int local_state;            /* 0 no local work, 1 waits for the worker, 2 on the worker, 3 done */
    unsigned slot,uses_slices,uses_cache[FR_PLANE_CACHES],resend_turn;
    uint32_t step,alt[FR_CONNECTION_NODES]; /* alt: a fresh step a node was asked again under */
    uint32_t slices,cache,warm,asked,missing;
    uint64_t order,deadline_ns,total_ns,local_ns,rx_ns,datagrams;
    int64_t path_start,start,send_start,sent,resend_at,resend_ns;
    fr_accumulator acc;
    fr_req rounded,cold,warm_req[FR_PLANE_CACHES];
    uint16_t half[FR_MAX_BURST][FR_HIDDEN];
    _Alignas(float) unsigned char xq[FR_MAX_BURST][FR_Q8_TOKEN_BYTES];
    unsigned char frame[FR_FRAME_MAX]; size_t lengths[4*FR_MAX_BURST],frame_len;
    unsigned char frame_warm[FR_PLANE_CACHES][FR_FRAME_MAX]; size_t lengths_warm[FR_PLANE_CACHES][4*FR_MAX_BURST],frame_warm_len[FR_PLANE_CACHES];
} fr_call;
typedef struct {
    fr_fanout fanout;
    const fr_connection *links;
    fr_compute local;
    int has_local,q8_k,q8_capable;
    unsigned local_blocks;
    fr_call call[2];             /* the calls in flight (fr_head_plane_begin) */
    const fr_accumulator *last_acc; /* the call that ended last, for fr_head_plane_missing_text */
    int local_owner;             /* the slot whose local view is on the worker, or -1 */
    uint32_t step_hi;            /* the highest step sent: steps only grow, at every node */
    uint64_t order_next,ended_upto,ended_early; /* runs_done counts calls in the order they began */
    unsigned char resend_frame[FR_FRAME_MAX]; size_t resend_lengths[4*FR_MAX_BURST];
    double (*exact_remote)[FR_HIDDEN]; /* deterministic sums for speculative verification */
    /* Caches (hot.h), in the policy's order: cache_link[c] is a link that holds no
     * unit, or -1 when cache c is this process. A call is sent as one view per side
     * of the same activations: `cold` to the slice nodes (gate 0 for every cached
     * expert) and warm[c] to cache c (gate 0 for the experts it does not hold). */
    unsigned caches; int cache_link[FR_PLANE_CACHES]; char group[16]; unsigned port;
    int cache_local;   /* 1 + the number of the cache that is this process (hot.c store, computed by the local worker), or 0 */
    /* unicast: every node has its own request socket, so one node can be asked again
     * alone. resends: requests repeated to a node whose reply had not arrived whole. */
    int unicast; uint64_t resends;
    uint32_t paced;   /* links told to keep large replies until asked (control PACE), one at a time per interface */
    /* Per cache: calls that needed the slice nodes, and those where that cache answered
     * last. hot.c's policy keeps each cache and the slices finishing together. */
    _Atomic uint64_t cache_calls[FR_PLANE_CACHES],cache_last[FR_PLANE_CACHES];
    /* Barriers begun and finished: hot.c frees an expert on the cache node only when
     * every barrier that could still name it there is over. */
    _Atomic uint64_t runs_begun,runs_done;
    unsigned char rx[FR_FRAME_MAX];
    uint64_t barriers,deadline_hits,missing[FR_CONNECTION_NODES],local_ns,total_ns,send_ns;
    uint64_t width_calls[FR_MAX_BURST+1]; /* reuse request metadata; no new clock reads */
    uint64_t last_local_ns,last_total_ns;
    uint32_t last_missing,active; /* active: link indexes with blocks (default all) */
    /* Per participant (links, then head at index links->count): summed arrival
     * time and how often it completed last, for REPORT's bottleneck. */
    uint64_t arrive_ns[FR_CONNECTION_NODES+1],last[FR_CONNECTION_NODES+1];
    /* Per-call timings by width, printed every 480 calls while `profile` is set (the
     * head sets it when its state directory holds a file named path-profile): where
     * the serial stages of a call go, how long each participant takes to answer
     * whole, and how the expert uses were split. A few clock reads a call when on,
     * one branch when off. rx: time spent reading and adding replies. */
    int profile;
    struct {
        uint64_t calls, prepare, submit, send, wait, finish, local, rx, datagrams;
        uint64_t uses_slices, uses_cache[FR_PLANE_CACHES];
        uint64_t arrive[FR_CONNECTION_NODES+1], answered[FR_CONNECTION_NODES+1], last[FR_CONNECTION_NODES+1];
    } path[FR_MAX_BURST+1];
    unsigned path_calls;
#ifdef FRONK_PATH_PROFILE
    FILE *capture;   /* diagnostic builds only: a replay fixture of real calls */
    unsigned capture_width,capture_count,capture_decode;
#endif
} fr_head_plane;
void fr_plane_stats_init(fr_plane_stats *s);
int fr_compute_open(fr_compute *w,const fr_backend *backend);
int fr_compute_submit(fr_compute *w,const fr_req *request,uint32_t step);
int fr_compute_close(fr_compute *w);
int fr_compute_submit_q8(fr_compute *w,const fr_req *request,
                        const unsigned char xq[FR_MAX_BURST][FR_Q8_TOKEN_BYTES],uint32_t step);
/* 1 complete, 0 incomplete/duplicate/old, -1 invalid. */
int fr_assemble(fr_assembly *a,const fr_fragment *f,fr_plane_stats *stats);
void fr_accumulator_begin(fr_accumulator *a,uint32_t session,uint32_t step,unsigned layer,unsigned burst,unsigned nodes);
int fr_accumulator_add(fr_accumulator *a,const fr_fragment *f);
uint32_t fr_accumulator_missing(const fr_accumulator *a);
int fr_receiver_open(fr_receiver *r,int fd,int multicast_fd,uint32_t session,unsigned node,unsigned burst,
                     const struct sockaddr_in *head,const fr_udp_info *local,const fr_backend *backend);
int fr_receiver_close(fr_receiver *r);
int fr_head_plane_open(fr_head_plane *h,const fr_connection *links,const char *group,unsigned port,
                       const fr_backend *local,unsigned blocks);
int fr_head_plane_run(fr_head_plane *h,const fr_req *req,uint32_t step,uint64_t deadline_ns,fr_part *out);
/* The same call in three steps, so that two can be in flight (slots 0 and 1). begin stages
 * and sends it (FR_PLANE_BUSY: the other slot's call cannot share the wire with it, end
 * that one first); poll receives replies for both and returns the slots that are complete,
 * one bit each; end hands out the sum (last_missing, last_total_ns: this call's). A node
 * that lost a datagram while two calls are in flight is asked again under a fresh step:
 * it keeps one result, and the other call may have replaced it. */
#define FR_PLANE_BUSY (-2)
int fr_head_plane_begin(fr_head_plane *h,unsigned slot,const fr_req *req,uint32_t step,uint64_t deadline_ns);
unsigned fr_head_plane_poll(fr_head_plane *h);
int fr_head_plane_end(fr_head_plane *h,unsigned slot,fr_part *out);
/* After a call that left nodes missing: " name got/expected" reply parts for each of
 * them. Nothing arrived: the request or the node was lost; some arrived: the reply was. */
void fr_head_plane_missing_text(const fr_head_plane *h,char *text,size_t capacity);
/* After open: one request socket per node (fr_head_plane_run's recovery needs it). */
int fr_head_plane_unicast(fr_head_plane *h);
/* After open, with the hot store open here and no unit on the head: the head's own
 * worker computes the cached experts of every call. Returns the cache's number. */
int fr_head_plane_cache_local(fr_head_plane *h);
/* Q8_K requests are computed by the gsq backend. A node whose cache lives elsewhere
 * (hotgpu.h: a GPU) points this at its own run for the session. */
extern int (*fr_plane_q8_run)(const fr_req *req,const void *xq,fr_part *out);
/* Set by such a node: its work waits for a GPU, not for a core, so its threads sleep
 * between calls instead of spinning on the cores the slices of the same Mac need. */
extern int fr_plane_patient;
/* After open: one more cache on this link, which holds no unit. Returns its number. */
int fr_head_plane_cache(fr_head_plane *h,unsigned link);
int fr_head_plane_close(fr_head_plane *h);
int fr_head_plane_report(const fr_head_plane *h,FILE *output);
#endif
