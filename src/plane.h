/* Session-owned storage. Backend calls stay serialized; no hot-path allocation.
 * Node RX and compute are separate threads. The head uses the same compute
 * worker in-process while draining PART on every active head interface. */
#ifndef FR_PLANE_H
#define FR_PLANE_H
#include "fanout.h"
#include "wire.h"
#include <pthread.h>
#include <stdatomic.h>
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
    fr_part part;
    uint32_t step;
    int result;
    uint64_t elapsed;
    int64_t submitted,started;
} fr_compute;
typedef struct {
    fr_req request;
    uint64_t sequence;
    uint32_t bitmap,step;
    int submitted;
} fr_assembly;
typedef struct {
    fr_part sum;
    uint32_t session,step,expected,active,bitmap[FR_CONNECTION_NODES];
    unsigned nodes;
    uint64_t duplicates,late,invalid;
} fr_accumulator;
typedef struct {
    pthread_t thread;
    atomic_int stop;
    int fd,multicast_fd,reply_fd; /* reply_fd: connected to the head, ephemeral port */
    uint32_t session;
    unsigned node,burst;
    struct sockaddr_in head;
    fr_udp_info local;
    fr_compute compute;
    fr_assembly assembly;
    fr_plane_stats stats;
    int64_t first_rx;
} fr_receiver;
typedef struct {
    fr_fanout fanout;
    const fr_connection *links;
    fr_compute local;
    int has_local;
    unsigned local_blocks;
    fr_accumulator accumulator;
    double (*exact_remote)[FR_HIDDEN]; /* deterministic sums for speculative verification */
    fr_req rounded;
    uint16_t half[FR_MAX_BURST][FR_HIDDEN];
    unsigned char packets[4*FR_MAX_BURST][FR_REQ_FRAGMENT_MAX];
    size_t lengths[4*FR_MAX_BURST];
    unsigned char frame[FR_FRAME_MAX],rx[FR_FRAME_MAX];
    uint64_t barriers,deadline_hits,missing[FR_CONNECTION_NODES],local_ns,total_ns,send_ns;
    uint64_t width_calls[FR_MAX_BURST+1]; /* reuse request metadata; no new clock reads */
    uint64_t last_local_ns,last_total_ns;
    uint32_t last_missing,active; /* active: link indexes with blocks (default all) */
    /* Per participant (links, then head at index links->count): summed arrival
     * time and how often it completed last, for REPORT's bottleneck. */
    uint64_t arrive_ns[FR_CONNECTION_NODES+1],last[FR_CONNECTION_NODES+1];
#ifdef FRONK_PATH_PROFILE
    /* Diagnostic builds only: separate widths, never add overlapping node
     * arrivals to the serial critical-path stages. No production timers. */
    struct {
        uint64_t calls, prepare, submit, send, wait, finish, local;
        uint64_t arrive[FR_CONNECTION_NODES+1], last[FR_CONNECTION_NODES+1];
    } path[FR_MAX_BURST+1];
    unsigned path_calls;
    FILE *capture;
    unsigned capture_width,capture_count,capture_decode;
#endif
} fr_head_plane;
void fr_plane_stats_init(fr_plane_stats *s);
int fr_compute_open(fr_compute *w,const fr_backend *backend);
int fr_compute_submit(fr_compute *w,const fr_req *request,uint32_t step);
int fr_compute_close(fr_compute *w);
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
int fr_head_plane_close(fr_head_plane *h);
int fr_head_plane_report(const fr_head_plane *h,FILE *output);
#endif
