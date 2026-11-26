/* hot.h - whole hot experts in caches (GSQ layout).
 *
 * The slice nodes hold every expert, cut in 32-neuron slices (units). Routing is
 * skewed: about a tenth of a layer's experts take half of its uses. A cache keeps
 * some experts whole in RAM (all 20 slices) and computes them alone; the head asks
 * the slice nodes only for the experts that are in no cache. Each side receives
 * gate 0 for the others' experts, which the gsq backend skips, so the partials add
 * up to the same sum as before. An expert missing from every cache is never an
 * error and never a disk read on the request path: the slices cover it.
 *
 * There may be several caches (FR_HOT_CACHES): the head itself (its experts then
 * need no network at all) and machines that hold no unit. Their catalogues are
 * exclusive: every expert has one cache that may hold it, so the catalogues
 * together are the whole model and each cache machine needs on its disk only its
 * part. The head knows the catalogue of every cache and, because it is the head
 * that orders each copy and each release, exactly what is in each one now
 * (fr_hot_owner): an expert in its cache is computed there alone, any other by
 * the slices alone. Nothing is asked twice and there is no list to keep in step.
 *
 * Two halves, in two processes when a cache is another machine:
 *   store  (the cache): 20 anonymous units loaded into the gsq backend, filled
 *          from their start with slots; an expert's slice is at the same slot in
 *          each unit (gsq.h fr_gsq_backend_slots). Experts are copied from the
 *          machine's own model file when the head says so (control command HOT).
 *   policy (head): scores every call's routing, and the scores fade (they halve
 *          in about 130 tokens), so the caches follow the chat in progress and
 *          not the history: measured on a real chat, 7000 experts chosen from
 *          history alone served 57% of the uses, the same memory following the
 *          chat 73%. Two movements:
 *          - how much each cache computes follows the machines' real speed: a
 *            cache that seldom answers last receives more of its catalogue's most
 *            used experts while it has room; one that answers last too often gives
 *            its least used ones back to the slices. Its memory is an upper bound.
 *          - what a full cache holds follows the work: in batches, its least used
 *            experts leave (control command COLD) and experts used more are copied
 *            from its own model file (HOT). So many per token of work, no more
 *            (state-dir/hot-loads-per-token, default 3: reads of the cache's disk,
 *            never writes).
 *          Use counts persist in state-dir/hot-experts.bin (98 KB, written when the
 *          head stops, never during a session): a start resumes from them.
 * The store's memory is wired (mlock) a piece at a time, before it is written, up to
 * its budget, and stays wired until the store closes: an expert that leaves frees
 * its slot for another one. Left pageable, macOS compresses the experts used less
 * often; given back page by page, it swaps them.
 */
#ifndef FR_HOT_H
#define FR_HOT_H
#include "backend.h"
#include "prep.h"
#include <stdatomic.h>
#include <stdint.h>

/* ---- store: the cache node ---- */
int fr_hot_open(const fr_gguf *g,uint64_t budget_bytes);
void fr_hot_close(void);
/* 0: the expert is in memory; 1: the budget is full; -1: error. */
int fr_hot_admit(unsigned layer,unsigned expert);
/* Free the expert's slot for another one. Only when no request for it can still arrive. */
int fr_hot_evict(unsigned layer,unsigned expert);
void fr_hot_usage(uint64_t *bytes,int *held);   /* memory held; whether it is wired */
/* The budget may change while the store is open (an elastic cache, see the policy).
 * grow: a larger budget; memory is taken only as experts come. above: the experts
 * whose slot ends above a smaller budget (the count; the first max of them in who,
 * as layer and expert). shrink: when those have all been evicted, the memory above
 * the new budget goes back to the system at once; -1 if a slot up there is in use. */
uint64_t fr_hot_capacity(void);
int fr_hot_grow(uint64_t budget_bytes);
unsigned fr_hot_above(uint64_t budget_bytes,uint16_t (*who)[2],unsigned max);
int fr_hot_shrink(uint64_t budget_bytes);
const unsigned char *fr_hot_peek(unsigned block,unsigned layer,unsigned expert);   /* an admitted slice, for tests */

/* ---- policy: the head ---- */
#define FR_HOT_CACHES 3
typedef int (*fr_hot_admit_fn)(void *ctx,unsigned layer,unsigned expert);  /* as fr_hot_admit */
typedef int (*fr_hot_evict_fn)(void *ctx,unsigned layer,unsigned expert);  /* as fr_hot_evict */
/* start, then add each cache (and test it), then go. begun/done: the head plane's
 * runs_begun and runs_done. */
int fr_hot_policy_start(const char *state_dir,const _Atomic uint64_t *begun,const _Atomic uint64_t *done);
/* One more cache; its number (0 first) or -1. share: the size of its catalogue against
 * the others' (its GB, or less when its speed is the limit). calls/last: the plane's counters for
 * it (plane.h cache_calls, cache_last). evict may be NULL: it then keeps what it loads. */
int fr_hot_policy_add(const char *name,double share,fr_hot_admit_fn admit,fr_hot_evict_fn evict,void *ctx,
                      const _Atomic uint64_t *calls,const _Atomic uint64_t *last);
/* Before go: cache k is the store of this process (fr_hot_open) and may use between
 * floor and ceiling bytes. It takes more while the machine has memory to spare and
 * gives memory back before the machine runs short, the slots filled last first: the
 * engine on the same machine grows with the chat, and a cache of fixed size either
 * wastes memory or ends in swap. */
int fr_hot_policy_elastic(unsigned cache,uint64_t floor_bytes,uint64_t ceiling_bytes);
/* After go: ask an elastic cache for a size (clamped to its bounds); the balance
 * thread does it within half a second. For tests and for an operator. */
int fr_hot_policy_resize(unsigned cache,uint64_t bytes);
int fr_hot_policy_go(void);   /* resume from the saved counts and follow the workload */
void fr_hot_policy_stop(void);
void fr_hot_save(void);   /* the counts, to state-dir/hot-experts.bin; safe in a signal handler */
/* That cache stopped answering: route its experts back to the slices for the rest
 * of this run. The slices hold everything, so the chat goes on. */
void fr_hot_policy_disable(unsigned cache);
/* Admit now and route there (the self-test's experts). */
int fr_hot_policy_admit(unsigned cache,unsigned layer,unsigned expert);
/* Route back to the slices, wait for the calls under way, free it where it is held. */
int fr_hot_policy_evict(unsigned layer,unsigned expert);
void fr_hot_count(const fr_req *req);   /* one call's routing; the caller serializes */
/* Experts routed to that cache and the share of counted uses it served. */
void fr_hot_report(unsigned cache,unsigned *experts,double *served);
/* Where each expert is computed: 0 the slices, c+1 cache c. */
extern _Atomic uint8_t fr_hot_owner[FR_LAYERS][FR_ROUTED_EXPERTS];
static inline unsigned fr_hot_where(unsigned layer,unsigned expert) {
    return atomic_load_explicit(&fr_hot_owner[layer][expert],memory_order_acquire);
}
static inline int fr_hot_has(unsigned layer,unsigned expert) { return fr_hot_where(layer,expert)!=0; }
#endif
