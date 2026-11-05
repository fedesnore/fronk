/* selection.h - measured per-share backend choices. Pure bounded tables keep
 * eligibility, measurements and assigned backend distinct. No kernel ABI change.
 */
#ifndef FR_SELECTION_H
#define FR_SELECTION_H
#include "hw.h"
#include "model.h"
#include "ssd.h"
#include <stdatomic.h>
#include <stdio.h>
#define FR_CANDIDATES 16
#define FR_SWEEP 5
#define FR_SELECTION_TEXT 16384
typedef struct {
    char name[16];
    unsigned capacity, measured;
    unsigned workers[FR_SWEEP],prefetch[FR_SWEEP];
    double ns[FR_SWEEP],p99[FR_SWEEP];
    uint64_t resident[FR_SWEEP];
    /* An explicitly measured RAM+SSD mode; never extrapolate past these
     * streamed points. Cold costs govern admission; warm costs describe reuse. */
    unsigned ram_blocks,ssd_blocks;
    uint64_t cache_bytes;
    double cold_ns[FR_SSD_MAX_BLOCKS],warm_ns[FR_SSD_MAX_BLOCKS],tail_ns[FR_SSD_MAX_BLOCKS];
} fr_candidate;
typedef struct {
    unsigned count,burst,assigned;
    fr_candidate candidate[FR_CANDIDATES];
} fr_selection;
uint64_t selection_capacity(const fr_hwinfo *hw,const char *backend,int head);
int selection_add(fr_selection *s,const char *backend,unsigned capacity);
int selection_record(fr_selection *s,unsigned candidate,unsigned blocks,double p50,double p99,uint64_t resident);
int selection_limit(fr_selection *s,unsigned candidate,unsigned capacity);
int selection_settings(fr_selection *s,unsigned candidate,unsigned blocks,unsigned workers,unsigned prefetch);
int selection_hybrid(fr_selection *s,unsigned ram,unsigned streamed,uint64_t cache_bytes,
                     const double *cold,const double *warm,const double *tail);
uint32_t selection_paged(const fr_candidate *c,uint32_t mask);
int selection_fits(const fr_candidate *c,const fr_hwinfo *hw,int head,unsigned blocks);
/* Exact measured points through five; larger shares use a nonnegative linear
 * fit to the block sweep. Report such costs as estimates, never measurements. */
double selection_cost(const fr_candidate *c,unsigned blocks);
int selection_pick(const fr_selection *s,unsigned blocks);
int selection_complete(const fr_selection *s);
int selection_format(const fr_selection *s,char *out,size_t size);
int selection_parse(const char *text,size_t len,fr_selection *out);
int selection_reason(const fr_selection *s,unsigned blocks,char *out,size_t size);
/* Native setup: probes only compiled backends; never runs a kernel to infer ISA. */
int selection_discover(const fr_hwinfo *hw,unsigned burst,int head,fr_selection *out);
/* Config "backend": only NAME is a candidate from now on (it still passes SELFTEST
 * and gets its block sweep, which the planner needs). NULL or "" restores the race.
 * Fails, with a message on ERR, when NAME is not built in or cannot run here. */
int selection_backend(const char *name,FILE *err);
const char *selection_chosen(void);
/* The backend to suggest without measuring anything (fronk backends). */
const char *selection_recommend(const fr_hwinfo *hw);
const char *selection_recommend_cpu(const fr_hwinfo *hw);
/* Native backends that pass SELFTEST exclude ref; a 200-sample probe picks one
 * winner. Missing sweep points use 256 samples per width, all widths sharing
 * one resident fixture. Existing full reports remain valid. CACHE reuses results,
 * probes and SELFTEST passes keyed by selection_cache_identity (compute build SHA-256 +
 * hardware); REFRESH re-measures. CANCEL (may be NULL) stops between and within
 * points; a cancelled point is not saved. */
int selection_sweep_local(const fr_hwinfo *hw,unsigned burst,int head,fr_selection *out,FILE *progress,const char *cache,int refresh,const atomic_int *cancel);
/* 1 when every native candidate's SELFTEST pass is saved in CACHE: the sweep then
 * needs no idle backend, so a node can keep its loaded blocks through it. */
int selection_self_tests_cached(const fr_hwinfo *hw,unsigned burst,int head,const char *cache);
void selection_cache_identity(const char *identity);
const char *selection_cache_identity_get(void);
/* Installation calibrates; reconnect only reads compatible saved profiles. */
int selection_hybrid_calibrate(const fr_hwinfo *hw,int head,const fr_selection *resident,
                               const char *dir,int refresh,FILE *progress);
void selection_hybrid_restore(const fr_hwinfo *hw,int head,fr_selection *s,const char *dir,FILE *progress);
#endif
