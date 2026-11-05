/* head.h - `fronk head`: CONNECTION → DISCOVERY → PLAN → [p]roceed/[m]odified →
 * FRONK (units, sessions, selftest, expert service, test generation) → REPORT.
 * Any [m], restart or failed plan returns to CONNECTION; caches are reused.
 */
#ifndef FR_HEAD_H
#define FR_HEAD_H
#include "config.h"
#include "planner.h"
#include "probe.h"
#include <stdint.h>
#define FR_PLAN_HEAD FR_CONNECTION_NODES
typedef struct {
    unsigned count;          /* participants; head last when present */
    struct { unsigned node, share, cap; uint32_t cached, mask, paged; uint64_t cache_bytes; } p[FR_PLAN_NODES];
    double cache_read_bps;    /* sampled on the filesystem holding unit files */
    uint32_t ssd;            /* unowned blocks the head reads from its SSD slices */
} fr_plan;
/* Pure: keep each participant's cached blocks up to its share, then fill the
 * rest, favoring blocks absent from head storage when a spill is needed. */
int head_assign(fr_plan *plan);
/* Water-filling shares within capacities; leftover blocks go to head-ssd. */
/* Network-time calibration: the per-call network time measured by REPORT for a
 * node on a given address and burst, reused by the next plan. */
int fr_netcal_get(const char *dir,uint64_t id,const char *ip,unsigned burst,double *us);
void fr_netcal_put(const char *dir,uint64_t id,const char *ip,unsigned burst,double us);
int head_shares(const fr_selection *curves,const double *net_ns,unsigned count,unsigned *shares,unsigned *ssd);
int head_main(const fr_config *cfg);
/* `fronk demo`: plan and time the cluster on synthetic units (no model needed). */
int head_demo_main(const fr_config *cfg,unsigned passes);
#endif
