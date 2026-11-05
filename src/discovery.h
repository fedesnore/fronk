/* Gather hardware and backend costs over authenticated control links.
 * Hardware availability is current; compatible saved compute curves are reused.
 * The split start/collect API lets the head do its own setup while nodes answer.
 * Planning consumes these profiles; it does not measure kernels itself.
 */
#ifndef FR_DISCOVERY_H
#define FR_DISCOVERY_H
#include "connection.h"
#include "hw.h"
#include "bench.h"
#include "selection.h"
typedef struct {
    fr_hwinfo nodes[FR_CONNECTION_NODES];
    fr_bench_report compute[FR_CONNECTION_NODES];
    fr_selection selection[FR_CONNECTION_NODES],head_selection;
    char last_backend[FR_CONNECTION_NODES][16];
    uint32_t measured;
    unsigned count, excluded,burst;
} fr_discovery;
int discovery_fetch(int fd, fr_hwinfo *out);
int discovery_run(const fr_connection *links, fr_discovery *out, FILE *output);
/* Print capacity first in planning output; 0 fits, 1 refuses, -1 invalid. */
int discovery_capacity(const fr_config *cfg,const fr_discovery *hw,FILE *output);
/* Request each node's backend curves, from cache or a measured sweep. A missing,
 * failed or wrong-width profile fails the phase instead of inventing a cost. */
int discovery_bench(const fr_connection *links,fr_discovery *out,FILE *output,int refresh);
int discovery_bench_start(const fr_connection *links,fr_discovery *out,FILE *output,int refresh);
int discovery_bench_collect(const fr_connection *links,fr_discovery *out,FILE *output);
/* Called by the planner with integer shares; verifies all 20 blocks, selected
 * backend capacities and node acknowledgments before accepting configuration. */
int discovery_configure(const fr_config *cfg,const fr_connection *links,fr_discovery *out,
                        unsigned head_blocks,const unsigned *shares,unsigned ssd,FILE *output);
/* Atomic consolidated profile, with head capacity and one section per node. */
int discovery_save(const fr_config *cfg, const fr_connection *links, const fr_discovery *hw);
#endif
