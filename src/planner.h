/* Setup-time capacity checks and measured share selection. Pure, no I/O.
 * This module decides how many units each participant should own. head_assign
 * in head.c chooses their identities, preferring already cached units. */
#ifndef FR_PLANNER_H
#define FR_PLANNER_H
#include "model.h"
#include "selection.h"
#include <stddef.h>
#define FR_PLAN_NODES 21 /* head plus at most twenty distinct remote IDs */
typedef struct { uint64_t id,blocks; } fr_capacity_node;
typedef struct { uint64_t total; unsigned nodes; uint32_t missing; } fr_capacity;
/* Returns 0 if feasible, 1 if insufficient, -1 for invalid input (atomic out).
 * Missing IDs use provisional ascending packing, not a performance assignment:
 * N usable blocks cover 0..N-1 and leave N..19 unplaced. */
int planner_capacity(const fr_capacity_node *nodes,unsigned count,fr_capacity *out);
int planner_capacity_format(const fr_capacity *capacity,char *out,size_t size);
/* Exact share split. A call lasts as long as its slowest node, and a node takes
 * its link's network time plus its measured compute time at its share; the
 * shares minimise that maximum (an integer min-max, solved exactly over the
 * measured times). net_ns may be NULL (compute only). */
int planner_compute_shares(const fr_selection *nodes,unsigned count,unsigned *shares);
int planner_shares(const fr_selection *nodes,const double *net_ns,unsigned count,unsigned *shares);
/* Select a measured width from explicit engine settings. Unknown modes retain
 * the configured width; deep unsplit batches are not labelled as measured. */
unsigned planner_burst(unsigned configured,const char *mtp,const char *native,const char *wave);
#endif
