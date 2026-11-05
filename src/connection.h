/* connection.h - phase 1: discover, authenticate, measure links, check multicast.
 * run initializes an empty result; close releases all live descriptors before
 * reuse. Failures retain partial results for diagnosis, never report success.
 */
#ifndef FR_CONNECTION_H
#define FR_CONNECTION_H
#include "config.h"
#include "probe.h"
#include "stats.h"
#include "interfaces.h"
#include <stdio.h>
#define FR_DISCOVERY_GROUP "239.7.7.6"
#define FR_CONNECTION_PATHS (FR_CONNECTION_NODES*FR_INTERFACES)
/* The largest currently measured barrier (burst 4, 20 KiB) ranks paths. Keep
 * all three sizes in the result so callers can inspect each workload separately. */
#define FR_PATH_RTT_KIND 2
typedef struct {
    fr_identity identity;
    char ip[16];
    fr_interface via;
    unsigned interface_slot,selected,ready,skipped;
    uint64_t percentile[3][3];
    unsigned samples[3],losses[3];
} fr_path;
typedef struct {
    fr_identity identity;
    char ip[16];
    int fd, authenticated, multicast;
    fr_stats rtt[3];
    unsigned losses[3];
    fr_interface via;
    unsigned interface_slot,path;
} fr_link;
typedef struct {
    fr_link links[FR_CONNECTION_NODES];
    unsigned count, ignored, duplicates;
    uint32_t session, step;
    int udp, local;
    fr_interface interfaces[FR_INTERFACES];
    int sockets[FR_INTERFACES];
    unsigned interface_count,path_count;
    fr_path paths[FR_CONNECTION_PATHS];
} fr_connection;
/* Returns the best complete path for an ID, or -1. Ties are deterministic. */
int connection_best_path(const fr_path *paths,unsigned count,uint64_t id,unsigned kind);
int connection_run(const fr_config *config, fr_connection *result, FILE *output);
int connection_close(fr_connection *result);
#endif
