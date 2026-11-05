/* node.h - one control-plane event loop, owned by the foreground process.
 * Requires a nonempty password. Runs until SIGINT/SIGTERM; returns zero on
 * clean shutdown, one on failure. Each TCP connection authenticates separately.
 */
#ifndef FR_NODE_H
#define FR_NODE_H
#include "config.h"
int node_run(const fr_config *config, int debug);
#endif
