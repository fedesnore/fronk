/* config.h - the single, bounded key/value configuration for head and nodes.
 * Defaults are local-development values. A failed load leaves output intact.
 */
#ifndef FR_CONFIG_H
#define FR_CONFIG_H
#include <stddef.h>
#include <stdint.h>
#define FR_DEFAULT_MARGIN UINT64_MAX
#define FR_CONFIG_NODES 16
typedef struct {
    char name[64], bind[16], multicast[16], password[129], interface[16], broadcast[16];
    char state_dir[256], profile_file[256];
    char model[512], ds4[512]; /* head: local Q4 GGUF and ds4-server command line */
    char interfaces[128]; /* optional auto or local IPv4 list; empty uses interface */
    struct { char name[64],address[16]; } peers[FR_CONFIG_NODES];
    unsigned peer_count;
    uint64_t mem_margin, head_reserve;
    unsigned head_blocks; /* head: fixed resident block count; 0 measures memory */
    char backend[32]; /* compute backend this machine uses; empty (auto) races the compatible ones */
    unsigned control_port, data_port, discovery_port, burst;
} fr_config;
void config_defaults(fr_config *out);
int config_load(const char *path, fr_config *out, char *error, size_t error_cap);
#endif
