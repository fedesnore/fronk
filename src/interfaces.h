/* interfaces.h - head-side IPv4 link inventory, never driver/bus discovery.
 * Setup-only snapshots. Unknown link speed is zero, not an estimate. Explicit
 * lists contain existing local IPv4 addresses; auto enumerates up interfaces.
 */
#ifndef FR_INTERFACES_H
#define FR_INTERFACES_H
#include <stdint.h>
#define FR_INTERFACES 8
#define FR_IFNAME 64
typedef enum { FR_LINK_OTHER, FR_LINK_ETHERNET, FR_LINK_THUNDERBOLT, FR_LINK_WIFI } fr_link_type;
typedef struct {
    char name[FR_IFNAME],address[16],broadcast[16],netmask[16];
    unsigned index,loopback,running;
    fr_link_type type;
    uint64_t speed_bps;
} fr_interface;
const char *interface_type(fr_link_type type);
int interface_multicast_wanted(fr_link_type type,unsigned nodes);
int interface_contains(const fr_interface *interface,const char *address);
/* Pure selection over an injected inventory; explicit loopback remains valid. */
int interfaces_select(const char *spec,const char *fallback,const fr_interface *inventory,unsigned size,
                      fr_interface out[FR_INTERFACES],unsigned *count);
/* Empty spec selects fallback; otherwise auto or comma-separated IPv4s.
 * One address per physical interface, at most FR_INTERFACES. Output is atomic. */
int interfaces_read(const char *spec,const char *fallback,fr_interface out[FR_INTERFACES],unsigned *count);
#endif
