/* Group request destinations once, when a session starts.
 * One group per active head interface. Each call sends an already-built
 * frame by unicast for one member or multicast for multiple wired members.
 * Wi-Fi always unicasts separately to every member.
 * No allocation or path discovery on the send path.
 */
#ifndef FR_FANOUT_H
#define FR_FANOUT_H
#include "connection.h"
#include "net.h"
typedef struct {
    fr_interface via;
    uint32_t members;
    unsigned count;
    int fd,unicast_fd[FR_CONNECTION_NODES]; /* connected; unicast_fd used unless multicast */
    int multicast;
    struct sockaddr_in destination;
    struct sockaddr_in unicast[FR_CONNECTION_NODES];
    uint64_t packets,bytes;
} fr_fanout_group;
typedef struct { unsigned count; fr_fanout_group groups[FR_INTERFACES]; } fr_fanout;
/* Open into fresh storage. Failure leaves output unchanged. Group port is
 * the common REQ receive port; singleton destinations use their node's port. */
int fanout_open(const fr_connection *selected,const char *multicast,unsigned group_port,fr_fanout *out);
int fanout_send(fr_fanout *fanout,const void *packet,size_t bytes);
int fanout_close(fr_fanout *fanout);
#endif
