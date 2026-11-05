/* Preconfigured request fanout per head interface, with Wi-Fi unicast.
 * Socket/group setup stays outside the allocation-free send path. */
#include "fanout.h"
#include "wire.h"
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int fanout_close(fr_fanout *fanout) {
    if (!fanout || fanout->count>FR_INTERFACES) return -1;
    int rc=0;
    for (unsigned i=0;i<fanout->count;++i) {
        fr_fanout_group *g=&fanout->groups[i];
        if (g->fd>=0 && close(g->fd)) rc=-1;
        g->fd=-1;
        for (unsigned j=0;j<g->count;++j) { if (g->unicast_fd[j]>=0 && close(g->unicast_fd[j])) rc=-1; g->unicast_fd[j]=-1; }
    }
    fanout->count=0; return rc;
}
int fanout_open(const fr_connection *selected,const char *multicast,unsigned group_port,fr_fanout *out) {
    if (!selected || !out || !selected->count || selected->count>FR_CONNECTION_NODES || !group_port) return -1;
    fr_fanout plan={0}; struct sockaddr_in group;
    if (net_address(multicast,group_port,&group) || !IN_MULTICAST(ntohl(group.sin_addr.s_addr))) return -1;
    for (unsigned i=0;i<selected->count;++i) {
        const fr_link *link=&selected->links[i];
        if (!link->identity.id || !link->authenticated || !link->via.index || !link->via.name[0]) goto failed;
        for (unsigned j=0;j<i;++j) if (selected->links[j].identity.id==link->identity.id) goto failed;
        unsigned slot;
        for (slot=0;slot<plan.count && plan.groups[slot].via.index!=link->via.index;++slot) {}
        if (slot==plan.count) {
            if (plan.count==FR_INTERFACES) goto failed;
            fr_fanout_group *g=&plan.groups[plan.count++]; g->fd=-1; g->via=link->via;
            for (unsigned j=0;j<FR_CONNECTION_NODES;++j) g->unicast_fd[j]=-1;
            if (net_address(link->ip,link->identity.data_port,&g->destination) || !link->identity.data_port) goto failed;
        }
        fr_fanout_group *g=&plan.groups[slot];
        if (strcmp(g->via.address,link->via.address) || strcmp(g->via.name,link->via.name) || g->via.type!=link->via.type) goto failed;
        if (!link->identity.data_port || net_address(link->ip,link->identity.data_port,&g->unicast[g->count])) goto failed;
        g->members|=1u<<i; if (++g->count>1) g->destination=group;
    }
    for (unsigned i=0;i<plan.count;++i) {
        fr_fanout_group *g=&plan.groups[i];
        int wanted=interface_multicast_wanted(g->via.type,g->count);
        for (unsigned j=0;wanted && j<selected->count;++j)
            if ((g->members&(1u<<j)) && !selected->links[j].multicast) wanted=0; /* unicast to each */
        g->multicast=wanted;
        unsigned sockets=g->multicast?1:g->count;
        for (unsigned j=0;j<sockets;++j) {
            int fd=net_udp_bind(g->via.address,0,0);
            if (g->multicast) g->fd=fd; else g->unicast_fd[j]=fd;
            const struct sockaddr_in *to=g->multicast?&g->destination:&g->unicast[j];
            if (fd<0 || net_bind_interface(fd,g->via.index) ||
                (g->multicast && net_multicast(fd,multicast,g->via.address,0)) ||
                connect(fd,(const struct sockaddr *)to,sizeof(*to))) goto failed;
        }
    }
    *out=plan; return 0;
failed:
    (void)fanout_close(&plan); return -1;
}
int fanout_send(fr_fanout *fanout,const void *packet,size_t bytes) {
    if (!fanout || !packet || !bytes || bytes>FR_FRAME_MAX || !fanout->count || fanout->count>FR_INTERFACES) return -1;
    for (unsigned i=0;i<fanout->count;++i) {
        fr_fanout_group *g=&fanout->groups[i];
        for (unsigned j=0;j<(g->multicast?1u:g->count);++j) {
            ssize_t n; int fd=g->multicast?g->fd:g->unicast_fd[j];
            do { n=send(fd,packet,bytes,0); } while (n<0 && errno==EINTR);
            /* A dead node's ICMP port-unreachable surfaces here; the deadline handles it. */
            if (n!=(ssize_t)bytes && !(n<0 && (errno==ECONNREFUSED || errno==EHOSTDOWN || errno==EHOSTUNREACH))) return -1;
            ++g->packets; g->bytes+=bytes;
        }
    }
    return 0;
}
