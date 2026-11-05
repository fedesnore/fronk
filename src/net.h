/* net.h - IPv4 sockets for control connections and CONNECTION probes.
 *
 * Returned sockets are nonblocking and close-on-exec. TCP additionally uses
 * TCP_NODELAY and SIGPIPE protection; UDP has bounded datagram reads.
 */
#ifndef FR_NET_H
#define FR_NET_H
#include <stddef.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <stdint.h>
int64_t net_now_ns(void);
int net_address(const char *ip, unsigned port, struct sockaddr_in *out);
int net_udp_bind(const char *ip, unsigned port, int shared);
int net_multicast(int fd, const char *group, const char *interface, int join);
int net_multicast_leave(int fd,const char *group,const char *interface);
int net_multicast_refresh(int fd,const char *group,const char *interface);
ssize_t net_udp_recv(int fd, void *data, size_t cap, struct sockaddr_in *peer);
ssize_t net_udp_send(int fd, const void *data, size_t len, const struct sockaddr_in *peer);
int net_tcp_listen(const char *ip, unsigned port); /* Zero asks for an ephemeral port. */
int net_tcp_accept(int listener);
int net_tcp_connect(const char *ip, unsigned port, int timeout_ms);
int net_tcp_connect_from(const char *ip,unsigned port,int timeout_ms,const char *local,unsigned interface_index);
int net_bind_interface(int fd,unsigned interface_index);
int net_port(int fd);
ssize_t net_send(int fd, const void *data, size_t len);
typedef struct { struct in_addr address; unsigned index; } fr_udp_info;
int net_udp_info_enable(int fd);
ssize_t net_udp_recv_info(int fd,void *data,size_t cap,struct sockaddr_in *peer,fr_udp_info *local);
ssize_t net_udp_send_from(int fd,const void *data,size_t len,const struct sockaddr_in *peer,const fr_udp_info *local);
/* Wait for a client read/write: 1 ready, 0 timed out, -1 error. Server uses ae. */
int net_wait(int fd, int writing, int timeout_ms);
#endif
