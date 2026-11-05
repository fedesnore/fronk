/* net.c - checked, nonblocking IPv4 TCP/UDP setup for the control plane.
 *
 * Address resolution and alternate transports are intentionally absent.
 * Client waits use poll with a monotonic budget; the server uses ae instead.
 */
#include "net.h"
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int fail(int fd) {
    int error=errno;
    if (fd>=0) close(fd);
    errno=error;
    return -1;
}
static int setup(int fd) {
    int flags=fcntl(fd,F_GETFL), one=1;
    if (flags<0 || fcntl(fd,F_SETFL,flags|O_NONBLOCK)<0 ||
        fcntl(fd,F_SETFD,FD_CLOEXEC)<0 ||
        setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one))<0) return fail(fd);
#ifdef __APPLE__
    if (setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof(one))<0) return fail(fd);
#endif
    return fd;
}
int net_address(const char *ip, unsigned port, struct sockaddr_in *out) {
    *out=(struct sockaddr_in){.sin_family=AF_INET};
    if (!ip || port>65535 || inet_pton(AF_INET,ip,&out->sin_addr)!=1) {
        errno=EINVAL; return -1;
    }
    out->sin_port=htons((uint16_t)port);
    return 0;
}
int net_tcp_listen(const char *ip, unsigned port) {
    struct sockaddr_in addr;
    if (net_address(ip,port,&addr)<0) return -1;
    int fd=socket(AF_INET,SOCK_STREAM,0), one=1;
    if (fd<0) return -1;
    if (setup(fd)<0) return -1;
    if (setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one))<0 ||
        bind(fd,(struct sockaddr *)&addr,sizeof(addr))<0 || listen(fd,64)<0) return fail(fd);
    return fd;
}
int net_tcp_accept(int listener) {
    int fd;
    do { fd=accept(listener,NULL,NULL); } while (fd<0 && errno==EINTR);
    return fd<0 ? -1 : setup(fd);
}
int64_t net_now_ns(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC,&now)<0) return -1;
    return (int64_t)now.tv_sec*1000000000 + now.tv_nsec;
}
static int64_t now_ms(void) { int64_t n=net_now_ns(); return n<0?-1:n/1000000; }
int net_wait(int fd, int writing, int timeout_ms) {
    if (fd<0 || timeout_ms<0 || (writing!=0 && writing!=1)) { errno=EINVAL; return -1; }
    int64_t start=now_ms();
    if (start<0) return -1;
    int remaining=timeout_ms;
    for (;;) {
        struct pollfd item={.fd=fd,.events=writing?POLLOUT:POLLIN};
        int rc=poll(&item,1,remaining);
        if (rc>=0) {
            if (item.revents&POLLNVAL) { errno=EBADF; return -1; }
            return rc;
        }
        if (errno!=EINTR) return -1;
        int64_t now=now_ms();
        if (now<0) return -1;
        if (now-start>=timeout_ms) return 0;
        remaining=timeout_ms-(int)(now-start);
    }
}
int net_bind_interface(int fd,unsigned index) {
    if (!index) { errno=EINVAL; return -1; }
#ifdef __APPLE__
    return setsockopt(fd,IPPROTO_IP,IP_BOUND_IF,&index,sizeof(index));
#else
    uint32_t network_index=htonl(index);
    return setsockopt(fd,IPPROTO_IP,IP_UNICAST_IF,&network_index,sizeof(network_index));
#endif
}
int net_tcp_connect_from(const char *ip,unsigned port,int timeout_ms,const char *local,unsigned index) {
    struct sockaddr_in addr;
    if (!port || timeout_ms<0) { errno=EINVAL; return -1; }
    if (net_address(ip,port,&addr)<0) return -1;
    int fd=socket(AF_INET,SOCK_STREAM,0);
    if (fd<0) return -1;
    if (setup(fd)<0) return -1;
    if (local) {
        struct sockaddr_in source;
        if (net_address(local,0,&source) || bind(fd,(struct sockaddr *)&source,sizeof(source)) ||
            (index && net_bind_interface(fd,index))) return fail(fd);
    }
    if (connect(fd,(struct sockaddr *)&addr,sizeof(addr))==0) return fd;
    if (errno!=EINPROGRESS && errno!=EINTR) return fail(fd);
    int rc=net_wait(fd,1,timeout_ms);
    if (rc<=0) { if (!rc) errno=ETIMEDOUT; return fail(fd); }
    int error=0; socklen_t len=sizeof(error);
    if (getsockopt(fd,SOL_SOCKET,SO_ERROR,&error,&len)<0) return fail(fd);
    if (error) { errno=error; return fail(fd); }
    return fd;
}
int net_tcp_connect(const char *ip,unsigned port,int timeout_ms) {
    return net_tcp_connect_from(ip,port,timeout_ms,NULL,0);
}
int net_port(int fd) {
    struct sockaddr_in addr; socklen_t len=sizeof(addr);
    if (getsockname(fd,(struct sockaddr *)&addr,&len)<0) return -1;
    if (addr.sin_family!=AF_INET) { errno=EINVAL; return -1; }
    return ntohs(addr.sin_port);
}
ssize_t net_send(int fd, const void *data, size_t len) {
    ssize_t n;
    do {
#ifdef __APPLE__
        n=send(fd,data,len,0);
#else
        n=send(fd,data,len,MSG_NOSIGNAL);
#endif
    } while (n<0 && errno==EINTR);
    return n;
}
int net_udp_bind(const char *ip, unsigned port, int shared) {
    struct sockaddr_in addr;
    if (net_address(ip,port,&addr)<0) return -1;
    int fd=socket(AF_INET,SOCK_DGRAM,0), one=1;
    if (fd<0) return -1;
    int flags=fcntl(fd,F_GETFL);
    if (flags<0 || fcntl(fd,F_SETFL,flags|O_NONBLOCK)<0 || fcntl(fd,F_SETFD,FD_CLOEXEC)<0 ||
        setsockopt(fd,SOL_SOCKET,SO_BROADCAST,&one,sizeof(one))<0) return fail(fd);
    /* Whole-message datagrams (up to FR_FRAME_MAX) queue several deep; best effort. */
    int buffer=4<<20;
    (void)setsockopt(fd,SOL_SOCKET,SO_RCVBUF,&buffer,sizeof(buffer));
    (void)setsockopt(fd,SOL_SOCKET,SO_SNDBUF,&buffer,sizeof(buffer));
    if (shared) {
        if (setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one))<0) return fail(fd);
#ifdef __APPLE__
        if (setsockopt(fd,SOL_SOCKET,SO_REUSEPORT,&one,sizeof(one))<0) return fail(fd);
#endif
    }
    if (bind(fd,(struct sockaddr *)&addr,sizeof(addr))<0) return fail(fd);
    return fd;
}
int net_multicast(int fd, const char *group, const char *interface, int join) {
    struct ip_mreq membership; unsigned char one=1;
    if (inet_pton(AF_INET,group,&membership.imr_multiaddr)!=1 ||
        !IN_MULTICAST(ntohl(membership.imr_multiaddr.s_addr)) ||
        inet_pton(AF_INET,interface,&membership.imr_interface)!=1) { errno=EINVAL; return -1; }
    if (setsockopt(fd,IPPROTO_IP,IP_MULTICAST_IF,&membership.imr_interface,sizeof(struct in_addr))<0 ||
        setsockopt(fd,IPPROTO_IP,IP_MULTICAST_TTL,&one,sizeof(one))<0 ||
        setsockopt(fd,IPPROTO_IP,IP_MULTICAST_LOOP,&one,sizeof(one))<0) return -1;
    return join?setsockopt(fd,IPPROTO_IP,IP_ADD_MEMBERSHIP,&membership,sizeof(membership)):0;
}
/* An IGMP-snooping switch with no querier forgets a group a few minutes after
 * the last membership report. Blocking then unblocking an unused source
 * (192.0.2.1, TEST-NET-1) makes the kernel send two IGMPv3 state-change reports
 * for the group without leaving it, so no datagram is lost. */
int net_multicast_refresh(int fd,const char *group,const char *interface) {
    struct ip_mreq_source m; memset(&m,0,sizeof(m));
    if (inet_pton(AF_INET,group,&m.imr_multiaddr)!=1 || !IN_MULTICAST(ntohl(m.imr_multiaddr.s_addr)) ||
        inet_pton(AF_INET,interface,&m.imr_interface)!=1 || inet_pton(AF_INET,"192.0.2.1",&m.imr_sourceaddr)!=1) { errno=EINVAL; return -1; }
    if (setsockopt(fd,IPPROTO_IP,IP_BLOCK_SOURCE,&m,sizeof(m))<0) return -1;
    return setsockopt(fd,IPPROTO_IP,IP_UNBLOCK_SOURCE,&m,sizeof(m));
}
int net_multicast_leave(int fd,const char *group,const char *interface) {
    struct ip_mreq membership;
    if (inet_pton(AF_INET,group,&membership.imr_multiaddr)!=1 || inet_pton(AF_INET,interface,&membership.imr_interface)!=1) { errno=EINVAL; return -1; }
    return setsockopt(fd,IPPROTO_IP,IP_DROP_MEMBERSHIP,&membership,sizeof(membership));
}
int net_udp_info_enable(int fd) {
    int one=1; return setsockopt(fd,IPPROTO_IP,IP_PKTINFO,&one,sizeof(one));
}
ssize_t net_udp_recv_info(int fd,void *data,size_t cap,struct sockaddr_in *peer,fr_udp_info *local) {
    /* Darwin Thunderbolt reports MSG_TRUNC when a datagram exactly fills the
     * supplied iovec. A separate sentinel byte distinguishes exact fit from
     * true overflow without allocating or writing beyond caller storage. */
    unsigned char excess;
    struct iovec iov[2]={{.iov_base=data,.iov_len=cap},{.iov_base=&excess,.iov_len=1}};
    union { struct cmsghdr align; unsigned char bytes[CMSG_SPACE(sizeof(struct in_pktinfo))]; } control;
    struct msghdr msg={.msg_name=peer,.msg_namelen=sizeof(*peer),.msg_iov=iov,.msg_iovlen=2,
        .msg_control=control.bytes,.msg_controllen=sizeof(control.bytes)};
    ssize_t n;
    do { n=recvmsg(fd,&msg,0); } while (n<0 && errno==EINTR);
    if (n>=0 && ((size_t)n>cap || (msg.msg_flags&(MSG_TRUNC|MSG_CTRUNC)))) { errno=EMSGSIZE; return -1; }
    if (local) {
        memset(local,0,sizeof(*local));
        for (struct cmsghdr *c=CMSG_FIRSTHDR(&msg);n>=0 && c;c=CMSG_NXTHDR(&msg,c))
            if (c->cmsg_level==IPPROTO_IP && c->cmsg_type==IP_PKTINFO && c->cmsg_len>=CMSG_LEN(sizeof(struct in_pktinfo))) {
                struct in_pktinfo info; memcpy(&info,CMSG_DATA(c),sizeof(info));
                local->address=info.ipi_addr; local->index=info.ipi_ifindex;
            }
    }
    return n;
}
ssize_t net_udp_recv(int fd,void *data,size_t cap,struct sockaddr_in *peer) {
    return net_udp_recv_info(fd,data,cap,peer,NULL);
}
ssize_t net_udp_send_from(int fd,const void *data,size_t len,const struct sockaddr_in *peer,const fr_udp_info *local) {
    if (!local) return net_udp_send(fd,data,len,peer);
    struct iovec iov={.iov_base=(void *)data,.iov_len=len};
    union { struct cmsghdr align; unsigned char bytes[CMSG_SPACE(sizeof(struct in_pktinfo))]; } control={0};
    struct msghdr msg={.msg_name=(void *)peer,.msg_namelen=sizeof(*peer),.msg_iov=&iov,.msg_iovlen=1,
        .msg_control=control.bytes,.msg_controllen=sizeof(control.bytes)};
    struct cmsghdr *c=CMSG_FIRSTHDR(&msg); c->cmsg_level=IPPROTO_IP; c->cmsg_type=IP_PKTINFO; c->cmsg_len=CMSG_LEN(sizeof(struct in_pktinfo));
    struct in_pktinfo info={.ipi_ifindex=local->index,.ipi_spec_dst=local->address};
    memcpy(CMSG_DATA(c),&info,sizeof(info)); ssize_t n;
    do { n=sendmsg(fd,&msg,0); } while (n<0 && errno==EINTR);
    return n;
}
ssize_t net_udp_send(int fd, const void *data, size_t len, const struct sockaddr_in *peer) {
    ssize_t n;
    do { n=sendto(fd,data,len,0,(const struct sockaddr *)peer,sizeof(*peer)); } while (n<0 && errno==EINTR);
    return n;
}
