/* test_net.c - actual loopback sockets, descriptor flags and error behavior. */
#include "test.h"
#include "../src/net.h"
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int flags(int fd) {
    int nodelay=0; socklen_t size=sizeof(nodelay);
    int status=fcntl(fd,F_GETFL), descriptor=fcntl(fd,F_GETFD);
    int rc=getsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&nodelay,&size);
    if (!(status&O_NONBLOCK) || !(descriptor&FD_CLOEXEC) || rc || !nodelay)
        fprintf(stderr,"socket flags: status=%d descriptor=%d nodelay=%d rc=%d\n",status,descriptor,nodelay,rc);
    return status>=0 && descriptor>=0 && (status&O_NONBLOCK) && (descriptor&FD_CLOEXEC) &&
           rc==0 && nodelay!=0;
}
int main(void) {
    test_begin("net");
    check(net_tcp_listen("bad",7700)==-1 && errno==EINVAL, "reject non-IPv4 bind");
    check(net_tcp_listen("127.0.0.1",65536)==-1, "reject oversized port");
    check(net_tcp_connect("127.0.0.1",0,100)==-1, "reject zero remote port");
    check(net_tcp_connect("127.0.0.1",7700,-1)==-1, "reject negative connect timeout");
    int listener=net_tcp_listen("127.0.0.1",0);
    check(listener>=0, "listen on ephemeral loopback port");
    if (listener<0) { perror("listen"); return test_end(); }
    check(flags(listener), "listener flags");
    int port=net_port(listener);
    check(port>0 && port<=65535, "discover allocated port");
    check(net_tcp_accept(listener)==-1 && (errno==EAGAIN || errno==EWOULDBLOCK), "empty accept never blocks");
    int duplicate=net_tcp_listen("127.0.0.1",(unsigned)port);
    check(duplicate==-1 && errno==EADDRINUSE, "occupied port rejected");
    if (duplicate>=0) close(duplicate);
    int client=net_tcp_connect("127.0.0.1",(unsigned)port,1000);
    check(client>=0, "connect completes");
    if (client<0) { close(listener); return test_end(); }
    check(net_wait(listener,0,1000)==1, "listener becomes readable");
    int peer=net_tcp_accept(listener);
    check(peer>=0, "accept connection");
    if (peer<0) { close(client); close(listener); return test_end(); }
    check(flags(client) && flags(peer), "connected and accepted flags");
    check(net_wait(peer,0,0)==0, "empty read times out");
    check(net_send(client,"hello",5)==5, "send request");
    check(net_wait(peer,0,1000)==1, "read readiness");
    char bytes[16]={0};
    check(recv(peer,bytes,sizeof(bytes),0)==5 && !memcmp(bytes,"hello",5), "receive request");
    check(shutdown(client,SHUT_WR)==0, "client half-close");
    check(net_wait(peer,0,1000)==1 && recv(peer,bytes,sizeof(bytes),0)==0, "half-close is readable EOF");
    check(net_send(peer,"reply",5)==5 && net_wait(client,0,1000)==1 &&
          recv(client,bytes,sizeof(bytes),0)==5, "reply still possible after half-close");
    check(shutdown(peer,SHUT_WR)==0 && net_send(peer,"x",1)==-1,
          "closed send returns an error without SIGPIPE termination");
    check(net_wait(-1,0,0)==-1 && errno==EINVAL, "invalid wait descriptor");
    check(net_port(-1)==-1, "invalid local-port descriptor");
    check(close(peer)==0 && close(client)==0 && close(listener)==0, "close all sockets");
    int refused=net_tcp_connect("127.0.0.1",(unsigned)port,1000);
    check(refused==-1 && errno==ECONNREFUSED, "connection refusal reported");
    if (refused>=0) close(refused);
    return test_end();
}
