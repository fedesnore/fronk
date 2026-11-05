/* test_udp.c - loopback multicast fanout, UDP flags and truncation checks.
 * Darwin lo0 does not deliver directed broadcasts; LAN broadcast delivery is
 * verified at the hardware trial. Local discovery uses multicast explicitly.
 */
#include "test.h"
#include "../src/net.h"
#include <errno.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
int main(void) {
    test_begin("udp"); int fds[3]={-1,-1,-1}, sender=-1;
    struct sockaddr_in destination, peer; char data[32];
    check(net_address("bad",1,&destination)<0,"invalid destination");
    fds[0]=net_udp_bind("0.0.0.0",0,1);
    check(fds[0]>=0,"shared receive socket"); if (fds[0]<0) return test_end();
    unsigned port=(unsigned)net_port(fds[0]);
    fds[1]=net_udp_bind("0.0.0.0",port,1); fds[2]=net_udp_bind("0.0.0.0",port,1);
    sender=net_udp_bind("127.0.0.1",0,0);
    check(fds[1]>=0 && fds[2]>=0 && sender>=0,"three shared listeners and private sender");
    if (fds[1]<0 || fds[2]<0 || sender<0) goto done;
    check((fcntl(sender,F_GETFL)&O_NONBLOCK) && (fcntl(sender,F_GETFD)&FD_CLOEXEC),"UDP descriptor flags");
    check(net_udp_recv(sender,data,sizeof(data),&peer)<0 && (errno==EAGAIN || errno==EWOULDBLOCK),"empty receive nonblocking");
    int broadcast=0; socklen_t option_size=sizeof(broadcast);
    check(!getsockopt(sender,SOL_SOCKET,SO_BROADCAST,&broadcast,&option_size) && broadcast,"broadcast enabled for LAN discovery");
    int joined=net_multicast(sender,"239.7.7.7","127.0.0.1",0)==0;
    for (int i=0; i<3; ++i) if (net_multicast(fds[i],"239.7.7.7","127.0.0.1",1)<0) joined=0;
    check(joined,"join multicast on loopback and select sender interface");
    check(!net_udp_info_enable(fds[0]),"enable destination/interface packet metadata");
    check(!net_address("239.7.7.7",port,&destination) && net_udp_send(sender,"GROUP",5,&destination)==5,"send one multicast datagram");
    int multicast=1; fr_udp_info local={0};
    for (int i=0; i<3; ++i) if (net_wait(fds[i],0,1000)!=1 || net_udp_recv_info(fds[i],data,sizeof(data),&peer,i==0?&local:NULL)!=5 || memcmp(data,"GROUP",5)) multicast=0;
    check(multicast,"one multicast reaches all three listeners");
    check(local.index>0 && local.address.s_addr==destination.sin_addr.s_addr,"multicast preserves receiving interface and destination");
    multicast=!net_multicast_refresh(fds[1],"239.7.7.7","127.0.0.1") && net_udp_send(sender,"AGAIN",5,&destination)==5;
    for (int i=0; multicast && i<3; ++i) if (net_wait(fds[i],0,1000)!=1 || net_udp_recv(fds[i],data,sizeof(data),&peer)!=5 || memcmp(data,"AGAIN",5)) multicast=0;
    check(multicast,"an IGMP refresh keeps the group joined: the next datagram still reaches all listeners");
    check(net_multicast_refresh(fds[1],"127.0.0.1","127.0.0.1")<0 && errno==EINVAL,"refresh rejects a nonmulticast group");
    inet_pton(AF_INET,"127.0.0.1",&local.address);
    check(!net_address("127.0.0.1",(unsigned)net_port(sender),&destination) &&
          net_udp_send_from(fds[0],"PATH",4,&destination,&local)==4 && net_wait(sender,0,1000)==1 &&
          net_udp_recv(sender,data,sizeof(data),&peer)==4 && !memcmp(data,"PATH",4) && peer.sin_addr.s_addr==local.address.s_addr,
          "reply uses the chosen local source interface/address");
    check(!net_address("127.0.0.1",(unsigned)net_port(sender),&destination) &&
          net_udp_send(fds[0],"too long",8,&destination)==8 && net_wait(sender,0,1000)==1 &&
          net_udp_recv(sender,data,2,&peer)<0 && errno==EMSGSIZE,"truncated datagrams rejected");
    check(net_multicast(sender,"127.0.0.1","127.0.0.1",1)<0,"nonmulticast group rejected");
    data[5]='X';
    check(net_udp_send(fds[0],"exact",5,&destination)==5 && net_wait(sender,0,1000)==1 &&
          net_udp_recv(sender,data,5,&peer)==5 && !memcmp(data,"exact",5) && data[5]=='X',"exact-size datagram accepted without overwriting sentinel");
    check(net_udp_send(fds[0],"excess",6,&destination)==6 && net_wait(sender,0,1000)==1 &&
          net_udp_recv(sender,data,5,&peer)<0 && errno==EMSGSIZE && data[5]=='X',"one-byte overflow rejected without overwriting caller buffer");
    check(net_now_ns()>0,"monotonic clock");
done:
    for (int i=0; i<3; ++i) if (fds[i]>=0) close(fds[i]);
    if (sender>=0) close(sender);
    return test_end();
}
