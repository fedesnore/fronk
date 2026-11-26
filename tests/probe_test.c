/* net_tcp_probe against a peer that stays, and against one that vanishes without
 * closing (its address is removed: nothing answers, nothing is refused).
 * Run inside a private network sandbox: probe_test alive|dead */
#include "net.h"
#include <arpa/inet.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
static double seconds(void) { return net_now_ns()/1e9; }
int main(int argc,char **argv) {
    int dead=argc>1 && !strcmp(argv[1],"dead");
    int listener=net_tcp_listen("10.9.9.1",7999); if (listener<0) { perror("listen"); return 2; }
    pid_t peer=fork();
    if (!peer) {   /* the holder: connects from 10.9.9.2 and never says anything */
        int fd=socket(AF_INET,SOCK_STREAM,0); struct sockaddr_in a={0},b={0};
        a.sin_family=b.sin_family=AF_INET; inet_pton(AF_INET,"10.9.9.2",&a.sin_addr); inet_pton(AF_INET,"10.9.9.1",&b.sin_addr); b.sin_port=htons(7999);
        if (bind(fd,(struct sockaddr *)&a,sizeof(a)) || connect(fd,(struct sockaddr *)&b,sizeof(b))) { perror("holder"); _exit(2); }
        pause(); _exit(0);
    }
    int fd=-1; for (int i=0;i<200 && fd<0;++i) { fd=net_tcp_accept(listener); if (fd<0) usleep(10000); }
    if (fd<0) { puts("no connection"); return 2; }
    sleep(2);   /* an idle control connection, as between a head and its node */
    if (dead && system("ip addr del 10.9.9.2/32 dev lo && ip route add blackhole 10.9.9.2/32")) { puts("cannot remove the holder"); return 2; }
    double t0=seconds(); net_tcp_probe(fd,1);
    int result=2;
    for (;;) {
        struct pollfd p={fd,POLLIN,0}; int n=poll(&p,1,200); double t=seconds()-t0;
        if (n>0) { char c; ssize_t k=recv(fd,&c,1,0); int e=errno;
            printf("%s holder: read fails after %.1f s (%s)\n",dead?"vanished":"live",t,k<0?strerror(e):"closed");
            result=dead?0:1; break; }
        if (t>12) { printf("%s holder: still connected after %.0f s of probes\n",dead?"vanished":"live",t); result=dead?1:0; break; }
    }
    if (!dead) { net_tcp_probe(fd,0); int on=-1; socklen_t len=sizeof(on); getsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&on,&len); printf("probes off again: %s\n",on?"no":"yes"); }
    kill(peer,9); return result;
}
