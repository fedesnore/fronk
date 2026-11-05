#include "test.h"
#include "../src/fanout.h"
#include "../src/wire.h"
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
int main(int argc,char **argv) {
    test_begin("fanout"); fr_interface interfaces[FR_INTERFACES]; unsigned n=0;
    char spec[40]="127.0.0.1";
    if (argc==2) snprintf(spec,sizeof(spec),"127.0.0.1,%s",argv[1]);
    int ready=!interfaces_read(spec,NULL,interfaces,&n);
    check(ready && n==(argc==2?2u:1u),"select existing local sender interfaces");
    if (!ready) return test_end();
    int receivers[3]={-1,-1,-1}; const char *group="239.77.6.28";
    receivers[0]=net_udp_bind(group,0,1);
    if (receivers[0]<0) { check(0,"create local multicast receiver"); return test_end(); }
    unsigned port=(unsigned)net_port(receivers[0]);
    receivers[1]=net_udp_bind(group,port,1); receivers[2]=net_udp_bind(interfaces[n-1].address,0,0);
    int joined=receivers[1]>=0 && receivers[2]>=0 && !net_multicast(receivers[0],group,interfaces[0].address,1) &&
        !net_multicast(receivers[1],group,interfaces[0].address,1);
    check(joined,"two multicast listeners plus independent unicast listener");
    fr_connection *selected=calloc(1,sizeof(*selected));
    check(selected!=NULL,"allocate setup-only selected-path fixture");
    if (!selected || !joined) goto done;
    selected->count=n==2?3:2;
    for (unsigned i=0;i<3;++i) {
        fr_link *l=&selected->links[i]; l->identity.id=i+1; l->authenticated=l->multicast=1;
        l->identity.data_port=i==2?(unsigned)net_port(receivers[2]):port;
        l->via=interfaces[i==2?n-1:0]; strcpy(l->ip,l->via.address);
    }
    fr_fanout plan; ready=!fanout_open(selected,group,port,&plan);
    check(ready && plan.count==n && plan.groups[0].count==2 && plan.groups[0].members==3 &&
          (n==1 || (plan.groups[1].count==1 && plan.groups[1].members==4)),"one group per interface, multicast for two nodes and unicast for singleton");
    if (ready) {
        int bound=1;
        for (unsigned i=0;i<plan.count;++i) {
            struct in_addr actual,wanted; socklen_t len=sizeof(actual); inet_pton(AF_INET,plan.groups[i].via.address,&wanted);
            if (interface_multicast_wanted(plan.groups[i].via.type,plan.groups[i].count) &&
                (getsockopt(plan.groups[i].fd,IPPROTO_IP,IP_MULTICAST_IF,&actual,&len) || actual.s_addr!=wanted.s_addr)) bound=0;
        }
        check(bound,"each multicast socket selects its own IP_MULTICAST_IF");
        fr_req request={.layer=47,.burst=8}; uint16_t values[FR_MAX_BURST][FR_HIDDEN]={{0}};
        for (unsigned t=0;t<8;++t) for (unsigned e=0;e<10;++e) request.expert[t][e]=(uint16_t)e;
        unsigned char packet[FR_REQ_FRAGMENT_MAX],received[FR_UDP_PAYLOAD_MAX]; int delivered=1; uint32_t seen[3]={0};
        for (unsigned frag=0;frag<32;++frag) {
            size_t size=fr_req_encode(packet,sizeof(packet),&request,values,7,1,frag);
            if (!size || fanout_send(&plan,packet,size)) { delivered=0; break; }
            for (unsigned node=0;node<selected->count;++node) {
                struct sockaddr_in peer; struct in_addr expected; inet_pton(AF_INET,selected->links[node].via.address,&expected);
                if (net_wait(receivers[node],0,1000)!=1 || net_udp_recv(receivers[node],received,sizeof(received),&peer)!=(ssize_t)size ||
                    peer.sin_addr.s_addr!=expected.s_addr || memcmp(packet,received,size) || (seen[node]&(1u<<frag))) delivered=0;
                seen[node]|=1u<<frag;
            }
        }
        for (unsigned i=0;i<selected->count;++i) if (seen[i]!=UINT32_MAX || net_wait(receivers[i],0,10)!=0) delivered=0;
        check(delivered,"each node receives every REQ fragment exactly once across all active interfaces");
        int once=1;
        for (unsigned i=0;i<plan.count;++i) if (plan.groups[i].packets!=32 || plan.groups[i].bytes!=8*5312) once=0;
        check(once,"one burst-8 REQ per interface: 32 datagrams, never per-node duplicate multicast sends");
        check(!fanout_close(&plan),"close group sockets");
    }
    selected->links[0]=selected->links[2]; selected->count=1;
    selected->links[0].multicast=0; selected->links[0].via.type=FR_LINK_THUNDERBOLT;
    ready=!fanout_open(selected,group,port,&plan);
    check(ready && plan.count==1 && plan.groups[0].count==1 &&
          ntohs(plan.groups[0].destination.sin_port)==selected->links[0].identity.data_port,"singleton bridge uses unicast even without multicast support");
    if (ready) {
        struct sockaddr_in peer; char bytes[8];
        check(!fanout_send(&plan,"REQ",3) && net_wait(receivers[2],0,1000)==1 && net_udp_recv(receivers[2],bytes,sizeof(bytes),&peer)==3,
              "singleton receives one unicast request");
        check(!fanout_close(&plan),"close unicast group");
    }
    selected->count=2; selected->links[1]=selected->links[0]; plan.count=99;
    check(fanout_open(selected,group,port,&plan)<0 && plan.count==99,"duplicate persistent node IDs cannot produce duplicate sends");
    selected->links[1].identity.id=99; selected->links[0].via.type=selected->links[1].via.type=FR_LINK_ETHERNET;
    /* Model Wi-Fi, the Thunderbolt bridge and Ethernet whose nodes did not answer
     * multicast (a snooping switch) on real local sockets: no multicast
     * membership is created, and nodes have different unicast receive ports. */
    int wifi[2]={net_udp_bind("127.0.0.1",0,0),net_udp_bind("127.0.0.1",0,0)};
    static const char *const names[3]={"Wi-Fi","Thunderbolt","Ethernet without multicast replies"};
    for (unsigned kind=0;kind<3;++kind) {
    for (unsigned i=0;i<2;++i) {
        fr_link *l=&selected->links[i]; l->identity.id=i+1; l->authenticated=1; l->multicast=0;
        l->identity.data_port=(unsigned)net_port(wifi[i]); l->via=interfaces[0]; l->via.type=kind==2?FR_LINK_ETHERNET:kind?FR_LINK_THUNDERBOLT:FR_LINK_WIFI;
        strcpy(l->ip,"127.0.0.1");
    }
    ready=wifi[0]>=0 && wifi[1]>=0 && !fanout_open(selected,group,port,&plan);
    char what[96]; snprintf(what,sizeof(what),"two %s nodes are eligible, with unicast",names[kind]);
    check(ready && plan.count==1 && plan.groups[0].count==2 && !plan.groups[0].multicast,what);
    if (ready) {
        int delivered=!fanout_send(&plan,"REQ",3);
        for (unsigned i=0;i<2;++i) {
            char data[8]; struct sockaddr_in peer;
            if (net_wait(wifi[i],0,1000)!=1 || net_udp_recv(wifi[i],data,sizeof(data),&peer)!=3 ||
                memcmp(data,"REQ",3) || net_wait(wifi[i],0,10)!=0) delivered=0;
        }
        snprintf(what,sizeof(what),"%s sends one unicast per node, never one multicast per interface",names[kind]);
        check(delivered && plan.groups[0].packets==2 && plan.groups[0].bytes==6,what);
        check(!fanout_close(&plan),"close unicast sockets");
    }
    }
    for (unsigned i=0;i<2;++i) if (wifi[i]>=0) close(wifi[i]);
done:
    for (unsigned i=0;i<3;++i) if (receivers[i]>=0) close(receivers[i]);
    free(selected); return test_end();
}
