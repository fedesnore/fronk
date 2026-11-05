#include "test.h"
#include "../src/interfaces.h"
#include <string.h>
int main(void) {
    test_begin("interfaces"); fr_interface all[FR_INTERFACES],one[FR_INTERFACES],saved[FR_INTERFACES]; unsigned n=0,count=0;
    check(!strcmp(interface_type(FR_LINK_OTHER),"other") && !strcmp(interface_type(FR_LINK_ETHERNET),"ethernet") &&
          !strcmp(interface_type(FR_LINK_THUNDERBOLT),"thunderbolt") && !strcmp(interface_type(FR_LINK_WIFI),"wifi"),
          "stable wire names include the Wi-Fi unicast policy");
    check(!interfaces_read("auto",NULL,all,&n) && n>0 && n<=FR_INTERFACES,"read up local IPv4 interfaces without changing them");
    int physical=1; for (unsigned i=0;i<n;++i) {
        if (all[i].loopback) physical=0;
        printf("auto interface=%s address=%s type=%s\n",all[i].name,all[i].address,interface_type(all[i].type));
    }
    check(physical,"automatic selection excludes loopback");
    check(!interfaces_read("","127.0.0.1",one,&count) && count==1 && one[0].loopback && one[0].index &&
          one[0].name[0] && !strcmp(one[0].address,"127.0.0.1") && !one[0].speed_bps,"loopback has its real name and unknown physical speed");
    memcpy(saved,one,sizeof(saved)); unsigned before=count;
    check(interfaces_read("127.0.0.1,127.0.0.1",NULL,one,&count)<0 && count==before && !memcmp(saved,one,sizeof(saved)),"duplicate interface list rejected atomically");
    check(interfaces_read("127.0.0.1,",NULL,one,&count)<0 && interfaces_read(",127.0.0.1",NULL,one,&count)<0 &&
          interfaces_read("127.0.0.1,,192.0.2.1",NULL,one,&count)<0,"empty list elements rejected");
    check(interfaces_read("192.0.2.254",NULL,one,&count)<0 && interfaces_read("localhost",NULL,one,&count)<0,
          "nonlocal addresses and names cannot masquerade as measured local interfaces");
    unsigned other=n;
    for (unsigned i=0;i<n;++i) if (!all[i].loopback) { other=i; break; }
    if (other<n) {
        char spec[40]; snprintf(spec,sizeof(spec),"127.0.0.1,%s",all[other].address);
        check(!interfaces_read(spec,NULL,one,&count) && count==2 && one[0].index!=one[1].index,
              "existing loopback and second local interface selected together");
        for (unsigned i=0;i<count;++i) printf("interface=%s address=%s type=%s speed_bps=%llu\n",one[i].name,one[i].address,
                                            interface_type(one[i].type),(unsigned long long)one[i].speed_bps);
    }
    fr_interface fake[]={
        {.name="lo",.address="127.0.0.1",.index=1,.loopback=1,.running=1},
        {.name="eth",.address="192.0.2.1",.netmask="255.255.255.0",.index=2,.running=1,.type=FR_LINK_ETHERNET},
        {.name="bridge",.address="198.51.100.1",.netmask="255.255.255.0",.index=3,.running=1,.type=FR_LINK_THUNDERBOLT},
        {.name="wifi",.address="192.168.8.1",.index=4,.running=1,.type=FR_LINK_WIFI},
        {.name="down",.address="10.30.30.1",.index=5},
        {.name="eth",.address="192.0.2.2",.index=2,.running=1,.type=FR_LINK_ETHERNET}
    };
    check(!interfaces_select("auto",NULL,fake,6,one,&count) && count==3 && one[0].index==2 && one[1].index==3 && one[2].index==4,
          "fake inventory: skip loopback/down and deduplicate address aliases");
    check(!interfaces_select("127.0.0.1",NULL,fake,6,one,&count) && count==1 && one[0].loopback,"fake inventory: explicit localhost remains available");
    memcpy(saved,one,sizeof(saved)); before=count;
    check(interfaces_select("auto",NULL,fake,1,one,&count)<0 && count==before && !memcmp(saved,one,sizeof(saved)),"auto never falls back silently to loopback");
    check(interfaces_select("192.0.2.1,192.0.2.2",NULL,fake,6,one,&count)<0 && !memcmp(saved,one,sizeof(saved)),"two aliases of one interface rejected atomically");
    check(interface_contains(&fake[1],"192.0.2.4") && !interface_contains(&fake[1],"198.51.100.2"),"configured peers use their OS-reported subnet");
    check(!interface_multicast_wanted(FR_LINK_THUNDERBOLT,1) && !interface_multicast_wanted(FR_LINK_THUNDERBOLT,2) &&
          !interface_multicast_wanted(FR_LINK_WIFI,2) && interface_multicast_wanted(FR_LINK_ETHERNET,2),
          "mixed cluster: Thunderbolt and Wi-Fi unicast per node, Ethernet pair multicast");
    check(!interfaces_select("192.0.2.2",NULL,fake,6,one,&count) && count==1 && one[0].index==2 && !strcmp(one[0].address,"192.0.2.2"),
          "session TCP endpoint resolves a secondary alias even when auto chose the primary address");
    return test_end();
}
