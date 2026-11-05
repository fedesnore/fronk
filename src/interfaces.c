/* Interface metadata comes from the head OS. Unknown speed stays unknown;
 * a USB adapter or PCI bus is deliberately not inspected. */
#include "interfaces.h"
#include <arpa/inet.h>
#include <errno.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#ifdef __APPLE__
#include <net/if_media.h>
#include <sys/ioctl.h>
#include <SystemConfiguration/SystemConfiguration.h>
#else
#include <sys/stat.h>
#endif
const char *interface_type(fr_link_type type) {
    return type==FR_LINK_THUNDERBOLT?"thunderbolt":type==FR_LINK_ETHERNET?"ethernet":type==FR_LINK_WIFI?"wifi":"other";
}
/* Multicast only on shared wired Ethernet, and only when every node there
 * answers it; otherwise that interface falls back to one unicast per node.
 * Wi-Fi and the Thunderbolt bridge always get unicast: the bridge floods every
 * port anyway, and on it macOS delivered only 4 of 14 multicast probes to the
 * M1 Pro while unicast was clean. Some IGMP-snooping switches without a querier
 * also stop forwarding after the membership expires; test reception each time. */
int interface_multicast_wanted(fr_link_type type,unsigned nodes) { return nodes>=2 && type!=FR_LINK_WIFI && type!=FR_LINK_THUNDERBOLT; }
int interface_contains(const fr_interface *interface,const char *address) {
    struct in_addr local,peer,mask;
    return interface && address && inet_pton(AF_INET,interface->address,&local)==1 &&
        inet_pton(AF_INET,address,&peer)==1 && inet_pton(AF_INET,interface->netmask,&mask)==1 && mask.s_addr &&
        (local.s_addr&mask.s_addr)==(peer.s_addr&mask.s_addr);
}
#ifdef __APPLE__
static void metadata(fr_interface *out) {
    /* The OS's configured interface identity distinguishes Wi-Fi and the
     * Thunderbolt Bridge without looking at a driver or its parent bus. */
    CFArrayRef interfaces=SCNetworkInterfaceCopyAll();
    if (interfaces) {
        for (CFIndex i=0;i<CFArrayGetCount(interfaces);++i) {
            SCNetworkInterfaceRef interface=CFArrayGetValueAtIndex(interfaces,i);
            CFStringRef name=SCNetworkInterfaceGetBSDName(interface),type=SCNetworkInterfaceGetInterfaceType(interface);
            char bsd[FR_IFNAME],label[128];
            if (!name || !CFStringGetCString(name,bsd,sizeof(bsd),kCFStringEncodingUTF8) || strcmp(bsd,out->name)) continue;
            if (type && CFEqual(type,kSCNetworkInterfaceTypeEthernet)) out->type=FR_LINK_ETHERNET;
            if (type && CFEqual(type,kSCNetworkInterfaceTypeIEEE80211)) out->type=FR_LINK_WIFI;
            CFStringRef display=SCNetworkInterfaceGetLocalizedDisplayName(interface);
            if (type && CFEqual(type,CFSTR("Bridge")) && display &&
                CFStringGetCString(display,label,sizeof(label),kCFStringEncodingUTF8) && strstr(label,"Thunderbolt")) out->type=FR_LINK_THUNDERBOLT;
            break;
        }
        CFRelease(interfaces);
    }
    int fd=socket(AF_INET,SOCK_DGRAM,0); if (fd<0) return;
    struct ifmediareq media={0};
    if (strlen(out->name)>=sizeof(media.ifm_name)) { close(fd); return; }
    strcpy(media.ifm_name,out->name);
    if (!ioctl(fd,SIOCGIFMEDIA,&media) && (media.ifm_status&IFM_AVALID) && (media.ifm_status&IFM_ACTIVE) &&
        IFM_TYPE(media.ifm_active)==IFM_ETHER) {
        static const struct ifmedia_description descriptions[]=IFM_SUBTYPE_ETHERNET_DESCRIPTIONS;
        for (unsigned i=0;descriptions[i].ifmt_string;++i) if (descriptions[i].ifmt_word==IFM_SUBTYPE(media.ifm_active)) {
            /* Parse the SDK's canonical rate names (1000baseT, 10Gbase-T,
             * 2.5GBase-T, ...), not user-provided or localized strings. */
            char *end; double rate=strtod(descriptions[i].ifmt_string,&end);
            if (end!=descriptions[i].ifmt_string && rate>0 && rate<=1000000)
                out->speed_bps=(uint64_t)(rate*((*end=='G'||*end=='g')?1e9:1e6));
            break;
        }
    }
    close(fd);
}
#else
static int number(const char *name,const char *field,uint64_t *out) {
    char path[160],text[64],*end; snprintf(path,sizeof(path),"/sys/class/net/%s/%s",name,field);
    FILE *file=fopen(path,"r"); if (!file) return -1;
    int ok=fgets(text,sizeof(text),file)!=NULL; fclose(file);
    if (!ok || text[0]<'0' || text[0]>'9') return -1;
    errno=0; unsigned long long value=strtoull(text,&end,10);
    if (errno || (*end!='\n' && *end)) return -1;
    *out=(uint64_t)value; return 0;
}
static void metadata(fr_interface *out) {
    uint64_t type=0,speed=0; char wireless[160]; struct stat st;
    snprintf(wireless,sizeof(wireless),"/sys/class/net/%s/wireless",out->name);
    if (!stat(wireless,&st)) out->type=FR_LINK_WIFI;
    else if (errno==ENOENT && !number(out->name,"type",&type) && type==1) out->type=FR_LINK_ETHERNET;
    if (!number(out->name,"speed",&speed) && speed>0 && speed<1000000) out->speed_bps=speed*UINT64_C(1000000);
}
#endif
static int read_one(const struct ifaddrs *ifa,fr_interface *out) {
    if (!ifa->ifa_addr || ifa->ifa_addr->sa_family!=AF_INET || !(ifa->ifa_flags&IFF_UP) ||
        strlen(ifa->ifa_name)>=sizeof(out->name)) return -1;
    memset(out,0,sizeof(*out)); strcpy(out->name,ifa->ifa_name); out->index=if_nametoindex(out->name);
    out->loopback=(ifa->ifa_flags&IFF_LOOPBACK)!=0;
    out->running=(ifa->ifa_flags&IFF_RUNNING)!=0;
    const struct sockaddr_in *addr=(const struct sockaddr_in *)ifa->ifa_addr;
    if (!out->index || !inet_ntop(AF_INET,&addr->sin_addr,out->address,sizeof(out->address))) return -1;
    if (ifa->ifa_netmask) {
        addr=(const struct sockaddr_in *)ifa->ifa_netmask;
        if (!inet_ntop(AF_INET,&addr->sin_addr,out->netmask,sizeof(out->netmask))) return -1;
    }
    strcpy(out->broadcast,"255.255.255.255");
    if ((ifa->ifa_flags&IFF_BROADCAST) && ifa->ifa_broadaddr) {
        addr=(const struct sockaddr_in *)ifa->ifa_broadaddr;
        if (!inet_ntop(AF_INET,&addr->sin_addr,out->broadcast,sizeof(out->broadcast))) return -1;
    }
    if (!out->loopback) metadata(out);
    return 0;
}
int interfaces_select(const char *spec,const char *fallback,const fr_interface *inventory,unsigned size,
                      fr_interface out[FR_INTERFACES],unsigned *count) {
    if (!spec || !out || !count || (!inventory && size)) return -1;
    if (!*spec) spec=fallback;
    if (!spec || !*spec || strlen(spec)>=FR_INTERFACES*16) return -1;
    fr_interface selected[FR_INTERFACES]={0}; unsigned n=0; int rc=0;
    if (!strcmp(spec,"auto")) {
        for (unsigned i=0;i<size;++i) {
            const fr_interface *item=&inventory[i];
            if (item->loopback || !item->running || !item->index) continue;
            unsigned index=item->index,j;
            for (j=0;j<n && selected[j].index!=index;++j) {}
            if (j<n) continue;
            if (n==FR_INTERFACES) { rc=-1; break; } selected[n++]=*item;
        }
    } else {
        const char *p=spec;
        while (!rc && *p) {
            const char *end=strchr(p,','); size_t len=end?(size_t)(end-p):strlen(p); char address[16];
            if (!len || len>=sizeof(address) || n==FR_INTERFACES || (end && !end[1])) { rc=-1; break; }
            memcpy(address,p,len); address[len]=0; struct in_addr ip;
            if (inet_pton(AF_INET,address,&ip)!=1) { rc=-1; break; }
            int found=0;
            for (unsigned i=0;i<size;++i) {
                if (strcmp(inventory[i].address,address) || !inventory[i].index) continue;
                selected[n]=inventory[i]; found=1;
                break;
            }
            if (!found) { rc=-1; break; }
            for (unsigned j=0;j<n;++j) if (selected[j].index==selected[n].index) rc=-1;
            ++n; p=end?end+1:p+len;
        }
    }
    if (rc || !n) return -1;
    memcpy(out,selected,sizeof(selected)); *count=n; return 0;
}
int interfaces_read(const char *spec,const char *fallback,fr_interface out[FR_INTERFACES],unsigned *count) {
    struct ifaddrs *all=NULL; if (getifaddrs(&all)) return -1;
    unsigned capacity=0,n=0;
    for (const struct ifaddrs *ifa=all;ifa;ifa=ifa->ifa_next) ++capacity;
    fr_interface *inventory=calloc(capacity?capacity:1,sizeof(*inventory));
    if (!inventory) { freeifaddrs(all); return -1; }
    for (const struct ifaddrs *ifa=all;ifa;ifa=ifa->ifa_next)
        if (!read_one(ifa,&inventory[n])) ++n;
    int rc=interfaces_select(spec,fallback,inventory,n,out,count);
    free(inventory); freeifaddrs(all); return rc;
}
