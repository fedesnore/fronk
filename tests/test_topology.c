/* Opt-in two-interface integration. Uses existing addresses only; no aliases
 * or system/network settings are created. local_cluster.sh owns real nodes. */
#include "test.h"
#include "../src/discovery.h"
#include "../src/control.h"
#include "../src/net.h"
#include <string.h>
int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"--interfaces")) {
        fr_interface list[FR_INTERFACES]; unsigned n=0,best=FR_INTERFACES;
        if (interfaces_read("auto",NULL,list,&n)) return 1;
        for (unsigned i=0;i<n;++i) if (!list[i].loopback && (best==FR_INTERFACES || list[i].speed_bps>list[best].speed_bps)) best=i;
        if (best==FR_INTERFACES) return 1;
        printf("127.0.0.1,%s\n",list[best].address); return 0;
    }
    test_begin("topology");
    if (argc!=2) { check(0,"generated head configuration required"); return test_end(); }
    fr_config cfg; char error[512]; fr_connection *links=calloc(1,sizeof(*links)); fr_discovery hw;
    int ready=links && !config_load(argv[1],&cfg,error,sizeof(error));
    check(ready,"load real two-interface local cluster configuration");
    if (!ready) { free(links); return test_end(); }
    /* A single legacy broadcast override cannot override both interfaces. */
    strcpy(cfg.broadcast,"192.0.2.255");
    int rc=connection_run(&cfg,links,stdout);
    check(!rc && links->count==3 && links->interface_count==2 && links->path_count==6,"three persistent node IDs discovered on six actual interface paths");
    char wifi[16]={0};
    for (unsigned i=0;i<links->interface_count;++i) if (links->interfaces[i].type==FR_LINK_WIFI)
        strcpy(wifi,links->interfaces[i].address);
    uint64_t ids[3]={0};
    if (!rc) {
        int chosen=1,all_measured=1,info=1;
        for (unsigned i=0;i<links->count;++i) {
            fr_link *l=&links->links[i]; ids[i]=l->identity.id; unsigned paths=0,selections=0;
            uint64_t p99=links->paths[l->path].percentile[FR_PATH_RTT_KIND][1];
            for (unsigned j=0;j<links->path_count;++j) if (links->paths[j].identity.id==l->identity.id) {
                fr_path *p=&links->paths[j]; ++paths; selections+=p->selected;
                if (!p->ready || p->samples[0]!=1000 || p->samples[1]!=1000 || p->samples[2]!=1000) all_measured=0;
                if (p->ready && p99>p->percentile[FR_PATH_RTT_KIND][1]) chosen=0;
            }
            if (paths!=2 || selections!=1 || (i && ids[i]<=ids[i-1])) chosen=0;
            char text[RESP_MAX_MESSAGE],expected[100]; resp_value v[RESP_MAX_VALUES];
            const char *e=control_exchange(l->fd,"INFO system\r\n",13,text,v,net_now_ns()/1000000+3000);
            if (e || v[0].type!=RESP_BULK) { info=0; continue; }
            text[v[0].data-text+v[0].len]=0;
            snprintf(expected,sizeof(expected),"head_interface:%s\r\n",l->via.name);
            if (!strstr(text,expected)) info=0;
            snprintf(expected,sizeof(expected),"head_interface_address:%s\r\n",l->via.address);
            if (!strstr(text,expected) || !strstr(text,"head_link_type:") || !strstr(text,"head_link_speed_bps:")) info=0;
        }
        check(all_measured,"both paths have all 1000 successful samples at every unchanged loss/time limit");
        check(chosen,"lowest barrier p99 wins once per persistent ID; unchosen paths retained");
        check(info,"every selected node reports its head-side path in INFO");
        check(!discovery_run(links,&hw,NULL) && hw.count==3 && !discovery_save(&cfg,links,&hw),"hardware and selected head topology persist together");
        FILE *file=fopen(cfg.profile_file,"r"); char profile[16384]={0}; size_t size=file?fread(profile,1,sizeof(profile)-1,file):0;
        if (file) fclose(file);
        unsigned fields=0; const char *p=profile;
        while ((p=strstr(p,"head_interface:"))) { ++fields; p+=15; }
        check(size && fields==3 && strstr(profile,"head_link_speed_bps:"),"saved profile has one head-interface entry per unique node");
    }
    check(!connection_close(links),"close selected paths and all interface sockets");
    strcpy(cfg.interfaces,"127.0.0.1");
    rc=connection_run(&cfg,links,NULL); int stable=!rc && links->count==3 && links->path_count==3;
    for (unsigned i=0;i<links->count && i<3;++i) if (links->links[i].identity.id!=ids[i] || !links->links[i].via.loopback) stable=0;
    check(stable,"a fresh CONNECTION reselects available paths between sessions without changing node identity");
    check(!connection_close(links),"close rerun topology");
    if (*wifi) {
        strcpy(cfg.interfaces,wifi); strcpy(cfg.broadcast,"255.255.255.255"); strcpy(cfg.multicast,"239.7.7.8");
        rc=connection_run(&cfg,links,NULL); int unicast=!rc && links->count==3;
        for (unsigned i=0;i<links->count;++i)
            if (links->links[i].multicast || links->links[i].via.type!=FR_LINK_WIFI) unicast=0;
        check(unicast,"Wi-Fi-only CONNECTION succeeds with an unjoined multicast group and never claims multicast passed");
        check(!connection_close(links),"close Wi-Fi-only paths");
    }
    free(links);
    return test_end();
}
