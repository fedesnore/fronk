/* test_connection.c - three actual nodes: discovery, auth, RTT trains and group.
 * local_cluster.sh owns the nodes. Failed phases must never become success.
 */
#include "test.h"
#include "../src/connection.h"
#include "../src/control.h"
#include "../src/net.h"
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    test_begin("connection");
    fr_path paths[3]={0};
    for (unsigned i=0;i<3;++i) { paths[i].identity.id=7; paths[i].ready=1; paths[i].percentile[2][1]=100+i*10; }
    paths[1].percentile[2][1]=80; paths[1].percentile[0][1]=500;
    check(connection_best_path(paths,3,7,2)==1,"rank all addresses for one persistent ID by barrier p99, not small-probe latency");
    paths[1].ready=0;
    check(connection_best_path(paths,3,7,2)==0 && connection_best_path(paths,3,8,2)<0,"failed paths and other IDs cannot be selected");
    paths[0].percentile[2][1]=120; strcpy(paths[0].via.name,"en9"); strcpy(paths[2].via.name,"en1");
    check(connection_best_path(paths,3,7,2)==2,"equal RTT paths have deterministic interface-name tie breaking");
    if (argc!=2) { check(0,"head config required"); return test_end(); }
    fr_config cfg; char error[512];
    check(!config_load(argv[1],&cfg,error,sizeof(error)),"read generated head config");
    fr_connection *run=calloc(1,sizeof(*run));
    check(run!=NULL,"allocate bounded run storage"); if (!run) return test_end();
    int rc=connection_run(&cfg,run,stdout);
    check(rc==0 && run->count==3 && run->local,"three loopback nodes found through real UDP discovery");
    int ready=1, measured=1, totals=1;
    for (unsigned i=0; i<run->count; ++i) {
        fr_link *link=&run->links[i];
        if (!link->authenticated || !link->multicast || link->fd<0) ready=0;
        for (int k=0; k<3; ++k) {
            uint64_t p[3]; unsigned sum=0;
            if (link->rtt[k].count!=1000 || stats_percentiles(&link->rtt[k],p) || !p[0] || p[0]>p[1] || p[1]>p[2]) measured=0;
            for (int b=0; b<FR_HIST_BINS; ++b) sum+=link->rtt[k].bins[b];
            if (sum!=link->rtt[k].count) totals=0;
        }
    }
    check(ready,"all discovered links authenticated and multicast confirmed");
    check(measured,"1000 successful samples for every node and size, ordered percentiles");
    check(totals,"histogram counts equal successful sample counts");
    if (!rc && run->count) {
        char reply[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES];
        const char *e=control_exchange(run->links[0].fd,"INFO system\r\n",13,reply,v,net_now_ns()/1000000+3000);
        check(!e && v[0].type==RESP_BULK && v[0].len<sizeof(reply) &&
              (reply[v[0].data-reply+v[0].len]='\0',strstr(reply,"link_rtt_64_samples:1000")) && strstr(reply,"link_rtt_20k_hist:"),"measurements and histograms inspectable in INFO");
    }
    uint64_t ids[FR_CONNECTION_NODES]={0}; int sorted=1;
    for (unsigned i=0;i<run->count;++i) {
        ids[i]=run->links[i].identity.id;
        if (!ids[i] || (i && ids[i]<=ids[i-1])) sorted=0;
    }
    check(sorted,"persistent machine IDs are nonzero, distinct and sorted independently of names");
    check(!connection_close(run),"close all authenticated links");
    rc=connection_run(&cfg,run,NULL); int stable=!rc && run->count==3;
    for(unsigned i=0;i<run->count;++i) if(run->links[i].identity.id!=ids[i]) stable=0;
    check(stable,"new sessions retain the same machine IDs");
    check(!connection_close(run),"close repeated session");
    fr_config wrong=cfg; strcpy(wrong.password,"wrong-test-password");
    check(connection_run(&wrong,run,stdout)!=0 && run->count==3,"wrong password cannot produce a successful CONNECTION phase");
    check(!connection_close(run),"failed authentication cleanup");
    fr_config missing=cfg; strcpy(missing.multicast,"239.7.7.8");
    int unicast=!connection_run(&missing,run,stdout) && run->count==3;
    for (unsigned i=0;unicast && i<run->count;++i) if (run->links[i].multicast) unicast=0;
    check(unicast,"nodes that miss multicast keep their paths and fall back to unicast");
    check(!connection_close(run),"unicast fallback cleanup");
    missing.peer_count=1; strcpy(missing.peers[0].name,"local-1"); strcpy(missing.peers[0].address,"127.0.0.1");
    check(!connection_run(&missing,run,NULL) && run->count==1 && !run->links[0].multicast,
          "configured singleton succeeds with an unjoined multicast group and never claims multicast passed");
    check(!connection_close(run),"singleton unicast cleanup");
    int silent=net_udp_bind("127.0.0.1",0,0);
    check(silent>=0,"reserve an unused discovery destination");
    if (silent>=0) {
        fr_config absent=cfg; absent.discovery_port=(unsigned)net_port(silent);
        check(connection_run(&absent,run,stdout)!=0 && !run->count,"no advertisements is a failed phase");
        check(!connection_close(run) && !close(silent),"empty-discovery cleanup");
    }
    free(run); return test_end();
}
