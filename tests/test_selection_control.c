/* Opt-in native sweep/control test. A configured memory reserve limits this
 * disposable node to one real resident block; no sparse timing shortcuts.
 */
#include "test.h"
#include "server.h"
#include "../src/discovery.h"
#include "../src/control.h"
#include "../src/net.h"
#include <inttypes.h>
int main(int argc,char **argv) {
    test_begin("selection-control");
    if (argc!=2) return 2;
    fr_hwinfo host; int rc=hw_probe(&host);
    check(!rc && host.available>2*FR_WEIGHT_BLOCK_BYTES,"room for fully resident native control fixture");
    if (rc || host.available<=2*FR_WEIGHT_BLOCK_BYTES) return test_end();
    uint64_t reserve=host.available-FR_WEIGHT_BLOCK_BYTES-FR_WEIGHT_BLOCK_BYTES/2;
    char extra[96];
    snprintf(extra,sizeof(extra),"mem-margin-gb %" PRIu64 ".%09" PRIu64 "\n",reserve/1000000000,reserve%1000000000);
    test_server server; check(server_start_extra(&server,argv[1],extra),"normal native node with explicit test reserve");
    int fd=net_tcp_connect("127.0.0.1",server.port,3000); char reply[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES];
    int ready=fd>=0 && !control_authenticate(fd,"fixture-secret",reply,v,net_now_ns()/1000000+3000);
    check(ready,"authenticate native control connection");
    fr_discovery d={.count=1,.burst=1}; fr_connection *links=calloc(1,sizeof(*links));
    if (ready && links) {
        check(!discovery_fetch(fd,&d.nodes[0]) && d.nodes[0].blocks==1,"test reserve yields one measured block capacity");
        links->count=1; links->links[0].fd=fd; links->links[0].authenticated=1; links->links[0].identity.id=d.nodes[0].id;
        rc=d.nodes[0].blocks==1?discovery_bench(links,&d,stdout,0):-1;
        check(!rc && selection_complete(&d.selection[0]),"DISCOVERY measures every eligible backend with resident point reports");
        if (!rc) {
            int best=selection_pick(&d.selection[0],1); char reason[1024];
            int n=selection_reason(&d.selection[0],1,reason,sizeof(reason));
            check(best>=0 && n>0,"measured winner available");
            int ok=!control_exchange(fd,"BACKEND 1\r\n",11,reply,v,net_now_ns()/1000000+3000) && v[0].type==RESP_BULK &&
                   v[0].len==(size_t)n && !memcmp(v[0].data,reason,(size_t)n);
            check(ok,"node configures exact measured backend and explains alternatives");
            ok=!control_exchange(fd,"INFO backend\r\n",14,reply,v,net_now_ns()/1000000+3000) && v[0].type==RESP_BULK;
            if (ok) { memmove(reply,v[0].data,v[0].len); reply[v[0].len]=0; }
            check(ok && strstr(reply,"backend_ready:1") && strstr(reply,reason) && strstr(reply,"candidate_ref_1_p50_ns:"),"INFO exposes selected share and every measured candidate");
        }
    }
    if (fd>=0) close(fd);
    free(links); check(server_stop(&server),"clean native shutdown"); return test_end();
}
