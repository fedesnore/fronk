/* Same test driver for disposable localhost nodes and the physical cluster.
 * Sparse native fixtures validate math/wire/lifecycle, never capacity or tok/s. */
#include "test.h"
#include "../src/plane.h"
#include "../src/units.h"
#include "../src/control.h"
#include "../src/sha256.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
static int command(int fd,const char *text,char *response,resp_value *v) {
    const char *error=control_exchange(fd,text,strlen(text),response,v,net_now_ns()/1000000+30000);
    if (error) fprintf(stderr,"control: %s\n",error);
    if (!error && v[0].type==RESP_ERROR) fprintf(stderr,"control: %.*s\n",(int)v[0].len,v[0].data);
    return error!=NULL;
}
int main(int argc,char **argv) {
    test_begin("plane-cluster");
    if (argc<2 || argc>3) { check(0,"config [local-backend] required"); return test_end(); }
    fr_config cfg; char error[512];
    if (config_load(argv[1],&cfg,error,sizeof(error))) { check(0,"config load"); return test_end(); }
    fr_connection *links=calloc(1,sizeof(*links)); fr_head_plane *h=calloc(1,sizeof(*h));
    fr_units local={0},reference[FR_CONNECTION_NODES+1]={0}; int opened=0;
#if defined(__x86_64__)
    const char *native="avx2";
#else
    const char *native="neon";
#endif
    const fr_backend *backend=fr_backend_find(argc==3?argv[2]:native);
    check(links && h && backend,"test storage and local backend available"); if (!links || !h || !backend) goto done;
    int rc=connection_run(&cfg,links,stdout);
    check(!rc && links->count>0,"CONNECTION on actual selected interfaces"); if (rc || !links->count) goto done;
    check(!fr_units_fixture(&local,backend,0),"head native fixture in process"); if (!local.mask) goto done;
    for (unsigned i=0;i<=links->count;++i) if (fr_units_fixture(&reference[i],&fr_ref,i)) { check(0,"oracle fixture"); goto done; }
    printf("PLAN node=head link=local backend=%s blocks=1 fixture=sparse\n",backend->name);
    char response[RESP_MAX_MESSAGE],request[128]; resp_value v[RESP_MAX_VALUES];
    for (unsigned i=0;i<links->count;++i) {
        fr_link *l=&links->links[i];
        if (command(l->fd,"INFO backend\r\n",response,v) || v[0].type!=RESP_BULK) { check(0,"node backend inventory"); goto done; }
        response[v[0].data-response+v[0].len]=0;
        const char *remote=strstr(v[0].data,"avx2")?"avx2":strstr(v[0].data,"neon")?"neon":"ref";
        snprintf(request,sizeof(request),"SESSION TEST %u %u 8 %s\r\n",links->session,i+1,remote);
        if (command(l->fd,request,response,v) || v[0].type==RESP_ERROR) { check(0,"start authenticated fixture session"); goto done; }
        printf("PLAN node=%s link=%s interface=%s backend=%s blocks=1 fixture=sparse\n",l->identity.name,interface_type(l->via.type),l->via.name,remote);
    }
    rc=fr_head_plane_open(h,links,cfg.multicast,cfg.discovery_port,backend,1); opened=!rc;
    check(!rc,"per-interface fanout and local compute worker open"); if (rc) goto done;
    for (unsigned i=0;i<h->fanout.count;++i) printf("FANOUT interface=%s address=%s nodes=%u mode=%s\n",
        h->fanout.groups[i].via.name,h->fanout.groups[i].via.address,h->fanout.groups[i].count,
        h->fanout.groups[i].count>=2?"multicast":"unicast");
    fr_req req={.layer=47}; fr_part expected,actual;
    unsigned step=0; double worst=0;
    for (unsigned mode=0;mode<2;++mode) for (unsigned burst=1;burst<=8;++burst) {
        req.burst=burst;
        for (unsigned t=0;t<burst;++t) {
            for (unsigned e=0;e<10;++e) req.expert[t][e]=(uint16_t)(e+(mode?0:t*10));
            for (unsigned e=0;e<11;++e) req.gate[t][e]=e==10?.5f:.0625f;
            for (unsigned d=0;d<2560;++d) req.x[t][d]=(float)((int)((d*17+t*31)%257)-128)/128;
        }
        rc=fr_ref.run(&req,&expected);
        if (!rc) rc=fr_head_plane_run(h,&req,++step,2000000000,&actual);
        int ok=!rc && !h->last_missing;
        for (unsigned t=0;ok && t<burst;++t) {
            fr_error e;
            if (fr_compare(expected.y[t],actual.y[t],2560,&e) || !fr_conforms(&e,&fr_backend_tolerance)) ok=0;
            if (e.relative_l2>worst) worst=e.relative_l2;
        }
        char label[100]; snprintf(label,sizeof(label),"%s burst %u: remote native slices + local head match scalar sum",mode?"reused":"distinct",burst);
        check(ok,label); if (!ok) { printf("failure status=%d missing=%u worst_l2=%.9g\n",rc,h->last_missing,worst); goto done; }
        if (h->exact_remote) {
            fr_part repeated;
            rc=fr_head_plane_run(h,&req,++step,2000000000,&repeated);
            check(!rc && !h->last_missing && !memcmp(actual.y,repeated.y,burst*sizeof(actual.y[0])),
                  "strict remote sums reproduce identical bits on repeated requests");
            if (rc || h->last_missing) goto done;
        }
    }
    fr_head_plane_report(h,stdout);
    printf("VALIDATION native sparse fixtures worst_relative_l2=%.9g\n",worst);
    for (unsigned i=0;i<links->count;++i) {
        rc=command(links->links[i].fd,"INFO dataplane\r\n",response,v);
        if (!rc && v[0].type==RESP_BULK) { response[v[0].data-response+v[0].len]=0; rc=!strstr(v[0].data,"compute_errors:0"); }
        else rc=1;
        check(!rc,"node INFO exposes active data plane and no compute errors");
    }
    check(!command(links->links[0].fd,"SESSION STOP\r\n",response,v) && v[0].type!=RESP_ERROR,"stop one node without closing head session");
    /* Only the deliberately stopped node should miss this correctness check.
     * The four workers share one test host; the last width-eight request can
     * exceed 100 ms on a busy machine. Use the same bound as the checks above,
     * rather than turning host scheduling into an accidental speed requirement. */
    rc=fr_head_plane_run(h,&req,++step,2000000000,&actual);
    int deadline_ok=!rc && h->last_missing==1 && h->deadline_hits==1 && h->last_local_ns>0;
    check(deadline_ok,"deadline records exactly the stopped node and drains local work");
    if (!deadline_ok) printf("deadline status=%d missing=%u hits=%llu local_ns=%llu\n",rc,h->last_missing,
        (unsigned long long)h->deadline_hits,(unsigned long long)h->last_local_ns);
    /* The low-level plane reports a partial sum plus its missing-node mask;
     * ordinary serving refuses to commit it. Check the surviving values too. */
    fr_units_close(&reference[1]);
    int partial_ok=deadline_ok && !fr_ref.run(&req,&expected);
    for (unsigned t=0;partial_ok && t<req.burst;++t) {
        fr_error e;
        partial_ok=!fr_compare(expected.y[t],actual.y[t],2560,&e) && fr_conforms(&e,&fr_backend_tolerance);
    }
    check(partial_ok,"partial result contains every surviving slice including the head");
done:
    if (opened) check(!fr_head_plane_close(h),"stop head worker");
    if (links) connection_close(links);
    fr_units_close(&local);
    for (unsigned i=0;i<=FR_CONNECTION_NODES;++i) fr_units_close(&reference[i]);
    free(h); free(links); secure_clear(&cfg,sizeof(cfg)); return test_end();
}
