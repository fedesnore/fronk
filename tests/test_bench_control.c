/* Real, opt-in Metal control integration: requires room for three resident
 * blocks. Performance is recorded, never asserted against a speed threshold. */
#include "test.h"
#include "../src/discovery.h"
#include "../src/control.h"
#include "../src/net.h"
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
typedef struct { fr_connection *links; fr_discovery *hw; atomic_int done; int result; } job;
static void *measure(void *arg) {
    job *j=arg; char reply[RESP_MAX_MESSAGE]; resp_value values[RESP_MAX_VALUES];
    const char *error=control_exchange(j->links->links[0].fd,"BENCH\r\n",7,reply,values,net_now_ns()/1000000+300000);
    j->result=error || values[0].type!=RESP_BULK || values[0].is_null || fr_bench_parse(values[0].data,values[0].len,&j->hw->compute[0]);
    if (!j->result) j->hw->measured=1;
    atomic_store(&j->done,1); return NULL;
}
static int request(int fd,const char *command,char text[RESP_MAX_MESSAGE]) {
    resp_value v[RESP_MAX_VALUES];
    if (control_exchange(fd,command,strlen(command),text,v,net_now_ns()/1000000+3000)) return -1;
    if (v[0].is_null || v[0].len>=RESP_MAX_MESSAGE ||
        (v[0].type!=RESP_STRING && v[0].type!=RESP_BULK && v[0].type!=RESP_ERROR)) return -1;
    memmove(text,v[0].data,v[0].len); text[v[0].len]=0; return v[0].type;
}
static void pause_ms(void) { struct timespec t={0,100000000}; nanosleep(&t,NULL); }
static int authenticated(const fr_link *link,const char *password) {
    int fd=net_tcp_connect(link->ip,link->identity.control_port,3000);
    char reply[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES];
    if (fd>=0 && control_authenticate(fd,password,reply,v,net_now_ns()/1000000+3000)) { close(fd); return -1; }
    return fd;
}
int main(int argc,char **argv) {
    test_begin("bench-control");
    if (argc!=2) { check(0,"generated one-node head config required"); return test_end(); }
    fr_config cfg; char error[512],text[RESP_MAX_MESSAGE];
    fr_connection *links=calloc(1,sizeof(*links)); fr_discovery hw={0};
    int configured=links && !config_load(argv[1],&cfg,error,sizeof(error));
    check(configured,"load isolated benchmark cluster");
    if (!configured) { free(links); return test_end(); }
    int ready=!connection_run(&cfg,links,NULL) && links->count==1 && !discovery_run(links,&hw,NULL);
    check(ready,"authenticated CONNECTION and HW precede BENCH");
    int observer=ready?authenticated(&links->links[0],cfg.password):-1;
    check(observer>=0,"independent authenticated observer connects");
    if (observer>=0) {
        job j={.links=links,.hw=&hw}; atomic_init(&j.done,0); pthread_t thread;
        int started=!pthread_create(&thread,NULL,measure,&j);
        check(started,"collect real BENCH through discovery on a separate control link");
        if (started) {
            int running=0;
            for (unsigned i=0;i<100 && !atomic_load(&j.done);++i) {
                if (request(observer,"INFO backend\r\n",text)==RESP_BULK && strstr(text,"bench_status:running")) { running=1; break; }
                pause_ms();
            }
            check(running,"INFO remains responsive while resident weights are prepared");
            check(request(observer,"PING\r\n",text)==RESP_STRING && !strcmp(text,"PONG"),"PING responds within ordinary control deadline during BENCH");
            check(request(observer,"BENCH\r\n",text)==RESP_ERROR && strstr(text,"BUSY"),"concurrent benchmark refused without a second working set");
            int responsive=1; unsigned polls=0;
            while (!atomic_load(&j.done) && polls++<3000) {
                if (!(polls%50) && (request(observer,"PING\r\n",text)!=RESP_STRING || strcmp(text,"PONG"))) responsive=0;
                pause_ms();
            }
            check(responsive && atomic_load(&j.done),"control stays responsive throughout bounded benchmark collection");
            check(!pthread_join(thread,NULL) && !j.result && hw.measured==1,"real measured profile received and parsed");
            if (!j.result) {
                fr_bench_report cached;
                check(request(observer,"BENCH GET\r\n",text)==RESP_BULK && !fr_bench_parse(text,strlen(text),&cached) &&
                      cached.barriers==hw.compute[0].barriers && cached.resident_bytes==3*FR_WEIGHT_BLOCK_BYTES && !cached.hot_allocations,
                      "cached benchmark retains full residency and zero own allocations");
                check(request(observer,"INFO\r\n",text)==RESP_BULK && strstr(text,"bench_status:ready") &&
                      strstr(text,"submission_distinct_4_overhead_p99_ns:") && strstr(text,"submission_reused_4_overhead_p50_ns:"),
                      "complete INFO fits and exposes measured overhead for both routing modes");
                check(!discovery_save(&cfg,links,&hw),"head atomically saves actual compute and network profile");
                FILE *file=fopen(cfg.profile_file,"r"); size_t n=file?fread(text,1,sizeof(text)-1,file):0; text[n]=0;
                if (file) fclose(file);
                check(n && strstr(text,"bench_resident_bytes:10212065280") && strstr(text,"bw_eff_gbs:") &&
                      strstr(text,"submission_distinct_4_gpu_p99_ns:") && strstr(text,"rtt_us_20k:"),"saved profile has measured phases, residency and unchanged link data");
                if (n) fputs(text,stdout);
            }
        }
        /* Cancellation must release a worker and its connection slot. Use the
         * still-authenticated discovery link to observe the cancelled owner. */
        check(net_send(observer,"BENCH\r\n",7)==7,"start cancellable benchmark");
        int running=0,monitor=links->links[0].fd;
        for (unsigned i=0;i<100;++i) {
            if (request(monitor,"INFO backend\r\n",text)==RESP_BULK && strstr(text,"bench_status:running")) { running=1; break; }
            pause_ms();
        }
        check(running,"cancellable worker started"); close(observer); observer=-1;
        int stopped=0;
        for (unsigned i=0;i<300;++i) {
            if (request(monitor,"INFO backend\r\n",text)==RESP_BULK && strstr(text,"bench_status:failed")) { stopped=1; break; }
            pause_ms();
        }
        check(stopped,"owner disconnect cancels and joins worker");
        observer=authenticated(&links->links[0],cfg.password);
        check(observer>=0 && request(observer,"PING\r\n",text)==RESP_STRING && !strcmp(text,"PONG"),"cancelled client slot can authenticate and serve a new client");
        if (observer>=0) close(observer);
    }
    check(!connection_close(links),"close benchmark control links"); free(links);
    return test_end();
}
