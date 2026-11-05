/* test_discovery.c - strict peer reports and three real authenticated nodes.
 * Fake peers cover malformed data without depending on a foreign machine.
 */
#include "test.h"
#include "../src/discovery.h"
#include "../src/net.h"
#include "../src/control.h"
#include "../src/identity.h"
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
static int exchange(const char *reply, size_t len, fr_hwinfo *h, fr_discovery *collected) {
    int fds[2]; if (socketpair(AF_UNIX,SOCK_STREAM,0,fds)) return -2;
    fflush(NULL); pid_t child=fork();
    if (!child) {
        close(fds[0]); char request[4]; size_t got=0;
        while (got<sizeof(request)) { ssize_t n=read(fds[1],request+got,sizeof(request)-got); if (n<=0) _exit(2); got+=(size_t)n; }
        size_t sent=0;
        while (sent<len) { ssize_t n=write(fds[1],reply+sent,len-sent); if (n<=0) _exit(3); sent+=(size_t)n; }
        close(fds[1]); _exit(memcmp(request,"HW\r\n",4)?4:0);
    }
    close(fds[1]);
    if (child<0) { close(fds[0]); return -2; }
    int rc, status;
    if (collected) {
        fr_connection *links=calloc(1,sizeof(*links));
        if (!links) { close(fds[0]); waitpid(child,&status,0); return -2; }
        links->count=1; fr_link *link=&links->links[0];
        link->fd=fds[0]; link->authenticated=link->multicast=1;
        link->identity.id=h->id; strcpy(link->identity.name,h->name);
        for (unsigned i=0;i<3;++i) link->rtt[i].count=FR_RTT_SAMPLES;
        rc=discovery_run(links,collected,NULL); free(links);
    } else rc=discovery_fetch(fds[0],h);
    close(fds[0]);
    if (waitpid(child,&status,0)!=child || !WIFEXITED(status) || WEXITSTATUS(status)) return -2;
    return rc;
}
static int fake(const char *reply, size_t len, fr_hwinfo *h) { return exchange(reply,len,h,NULL); }
int main(int argc, char **argv) {
    test_begin("discovery");
    fr_hwinfo h={.id=1,.name="fixture",.version="fixture",.os="linux",.arch="x86_64",.cpu="Fixture CPU",.gpu="none",.cores=4,.logical_cores=8,.isa=7,
        .total=UINT64_C(17179869184),.available=UINT64_C(12884901888)};
    char text[FR_HW_TEXT], reply[FR_HW_TEXT+32];
    check(!hw_finalize(&h),"prepare complete hardware fixture");
    int n=hw_format(&h,text,sizeof(text));
    int len=snprintf(reply,sizeof(reply),"$%d\r\n%s\r\n",n,text);
    fr_hwinfo got={0};
    check(n>0 && len>0 && !fake(reply,(size_t)len,&got) && got.isa==7 && got.cores==4,"valid HW bulk report accepted");
    check(fake("-NOAUTH authentication required\r\n",33,&got)==-1,"server error cannot become a profile");
    check(fake("$-1\r\n",5,&got)==-1,"null HW reply rejected");
    check(fake("$5\r\nnope!\r\n",11,&got)==-1,"malformed hardware body rejected");
    check(fake("$8\r\nshort",9,&got)==-1,"truncated response rejected");
    fr_discovery excluded={0}; strcpy(h.os,"macos"); h.translated=1;
    n=hw_format(&h,text,sizeof(text)); len=snprintf(reply,sizeof(reply),"$%d\r\n%s\r\n",n,text);
    check(n>0 && exchange(reply,(size_t)len,&h,&excluded)<0 && excluded.count==0 && excluded.excluded==1,
        "translated peer is ignored and cannot form a compute cluster");
    fr_connection *links=calloc(1,sizeof(*links)); fr_discovery result={0};
    check(links!=NULL,"allocate link storage"); if (!links) return test_end();
    check(discovery_run(links,&result,NULL)<0 && !result.count,"DISCOVERY requires a completed CONNECTION phase");
    if (argc!=2) { check(0,"generated head config required"); free(links); return test_end(); }
    fr_config cfg; char error[512];
    check(!config_load(argv[1],&cfg,error,sizeof(error)),"load local cluster config");
    int rc=connection_run(&cfg,links,NULL);
    check(!rc && links->count==3,"CONNECTION establishes three local links first");
    if (!rc) {
        check(!discovery_run(links,&result,stdout) && result.count==3,"three authenticated hardware profiles collected");
        fr_hwinfo local; int ok=!hw_probe(&local);
        for (unsigned i=0; i<result.count; ++i) {
            fr_hwinfo *p=&result.nodes[i];
            if (strcmp(p->cpu,local.cpu) || strcmp(p->arch,local.arch) || p->total!=local.total ||
                p->cores!=local.cores || p->metal_limit!=local.metal_limit || strcmp(hw_backend(p),hw_backend(&local))) ok=0;
        }
        check(ok,"all local profiles agree with independent native hardware query");
        links->links[0].authenticated=0;
        check(discovery_run(links,&result,NULL)<0 && !result.count,"unverified links cannot enter DISCOVERY");
        links->links[0].authenticated=1;
        check(!discovery_run(links,&result,NULL) && result.count==3,"DISCOVERY can rerun on the existing links");
        fr_link *wifi=&links->links[0]; fr_link_type old_type=wifi->via.type;
        wifi->via.type=FR_LINK_WIFI; wifi->multicast=0;
        check(!discovery_run(links,&result,NULL) && !discovery_save(&cfg,links,&result),
              "Wi-Fi unicast paths enter DISCOVERY and saved profiles without multicast receipt");
        wifi->via.type=old_type; wifi->multicast=1;
        check(!discovery_save(&cfg,links,&result),"atomic hardware profile persisted");
        FILE *saved=fopen(cfg.profile_file,"r"); char profile[16384]={0};
        size_t size=saved?fread(profile,1,sizeof(profile)-1,saved):0;
        if (saved) fclose(saved);
        check(size && strstr(profile,"# head") && strstr(profile,"head_reserve_bytes:10000000000") &&
            strstr(profile,"mem_cap_blocks:") && strstr(profile,"rtt_us_20k:") &&
            !strstr(profile,"bw_eff_gbs") && !strstr(profile,"t0_us") && !strstr(profile,"burst_slope") &&
            !strstr(profile,"sustained_ratio"),"profile includes head reserve, whole blocks and RTTs without compute placeholders");
        fr_config capacity_cfg=cfg; capacity_cfg.head_reserve=UINT64_MAX;
        fr_discovery snapshot={.count=4}; uint64_t head_id=0;
        int valid_id=!identity_load(cfg.state_dir,&head_id); unsigned capacities[]={2,1,4,10};
        for (unsigned i=0;i<4;++i) { snapshot.nodes[i].id=head_id^(i+1); snapshot.nodes[i].blocks=capacities[i]; }
        FILE *capacity_out=tmpfile(); char capacity_text[512]={0};
        int refused=capacity_out?discovery_capacity(&capacity_cfg,&snapshot,capacity_out):-1;
        if (capacity_out) { rewind(capacity_out); if (!fread(capacity_text,1,sizeof(capacity_text)-1,capacity_out)) capacity_text[0]=0; fclose(capacity_out); }
        check(valid_id && refused==1 && !strncmp(capacity_text,"CAPACITY total=17 required=20",29) &&
              strstr(capacity_text,"unplaced_blocks=17,18,19"),"head reserve is honored and live planning gate refuses the 17-block snapshot");
        snapshot.nodes[0].blocks=5; capacity_out=tmpfile();
        int fits=capacity_out?discovery_capacity(&capacity_cfg,&snapshot,capacity_out):-1;
        if (capacity_out) fclose(capacity_out);
        check(fits==0,"live planning gate accepts twenty blocks from the currently available node capacities");
        fr_discovery invalid=result; invalid.nodes[0].id^=1;
        check(discovery_save(&cfg,links,&invalid)<0,"invalid profile cannot replace previous file");
        saved=fopen(cfg.profile_file,"r"); char retained[16384]={0};
        size_t retained_size=saved?fread(retained,1,sizeof(retained)-1,saved):0;
        if(saved) fclose(saved);
        check(size==retained_size && !memcmp(profile,retained,size),"failed save preserves previous complete profile");
        fr_bench_report cost={.blocks=3,.samples=FR_BENCH_SAMPLES,.burst_max=4,.elapsed=60,.barriers=100000,.sustained_ratio=0.97};
        cost.working_set=cost.resident_bytes=3*FR_WEIGHT_BLOCK_BYTES;
        cost.rss_start=cost.rss_end=cost.working_set+1000000;
        for (unsigned m=0;m<2;++m) for (unsigned b=0;b<4;++b) {
            double total=(30000+(10*(b+1)+1)*3*FR_SLICE_BYTES/100.0)*(m?0.75:1);
            cost.mean[m][b]=total*1.1;
            for (unsigned p=0;p<FR_METAL_PHASES;++p) {
                double v=p==FR_METAL_TOTAL?total:p==FR_METAL_GPU?total-30000:p==FR_METAL_OVERHEAD?30000:10000;
                cost.percentile[m][b][p][0]=v; cost.percentile[m][b][p][1]=v*1.25;
            }
        }
        check(!fr_bench_fit(&cost),"prepare synthetic measured-profile fixture");
        result.compute[0]=cost; result.measured=1;
        check(!discovery_save(&cfg,links,&result),"persist a validated compute profile alongside hardware");
        saved=fopen(cfg.profile_file,"r"); size=saved?fread(profile,1,sizeof(profile)-1,saved):0; profile[size]=0;
        if (saved) fclose(saved);
        check(size && strstr(profile,"bw_eff_gbs:") && strstr(profile,"bench_resident_bytes:10212065280") &&
            strstr(profile,"submission_distinct_4_overhead_p99_ns:"),"saved profile contains residency, cost and submission percentiles");
        invalid=result; invalid.compute[0].resident_bytes=0;
        check(discovery_save(&cfg,links,&invalid)<0,"unverified compute result cannot enter the profile");
        uint64_t id=links->links[0].identity.id; links->links[0].identity.id^=1;
        check(discovery_run(links,&result,NULL)<0,"HW identity must match authenticated LINK identity");
        links->links[0].identity.id=id;
        check(!discovery_run(links,&result,NULL),"restore authenticated identities");
        char response[RESP_MAX_MESSAGE]; resp_value values[RESP_MAX_VALUES];
        const char *e=control_exchange(links->links[0].fd,"INFO system\r\n",13,response,values,net_now_ns()/1000000+3000);
        check(!e && values[0].type==RESP_BULK && values[0].len<FR_HW_TEXT*3 &&
            (response[values[0].data-response+values[0].len]=0,strstr(response,"cpu_model:")) && strstr(response,"hw_samples:"),"latest hardware snapshot visible in INFO");
    }
    check(!connection_close(links),"close discovery links"); free(links);
    return test_end();
}
