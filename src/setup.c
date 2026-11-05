/* setup.c - "fronk backends": the node installer's backend step.
 * Lists the compiled backends this machine can run, marks one recommended by
 * rule (no measuring), and saves the choice in the machine's own state-dir.
 * Upgrades preserve that directory. Saving a choice also measures it, so
 * installing pays the block sweep once and every head reads it from the cache.
 * --test gives backend authors SELFTEST and a short timing (BACKENDS.md).
 */
#include "setup.h"
#include "runtime.h"
#include "backend.h"
#include "bench.h"
#include "native.h"
#include "selection.h"
#include "config.h"
#include "sha256.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/stat.h>

void setup_cache_identity(const char *argv0) {
    fr_hwinfo hw; char id[192];
    if (hw_probe(&hw)) return;
#ifdef FR_BENCH_BUILD
    (void)argv0;
    if (strlen(FR_BENCH_BUILD)!=64) return;
    const char *digest=FR_BENCH_BUILD;
#else
    // Standalone builds without a compute fingerprint keep exact-binary caching.
    FILE *f=fopen(argv0,"rb"); sha256_ctx h; unsigned char d[32],buf[65536]; char digest[65]; size_t n;
    if (!f) return;
    sha256_init(&h);
    while ((n=fread(buf,1,sizeof(buf),f))>0) sha256_update(&h,buf,n);
    int failed=ferror(f); fclose(f); if (failed) return;
    sha256_final(&h,d);
    for (unsigned i=0;i<32;++i) snprintf(digest+2*i,3,"%02x",d[i]);
#endif
    snprintf(id,sizeof(id),"%.64s|%.16s|%.32s|%u|%u|%llu|%u",digest,hw.os,hw.version,hw.cores,hw.logical_cores,(unsigned long long)hw.total,hw.isa);
    selection_cache_identity(id);
}
/* state-dir from the config text; config_load would also need the cluster secret. */
static int state_dir(const char *config,char *out,size_t size) {
    FILE *f=fopen(config,"r"); char line[512];
    if (!f) { fprintf(stderr,"fronk backends: %s: cannot open config\n",config); return -1; }
    out[0]=0;
    while (fgets(line,sizeof(line),f)) {
        char key[32],value[256];
        if (sscanf(line," %31s %255s",key,value)==2 && !strcmp(key,"state-dir")) snprintf(out,size,"%s",value);
    }
    fclose(f);
    if (!out[0]) { fprintf(stderr,"fronk backends: %s has no state-dir\n",config); return -1; }
    return 0;
}
int setup_saved_backend(const char *dir,char *out,size_t size) {
    char path[300]; snprintf(path,sizeof(path),"%s/backend",dir);
    FILE *f=fopen(path,"r"); out[0]=0;
    if (!f) return 0;
    if (!fgets(out,(int)size,f)) out[0]=0;
    fclose(f); out[strcspn(out,"\r\n")]=0;
    return 0;
}
static int save(const char *dir,const char *name) {
    char path[300],temp[310]; snprintf(path,sizeof(path),"%s/backend",dir); snprintf(temp,sizeof(temp),"%s.tmp",path);
    if (!strcmp(name,"auto")) { if (remove(path) && errno!=ENOENT) return -1; printf("saved: auto (the node measures the compatible backends and keeps the fastest)\n"); return 0; }
    if (selection_backend(name,stderr)) return -1;
    mkdir(dir,0700);
    FILE *f=fopen(temp,"w");
    if (!f || fprintf(f,"%s\n",name)<0 || fclose(f) || rename(temp,path)) { fprintf(stderr,"fronk backends: cannot write %s\n",path); return -1; }
    printf("saved: %s in %s (the node uses it from its next start)\n",name,path);
    return 0;
}
/* The short block sweep requested by DISCOVERY, cached in DIR/bench for every
 * width. Never allocate a new fixture above current usable memory. */
static int measure(const char *config,const char *dir,const char *name,unsigned burst,int refresh) {
    FILE *pid=fopen("runs/cluster/node.pid","r");
    if (pid) { int n=0; if (fscanf(pid,"%d",&n)==1 && n>0 && !kill(n,0)) { fclose(pid);
        fprintf(stderr,"fronk backends: a node is running here (pid %d); stop it on this computer before measuring\n",n); return -1; } fclose(pid); }
    fr_config cfg; char error[512]; fr_hwinfo hw; fr_selection s;
    if (config_load(config,&cfg,error,sizeof(error))) { config_defaults(&cfg); }
    secure_clear(cfg.password,sizeof(cfg.password));
    /* A head config names the dense engine, even before model installation.
     * Use its reserve, head-blocks and burst so later cache keys match. */
    int head=cfg.ds4[0]!=0 || cfg.model[0]!=0;
    if (head && fr_head_defaults()) { perror("fronk: head defaults"); return -1; }
    if (head) burst=cfg.burst;
    if (selection_backend(!strcmp(name,"auto")?NULL:name,stderr) || hw_probe(&hw)) return -1;
    hw.margin=cfg.mem_margin; hw.head_reserve=cfg.head_reserve; if (head) hw.head_fixed=cfg.head_blocks;
    if (hw_node_available(&hw) || hw_finalize(&hw)) return -1;
    printf("measuring %s as %s at burst %u (cached profiles reused; new short profiles cover widths 1..4)\n",name,head?"head":"node",burst); fflush(stdout);
    if (selection_sweep_local(&hw,burst,head,&s,stdout,dir,refresh,NULL) || !s.count) { fprintf(stderr,"fronk backends: measuring failed\n"); return -1; }
    if (selection_hybrid_calibrate(&hw,head,&s,dir,refresh,stdout)) {
        fprintf(stderr,"fronk backends: hybrid calibration failed; no unmeasured SSD capacity will be used. Check free space and retry --measure.\n"); return -1;
    }
    const fr_candidate *c=&s.candidate[0];
    printf("measured %s:",c->name);
    for (unsigned b=1;b<=FR_SWEEP && b<=c->capacity;++b) if (c->measured&(1u<<(b-1))) printf(" %u block%s %.0f us%s",b,b>1?"s":"",c->ns[b-1]/1e3,b<FR_SWEEP && b<c->capacity?",":"");
    printf("\nsaved in %s/bench; the head reads it at DISCOVERY without measuring\n",dir);
    return 0;
}
/* Fills names[] with the compiled backends; returns the count and the recommended index. */
static unsigned list(const fr_hwinfo *hw,const char *current,const char **names,int *recommended) {
    const char *pick=selection_recommend(hw); unsigned n=0; *recommended=-1;
    printf("Backends in this fronk build, for %s %s (%s):\n",hw->os,hw->arch,hw->cpu);
    for (unsigned i=0;fr_backend_at(i);++i) {
        const fr_backend *b=fr_backend_at(i); fr_hwinfo probe; int rc=b->probe(&probe);
        int mine=pick && !strcmp(pick,b->name);
        if (mine) *recommended=(int)n;
        names[n]=b->name;
        printf("  %u) %-10s %-26s%s%s\n",n+1,b->name,
            rc==FR_OK?"runs here":rc==FR_UNSUPPORTED?"not supported here":"probe failed",
            mine?"  <- recommended":"",current[0] && !strcmp(current,b->name)?"  (saved)":"");
        ++n;
    }
    printf("  a) auto       measure the compatible ones and keep the fastest%s\n",current[0]?"":"  (saved)");
    return n;
}
static int test(const char *name,const fr_hwinfo *hw,int timing) {
    const fr_backend *b=fr_backend_find(name);
    if (!b) { fprintf(stderr,"fronk backends: %s is not in this build (built: %s)\n",name,fr_backends_compiled()); return 1; }
    fr_hwinfo probe; int rc=b->probe(&probe);
    if (rc) { printf("%s: probe %s\n",name,rc==FR_UNSUPPORTED?"says this machine is not supported":"failed"); return 1; }
    rc=b->self_test("tests/vectors");
    printf("%s: SELFTEST %s (rc %d)\n",name,rc?"FAILED":"passed",rc);
    if (rc || !timing) return rc?1:0;
    const char *pick=selection_recommend(hw), *order[2]={name,pick && strcmp(pick,name)?pick:NULL};
    for (unsigned i=0;i<2 && order[i];++i) {
        fr_bench_report r;
        if (fr_bench_measure_backend(fr_backend_find(order[i]),1,200,4,2,1,NULL,NULL,NULL,&r)) { printf("%s: timing failed\n",order[i]); return 1; }
        printf("%s: 1 block, burst 2: p50 %.0f us, p99 %.0f us per call (200 calls)%s\n",order[i],
            r.percentile[0][1][FR_METAL_TOTAL][0]/1e3,r.percentile[0][1][FR_METAL_TOTAL][1]/1e3,i?" (recommended, for comparison)":"");
    }
    return 0;
}
int setup_main(int argc,char **argv) {
    setup_cache_identity(argv[0]);
    const char *config="fronk.conf",*set=NULL,*tested=NULL; int choose=0,only=0,skip=0,refresh=0,timing=1,list_only=0; unsigned burst=2;
    for (int i=2;i<argc;++i) {
        if (!strcmp(argv[i],"--config") && i+1<argc) config=argv[++i];
        else if (!strcmp(argv[i],"--set") && i+1<argc) set=argv[++i];
        else if (!strcmp(argv[i],"--test") && i+1<argc) tested=argv[++i];
        else if (!strcmp(argv[i],"--check") && i+1<argc) { tested=argv[++i]; timing=0; }
        else if (!strcmp(argv[i],"--list")) list_only=1;
        else if (!strcmp(argv[i],"--choose")) choose=1;
        else if (!strcmp(argv[i],"--measure")) only=1;
        else if (!strcmp(argv[i],"--no-measure")) skip=1;
        else if (!strcmp(argv[i],"--refresh")) refresh=1;
        else if (!strcmp(argv[i],"--burst") && i+1<argc && argv[i+1][0]>='1' && argv[i+1][0]<='4' && !argv[i+1][1]) burst=(unsigned)(argv[++i][0]-'0');
        else {
            fputs("Usage: fronk backends [--config PATH]           list; mark the recommended one\n"
                  "       fronk backends --choose [--config PATH]  pick one from the list and save it\n"
                  "       fronk backends --set NAME|auto [--config PATH]\n"
                  "       fronk backends --measure [--burst N] [--refresh] [--config PATH]\n"
                  "       fronk backends --test NAME               SELFTEST and a short timing\n"
                  "       fronk backends --check NAME              SELFTEST only, no calibration\n"
                  "       fronk backends --list                    list without a configuration\n"
                  "--choose and --set also measure, unless --no-measure: a node at --burst\n"
                  "(default 2); a head config uses its configured burst and memory reserve.\n",stderr);
            return 2;
        }
    }
    fr_hwinfo hw;
    if (hw_probe(&hw)) { fprintf(stderr,"fronk backends: hardware probe failed\n"); return 1; }
    if (tested) return test(tested,&hw,timing);
    if (list_only) { const char *names[FR_BACKENDS_MAX]; int recommended; list(&hw,"",names,&recommended); return 0; }
    char dir[256],current[64];
    if (state_dir(config,dir,sizeof(dir))) return 1;
    setup_saved_backend(dir,current,sizeof(current));
    if (only) return measure(config,dir,current[0]?current:"auto",burst,refresh)?1:0;
    if (set) return save(dir,set) || (!skip && measure(config,dir,set,burst,refresh))?1:0;
    const char *names[FR_BACKENDS_MAX]; int recommended;
    unsigned n=list(&hw,current,names,&recommended);
    if (!choose) { printf("Save one with: ./fronk backends --choose%s%s\n",strcmp(config,"fronk.conf")?" --config ":"",strcmp(config,"fronk.conf")?config:""); return 0; }
    printf("Choice [%s]: ",recommended>=0?names[recommended]:"a"); fflush(stdout);
    char line[64];
    if (!fgets(line,sizeof(line),stdin)) return 1;
    line[strcspn(line,"\r\n")]=0;
    const char *name=NULL;
    if (!line[0]) name=recommended>=0?names[recommended]:"auto";
    else if (!strcmp(line,"a") || !strcmp(line,"auto")) name="auto";
    else { char *end; long k=strtol(line,&end,10);
        if (!*end && k>=1 && (unsigned long)k<=n) name=names[k-1];
        else for (unsigned i=0;i<n;++i) if (!strcmp(line,names[i])) name=names[i]; }
    if (!name) { fprintf(stderr,"fronk backends: no backend %s\n",line); return 1; }
    return save(dir,name) || (!skip && measure(config,dir,name,burst,refresh))?1:0;
}
