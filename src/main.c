/* main.c - dispatch the node, head and cluster setup commands. */
#include "config.h"
#include "backend.h"
#include "node.h"
#include "sha256.h"
#include "connection.h"
#include "discovery.h"
#include "identity.h"
#include "planner.h"
#include "head.h"
#include "selection.h"
#include "setup.h"
#include "runtime.h"
#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <IOKit/pwr_mgt/IOPMLib.h>
#endif

/* Calibration can leave the control connection idle for minutes. Hold sleep
 * assertions for the process lifetime; the OS releases them at exit. A process
 * started remotely may begin in DarkWake, so also request system wake on AC
 * power and declare activity once. This does not keep a closed laptop on
 * battery awake. Normal installation starts locally and needs no SSH. */
static void stay_awake(const char *role) {
#ifdef __APPLE__
    IOPMAssertionID id; CFStringRef why=CFStringCreateWithFormat(NULL,NULL,CFSTR("fronk %s serving the cluster"),role);
    if (!why || IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep,kIOPMAssertionLevelOn,why,&id)!=kIOReturnSuccess ||
        IOPMAssertionCreateWithName(kIOPMAssertionTypePreventSystemSleep,kIOPMAssertionLevelOn,why,&id)!=kIOReturnSuccess)
        fprintf(stderr,"fronk: could not hold the sleep assertions; the machine may sleep during long BENCH sweeps\n");
    if (why) { IOPMAssertionID wake; (void)IOPMAssertionDeclareUserActivity(why,kIOPMUserActiveLocal,&wake); CFRelease(why); }
#else
    (void)role;
#endif
}

/* CONNECTION restarts node sessions. A second head, demo or pairing pass
 * using this installation must not reset the running cluster. */
static int lock_head(const char *state) {
    char path[300];
    snprintf(path,sizeof(path),"%s/head.lock",state);
    if (mkdir(state,0700) && errno!=EEXIST) { perror("fronk: state-dir"); return -1; }
    int fd=open(path,O_RDWR|O_CREAT|O_CLOEXEC,0600);
    if (fd<0) { perror("fronk: head.lock"); return -1; }
    if (flock(fd,LOCK_EX|LOCK_NB)<0) {
        fputs("fronk: another head or setup check is using this state directory; quit it first\n",stderr);
        close(fd); return -1;
    }
    // The console follows the log of the process that actually owns the lock.
    if (ftruncate(fd,0) || dprintf(fd,"%ld\n",(long)getpid())<0) {
        perror("fronk: record head pid"); close(fd); return -1;
    }
    return fd;
}

static void usage(FILE *out) {
    fputs("Usage: fronk head|node [--config path] [--debug]\n"
          "       fronk peers [--config path]   discover and authenticate available nodes\n"
          "       fronk demo [--passes N] [--config path]   plan and time the cluster on synthetic units\n"
          "       fronk backends [--choose | --set NAME|auto | --test NAME] [--config path]\n"
          "       fronk --help | --version\n"
          "Head: CONNECTION, DISCOVERY, PLAN, units, selftest, ds4 expert service, REPORT.\n", out);
}
int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(stdout);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
#ifdef FRONK_DEBUG
        const char *build = "debug";
#else
        const char *build = "release";
#endif
        printf("fronk %s (%s %s %s)\n",
               FR_VERSION, build, FR_BUILD_OS, FR_BUILD_ARCH);
        return 0;
    }
    if (fr_backends_validate(stderr)) return 1;
    if (argc >= 2 && !strcmp(argv[1],"backends")) { stay_awake("calibration"); return setup_main(argc,argv); }
    if (argc >= 2 && (!strcmp(argv[1],"node") || !strcmp(argv[1],"head") || !strcmp(argv[1],"demo") || !strcmp(argv[1],"peers"))) {
        const char *path="fronk.conf"; int debug=0, have_config=0; unsigned passes=0;
        for (int i=2; i<argc; ++i) {
            if (!strcmp(argv[i],"--debug") && !debug) debug=1;
            else if (!strcmp(argv[1],"demo") && !strcmp(argv[i],"--passes") && !passes && i+1<argc) {
                char *end; const char *arg=argv[++i];
                errno=0; unsigned long n=strtoul(arg,&end,10);
                if (*arg<'0' || *arg>'9' || *end || errno || !n || n>UINT_MAX) { usage(stderr); return 2; }
                passes=(unsigned)n;
            }
            else if (!strcmp(argv[i],"--config") && !have_config && i+1<argc) {
                path=argv[++i]; have_config=1;
            } else { usage(stderr); return 2; }
        }
        setup_cache_identity(argv[0]);
        fr_config config; char error[512];
        if (config_load(path,&config,error,sizeof(error))<0) { fprintf(stderr,"fronk: %s\n",error); return 1; }
        if (strcmp(argv[1],"node") && fr_head_defaults()) {
            perror("fronk: head defaults"); secure_clear(&config,sizeof(config)); return 1;
        }
        /* The config's "backend" line wins over the choice saved by "fronk backends". */
        if (!config.backend[0] && config.state_dir[0]) setup_saved_backend(config.state_dir,config.backend,sizeof(config.backend));
        if (!strcmp(config.backend,"auto")) config.backend[0]=0;
        if (selection_backend(config.backend,stderr)) { secure_clear(&config,sizeof(config)); return 1; }
        if (config.backend[0]) fprintf(stderr,"fronk: backend %s (chosen; the others are not measured)\n",config.backend);
        int lock=-1;
        if (strcmp(argv[1],"node") && config.state_dir[0] && (lock=lock_head(config.state_dir))<0) {
            secure_clear(&config,sizeof(config)); return 1;
        }
        setvbuf(stdout,NULL,_IOLBF,0);
        int result=1; stay_awake(argv[1]);
        if (!strcmp(argv[1],"node")) result=node_run(&config,debug);
        else if (!strcmp(argv[1],"demo")) result=head_demo_main(&config,passes);
        else if (!strcmp(argv[1],"peers")) {
            fr_connection *links=calloc(1,sizeof(*links));
            if (links) {
                result=connection_run(&config,links,stdout) != 0;
                if (!result) for (unsigned i=0;i<links->count;++i)
                    printf("PEER %s %016" PRIx64 " %s\n",links->links[i].identity.name,
                           links->links[i].identity.id,links->links[i].ip);
                if (connection_close(links)) result=1;
                free(links);
            }
        } else result=head_main(&config);
        if (lock>=0) close(lock);
        secure_clear(&config,sizeof(config));
        return result;
    }
    usage(stderr);
    return 2;
}
