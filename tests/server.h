/* server.h - disposable real-node fixture for control-plane integration tests.
 * Ephemeral ports avoid races between test suites. Child diagnostics are kept
 * until shutdown, including sanitizer errors that must fail the parent test.
 */
#ifndef FR_TEST_SERVER_H
#define FR_TEST_SERVER_H
#include <signal.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
typedef struct { pid_t pid; unsigned port; FILE *log; char config[64], state[64]; } test_server;

static inline void server_pause(void) {
    struct timespec delay={0,10000000};
    (void)nanosleep(&delay,NULL);
}
static inline int server_stop(test_server *s) {
    int ok=1, status=0;
    if (s->pid>0) {
        if (kill(s->pid,SIGTERM)<0) ok=0;
        int i;
        for (i=0; i<300; ++i) {
            pid_t got=waitpid(s->pid,&status,WNOHANG);
            if (got==s->pid) break;
            if (got<0) { ok=0; break; }
            server_pause();
        }
        if (i==300) { kill(s->pid,SIGKILL); waitpid(s->pid,&status,0); ok=0; }
        if (!WIFEXITED(status) || WEXITSTATUS(status)!=0) ok=0;
        s->pid=-1;
    }
    if (s->log) {
        rewind(s->log); char line[1024];
        while (fgets(line,sizeof(line),s->log))
            if (strstr(line,"AddressSanitizer") || strstr(line,"runtime error:") || strstr(line,"LeakSanitizer")) ok=0;
        if (!ok) {
            rewind(s->log);
            while (fgets(line,sizeof(line),s->log)) fputs(line,stderr);
        }
        if (fclose(s->log)!=0) ok=0;
        s->log=NULL;
    }
    if (s->config[0] && unlink(s->config)<0) ok=0;
    s->config[0]='\0';
    if (s->state[0]) {
        char path[128]; snprintf(path,sizeof(path),"%s/node.id",s->state);
        if (unlink(path)<0 && errno!=ENOENT) ok=0;
        if (rmdir(s->state)<0) ok=0;
        s->state[0]=0;
    }
    return ok;
}
static inline int server_start_extra(test_server *s, const char *binary,const char *extra) {
    *s=(test_server){.pid=-1};
    strcpy(s->config,"/tmp/fronk-server.XXXXXX");
    int fd=mkstemp(s->config);
    if (fd<0) { s->config[0]='\0'; return 0; }
    strcpy(s->state,"/tmp/fronk-state.XXXXXX");
    if (!mkdtemp(s->state)) { s->state[0]=0; close(fd); server_stop(s); return 0; }
    const char *config="name test-node\nbind 127.0.0.1\ncontrol-port 0\ndata-port 0\ndiscovery-port 0\npassword fixture-secret\n";
    int ok=write(fd,config,strlen(config))==(ssize_t)strlen(config);
    if (extra && write(fd,extra,strlen(extra))!=(ssize_t)strlen(extra)) ok=0;
    char state[128]; int n=snprintf(state,sizeof(state),"state-dir %s\n",s->state);
    if (write(fd,state,(size_t)n)!=n) ok=0;
    if (close(fd)<0) ok=0;
    s->log=tmpfile();
    if (!ok || !s->log) { server_stop(s); return 0; }
    fflush(NULL);
    s->pid=fork();
    if (s->pid==0) {
        if (dup2(fileno(s->log),STDOUT_FILENO)<0 || dup2(fileno(s->log),STDERR_FILENO)<0) _exit(126);
        execl(binary,binary,"node","--config",s->config,"--debug",(char *)NULL);
        _exit(127);
    }
    if (s->pid<0) { server_stop(s); return 0; }
    for (int i=0; i<300; ++i) {
        rewind(s->log); char line[512];
        while (fgets(line,sizeof(line),s->log)) {
            char *port=strstr(line,"control_port=");
            if (port && sscanf(port,"control_port=%u",&s->port)==1 && s->port) return 1;
        }
        int status;
        if (waitpid(s->pid,&status,WNOHANG)==s->pid) { s->pid=-1; break; }
        server_pause();
    }
    rewind(s->log); char line[512];
    while (fgets(line,sizeof(line),s->log)) fputs(line,stderr);
    server_stop(s);
    return 0;
}
static inline int server_start(test_server *s,const char *binary) { return server_start_extra(s,binary,NULL); }
#endif
