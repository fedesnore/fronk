/* test_redis.c - drive an actual redis-cli process over one persistent connection.
 * Challenge-response needs a round trip; separate CLI invocations cannot share
 * a nonce. Pipes let the test compute the proof between interactive commands.
 */
#include "test.h"
#include "../src/auth.h"
#include "../src/net.h"
#include <errno.h>
#include <signal.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int line(int fd, char *out, size_t cap) {
    struct timespec start, now;
    if (clock_gettime(CLOCK_MONOTONIC,&start)<0) return 0;
    size_t used=0;
    while (used+1<cap) {
        if (clock_gettime(CLOCK_MONOTONIC,&now)<0) return 0;
        long elapsed=(now.tv_sec-start.tv_sec)*1000+(now.tv_nsec-start.tv_nsec)/1000000;
        if (elapsed>=3000 || net_wait(fd,0,(int)(3000-elapsed))!=1) return 0;
        char c; ssize_t n=read(fd,&c,1);
        if (n<0 && errno==EINTR) continue;
        if (n!=1) return 0;
        if (c=='\n') { out[used]='\0'; return 1; }
        if (c!='\r') out[used++]=c;
    }
    return 0;
}
static int send_command(int fd, const char *text) {
    size_t len=strlen(text); ssize_t n;
    do { n=write(fd,text,len); } while (n<0 && errno==EINTR);
    return n==(ssize_t)len;
}
static int answer(int input, int output, const char *request, const char *expected) {
    char text[1024]={0};
    int ok=send_command(input,request);
    /* redis-cli adds display separators around some error replies. */
    for (int i=0; ok && i<4; ++i) {
        ok=line(output,text,sizeof(text));
        if (text[0]) break;
    }
    ok=ok && strstr(text,expected);
    if (!ok) fprintf(stderr,"redis-cli: expected %s, received %s\n",expected,text);
    return ok;
}
int main(int argc, char **argv) {
    test_begin("redis");
    if (argc!=2 && argc!=3) { check(0,"node port and optional host required"); return test_end(); }
    if (signal(SIGPIPE,SIG_IGN)==SIG_ERR) { check(0,"ignore pipe shutdown signal"); return test_end(); }
    int input[2], output[2];
    if (pipe(input)<0) { check(0,"input pipe"); return test_end(); }
    if (pipe(output)<0) { close(input[0]); close(input[1]); check(0,"output pipe"); return test_end(); }
    fflush(NULL); pid_t pid=fork();
    if (!pid) {
        if (dup2(input[0],STDIN_FILENO)<0 || dup2(output[1],STDOUT_FILENO)<0) _exit(126);
        close(input[0]); close(input[1]); close(output[0]); close(output[1]);
        execlp("redis-cli","redis-cli","--raw","-h",argc==3?argv[2]:"127.0.0.1","-p",argv[1],(char *)NULL);
        _exit(127);
    }
    close(input[0]); close(output[1]);
    if (pid<0) { close(input[1]); close(output[0]); check(0,"fork redis-cli"); return test_end(); }
    check(answer(input[1],output[0],"PING\n","PONG"),"redis-cli PING before authentication");
    check(answer(input[1],output[0],"INFO\n","NOAUTH"),"redis-cli INFO refused before AUTH");
    check(answer(input[1],output[0],"HW\n","NOAUTH"),"redis-cli HW refused before AUTH");
    char hex[128], proof[65], request[128];
    int ok=send_command(input[1],"NONCE\n");
    for (int i=0; ok && i<4; ++i) { ok=line(output[0],hex,sizeof(hex)); if (hex[0]) break; }
    ok=ok && !auth_response("fronk-local-test",hex,strlen(hex),proof);
    check(ok,"redis-cli NONCE is usable hex");
    if (!ok) goto done;
    snprintf(request,sizeof(request),"AUTH %s\n",proof);
    check(answer(input[1],output[0],request,"OK"),"redis-cli AUTH accepts proof on same connection");
    ok=send_command(input[1],"HW\nPING\n"); unsigned hw_fields=0; int hw_pong=0;
    for (int lines=0; ok && lines<64; ++lines) {
        char text[1024];
        if (!line(output[0],text,sizeof(text))) { ok=0; break; }
        if (!strcmp(text,"PONG")) { hw_pong=1; break; }
        if (!strncmp(text,"cpu_model:",10)) hw_fields|=1;
        if (!strncmp(text,"mem_cap_bytes:",14)) hw_fields|=2;
        if (!strncmp(text,"backend:",8)) hw_fields|=4;
    }
    check(ok && hw_pong && hw_fields==7,"redis-cli receives authenticated HW text");
    ok=send_command(input[1],"INFO\nPING\n");
    unsigned sections=0; int authenticated=0, pong=0;
    const char *names[]={"# server","# backend","# memory","# session","# dataplane","# system"};
    for (int lines=0; ok && lines<192; ++lines) {
        char text[1024];
        if (!line(output[0],text,sizeof(text))) { ok=0; break; }
        if (!strcmp(text,"PONG")) { pong=1; break; }
        for (unsigned i=0; i<6; ++i) if (!strcmp(text,names[i])) sections|=1u<<i;
        if (!strcmp(text,"auth:hmac-sha256")) authenticated=1;
    }
    check(ok && pong && sections==63 && authenticated,"redis-cli reads all authenticated INFO sections");
    check(answer(input[1],output[0],request,"WRONGPASS"),"redis-cli replay rejected");
    check(answer(input[1],output[0],"INFO\n","NOAUTH"),"failed reauthentication revokes INFO");
done:
    close(input[1]); close(output[0]);
    int status=0, exited=0;
    for (int i=0; i<300; ++i) {
        pid_t got=waitpid(pid,&status,WNOHANG);
        if (got==pid) { exited=1; break; }
        if (got<0) break;
        struct timespec pause={0,10000000}; (void)nanosleep(&pause,NULL);
    }
    if (!exited) { kill(pid,SIGKILL); waitpid(pid,&status,0); }
    check(exited && WIFEXITED(status) && !WEXITSTATUS(status),"redis-cli exits cleanly");
    return test_end();
}
