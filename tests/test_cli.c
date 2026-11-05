/* test_cli.c - exercise the real CLI against a node and small reply fixtures.
 * Server errors, malformed replies and EOF must propagate as failed commands.
 */
#include "test.h"
#include "server.h"
#include "../src/net.h"
#include <errno.h>
#include <sys/socket.h>

static int run(const char *binary, char *const args[], int expected, const char *text) {
    FILE *log=tmpfile();
    if (!log) return 0;
    fflush(NULL);
    pid_t child=fork();
    if (!child) {
        if (dup2(fileno(log),STDOUT_FILENO)<0 || dup2(fileno(log),STDERR_FILENO)<0) _exit(126);
        execv(binary,args); _exit(127);
    }
    int status;
    int ok=child>0 && waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==expected;
    rewind(log); char output[8192]; size_t n=fread(output,1,sizeof(output)-1,log); output[n]='\0';
    ok=ok && !ferror(log) && strstr(output,text) && !strstr(output,"AddressSanitizer") && !strstr(output,"runtime error:");
    if (!ok) fputs(output,stderr);
    if (fclose(log)!=0) ok=0;
    return ok;
}
static int mock_config(const char *cli, const char *reply, int expected, const char *text, const char *config) {
    int listener=net_tcp_listen("127.0.0.1",0);
    if (listener<0) return 0;
    int port=net_port(listener);
    if (port<0) { close(listener); return 0; }
    fflush(NULL);
    pid_t child=fork();
    if (!child) {
        if (net_wait(listener,0,3000)!=1) _exit(2);
        int fd=net_tcp_accept(listener); close(listener);
        if (fd<0 || net_wait(fd,0,3000)!=1) _exit(2);
        char request[256];
        if (recv(fd,request,sizeof(request),0)<=0) _exit(2);
        if (!reply) {
            struct timespec hold={3,500000000};
            (void)nanosleep(&hold,NULL); close(fd); _exit(0);
        }
        size_t len=strlen(reply);
        if (len && net_send(fd,reply,1)!=1) _exit(2);
        server_pause();
        if (len>1 && net_send(fd,reply+1,len-1)!=(ssize_t)(len-1)) _exit(2);
        close(fd); _exit(0);
    }
    close(listener);
    if (child<0) return 0;
    char number[16]; snprintf(number,sizeof(number),"%d",port);
    char *args[]={"fronk-cli","-p",number,"PING",NULL};
    char *auth_args[]={"fronk-cli","-p",number,"--config",(char *)config,"PING",NULL};
    int ok=run(cli,config?auth_args:args,expected,text), status;
    if (waitpid(child,&status,0)!=child || !WIFEXITED(status) || WEXITSTATUS(status)) ok=0;
    return ok;
}
static int mock(const char *cli, const char *reply, int expected, const char *text) {
    return mock_config(cli,reply,expected,text,NULL);
}
int main(int argc, char **argv) {
    test_begin("cli");
    if (argc!=3) { check(0,"node and cli binaries required"); return test_end(); }
    const char *cli=argv[2];
    char *help[]={"fronk-cli","--help",NULL};
    char *version[]={"fronk-cli","--version",NULL};
    char *bad_port[]={"fronk-cli","-p","65536",NULL};
    char *missing[]={"fronk-cli","-h",NULL};
    check(run(cli,help,0,"Usage:"),"CLI help");
    check(run(cli,version,0,"fronk-cli " FR_VERSION),"CLI version");
    check(run(cli,bad_port,2,"Usage:"),"invalid port");
    check(run(cli,missing,2,"Usage:"),"missing address");
    test_server server;
    int started=server_start(&server,argv[1]); check(started,"start real node");
    if (!started) return test_end();
    char port[16]; snprintf(port,sizeof(port),"%u",server.port);
    char *ping[]={"fronk-cli","-h","127.0.0.1","-p",port,"PING",NULL};
    char *default_ping[]={"fronk-cli","-p",port,NULL};
    char *info[]={"fronk-cli","-p",port,"INFO","system",NULL};
    char *auth_info[]={"fronk-cli","-p",port,"--config",server.config,"INFO","system",NULL};
    char *unknown[]={"fronk-cli","-p",port,"--config",server.config,"NOPE",NULL};
    char *spaced[]={"fronk-cli","-p",port,"PING","one argument with spaces",NULL};
    char *bad_address[]={"fronk-cli","-h","not-an-ip","-p",port,"PING",NULL};
    check(run(cli,ping,0,"PONG\n"),"PING through real CLI");
    check(run(cli,default_ping,0,"PONG\n"),"default command is PING");
    check(run(cli,info,1,"NOAUTH"),"INFO without authentication refused");
    check(run(cli,auth_info,0,"event_loop:"),"config-driven NONCE/AUTH then INFO on the same connection");
    check(run(cli,unknown,1,"(error) ERR unknown command"),"server error is nonzero");
    check(run(cli,spaced,1,"wrong number of arguments"),"arguments encoded as RESP bulk strings");
    check(run(cli,bad_address,1,"connect:"),"invalid address diagnosed");
    char wrong_path[]="/tmp/fronk-cli-password.XXXXXX"; int wrong_fd=mkstemp(wrong_path), ok=wrong_fd>=0;
    if (wrong_fd>=0) {
        const char *config="password incorrect-fixture-password\n";
        ok=write(wrong_fd,config,strlen(config))==(ssize_t)strlen(config);
        if (close(wrong_fd)<0) ok=0;
        char *wrong[]={"fronk-cli","-p",port,"--config",wrong_path,"INFO",NULL};
        if (ok) ok=run(cli,wrong,1,"authentication failed");
        if (unlink(wrong_path)<0) ok=0;
    }
    check(ok,"wrong config password fails before sending the requested command");
    char *missing_config[]={"fronk-cli","--config","/nonexistent/fronk.conf",NULL};
    char *missing_path[]={"fronk-cli","--config",NULL};
    check(run(cli,missing_config,1,"cannot open config"),"missing authentication config diagnosed");
    check(run(cli,missing_path,2,"Usage:"),"missing config path rejected");
    check(mock_config(cli,"+OK\r\n",1,"invalid authentication challenge",server.config),"CLI rejects a non-bulk nonce");
    check(mock_config(cli,"$-1\r\n",1,"invalid authentication challenge",server.config),"CLI rejects a null nonce");
    check(mock_config(cli,"$32\r\ngggggggggggggggggggggggggggggggg\r\n",1,"invalid authentication challenge",server.config),"CLI rejects nonhex nonce bytes");
    check(server_stop(&server),"clean node shutdown");
    check(run(cli,ping,1,"connect:"),"connection refusal is nonzero");
    check(mock(cli,"+OK\r\n",0,"OK\n"),"fragmented response");
    check(mock(cli,":42\r\n",0,"42\n"),"integer response");
    check(mock(cli,"*2\r\n$-1\r\n$3\r\nabc\r\n",0,"(nil)\nabc\n"),"array/null/bulk response");
    check(mock(cli,"?",1,"invalid RESP"),"malformed response");
    check(mock(cli,"unframed text\r\n",1,"invalid RESP"),"inline syntax is not a valid reply");
    check(mock(cli,"$4\r\nab",1,"incomplete response"),"truncated response");
    check(mock(cli,"",1,"incomplete response"),"EOF before response");
    check(mock(cli,NULL,1,"timed out"),"unresponsive peer has a bounded wait");
    return test_end();
}
