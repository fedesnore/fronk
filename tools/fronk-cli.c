/* fronk-cli.c - one-shot RESP2 control client, with plain text output.
 *
 * argv becomes a bulk-string array, so spaces inside arguments are preserved.
 * Connect and exchange each have a three-second budget, including NONCE/AUTH
 * when --config supplies a password. Passwords never enter argv or the wire.
 */
#include "../src/net.h"
#include "../src/resp.h"
#include "../src/config.h"
#include "../src/control.h"
#include "../src/sha256.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static void usage(FILE *out) {
    fputs("Usage: fronk-cli [-h IPv4-address] [-p port] [--config path] [command [args ...]]\n"
          "       fronk-cli --help | --version\n"
          "Defaults: 127.0.0.1:7700, command PING; one command per invocation.\n"
          "--config reads only the password; -h/-p still select the target.\n",out);
}
static int parse_port(const char *s) {
    unsigned port=0;
    if (!*s) return -1;
    for (; *s; ++s) {
        if (*s<'0' || *s>'9') return -1;
        port=port*10+(unsigned)(*s-'0');
        if (port>65535) return -1;
    }
    return port?(int)port:-1;
}
static int64_t now_ms(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC,&time)<0) return -1;
    return (int64_t)time.tv_sec*1000+time.tv_nsec/1000000;
}
static int show(const resp_value *values, size_t index, int *server_error) {
    const resp_value *v=&values[index];
    if (v->is_null) return puts("(nil)")<0?-1:0;
    if (v->type==RESP_ARRAY) {
        if (!v->len) return puts("(empty array)")<0?-1:0;
        size_t child=index+1;
        for (size_t i=0; i<v->len; ++i) {
            if (show(values,child,server_error)<0) return -1;
            child=values[child].next;
        }
        return 0;
    }
    if (v->type==RESP_INTEGER) return printf("%" PRId64 "\n",v->integer)<0?-1:0;
    FILE *out=v->type==RESP_ERROR?stderr:stdout;
    if (v->type==RESP_ERROR) { *server_error=1; if (fputs("(error) ",out)==EOF) return -1; }
    if (v->len && fwrite(v->data,1,v->len,out)!=v->len) return -1;
    if ((!v->len || v->data[v->len-1]!='\n') && fputc('\n',out)==EOF) return -1;
    return 0;
}
int main(int argc, char **argv) {
    if (argc==2 && !strcmp(argv[1],"--help")) { usage(stdout); return 0; }
    if (argc==2 && !strcmp(argv[1],"--version")) { printf("fronk-cli %s\n",FR_VERSION); return 0; }
    const char *ip="127.0.0.1", *config_path=NULL; int port=7700, i=1;
    for (; i<argc; ++i) {
        if (!strcmp(argv[i],"--")) { ++i; break; }
        if (argv[i][0]!='-') break;
        if (!strcmp(argv[i],"-h") && i+1<argc) ip=argv[++i];
        else if (!strcmp(argv[i],"--config") && !config_path && i+1<argc) config_path=argv[++i];
        else if (!strcmp(argv[i],"-p") && i+1<argc) {
            port=parse_port(argv[++i]);
            if (port<0) { usage(stderr); return 2; }
        } else { usage(stderr); return 2; }
    }
    int count=argc-i;
    int long_bench=(count==1 || count==4) && !strcasecmp(argv[i],"BENCH");
    if (count>16) { usage(stderr); return 2; }
    char request[RESP_MAX_MESSAGE], response[RESP_MAX_MESSAGE];
    resp_buf out; resp_init(&out,request,sizeof(request));
    resp_array(&out,count?count:1);
    if (!count) resp_bulk(&out,"PING",4);
    for (; i<argc; ++i) resp_bulk(&out,argv[i],strlen(argv[i]));
    if (out.error) { fputs("fronk-cli: command exceeds protocol limit\n",stderr); return 2; }
    fr_config config={0}; char config_error[512];
    if (config_path && (config_load(config_path,&config,config_error,sizeof(config_error))<0 || !config.password[0])) {
        fprintf(stderr,"fronk-cli: %s\n",config_error[0]?config_error:"config password is required");
        secure_clear(&config,sizeof(config)); return 1;
    }
    int fd=net_tcp_connect(ip,(unsigned)port,3000);
    if (fd<0) {
        fprintf(stderr,"fronk-cli: connect: %s\n",strerror(errno));
        secure_clear(&config,sizeof(config)); return 1;
    }
    int result=1; const char *error=NULL;
    int64_t start=now_ms();
    if (start<0) { error="cannot read monotonic clock"; goto done; }
    int64_t deadline=start+3000;
    resp_value values[RESP_MAX_VALUES];
    if (config_path && (error=control_authenticate(fd,config.password,response,values,deadline))) goto done;
    secure_clear(&config,sizeof(config));
    if (long_bench) {
        int64_t current=now_ms(); if (current<0) { error="cannot read monotonic clock"; goto done; }
        deadline=current+(count==1?300000:1800000); /* BENCH only; auth/network deadlines unchanged. */
    }
    if ((error=control_exchange(fd,request,out.len,response,values,deadline))) goto done;
    int server_error=0;
    if (show(values,0,&server_error)<0 || fflush(stdout)!=0 || fflush(stderr)!=0) error="cannot write response";
    else result=server_error?1:0;
done:
    secure_clear(&config,sizeof(config)); secure_clear(request,sizeof(request));
    secure_clear(response,sizeof(response));
    if (error) fprintf(stderr,"fronk-cli: %s\n",error);
    if (close(fd)<0) result=1;
    return result;
}
