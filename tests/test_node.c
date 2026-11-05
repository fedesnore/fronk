/* test_node.c - real TCP framing, backpressure, isolation and shutdown.
 * These tests drive the sanitizer-built node executable on an ephemeral port.
 */
#include "test.h"
#include "server.h"
#include "../src/net.h"
#include "../src/resp.h"
#include "../src/auth.h"
#include "../src/probe.h"
#include <errno.h>
#include <sys/socket.h>

static char reply[RESP_MAX_MESSAGE];
static resp_value values[RESP_MAX_VALUES];
static int send_all(int fd, const char *data, size_t len) {
    while (len) {
        ssize_t n=net_send(fd,data,len);
        if (n<0 && (errno==EAGAIN || errno==EWOULDBLOCK)) { if (net_wait(fd,1,2000)!=1) return 0; continue; }
        if (n<=0) return 0;
        data+=n; len-=(size_t)n;
    }
    return 1;
}
static int read_exact(int fd, char *out, size_t len) {
    while (len) {
        if (net_wait(fd,0,2000)!=1) return 0;
        ssize_t n=recv(fd,out,len,0);
        if (n<0 && errno==EINTR) continue;
        if (n<=0) return 0;
        out+=n; len-=(size_t)n;
    }
    return 1;
}
static int response(int fd) {
    size_t used=0;
    while (used<sizeof(reply)-1) {
        if (net_wait(fd,0,2000)!=1) return 0;
        ssize_t n=recv(fd,reply+used,sizeof(reply)-1-used,0);
        if (n<0 && errno==EINTR) continue;
        if (n<=0) return 0;
        used+=(size_t)n; reply[used]='\0';
        ssize_t parsed=resp_parse(reply,used,values,RESP_MAX_VALUES);
        if (parsed) return parsed==(ssize_t)used;
    }
    return 0;
}
static int command(int fd, const char *s, resp_type type, const char *contains) {
    return send_all(fd,s,strlen(s)) && response(fd) && values[0].type==type && strstr(reply,contains);
}
static int closed(int fd) {
    char c;
    return net_wait(fd,0,2000)==1 && recv(fd,&c,1,0)==0;
}
static int nonce(int fd, char hex[33]) {
    if (!command(fd,"NONCE\r\n",RESP_BULK,"$32\r\n") || values[0].len!=32) return 0;
    memcpy(hex,values[0].data,32); hex[32]='\0'; return 1;
}
static int prove(int fd, const char *proof, int accepted) {
    char request[128]; snprintf(request,sizeof(request),"AUTH %s\r\n",proof);
    return command(fd,request,accepted?RESP_STRING:RESP_ERROR,accepted?"+OK\r\n":"WRONGPASS");
}
static int login(int fd) {
    char hex[33], proof[65];
    return nonce(fd,hex) && !auth_response("fixture-secret",hex,32,proof) && prove(fd,proof,1);
}
static int drain_info(int fd, unsigned count) {
    size_t used=0; unsigned got=0;
    while (got<count) {
        if (net_wait(fd,0,2000)!=1) return 0;
        ssize_t n=recv(fd,reply+used,sizeof(reply)-used,0);
        if (n<=0) return 0;
        used+=(size_t)n;
        for (;;) {
            ssize_t parsed=resp_parse(reply,used,values,RESP_MAX_VALUES);
            if (parsed<0) return 0;
            if (!parsed) break;
            if (values[0].type!=RESP_BULK) return 0;
            ++got; used-=(size_t)parsed; memmove(reply,reply+parsed,used);
        }
    }
    return got==count && used==0;
}
int main(int argc, char **argv) {
    test_begin("node");
    if (argc!=2) { check(0,"node binary required"); return test_end(); }
    test_server server;
    int started=server_start(&server,argv[1]); check(started,"start real node on ephemeral loopback port");
    if (!started) return test_end();
    int fd=net_tcp_connect("127.0.0.1",server.port,1000);
    check(fd>=0,"connect client");
    if (fd<0) { check(server_stop(&server),"stop node"); return test_end(); }
    check(net_wait(fd,0,0)==0,"no unsolicited greeting");
    const char *blocked[]={"BENCH-PROFILE","EXPERT-ORDER","INFO","INFO server","INFO wrong extra","HW","BENCH","PUT x 1 y",
        "BLOCKS","SELFTEST","SESSION START","SESSION STOP","DEBUG TRACE on","HELLO 2","LINK 1 9999 0","NOPE"};
    int denied=1; char request[256], hex[33], replacement[33], proof[65], stale[65];
    for (size_t i=0; i<sizeof(blocked)/sizeof(blocked[0]); ++i) {
        snprintf(request,sizeof(request),"%s\r\n",blocked[i]);
        if (!command(fd,request,RESP_ERROR,"-NOAUTH")) denied=0;
    }
    check(denied,"only PING/NONCE/AUTH allowed before authentication");
    check(prove(fd,"fixture-secret",0),"plaintext password and AUTH without NONCE rejected");
    check(nonce(fd,hex) && !auth_response("fixture-secret",hex,32,proof),"explicit nonce and client proof");
    int other=net_tcp_connect("127.0.0.1",server.port,1000);
    check(other>=0 && nonce(other,replacement) && strcmp(hex,replacement) && prove(other,proof,0),"challenge cannot be used on another connection");
    if (other>=0) close(other);
    check(prove(fd,proof,1),"failed replay elsewhere does not consume this connection's nonce");
    check(command(fd,"HW extra\r\n",RESP_ERROR,"wrong number"),"HW rejects extra arguments");
    check(command(fd,"EXPERT-ORDER\r\n",RESP_STRING,"legacy"),"unranked worker does not advertise stable partial reuse");
    check(command(fd,"EXPERT-ORDER extra\r\n",RESP_ERROR,"ERR"),"expert capability query validates arity");
    check(command(fd,"BENCH-PROFILE\r\n",RESP_STRING,"quick-widths-v1"),"new worker advertises bounded width calibration without running it");
    check(command(fd,"BENCH-PROFILE extra\r\n",RESP_ERROR,"ERR"),"planning capability query validates arity");
    check(command(fd,"HW\r\n",RESP_BULK,"mem_cap_bytes:") && strstr(reply,"cpu_model:") && strstr(reply,"metal_working_set_bytes:"),"authenticated HW returns real capacity and identity");
    check(command(fd,"INFO system\r\n",RESP_BULK,"hw_samples:1") && strstr(reply,"cpu_model:"),"HW snapshot cached for INFO inspection");
    check(command(fd,"BENCH GET\r\n",RESP_ERROR,"no measurement"),"unmeasured BENCH cannot fabricate a profile");
    check(command(fd,"BENCH invalid\r\n",RESP_ERROR,"wrong number"),"BENCH validates command arguments");
    check(command(fd,"INFO backend\r\n",RESP_BULK,"backends_compiled:ref ref-q8") &&
          strstr(reply,(strstr(reply,"backend:ref\r\n") || (strstr(reply,"backend:metal\r\n") && strstr(reply,"ref ref-q8 metal")))?"backend_ready:1\r\n":"backend_ready:0\r\n") && strstr(reply,"reference_ready:1\r\n") &&
          strstr(reply,"golden_vectors_version:1\r\n"),"INFO distinguishes scalar reference from selected accelerator readiness");
    check(strstr(reply,"bench_status:unmeasured\r\n") && !strstr(reply,"bw_eff_gbs:"),"INFO omits unmeasured compute fields");
    check(command(fd,"BENCH LIST 1\r\n",RESP_BULK,"FRCHOICE1 1") && strstr(reply,"ref ") && strstr(reply,"ref-q8 "),"compiled eligible backends and individual capacities exposed");
    check(command(fd,"BACKEND 1\r\n",RESP_ERROR,"complete backend sweep"),"node refuses configuration before every eligible backend is measured");
    check(command(fd,"BENCH missing 1 1\r\n",RESP_ERROR,"ineligible") &&
          command(fd,"BENCH ref 6 1\r\n",RESP_ERROR,"ineligible"),"uncompiled candidate and sweep beyond five blocks refused");
    check(command(fd,"BENCH ref 1 2\r\n",RESP_ERROR,"differs from LIST"),"burst cannot change halfway through selection");
    int whole=command(fd,"INFO all\r\n",RESP_BULK,"# backend"); char *policy=strstr(reply,"\r\nbackend:");
    check(whole && policy && !strstr(policy+1,"\r\nbackend:"),"hardware section cannot shadow the selected backend in INFO all");
    check(command(fd,"HW\r\n",RESP_BULK,"mem_available_bytes:") && command(fd,"INFO system\r\n",RESP_BULK,"hw_samples:2"),"HW refreshes capacity on each request");
    check(prove(fd,proof,0) && command(fd,"INFO\r\n",RESP_ERROR,"NOAUTH"),"second AUTH needs NONCE and revokes access on failure");
    check(nonce(fd,hex) && !auth_response("wrong-password",hex,32,proof) && prove(fd,proof,0),"wrong password rejected");
    check(!auth_response("fixture-secret",hex,32,proof) && prove(fd,proof,0),"failed AUTH consumed the nonce");
    const char *malformed[]={"AUTH\r\n","AUTH one two\r\n","*2\r\n$4\r\nAUTH\r\n:1\r\n",
        "*2\r\n$4\r\nAUTH\r\n$-1\r\n","*2\r\n$4\r\nAUTH\r\n*0\r\n"};
    int consumed=1;
    for (size_t i=0; i<sizeof(malformed)/sizeof(malformed[0]); ++i) {
        if (!nonce(fd,hex) || auth_response("fixture-secret",hex,32,proof) ||
            !command(fd,malformed[i],RESP_ERROR,"WRONGPASS") || !prove(fd,proof,0)) consumed=0;
    }
    check(consumed,"AUTH arity/null/numeric/nested errors all consume the challenge");
    check(nonce(fd,hex) && !auth_response("fixture-secret",hex,32,stale) && nonce(fd,replacement) &&
          strcmp(hex,replacement) && prove(fd,stale,0),"new NONCE invalidates its predecessor");
    check(nonce(fd,hex) && !auth_response("fixture-secret",hex,32,proof),"fresh proof for ordered pipeline");
    snprintf(request,sizeof(request),"AUTH %s\r\nPING\r\nAUTH %s\r\nINFO\r\n",proof,proof);
    const char *expected="+OK\r\n+PONG\r\n-WRONGPASS invalid or consumed challenge\r\n-NOAUTH authentication required\r\n";
    char pipeline_reply[256];
    check(send_all(fd,request,strlen(request)) && read_exact(fd,pipeline_reply,strlen(expected)) &&
          !memcmp(pipeline_reply,expected,strlen(expected)),"pipeline applies authentication changes in command order");
    check(login(fd),"fresh NONCE/AUTH restores access");
    const char *prefix="*1\r\n$4\r\nPI";
    check(send_all(fd,prefix,strlen(prefix)) && net_wait(fd,0,0)==0,"partial frame waits");
    check(command(fd,"NG\r\n",RESP_STRING,"PONG"),"complete fragmented RESP request");
    check(command(fd,"ping\n",RESP_STRING,"PONG"),"case-insensitive nc request");
    check(send_all(fd,"PING\r\nINFO server\r\n",19),"pipeline two requests");
    char pong[7];
    check(read_exact(fd,pong,7) && !memcmp(pong,"+PONG\r\n",7),"pipeline first response");
    check(response(fd) && values[0].type==RESP_BULK && strstr(reply,"# server\r\n") &&
          !strstr(reply,"# backend"),"filtered INFO response follows in order");
    check(command(fd,"INFO\r\n",RESP_BULK,"# system\r\n") &&
          strstr(reply,"# backend\r\n") && strstr(reply,"# memory\r\n") &&
          strstr(reply,"# session\r\n") && strstr(reply,"# dataplane\r\n"),"all INFO sections");
    check(strstr(reply,"auth:hmac-sha256") && strstr(reply,"event_loop:") &&
          strstr(reply,"connected_clients:1") && strstr(reply,"authenticated_clients:1") &&
          strstr(reply,"auth_failures:") && strstr(reply,"noauth_commands:") &&
          !strstr(reply,"fixture-secret") && !strstr(reply,hex) && !strstr(reply,proof),"authentication telemetry without secrets or proofs");
    check(command(fd,"INFO unknown\r\n",RESP_ERROR,"ERR"),"unknown section");
    check(command(fd,"PING extra\r\n",RESP_ERROR,"ERR"),"wrong arity");
    check(command(fd,"NOPE\r\n",RESP_ERROR,"ERR"),"unknown command");
    check(command(fd,"SESSION\r\n",RESP_ERROR,"ERR") && command(fd,"SESSION TEST\r\n",RESP_ERROR,"ERR"),"short SESSION commands rejected without reading absent arguments");
    check(command(fd,":5\r\n",RESP_ERROR,"ERR"),"non-array command rejected");
    check(command(fd,"*1\r\n$-1\r\n",RESP_ERROR,"ERR"),"null command rejected");
    check(command(fd,"PING\r\n",RESP_STRING,"PONG"),"semantic errors leave connection usable");

    unsigned data_port=0;
    int udp=net_udp_bind("127.0.0.1",0,0), spoof=net_udp_bind("127.0.0.1",0,0);
    check(command(fd,"INFO dataplane\r\n",RESP_BULK,"data_port:") &&
          sscanf(strstr(reply,"data_port:"),"data_port:%u",&data_port)==1 && data_port && udp>=0 && spoof>=0,"obtain actual UDP port and probe endpoints");
    if (udp>=0 && spoof>=0 && data_port) {
        struct sockaddr_in target, from; unsigned char packet[FR_PROBE_BYTES];
        fr_probe probe={.session=12345,.step=1,.node=255}, parsed;
        size_t size=probe_make(packet,&probe);
        check(!net_address("127.0.0.1",data_port,&target) && net_udp_send(udp,packet,size,&target)==(ssize_t)size &&
              net_wait(udp,0,30)==0,"authenticated TCP alone does not authorize UDP probes");
        snprintf(request,sizeof(request),"LINK 12345 %d 0\r\n",net_port(udp));
        check(command(fd,request,RESP_ARRAY,"test-node"),"LINK binds session, UDP port and authenticated peer");
        check(net_udp_send(udp,packet,size,&target)==(ssize_t)size && net_wait(udp,0,1000)==1 &&
              net_udp_recv(udp,packet,sizeof(packet),&from)==64 && !probe_read(packet,64,&parsed) &&
              parsed.reply && !parsed.node && parsed.session==12345,"registered UDP PING returns exact PONG");
        check(command(fd,"LINK PATH lo0 other 0 127.0.0.1\r\n",RESP_STRING,"OK") &&
              command(fd,"INFO system\r\n",RESP_BULK,"head_interface:lo0\r\n") && strstr(reply,"head_link_speed_bps:0\r\n"),"authenticated owner publishes head-side path metadata");
        check(command(fd,"LINK PATH lo0 bogus 0 127.0.0.1\r\n",RESP_ERROR,"invalid LINK") &&
              command(fd,"LINK PATH en5 ethernet 1000000000 192.0.2.1\r\n",RESP_ERROR,"invalid LINK") &&
              command(fd,"INFO system\r\n",RESP_BULK,"head_interface:lo0\r\n"),"malformed path or wrong authenticated source cannot replace cached metadata");
        check(command(fd,"LINK PATH lo0 wifi 0 127.0.0.1\r\n",RESP_STRING,"OK") &&
              command(fd,"INFO system\r\n",RESP_BULK,"head_link_type:wifi\r\n"),"INFO retains the head Wi-Fi transport policy");
        check(command(fd,"LINK RELEASE\r\n",RESP_STRING,"OK"),"release probe ownership before measuring another path");
        size=probe_make(packet,&probe);
        check(net_udp_send(udp,packet,size,&target)==(ssize_t)size && net_wait(udp,0,30)==0,"released path cannot send authenticated probes");
        check(command(fd,request,RESP_ARRAY,"test-node") && command(fd,"INFO system\r\n",RESP_BULK,"link_active:1\r\n") &&
              !strstr(reply,"head_interface:"),"new LINK clears old path metadata until the head reports its selection");
        size=probe_make(packet,&probe);
        check(net_udp_send(spoof,packet,size,&target)==(ssize_t)size && net_wait(spoof,0,30)==0,"another source port cannot use the link");
        probe.session=54321; size=probe_make(packet,&probe);
        check(net_udp_send(udp,packet,size,&target)==(ssize_t)size && net_wait(udp,0,30)==0,"stale or wrong session ignored");
        check(command(fd,"LINK REPORT 0 1000 0 1 2 3 0\r\n",RESP_ERROR,"invalid LINK"),"invalid histogram cannot enter INFO");
        check(login(fd),"reauthenticate existing control connection");
        probe.session=12345; size=probe_make(packet,&probe);
        check(net_udp_send(udp,packet,size,&target)==(ssize_t)size && net_wait(udp,0,30)==0,"reauthentication invalidates old UDP registration");
    }
    if (udp>=0) close(udp);
    if (spoof>=0) close(spoof);

    int held[63], held_count=0;
    for (; held_count<63; ++held_count) {
        held[held_count]=net_tcp_connect("127.0.0.1",server.port,1000);
        if (held[held_count]<0) break;
        if (!command(held[held_count],"PING\r\n",RESP_STRING,"PONG")) { close(held[held_count]); break; }
    }
    check(held_count==63,"fill connection budget without losing existing clients");
    int excess=net_tcp_connect("127.0.0.1",server.port,1000);
    check(excess>=0 && closed(excess),"excess connection rejected without allocation");
    if (excess>=0) close(excess);
    check(command(fd,"INFO server\r\n",RESP_BULK,"connected_clients:64\r\n") &&
          strstr(reply,"rejected_connections:1\r\n"),"connection cap and rejection exposed in INFO");
    for (int i=0; i<held_count; ++i) close(held[i]);

    int slow=net_tcp_connect("127.0.0.1",server.port,1000), small=4096;
    check(slow>=0,"connect slow client");
    if (slow>=0) {
        check(command(slow,"INFO\r\n",RESP_ERROR,"NOAUTH") && login(slow),"new/reused connection starts unauthenticated");
        check(setsockopt(slow,SOL_SOCKET,SO_RCVBUF,&small,sizeof(small))==0,"limit slow-client receive buffer");
        char pipeline[512*6];
        for (size_t i=0; i<sizeof(pipeline); i+=6) memcpy(pipeline+i,"INFO\r\n",6);
        check(send_all(slow,pipeline,sizeof(pipeline)),"queue replies larger than server output buffer");
        server_pause();
        check(command(fd,"PING\r\n",RESP_STRING,"PONG"),"slow client cannot block other clients");
        check(drain_info(slow,512),"all pipelined replies survive backpressure");
        close(slow);
    }
    check(send_all(fd,"PING\r\n",6) && shutdown(fd,SHUT_WR)==0 && response(fd) &&
          values[0].type==RESP_STRING && closed(fd),"half-close flushes pending response then EOF");
    close(fd);
    fd=net_tcp_connect("127.0.0.1",server.port,1000);
    check(fd>=0 && command(fd,"?bad\r\n",RESP_ERROR,"Protocol") && closed(fd),"malformed frame errors then closes");
    if (fd>=0) close(fd);
    fd=net_tcp_connect("127.0.0.1",server.port,1000);
    check(fd>=0 && command(fd,"$999999\r\n",RESP_ERROR,"Protocol") && closed(fd),"declared oversize rejected without payload");
    if (fd>=0) close(fd);
    fd=net_tcp_connect("127.0.0.1",server.port,1000);
    check(fd>=0 && send_all(fd,prefix,strlen(prefix)) && shutdown(fd,SHUT_WR)==0 &&
          response(fd) && values[0].type==RESP_ERROR && closed(fd),"truncated request at EOF");
    if (fd>=0) close(fd);
    fd=net_tcp_connect("127.0.0.1",server.port,1000);
    static char full[RESP_MAX_MESSAGE]; memset(full,'X',sizeof(full));
    check(fd>=0 && send_all(fd,full,sizeof(full)) && response(fd) &&
          values[0].type==RESP_ERROR && closed(fd),"full incomplete input buffer rejected");
    if (fd>=0) close(fd);
    check(server_stop(&server),"clean SIGTERM shutdown without sanitizer errors");
    return test_end();
}
