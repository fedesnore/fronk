/* test_control.c - shared client exchange framing and timeout behavior.
 * The full NONCE/AUTH exchange is also exercised by test-cli and test-node.
 */
#include "test.h"
#include "../src/control.h"
#include "../src/net.h"
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int main(void) {
    test_begin("control"); int fd[2];
    check(socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"socket pair");
    char response[RESP_MAX_MESSAGE]; resp_value values[RESP_MAX_VALUES];
    check(write(fd[1],"+OK\r\n",5)==5 && !control_exchange(fd[0],"PING\r\n",6,response,values,net_now_ns()/1000000+1000) &&
          values[0].type==RESP_STRING && values[0].len==2,"framed reply accepted");
    check(write(fd[1],"bad\r\n",5)==5 && control_exchange(fd[0],"PING\r\n",6,response,values,net_now_ns()/1000000+1000),"unframed reply rejected");
    check(control_exchange(fd[0],"PING\r\n",6,response,values,net_now_ns()/1000000)!=NULL,"expired budget rejected");
    check(write(fd[1],"$-1\r\n",5)==5 && control_authenticate(fd[0],"fixture-secret",response,values,net_now_ns()/1000000+1000),"invalid authentication challenge rejected");
    check(write(fd[1],"-ERR storage\r\n",14)==14 &&
          !control_exchange(fd[0],"",0,response,values,net_now_ns()/1000000+1000) &&
          values[0].type==RESP_ERROR,"complete server error is available to the caller");
    check(control_exchange(fd[0],"",0,response,values,net_now_ns()/1000000) &&
          values[0].type!=RESP_ERROR && !values[0].data && !values[0].len,
          "timeout cannot reuse the previous server error");
    check(write(fd[1],"*2\r\n-ERR storage\r\n",18)==18 && shutdown(fd[1],SHUT_WR)==0 &&
          control_exchange(fd[0],"",0,response,values,net_now_ns()/1000000+1000) &&
          !values[0].data && !values[0].len,"closed partial response publishes no diagnostic payload");
    check(close(fd[0])==0 && close(fd[1])==0,"cleanup");
    return test_end();
}
