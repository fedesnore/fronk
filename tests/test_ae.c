/* test_ae.c - real kernel readiness, updates, EOF, removal and stale tags. */
#include "test.h"
#include "../src/ae.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

int main(void) {
    test_begin("ae");
    int loop = ae_open(), pair[2];
    check(loop >= 0, "create event queue");
    if (loop < 0) return test_end();
    check((fcntl(loop, F_GETFD) & FD_CLOEXEC) != 0, "queue is close-on-exec");
    int made = socketpair(AF_UNIX, SOCK_STREAM, 0, pair);
    check(made == 0, "create socket fixture");
    if (made < 0) { close(loop); return test_end(); }
    ae_event events[AE_MAX_EVENTS];
    check(ae_set(loop,pair[0],0,AE_READ,41)==0, "register readable interest");
    check(ae_wait(loop,events,AE_MAX_EVENTS,0)==0, "idle poll returns no events");
    check(write(pair[1],"x",1)==1, "make socket readable");
    int n = ae_wait(loop,events,AE_MAX_EVENTS,1000);
    check(n>0 && events[0].tag==41 && (events[0].mask&AE_READ), "read event has caller tag");
    uint64_t old_tag = n>0 ? events[0].tag : 0;
    check(ae_set(loop,pair[0],AE_READ,AE_READ,42)==0, "replace generation tag");
    n = ae_wait(loop,events,AE_MAX_EVENTS,1000);
    check(n>0 && events[0].tag==42 && old_tag!=events[0].tag, "caller can reject prior-generation events");
    char c;
    check(read(pair[0],&c,1)==1 && c=='x', "drain readable data");
    check(ae_wait(loop,events,AE_MAX_EVENTS,0)==0, "drained socket is not readable");
    check(ae_set(loop,pair[0],AE_READ,AE_WRITE,42)==0, "change interest to writes");
    n = ae_wait(loop,events,1,1000);
    check(n==1 && (events[0].mask&AE_WRITE), "writable event with caller-sized batch");
    check(ae_set(loop,pair[0],AE_WRITE,0,42)==0, "remove interest");
    check(ae_wait(loop,events,AE_MAX_EVENTS,0)==0, "removed fd stays silent");
    check(ae_set(loop,pair[0],0,AE_READ,43)==0, "re-register fd");
    check(close(pair[1])==0, "close peer");
    n = ae_wait(loop,events,AE_MAX_EVENTS,1000);
    check(n>0 && (events[0].mask&AE_READ) && read(pair[0],&c,1)==0, "peer EOF wakes reader");
    check(ae_set(loop,pair[0],AE_READ,0,43)==0, "remove EOF descriptor");
    check(ae_set(loop,-1,0,AE_READ,0)==-1, "reject invalid descriptor");
    check(ae_set(loop,pair[0],0,8,0)==-1 && errno==EINVAL, "reject unknown mask");
    check(ae_wait(loop,events,0,0)==-1 && errno==EINVAL, "reject empty event array");
    check(ae_wait(loop,events,AE_MAX_EVENTS+1,0)==-1, "reject oversized batch");
    check(ae_wait(loop,events,1,-2)==-1, "reject invalid timeout");
    check(close(pair[0])==0 && close(loop)==0, "release descriptors");
    return test_end();
}
