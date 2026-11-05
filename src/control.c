/* control.c - shared bounded RESP client and NONCE/AUTH handshake.
 * Extracted from fronk-cli so the head and CLI enforce identical framing,
 * authentication and timeout rules. Caller-provided buffers avoid allocation.
 */
#include "control.h"
#include "net.h"
#include "auth.h"
#include "sha256.h"
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
static int64_t now_ms(void) { int64_t ns=net_now_ns(); return ns<0?-1:ns/1000000; }
static int wait_until(int fd, int writing, int64_t deadline) {
    int64_t now=now_ms();
    if (now<0) return -1;
    if (now>=deadline) { errno=ETIMEDOUT; return -1; }
    int rc=net_wait(fd,writing,(int)(deadline-now));
    if (!rc) errno=ETIMEDOUT;
    return rc==1?0:-1;
}
const char *control_exchange(int fd, const char *request, size_t len, char response[RESP_MAX_MESSAGE],
                            resp_value *values, int64_t deadline) {
    size_t sent=0, received=0;
    const char *error;
    while (sent<len) {
        if (wait_until(fd,1,deadline)<0) { error="request write failed or timed out"; goto failed; }
        ssize_t n=net_send(fd,request+sent,len-sent);
        if (n<0 && (errno==EAGAIN || errno==EWOULDBLOCK)) continue;
        if (n<=0) { error="request write failed"; goto failed; }
        sent+=(size_t)n;
    }
    for (;;) {
        if (received==RESP_MAX_MESSAGE) { error="response exceeds protocol limit"; goto failed; }
        if (wait_until(fd,0,deadline)<0) { error="response read failed or timed out"; goto failed; }
        ssize_t n=recv(fd,response+received,RESP_MAX_MESSAGE-received,0);
        if (n<0 && (errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR)) continue;
        if (n<0) { error="response read failed"; goto failed; }
        if (!n) { error="server closed with incomplete response"; goto failed; }
        received+=(size_t)n;
        if (!response[0] || !strchr("+-:$*",response[0])) { error="invalid RESP response"; goto failed; }
        ssize_t parsed=resp_parse(response,received,values,RESP_MAX_VALUES);
        if (parsed<0) { error="invalid RESP response"; goto failed; }
        if (!parsed) continue;
        if ((size_t)parsed!=received) { error="unexpected extra response"; goto failed; }
        return NULL;
    }
failed:
    /* A failed transport must not look like a server error left over from a
     * previous exchange, or expose a partially parsed response to diagnostics. */
    values[0]=(resp_value){0};
    return error;
}
const char *control_authenticate(int fd, const char *password, char response[RESP_MAX_MESSAGE],
                                resp_value *values, int64_t deadline) {
    const char *error=control_exchange(fd,"*1\r\n$5\r\nNONCE\r\n",15,response,values,deadline);
    char proof[65]={0}, request[128]={0};
    if (error) goto done;
    if (values[0].type!=RESP_BULK || values[0].is_null ||
        auth_response(password,values[0].data,values[0].len,proof)<0) {
        error="invalid authentication challenge"; goto done;
    }
    resp_buf out; resp_init(&out,request,sizeof(request));
    resp_array(&out,2); resp_bulk(&out,"AUTH",4); resp_bulk(&out,proof,64);
    if (out.error) { error="cannot encode AUTH"; goto done; }
    error=control_exchange(fd,request,out.len,response,values,deadline);
    if (!error && (values[0].type!=RESP_STRING || values[0].len!=2 || memcmp(values[0].data,"OK",2)))
        error="authentication failed";
done:
    secure_clear(proof,sizeof(proof)); secure_clear(request,sizeof(request));
    return error;
}
