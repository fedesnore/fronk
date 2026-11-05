/* test_auth_entropy.c - replace only the OS entropy call at link time.
 * A partial write followed by failure must never expose a usable challenge.
 * There is no entropy override or test mode in the production executable.
 */
#include "test.h"
#include "../src/auth.h"
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#ifdef __APPLE__
int fr_test_getentropy(void *data, size_t len) {
#else
ssize_t fr_test_getrandom(void *data, size_t len, unsigned flags) {
    (void)flags;
#endif
    memset(data,0xa5,len); errno=EIO; return -1;
}
int main(void) {
    test_begin("auth_entropy");
    auth_state s={.pending=1,.authenticated=1}, zero={0}; char hex[33];
    memset(s.nonce,0xff,sizeof(s.nonce)); memset(hex,'x',sizeof(hex));
    check(auth_nonce(&s,hex)<0,"OS entropy failure propagated");
    check(!memcmp(&s,&zero,sizeof(s)) && hex[0]=='\0',"entropy failure clears challenge and authorization");
    return test_end();
}
