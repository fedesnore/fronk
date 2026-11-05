/* test_hw_failure.c - unavailable OS identity must not yield a usable snapshot.
 * Only uname is substituted; the production failure/transaction path is used.
 */
#include "test.h"
#include "../src/hw.h"
#include <sys/utsname.h>
#include <string.h>
#include <errno.h>
int fr_test_uname(struct utsname *out) { memset(out,0,sizeof(*out)); errno=EIO; return -1; }
int main(void) {
    test_begin("hw_failure");
    fr_hwinfo h, before; memset(&h,0x5a,sizeof(h)); before=h;
    check(hw_probe(&h)<0,"OS probe failure is reported");
    check(!memcmp(&h,&before,sizeof(h)),"failed probe cannot replace an existing snapshot with partial data");
    return test_end();
}
