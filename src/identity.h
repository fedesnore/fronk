/* identity.h - persistent machine identity, independent of names and addresses.
 * Empty directory selects ~/.fronk. Explicit directories isolate test nodes.
 * Existing IDs are never overwritten; concurrent first starts share one ID.
 */
#ifndef FR_IDENTITY_H
#define FR_IDENTITY_H
#include <stdint.h>
#include <string.h>
int identity_load(const char *directory, uint64_t *out);
/* Shared text parsing must not require identity-file or authentication setup. */
static inline int identity_parse(const char *s, uint64_t *out) {
    if (strlen(s)!=16) return -1;
    uint64_t n=0;
    for (unsigned i=0; i<16; ++i) {
        unsigned d=s[i]>='0' && s[i]<='9'?(unsigned)(s[i]-'0'):s[i]>='a' && s[i]<='f'?(unsigned)(s[i]-'a'+10):16;
        if (d==16) return -1;
        n=(n<<4)|d;
    }
    if (!n) return -1;
    *out=n; return 0;
}
#endif
