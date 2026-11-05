/* auth.c - single-use challenges backed by the operating system CSPRNG.
 * Nonces belong to a connection, never to a peer address. Every AUTH attempt
 * consumes the pending nonce before validation. Digest comparison visits all
 * bytes; neither credentials nor proofs are logged or exposed in INFO.
 */
#include "auth.h"
#include "sha256.h"
#include <errno.h>
#include <string.h>
#include <sys/random.h>

static void encode(const unsigned char *bytes, size_t len, char *hex) {
    const char *digits="0123456789abcdef";
    for (size_t i=0; i<len; ++i) { hex[2*i]=digits[bytes[i]>>4]; hex[2*i+1]=digits[bytes[i]&15]; }
    hex[2*len]='\0';
}
static int digit(unsigned char c) {
    if (c>='0' && c<='9') return c-'0';
    if (c>='a' && c<='f') return c-'a'+10;
    if (c>='A' && c<='F') return c-'A'+10;
    return -1;
}
static int decode(const char *hex, size_t len, unsigned char *out, size_t bytes) {
    if (!hex || len!=2*bytes) return -1;
    for (size_t i=0; i<bytes; ++i) {
        int a=digit((unsigned char)hex[2*i]), b=digit((unsigned char)hex[2*i+1]);
        if (a<0 || b<0) return -1;
        out[i]=(unsigned char)(16*a+b);
    }
    return 0;
}
void auth_reset(auth_state *state) { secure_clear(state,sizeof(*state)); }
int auth_nonce(auth_state *state, char hex[AUTH_NONCE_HEX+1]) {
    state->pending=0; secure_clear(state->nonce,sizeof(state->nonce)); hex[0]='\0';
#ifdef __APPLE__
    int rc;
    do { rc=getentropy(state->nonce,sizeof(state->nonce)); } while (rc<0 && errno==EINTR);
    if (rc<0) { auth_reset(state); return -1; }
#else
    size_t used=0;
    while (used<sizeof(state->nonce)) {
        ssize_t n=getrandom(state->nonce+used,sizeof(state->nonce)-used,0);
        if (n<0 && errno==EINTR) continue;
        if (n<=0) { if (!n) errno=EIO; auth_reset(state); return -1; }
        used+=(size_t)n;
    }
#endif
    state->pending=1; encode(state->nonce,sizeof(state->nonce),hex);
    return 0;
}
int auth_verify(auth_state *state, const char *password, const char *proof, size_t len) {
    unsigned char nonce[16], expected[32], received[32];
    memcpy(nonce,state->nonce,sizeof(nonce)); int pending=state->pending;
    auth_reset(state);
    if (pending && password && *password && !decode(proof,len,received,sizeof(received)) &&
        !hmac_sha256(password,strlen(password),nonce,sizeof(nonce),expected)) {
        volatile unsigned difference=0;
        for (size_t i=0; i<sizeof(expected); ++i) difference|=(unsigned)(expected[i]^received[i]);
        state->authenticated=difference==0;
    }
    secure_clear(nonce,sizeof(nonce)); secure_clear(expected,sizeof(expected)); secure_clear(received,sizeof(received));
    return state->authenticated;
}
int auth_response(const char *password, const char *nonce_hex, size_t len,
                  char proof[AUTH_PROOF_HEX+1]) {
    unsigned char nonce[16], digest[32]; int rc=-1;
    proof[0]='\0';
    if (password && *password && !decode(nonce_hex,len,nonce,sizeof(nonce)) &&
        !hmac_sha256(password,strlen(password),nonce,sizeof(nonce),digest)) {
        encode(digest,sizeof(digest),proof); rc=0;
    }
    secure_clear(nonce,sizeof(nonce)); secure_clear(digest,sizeof(digest));
    return rc;
}
