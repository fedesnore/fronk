/* auth.h - connection-owned, single-use HMAC challenge state.
 * Zero initialization/reset starts unauthenticated. NONCE replaces the pending
 * challenge without changing authorization; every AUTH consumes it and replaces
 * authorization with that attempt's result, including malformed attempts.
 */
#ifndef FR_AUTH_H
#define FR_AUTH_H
#include <stddef.h>
#define AUTH_NONCE_BYTES 16
#define AUTH_NONCE_HEX 32
#define AUTH_PROOF_HEX 64
typedef struct {
    unsigned char nonce[AUTH_NONCE_BYTES];
    int pending, authenticated;
} auth_state;
void auth_reset(auth_state *state);
/* 0 success, -1 entropy failure; failure discards pending challenge and access. */
int auth_nonce(auth_state *state, char hex[AUTH_NONCE_HEX+1]);
/* 1 accepted, 0 rejected. Empty passwords are never accepted. */
int auth_verify(auth_state *state, const char *password, const char *proof, size_t len);
/* Client helper: MAC the decoded nonce, output lowercase hex; 0 or -1. */
int auth_response(const char *password, const char *nonce_hex, size_t len,
                  char proof[AUTH_PROOF_HEX+1]);
#endif
