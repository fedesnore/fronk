/* test_auth.c - challenge lifecycle, binary nonce semantics and replay rejection.
 * Fixed proof independently generated with OpenSSL using key fixture-secret
 * and the 16 input bytes 00..0f (not their textual hexadecimal representation).
 */
#include "test.h"
#include "../src/auth.h"
#include <ctype.h>
#include <string.h>
static const char *password="fixture-secret";
static const char *nonce="000102030405060708090a0b0c0d0e0f";
static const char *gold="f99cf9130923781a8866ba7f9a2b199c9f5dcdbce3a76a7a5232c6f3a16b2aa3";
static void fixed(auth_state *s) {
    auth_reset(s); for (int i=0; i<16; ++i) s->nonce[i]=(unsigned char)i;
    s->pending=1;
}
int main(void) {
    test_begin("auth"); auth_state s={0}, other={0}; char hex[33], old[33], proof[65];
    check(!auth_response(password,nonce,32,proof) && !strcmp(proof,gold),"HMAC over raw nonce matches independent fixture");
    check(!auth_verify(&s,password,gold,64) && !s.authenticated,"AUTH requires NONCE");
    fixed(&s);
    check(auth_verify(&s,password,gold,64) && s.authenticated && !s.pending,"valid proof consumes challenge");
    check(!auth_verify(&s,password,gold,64) && !s.authenticated,"replay rejected and authorization revoked");
    fixed(&s); s.authenticated=1;
    check(!auth_verify(&s,"wrong-password",gold,64) && !s.pending && !s.authenticated,"wrong password consumes and revokes");
    check(!auth_verify(&s,password,gold,64),"correct retry still needs a new nonce");
    fixed(&s);
    check(!auth_verify(&s,password,NULL,0) && !s.pending,"malformed attempt consumes challenge");
    fixed(&s); memcpy(proof,gold,65); proof[63]='z';
    check(!auth_verify(&s,password,proof,64) && !s.pending,"nonhex digest rejected");
    fixed(&s);
    check(!auth_verify(&s,password,gold,63) && !s.pending,"truncated digest rejected");
    fixed(&s);
    check(!auth_verify(&s,"",gold,64) && !s.pending,"empty password cannot authenticate");
    fixed(&s); for (int i=0; i<65; ++i) proof[i]=(char)toupper((unsigned char)gold[i]);
    check(auth_verify(&s,password,proof,64),"uppercase proof hex accepted");
    check(auth_response(password,"000102030405060708090a0b0c0d0ez",32,proof)<0 &&
          auth_response(password,nonce,31,proof)<0,"client rejects malformed challenges");
    check(!auth_nonce(&s,hex) && strlen(hex)==32 && s.pending && s.authenticated,"fresh OS nonce preserves authenticated session");
    memcpy(old,hex,33);
    check(!auth_response(password,hex,32,proof) && !auth_nonce(&s,hex) && strcmp(hex,old),"NONCE replaces pending challenge");
    check(!auth_verify(&s,password,proof,64),"superseded challenge proof rejected");
    check(!auth_nonce(&s,hex) && !auth_response(password,hex,32,proof) &&
          !auth_nonce(&other,old) && strcmp(hex,old) && !auth_verify(&other,password,proof,64),"proof bound to its connection");
    check(auth_verify(&s,password,proof,64),"other connection's failure does not consume original challenge");
    auth_reset(&s); auth_state zero={0};
    check(!memcmp(&s,&zero,sizeof(s)),"disconnect/reset wipes challenge and authorization");
    return test_end();
}
