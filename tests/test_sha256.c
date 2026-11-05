/* test_sha256.c - known SHA-256 and RFC 4231 HMAC vectors, not self-roundtrips.
 * Sources: https://www.rfc-editor.org/rfc/rfc6234.txt (section 8.5),
 * https://www.rfc-editor.org/rfc/rfc4231.txt (all seven SHA-256 cases).
 */
#include "test.h"
#include "../src/sha256.h"
#include <string.h>

static int matches(const unsigned char *digest, const char *hex) {
    const char *digits="0123456789abcdef";
    for (size_t i=0; i<strlen(hex); ++i)
        if (digits[(digest[i/2] >> (i%2?0:4)) & 15]!=hex[i]) return 0;
    return 1;
}
static int hash(const void *data, size_t len, const char *hex) {
    unsigned char out[32];
    return sha256(data,len,out)==0 && matches(out,hex);
}
static int mac(const void *key, size_t key_len, const void *data, size_t len, const char *hex) {
    unsigned char out[32];
    return hmac_sha256(key,key_len,data,len,out)==0 && matches(out,hex);
}
int main(void) {
    test_begin("sha256");
    check(hash(NULL,0,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),"SHA empty input");
    check(hash("abc",3,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),"SHA RFC abc");
    const char *long_msg="abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    const char *long_hash="248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
    check(hash(long_msg,strlen(long_msg),long_hash),"SHA RFC two-block padding");
    sha256_ctx ctx; unsigned char out[32]; int ok=1;
    for (size_t split=0; split<=strlen(long_msg); ++split) {
        sha256_init(&ctx);
        if (sha256_update(&ctx,long_msg,split) || sha256_update(&ctx,NULL,0) ||
            sha256_update(&ctx,long_msg+split,strlen(long_msg)-split) ||
            sha256_final(&ctx,out) || !matches(out,long_hash)) ok=0;
    }
    check(ok,"every streaming split including empty updates");
    char chunk[1000]; memset(chunk,'a',sizeof(chunk)); sha256_init(&ctx); ok=1;
    for (int i=0; i<1000; ++i) if (sha256_update(&ctx,chunk,sizeof(chunk))) ok=0;
    check(ok && !sha256_final(&ctx,out) && matches(out,"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),"SHA RFC million a");
    check(sha256_final(&ctx,out)<0 && sha256_update(&ctx,"x",1)<0,"finalization consumes state");
    sha256_init(&ctx);
    check(sha256_update(&ctx,NULL,1)<0 && sha256_final(&ctx,out)<0,"invalid update poisons state");
    sha256_init(&ctx); ctx.bytes=UINT64_MAX/8;
    check(sha256_update(&ctx,"x",1)<0 && sha256_final(&ctx,out)<0,"bit length overflow fails before reading data");
    unsigned char key[131], data[50];
    memset(key,0x0b,20);
    check(mac(key,20,"Hi There",8,"b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"),"HMAC RFC 4231 case 1");
    check(mac("Jefe",4,"what do ya want for nothing?",28,"5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"),"HMAC RFC 4231 case 2");
    memset(key,0xaa,20); memset(data,0xdd,50);
    check(mac(key,20,data,50,"773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"),"HMAC RFC 4231 case 3");
    for (int i=0; i<25; ++i) key[i]=(unsigned char)(i+1);
    memset(data,0xcd,50);
    check(mac(key,25,data,50,"82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"),"HMAC RFC 4231 case 4");
    memset(key,0x0c,20);
    check(mac(key,20,"Test With Truncation",20,"a3b6167473100ee06e0c796c2955552b"),"HMAC RFC 4231 case 5 published prefix");
    memset(key,0xaa,sizeof(key));
    const char *six="Test Using Larger Than Block-Size Key - Hash Key First";
    check(mac(key,sizeof(key),six,strlen(six),"60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"),"HMAC RFC 4231 case 6 long key");
    const char *seven="This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm.";
    check(mac(key,sizeof(key),seven,strlen(seven),"9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"),"HMAC RFC 4231 case 7 long key and message");
    check(hmac_sha256(NULL,1,"x",1,out)<0 && hmac_sha256("x",1,NULL,1,out)<0,"invalid HMAC buffers rejected");
    memset(out,0xa5,sizeof(out)); secure_clear(out,sizeof(out));
    unsigned char zero[32]={0}; check(!memcmp(out,zero,32),"explicit clearing");
    return test_end();
}
