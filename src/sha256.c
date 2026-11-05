/* sha256.c - byte-oriented SHA-256 and HMAC-SHA256 for authentication/manifests.
 * Implements the compression and padding equations from RFC 6234. Explicit
 * big-endian loads avoid alignment and host-endian assumptions. No allocation.
 */
#include "sha256.h"
#include <string.h>

void secure_clear(void *data, size_t len) {
    volatile unsigned char *p=data;
    while (len--) *p++=0;
}
static uint32_t rotate(uint32_t x, unsigned n) { return (x>>n)|(x<<(32-n)); }
static void compress(sha256_ctx *ctx, const unsigned char *p) {
    static const uint32_t k[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t w[64];
    for (unsigned i=0; i<16; ++i)
        w[i]=(uint32_t)p[4*i]<<24 | (uint32_t)p[4*i+1]<<16 | (uint32_t)p[4*i+2]<<8 | p[4*i+3];
    for (unsigned i=16; i<64; ++i) {
        uint32_t x=w[i-15], y=w[i-2];
        w[i]=w[i-16]+(rotate(x,7)^rotate(x,18)^(x>>3))+w[i-7]+(rotate(y,17)^rotate(y,19)^(y>>10));
    }
    uint32_t a=ctx->h[0],b=ctx->h[1],c=ctx->h[2],d=ctx->h[3];
    uint32_t e=ctx->h[4],f=ctx->h[5],g=ctx->h[6],h=ctx->h[7];
    for (unsigned i=0; i<64; ++i) {
        uint32_t t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i];
        uint32_t t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    ctx->h[0]+=a; ctx->h[1]+=b; ctx->h[2]+=c; ctx->h[3]+=d;
    ctx->h[4]+=e; ctx->h[5]+=f; ctx->h[6]+=g; ctx->h[7]+=h;
    secure_clear(w,sizeof(w));
}
void sha256_init(sha256_ctx *ctx) {
    *ctx=(sha256_ctx){.h={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                              0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
}
int sha256_update(sha256_ctx *ctx, const void *data, size_t len) {
    if (ctx->closed || (!data && len) || len>UINT64_MAX/8-ctx->bytes) {
        ctx->closed=1; return -1;
    }
    ctx->bytes+=len;
    const unsigned char *p=data;
    while (len) {
        size_t n=sizeof(ctx->block)-ctx->used;
        if (n>len) n=len;
        memcpy(ctx->block+ctx->used,p,n); ctx->used+=n; p+=n; len-=n;
        if (ctx->used==64) { compress(ctx,ctx->block); ctx->used=0; }
    }
    return 0;
}
int sha256_final(sha256_ctx *ctx, unsigned char out[SHA256_BYTES]) {
    int rc=-1;
    if (!ctx->closed && out) {
        uint64_t bits=ctx->bytes*8;
        ctx->block[ctx->used++]=0x80;
        if (ctx->used>56) {
            memset(ctx->block+ctx->used,0,64-ctx->used); compress(ctx,ctx->block); ctx->used=0;
        }
        memset(ctx->block+ctx->used,0,56-ctx->used);
        for (unsigned i=0; i<8; ++i) ctx->block[63-i]=(unsigned char)(bits>>(8*i));
        compress(ctx,ctx->block);
        for (unsigned i=0; i<32; ++i) out[i]=(unsigned char)(ctx->h[i/4]>>(24-8*(i%4)));
        rc=0;
    }
    secure_clear(ctx,sizeof(*ctx)); ctx->closed=1;
    return rc;
}
int sha256(const void *data, size_t len, unsigned char out[SHA256_BYTES]) {
    sha256_ctx ctx; sha256_init(&ctx);
    int rc=sha256_update(&ctx,data,len);
    if (sha256_final(&ctx,out)<0) rc=-1;
    return rc;
}
int hmac_sha256(const void *key, size_t key_len, const void *data, size_t len,
                unsigned char out[SHA256_BYTES]) {
    if ((!key && key_len) || (!data && len) || !out) return -1;
    unsigned char pad[64]={0}, inner[32]; sha256_ctx ctx;
    int rc=-1;
    if (key_len>64) { if (sha256(key,key_len,pad)<0) goto done; }
    else if (key_len) memcpy(pad,key,key_len);
    for (unsigned i=0; i<64; ++i) pad[i]^=0x36;
    sha256_init(&ctx);
    if (sha256_update(&ctx,pad,64) || sha256_update(&ctx,data,len) || sha256_final(&ctx,inner)) goto done;
    for (unsigned i=0; i<64; ++i) pad[i]^=0x36^0x5c;
    sha256_init(&ctx);
    if (sha256_update(&ctx,pad,64) || sha256_update(&ctx,inner,32) || sha256_final(&ctx,out)) goto done;
    rc=0;
done:
    secure_clear(&ctx,sizeof(ctx)); secure_clear(pad,sizeof(pad)); secure_clear(inner,sizeof(inner));
    return rc;
}
