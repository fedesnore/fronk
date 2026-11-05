/* sha256.h - streaming SHA-256 and HMAC-SHA256, with no external dependency.
 * Byte-oriented input only. Init before update/final; final consumes the state.
 * Functions return 0 or -1 for invalid input, reuse or a 64-bit bit-count overflow.
 */
#ifndef FR_SHA256_H
#define FR_SHA256_H
#include <stddef.h>
#include <stdint.h>
#define SHA256_BYTES 32
typedef struct {
    uint32_t h[8];
    uint64_t bytes;
    unsigned char block[64];
    size_t used;
    int closed;
} sha256_ctx;
void sha256_init(sha256_ctx *ctx);
int sha256_update(sha256_ctx *ctx, const void *data, size_t len);
int sha256_final(sha256_ctx *ctx, unsigned char out[SHA256_BYTES]);
int sha256(const void *data, size_t len, unsigned char out[SHA256_BYTES]);
int hmac_sha256(const void *key, size_t key_len, const void *data, size_t len,
                unsigned char out[SHA256_BYTES]);
/* Volatile stores keep credential cleanup from being optimized away. */
void secure_clear(void *data, size_t len);
#endif
