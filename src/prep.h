/* prep.h - lossless native units straight from the local Q4 GGUF.
 * A unit is a deterministic gather of GGUF byte ranges (no requantization):
 * per layer, 512 routed slices (Q4_K gate rows, Q4_K up rows, MXFP4 down
 * columns) then the Q8_0 shared slice, exactly the native.h layout.
 * Units are streamed to a sink, so the head never needs all 62 GiB on disk.
 */
#ifndef FR_PREP_H
#define FR_PREP_H
#include "native.h"
typedef struct {
    int fd; const unsigned char *map; uint64_t size,mtime;
    uint64_t gate[FR_LAYERS],up[FR_LAYERS],down[FR_LAYERS],sgate[FR_LAYERS],sup[FR_LAYERS],sdown[FR_LAYERS];
} fr_gguf;
typedef int (*fr_sink)(void *ctx,const void *data,size_t n);
int fr_gguf_open(const char *path,fr_gguf *g); /* validates names, types and shapes */
void fr_gguf_close(fr_gguf *g);
/* Copy one routed/shared slice from a validated GGUF. Returns its exact byte
 * count, or 0 for invalid coordinates/capacity. Useful for format audits. */
size_t fr_slice_copy(const fr_gguf *g,unsigned block,unsigned layer,unsigned expert,void *out,size_t capacity);
/* Emits exactly FR_NATIVE_BLOCK_BYTES in order; stops at the first sink error. */
int fr_unit_stream(const fr_gguf *g,unsigned block,fr_sink sink,void *ctx);
/* Per-unit SHA-256 hex, cached in DIR/manifest.txt keyed by GGUF size and mtime. */
int fr_manifest(const fr_gguf *g,const char *dir,char hex[FR_BLOCKS][65]);
/* Sample expert-sized disk reads in the unit-cache directory, bytes/second. */
double fr_cache_read_rate(const char *directory);
/* DIR/unit-NN.bin + .sha256, written atomically; skipped if already matching. */
int fr_unit_write(const fr_gguf *g,unsigned block,const char *dir,const char *hex);
/* unit-NN.ok binds a verified SHA-256 to the file's size, mtime and inode, so
 * a later open trusts an unchanged file without rehashing 3.3 GB. */
int fr_units_stamp(const char *directory,unsigned block,const char *hex);
int fr_units_cached(const char *directory,unsigned block,char hex[65]); /* 0 if stamped and unchanged */
#endif
