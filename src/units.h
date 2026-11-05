#ifndef FR_UNITS_H
#define FR_UNITS_H
#include "native.h"
typedef struct { const fr_backend *backend; void *data[FR_BLOCKS]; uint32_t mask,paged; uint64_t cache_bytes; } fr_units;
/* Fixed local paths only: unit-NN.bin plus unit-NN.sha256 (64 hex + newline).
 * Whole-file SHA-256 is checked before any backend sees a mapping. */
int fr_units_open(fr_units *u,const fr_backend *backend,const char *directory,uint32_t mask,uint32_t paged);
int fr_units_open_cache(fr_units *u,const fr_backend *backend,const char *directory,uint32_t mask,uint32_t paged,uint64_t cache_bytes);
int fr_units_fixture(fr_units *u,const fr_backend *backend,unsigned block);
/* Synthetic units with real memory and valid encodings, for `fronk demo`. */
int fr_units_synthetic(fr_units *u,const fr_backend *backend,uint32_t mask);
int fr_units_synthetic_cache(fr_units *u,const fr_backend *backend,uint32_t mask,uint32_t paged,uint64_t bytes,const char *dir);
int fr_units_close(fr_units *u);
#endif
