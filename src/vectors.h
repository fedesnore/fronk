/* Deterministic synthetic fixtures for backend conformance; never on hot path. */
#ifndef FR_VECTORS_H
#define FR_VECTORS_H
#include "backend.h"
#include <stdio.h>
#define FR_VECTOR_COUNT 8
typedef struct {
    const char *name;
    uint32_t mask;
    fr_req req;
    unsigned char *data[FR_BLOCKS];
    char digest[65];
} fr_vector;
int fr_vector_open(unsigned index, fr_vector *v);
void fr_vector_close(fr_vector *v);
int fr_vectors_generate(const char *dir);
int fr_vectors_check(const fr_backend *backend, const char *dir,
                     const fr_tolerance *limit, FILE *report);
#endif
