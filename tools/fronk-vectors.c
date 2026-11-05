/* Explicit generation into a fresh directory; ordinary tests only read goldens. */
#include "../src/vectors.h"
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    int rc=FR_INVALID;
    if (argc==3 && !strcmp(argv[1],"generate")) rc=fr_vectors_generate(argv[2]);
    else if (argc==4 && !strcmp(argv[1],"check")) {
        const fr_backend *b=fr_backend_find(argv[2]);
        if (b) rc=fr_vectors_check(b,argv[3],b==&fr_ref?&fr_reference_tolerance:&fr_backend_tolerance,stdout);
    } else fputs("Usage: fronk-vectors generate DIR | check BACKEND DIR\n",stderr);
    if (rc) fprintf(stderr,"fronk-vectors: failed (%d)\n",rc);
    return rc?1:0;
}
