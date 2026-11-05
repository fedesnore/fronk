/* Compiled backend registry and frozen conformance metrics.
 * Eligibility and measured choice remain separate in selection.c. */
#include "backend.h"
#include "metal.h"
#include "neon.h"
#include "avx2.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

const fr_tolerance fr_reference_tolerance={1e-6,1e-6,0.999999999999};
/* Numerical contract: a new kernel must meet it, not loosen it to pass. */
const fr_tolerance fr_backend_tolerance={0.025,0.030,0.9998};

/* Keep the built-in order, then the legacy user slot and named source modules.
 * Selection resolves a table once; run keeps the same direct function call. */
static const fr_backend *const compiled[]={&fr_ref,&fr_ref_q8,
#ifdef FR_METAL_COMPUTE
    &fr_metal,
#endif
#ifdef FR_NEON_COMPUTE
    &fr_neon,
#endif
#ifdef FR_AVX2_COMPUTE
    &fr_avx2,
#endif
#ifdef FR_USER_BACKEND
    &fr_user,
#endif
#ifdef FR_BACKEND_MODULES
#define FR_BACKEND_MODULE(name) &fr_module_##name,
#include "registry.h"
#undef FR_BACKEND_MODULE
#endif
};
_Static_assert(sizeof(compiled)/sizeof(compiled[0])<=FR_BACKENDS_MAX,"too many compiled backends");
const fr_backend *fr_backend_at(unsigned i) { return i<sizeof(compiled)/sizeof(compiled[0])?compiled[i]:NULL; }
const char *fr_backends_compiled(void) {
    static char names[FR_BACKENDS_MAX*(FR_BACKEND_NAME_MAX+1)];
    if (!names[0]) for (unsigned i=0;fr_backend_at(i);++i) {
        size_t used=strlen(names);
        snprintf(names+used,sizeof(names)-used,"%s%s",i?" ":"",fr_backend_at(i)->name);
    }
    return names;
}
int fr_backends_validate(FILE *error) {
    const char *reason="invalid name or incomplete function table";
    for (unsigned i=0;fr_backend_at(i);++i) {
        const fr_backend *b=fr_backend_at(i);
        if (!b->name || !b->name[0] || strlen(b->name)>FR_BACKEND_NAME_MAX ||
            strspn(b->name,"abcdefghijklmnopqrstuvwxyz0123456789_-")!=strlen(b->name) ||
            !b->probe || !b->load || !b->unload || !b->run || !b->bench || !b->self_test) goto bad;
        for (unsigned j=0;j<i;++j) if (!strcmp(b->name,compiled[j]->name)) { reason="duplicate backend name"; goto bad; }
    }
#ifdef FR_BACKEND_MODULES
#define FR_BACKEND_MODULE(id) if (strcmp(fr_module_##id.name,#id)) { reason="module name differs from backend.json"; goto bad; }
#include "registry.h"
#undef FR_BACKEND_MODULE
#endif
    return FR_OK;
bad:
    if (error) fprintf(error,"fronk: %s; inspect the backend registration and module manifest\n",reason);
    return FR_INVALID;
}
const fr_backend *fr_backend_find(const char *name) {
    if (!name) return NULL;
#ifdef FR_METAL_COMPUTE
    /* A measured execution mode, not an independent backend to race at setup. */
    if (!strcmp(name,"metal-ssd")) return &fr_metal_ssd;
#endif
    for (unsigned i=0;fr_backend_at(i);++i) if (!strcmp(name,fr_backend_at(i)->name)) return fr_backend_at(i);
    return NULL;
}
int fr_compare(const float *ref, const float *actual, size_t n, fr_error *out) {
    if (!ref || !actual || !out || !n) return FR_INVALID;
    double r2=0,a2=0,d2=0,dot=0,rmax=0,dmax=0;
    for (size_t i=0; i<n; ++i) {
        double r=ref[i], a=actual[i], d=a-r;
        if (!isfinite(r) || !isfinite(a)) return FR_NUMERIC;
        r2+=r*r; a2+=a*a; d2+=d*d; dot+=r*a;
        if (fabs(r)>rmax) rmax=fabs(r);
        if (fabs(d)>dmax) dmax=fabs(d);
    }
    fr_error e;
    if (!r2) e=(fr_error){a2?INFINITY:0,a2?INFINITY:0,a2?0:1};
    else {
        e=(fr_error){sqrt(d2/r2),dmax/rmax,a2?dot/(sqrt(r2)*sqrt(a2)):0};
        if (e.cosine>1) e.cosine=1;
        if (e.cosine< -1) e.cosine=-1;
    }
    *out=e; return FR_OK;
}
int fr_conforms(const fr_error *e, const fr_tolerance *t) {
    return e && t && isfinite(e->relative_l2) && isfinite(e->normalized_max) && isfinite(e->cosine) &&
        e->relative_l2>=0 && e->normalized_max>=0 && e->cosine>=-1 && e->cosine<=1 &&
        e->relative_l2<=t->relative_l2 && e->normalized_max<=t->normalized_max && e->cosine>=t->cosine_min;
}
