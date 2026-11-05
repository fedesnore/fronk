/* Portable oracle, deliberately token-first. Loaded storage is borrowed and
 * immutable; scratch is bounded stack storage, with no run-time allocation.
 * The Q8 variant has independent slots and changes only activation precision.
 */
#include "../src/backend.h"
#include "../src/q4.h"
#include "../src/native.h"
#include "../src/vectors.h"
#include "../src/bench.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <string.h>

typedef struct { const unsigned char *data[FR_BLOCKS]; unsigned native[FR_BLOCKS]; } ref_state;
static ref_state state, q8_state;

static int probe(fr_hwinfo *out) { return out?(hw_probe(out)?FR_IO:FR_OK):FR_INVALID; }
static int load(ref_state *s, int block, const void *data, size_t len) {
    if (block<0 || block>=FR_BLOCKS || !data || (len!=FR_WEIGHT_BLOCK_BYTES && len!=FR_NATIVE_BLOCK_BYTES)) return FR_INVALID;
    if (s->data[block]) return FR_STATE;
    s->data[block]=data; s->native[block]=len==FR_NATIVE_BLOCK_BYTES; return FR_OK;
}
static int unload(ref_state *s, int block) {
    if (block<0 || block>=FR_BLOCKS) return FR_INVALID;
    if (!s->data[block]) return FR_STATE;
    s->data[block]=NULL; return FR_OK;
}
static int loaded(const ref_state *s) {
    for (int b=0; b<FR_BLOCKS; ++b) if (s->data[b]) return 1;
    return 0;
}
int fr_ref_idle(const fr_backend *b) {
    if (b!=&fr_ref && b!=&fr_ref_q8) return FR_INVALID;
    return loaded(b==&fr_ref?&state:&q8_state)?FR_STATE:FR_OK;
}
static int narrow(double value, float *out) {
    if (!isfinite(value) || fabs(value)>FLT_MAX) return FR_NUMERIC;
    *out=(float)value; return FR_OK;
}
static int validate(const fr_req *req) {
    if (!req || req->layer>=FR_LAYERS || req->burst<1 || req->burst>FR_MAX_BURST ||
        fegetround()!=FE_TONEAREST) return FR_INVALID;
    for (unsigned t=0; t<req->burst; ++t) {
        for (int e=0; e<FR_ACTIVE_ROUTED; ++e) {
            if (req->expert[t][e]>=FR_ROUTED_EXPERTS) return FR_INVALID;
            for (int j=0; j<e; ++j) if (req->expert[t][j]==req->expert[t][e]) return FR_INVALID;
        }
        for (int e=0; e<FR_ACTIVE_EXPERTS; ++e) {
            if (!isfinite(req->gate[t][e])) return FR_NUMERIC;
            if (req->gate[t][e]<0) return FR_INVALID;
        }
        for (int d=0; d<FR_HIDDEN; ++d) if (!isfinite(req->x[t][d])) return FR_NUMERIC;
    }
    return FR_OK;
}
static int run(const ref_state *s, int q8, const fr_req *req, fr_part *out) {
    if (!out) return FR_INVALID;
    int rc=validate(req); if (rc) return rc;
    if (!loaded(s)) return FR_STATE;
    fr_part result; memset(&result,0,sizeof(result));
    result.layer=req->layer; result.burst=req->burst;
    for (unsigned t=0; t<req->burst; ++t) {
        double y[FR_HIDDEN]={0}; float quantized[FR_HIDDEN];
        const float *x=req->x[t];
        if (q8) { rc=fr_q8_activation(x,quantized,FR_HIDDEN); if (rc) return rc; x=quantized; }
        for (int e=0; e<FR_ACTIVE_EXPERTS; ++e) {
            if (e==FR_ACTIVE_ROUTED && req->gate[t][e]==0) continue; /* shared expert computed by the head */
            unsigned id=e==FR_ACTIVE_ROUTED?FR_SHARED_EXPERT:req->expert[t][e];
            for (int b=0; b<FR_BLOCKS; ++b) if (s->data[b]) {
                unsigned native=s->native[b],type=id==FR_SHARED_EXPERT?FR_Q8_0:FR_Q4_K;
                unsigned down=id==FR_SHARED_EXPERT?FR_Q8_0:FR_MXFP4;
                size_t projection=native?(id==FR_SHARED_EXPERT?FR_NATIVE_SHARED_PROJECTION:FR_NATIVE_GATE_BYTES):FR_PROJECTION_BYTES;
                const unsigned char *p=s->data[b]+(native?fr_native_offset(req->layer,id):fr_slice_offset(req->layer,id));
                float h[FR_BLOCK_NEURONS];
                for (int n=0; n<FR_BLOCK_NEURONS; ++n) {
                    double ad,ud; float a,u;
                    size_t row=(size_t)n*(native?fr_native_row(type,FR_HIDDEN):(FR_HIDDEN/FR_Q4_WEIGHTS)*FR_Q4_BYTES);
                    rc=native?fr_native_dot(p+row,type,x,FR_HIDDEN,&ad):fr_q4_dot(p+row,x,FR_HIDDEN,&ad); if (rc) return rc;
                    rc=native?fr_native_dot(p+projection+row,type,x,FR_HIDDEN,&ud):fr_q4_dot(p+projection+row,x,FR_HIDDEN,&ud); if (rc) return rc;
                    if (narrow(ad,&a) || narrow(ud,&u)) return FR_NUMERIC;
                    double sigmoid=a>=0?1/(1+exp(-(double)a)):exp((double)a)/(1+exp((double)a));
                    if (narrow((double)a*sigmoid*u,&h[n])) return FR_NUMERIC;
                }
                if (q8) { rc=fr_q8_activation(h,h,FR_BLOCK_NEURONS); if (rc) return rc; }
                for (int d=0; d<FR_HIDDEN; ++d) {
                    double dot;
                    const unsigned char *row=p+2*projection+(size_t)d*(native?fr_native_row(down,32):FR_Q4_BYTES);
                    rc=native?fr_native_dot(row,down,h,FR_BLOCK_NEURONS,&dot):fr_q4_dot(row,h,FR_BLOCK_NEURONS,&dot);
                    if (rc) return rc;
                    y[d]+=(double)req->gate[t][e]*dot;
                }
            }
        }
        for (int d=0; d<FR_HIDDEN; ++d) if (narrow(y[d],&result.y[t][d])) return FR_NUMERIC;
    }
    *out=result; return FR_OK;
}
static int bench(const fr_backend *backend,int burst_max, fr_costmodel *out) {
    if (!out || burst_max<1 || burst_max>FR_BENCH_BURST_MAX) return FR_INVALID;
    if (fr_ref_idle(backend)) return FR_STATE;
#ifdef FR_BENCH_RUNTIME
    return fr_bench_execute_backend(backend,burst_max,out);
#else
    return FR_UNSUPPORTED;
#endif
}
static int ref_bench(int b,fr_costmodel *out) { return bench(&fr_ref,b,out); }
static int q8_bench(int b,fr_costmodel *out) { return bench(&fr_ref_q8,b,out); }
static int ref_load(int b,const void *p,size_t n) { return load(&state,b,p,n); }
static int q8_load(int b,const void *p,size_t n) { return load(&q8_state,b,p,n); }
static int ref_unload(int b) { return unload(&state,b); }
static int q8_unload(int b) { return unload(&q8_state,b); }
static int ref_run(const fr_req *r,fr_part *p) { return run(&state,0,r,p); }
static int q8_run(const fr_req *r,fr_part *p) { return run(&q8_state,1,r,p); }
static int ref_test(const char *dir) {
    if (loaded(&state)) return FR_STATE;
    return fr_vectors_check(&fr_ref,dir,&fr_reference_tolerance,NULL);
}
static int q8_test(const char *dir) {
    if (loaded(&q8_state)) return FR_STATE;
    return fr_vectors_check(&fr_ref_q8,dir,&fr_backend_tolerance,NULL);
}
const fr_backend fr_ref={"ref",probe,ref_load,ref_unload,ref_run,ref_bench,ref_test};
const fr_backend fr_ref_q8={"ref-q8",probe,q8_load,q8_unload,q8_run,q8_bench,q8_test};

/* Test of pull request labels; not merged. */
