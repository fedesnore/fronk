/* user-template.c - the starting point for your own backend (BACKENDS.md).
 * Copy it to backends/user.c, give it your own name below, then make run()
 * fast for your machine while ./fronk backends --test NAME keeps passing.
 * As written it is the plain scalar algorithm (no threads, no SIMD): correct
 * and slow, so every line of run() is the specification your version must meet.
 */
#include "../src/backend.h"
#include "../src/native.h"
#include "../src/bench.h"
#include <float.h>
#include <math.h>
#include <string.h>

/* Borrowed unit bytes, read-only and valid until unload. Never copy or free them. */
static const unsigned char *block[FR_BLOCKS];

/* FR_UNSUPPORTED when this machine lacks what the backend needs (an ISA, a GPU). */
static int probe(fr_hwinfo *out) { return out?(hw_probe(out)?FR_IO:FR_OK):FR_INVALID; }
static int load(int b,const void *data,size_t len) {
    if (b<0 || b>=FR_BLOCKS || !data || len!=FR_NATIVE_BLOCK_BYTES) return FR_INVALID;
    if (block[b]) return FR_STATE;
    block[b]=data; return FR_OK;
}
static int unload(int b) {
    if (b<0 || b>=FR_BLOCKS) return FR_INVALID;
    if (!block[b]) return FR_STATE;
    block[b]=NULL; return FR_OK;
}
static int loaded(void) {
    for (unsigned b=0;b<FR_BLOCKS;++b) if (block[b]) return 1;
    return 0;
}
static int narrow(double value,float *out) {
    if (!isfinite(value) || fabs(value)>FLT_MAX) return FR_NUMERIC;
    *out=(float)value; return FR_OK;
}
static int validate(const fr_req *r) {
    if (!r || r->layer>=FR_LAYERS || !r->burst || r->burst>FR_MAX_BURST) return FR_INVALID;
    for (unsigned t=0;t<r->burst;++t) {
        for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) {
            if (r->expert[t][e]>=FR_ROUTED_EXPERTS) return FR_INVALID;
            for (unsigned k=0;k<e;++k) if (r->expert[t][k]==r->expert[t][e]) return FR_INVALID;
        }
        for (unsigned e=0;e<FR_ACTIVE_EXPERTS;++e) {
            if (!isfinite(r->gate[t][e])) return FR_NUMERIC;
            if (r->gate[t][e]<0) return FR_INVALID;
        }
        for (unsigned d=0;d<FR_HIDDEN;++d) if (!isfinite(r->x[t][d])) return FR_NUMERIC;
    }
    return FR_OK;
}
/* For each token: for its 10 routed experts and the shared one, for every loaded
 * block (32 of the expert's 640 neurons), h = silu(gate . x) * (up . x), then
 * y += expert weight * (down . h). The head adds the partials of all machines. */
static int run(const fr_req *r,fr_part *out) {
    if (!out) return FR_INVALID;
    int rc=validate(r); if (rc) return rc;
    if (!loaded()) return FR_STATE;
    fr_part result; memset(&result,0,sizeof(result)); /* unused tokens stay zero */
    result.layer=r->layer; result.burst=r->burst;
    for (unsigned t=0;t<r->burst;++t) {
        double y[FR_HIDDEN]={0};
        for (unsigned e=0;e<FR_ACTIVE_EXPERTS;++e) {
            int shared=e==FR_ACTIVE_ROUTED;
            if (shared && r->gate[t][e]==0) continue; /* the head computes the shared expert */
            unsigned id=shared?FR_SHARED_EXPERT:r->expert[t][e];
            unsigned in=shared?FR_Q8_0:FR_Q4_K, down=shared?FR_Q8_0:FR_MXFP4;
            size_t projection=shared?FR_NATIVE_SHARED_PROJECTION:FR_NATIVE_GATE_BYTES;
            size_t in_row=fr_native_row(in,FR_HIDDEN), down_row=fr_native_row(down,FR_BLOCK_NEURONS);
            for (unsigned b=0;b<FR_BLOCKS;++b) if (block[b]) {
                /* One slice: 32 gate rows, 32 up rows (2560 weights each), 2560 down rows (32 each). */
                const unsigned char *p=block[b]+fr_native_offset(r->layer,id);
                float h[FR_BLOCK_NEURONS];
                for (unsigned n=0;n<FR_BLOCK_NEURONS;++n) {
                    double ad,ud; float a,u;
                    if ((rc=fr_native_dot(p+n*in_row,in,r->x[t],FR_HIDDEN,&ad)) ||
                        (rc=fr_native_dot(p+projection+n*in_row,in,r->x[t],FR_HIDDEN,&ud))) return rc;
                    if (narrow(ad,&a) || narrow(ud,&u)) return FR_NUMERIC;
                    double sigmoid=a>=0?1/(1+exp(-(double)a)):exp((double)a)/(1+exp((double)a));
                    if (narrow((double)a*sigmoid*u,&h[n])) return FR_NUMERIC;
                }
                for (unsigned d=0;d<FR_HIDDEN;++d) {
                    double dot;
                    if ((rc=fr_native_dot(p+2*projection+d*down_row,down,h,FR_BLOCK_NEURONS,&dot))) return rc;
                    y[d]+=(double)r->gate[t][e]*dot;
                }
            }
        }
        for (unsigned d=0;d<FR_HIDDEN;++d) if (narrow(y[d],&result.y[t][d])) return FR_NUMERIC;
    }
    *out=result; return FR_OK; /* on any failure above, *out is untouched */
}
static int bench(int burst,fr_costmodel *out) { return fr_bench_execute_backend(&fr_user,burst,out); }
/* Compares run() with the reference on a synthetic block, bursts 1..8, distinct
 * and repeated experts, the shared expert and corrupt scales. */
static int self_test(const char *dir) {
    (void)dir;
    return loaded()?FR_STATE:fr_native_self_test(&fr_user);
}
/* Your backend's name: lowercase letters, digits, - or _, at most 15, not starting with "ref". */
const fr_backend fr_user={"mine",probe,load,unload,run,bench,self_test};
