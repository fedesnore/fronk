#include "test.h"
#include "../src/backend.h"
#include "../src/q4.h"
#include "../src/vectors.h"
#include <math.h>
#include <fenv.h>
#include <float.h>
#include <string.h>
#include <sys/mman.h>

/* Independent expanded-matrix calculation: separate nibble indexing and half
 * decoder, full dense dot loops. Does not call the ref dot/SwiGLU helpers. */
static double weight(const unsigned char *row, unsigned j) {
    const unsigned char *q=row+(j/32)*18;
    unsigned half=q[0]+256u*q[1], e=(half>>10)&31, m=half&1023;
    double scale=e?ldexp(1024.0+m,(int)e-25):ldexp(m,-24);
    if (half&32768) scale=-scale;
    unsigned byte=q[2+(j%16)], nibble=j%32<16?byte&15:byte>>4;
    return ((int)nibble-8)*scale;
}
static void independent(const fr_vector *v, float out[FR_HIDDEN]) {
    double y[FR_HIDDEN]={0};
    for (int e=0; e<FR_ACTIVE_EXPERTS; ++e) {
        unsigned id=e==FR_ACTIVE_ROUTED?FR_SHARED_EXPERT:v->req.expert[0][e];
        for (int b=0; b<FR_BLOCKS; ++b) if (v->data[b]) {
            const unsigned char *p=v->data[b]+fr_slice_offset(v->req.layer,id);
            float h[FR_BLOCK_NEURONS];
            for (int n=0; n<FR_BLOCK_NEURONS; ++n) {
                double a=0,u=0;
                for (int d=0; d<FR_HIDDEN; ++d) {
                    a+=weight(p+n*(FR_HIDDEN/32)*18,(unsigned)d)*v->req.x[0][d];
                    u+=weight(p+FR_PROJECTION_BYTES+n*(FR_HIDDEN/32)*18,(unsigned)d)*v->req.x[0][d];
                }
                float af=(float)a, uf=(float)u;
                h[n]=(float)(((double)af/(1+exp(-(double)af)))*uf);
            }
            for (int d=0; d<FR_HIDDEN; ++d) {
                double dot=0;
                for (int n=0; n<FR_BLOCK_NEURONS; ++n)
                    dot+=weight(p+2*FR_PROJECTION_BYTES+d*18,(unsigned)n)*h[n];
                y[d]+=(double)v->req.gate[0][e]*dot;
            }
        }
    }
    for (int d=0; d<FR_HIDDEN; ++d) out[d]=(float)y[d];
}
static void lifecycle(const fr_backend *b) {
    fr_vector v;
    int rc=fr_vector_open(2,&v); check(!rc,"open deterministic signed fixture");
    if (rc) return;
    fr_part out, before; memset(&out,0x5a,sizeof(out)); before=out;
    check(b->run(&v.req,&out)==FR_STATE && !memcmp(&out,&before,sizeof(out)),"run requires weights, failure preserves output");
    check(b->load(-1,v.data[0],FR_WEIGHT_BLOCK_BYTES)==FR_INVALID &&
          b->load(20,v.data[0],FR_WEIGHT_BLOCK_BYTES)==FR_INVALID &&
          b->load(0,NULL,FR_WEIGHT_BLOCK_BYTES)==FR_INVALID &&
          b->load(0,v.data[0],1)==FR_INVALID,"load validates index, pointer and exact length");
    check(!b->load(0,v.data[0],FR_WEIGHT_BLOCK_BYTES),"load borrowed full block");
    check(b->load(0,v.data[0],FR_WEIGHT_BLOCK_BYTES)==FR_STATE,"duplicate load refused");
    check(b->self_test("tests/vectors")==FR_STATE,"self-test cannot replace live weights");
    fr_costmodel cost, saved; memset(&cost,0x5a,sizeof(cost)); saved=cost;
    check(b->bench(4,&cost)==FR_STATE && !memcmp(&cost,&saved,sizeof(cost)),"benchmark refuses to evict live reference weights");
    check(b->bench(0,&cost)==FR_INVALID && b->probe(NULL)==FR_INVALID,"probe/bench validate arguments");
    check(!mprotect(v.data[0],FR_WEIGHT_BLOCK_BYTES,PROT_READ),"make borrowed weights read-only");
    check(!b->run(&v.req,&out),"compute with read-only weights");
    if (b==&fr_ref) {
        float expected[FR_HIDDEN]; independent(&v,expected); fr_error error;
        check(!fr_compare(expected,out.y[0],FR_HIDDEN,&error) && fr_conforms(&error,&fr_reference_tolerance),"independent expanded-matrix oracle");
    }
    before=out;
    v.req.layer=48; check(b->run(&v.req,&out)==FR_INVALID,"invalid layer"); v.req.layer=0;
    v.req.burst=0; check(b->run(&v.req,&out)==FR_INVALID,"empty burst");
    v.req.burst=9; check(b->run(&v.req,&out)==FR_INVALID,"oversized burst"); v.req.burst=1;
    uint16_t id=v.req.expert[0][0]; v.req.expert[0][0]=512;
    check(b->run(&v.req,&out)==FR_INVALID,"shared expert cannot appear in routed list");
    v.req.expert[0][0]=v.req.expert[0][1];
    check(b->run(&v.req,&out)==FR_INVALID,"duplicate routed ids refused"); v.req.expert[0][0]=id;
    float gate=v.req.gate[0][0]; v.req.gate[0][0]=-1;
    check(b->run(&v.req,&out)==FR_INVALID,"negative gate refused"); v.req.gate[0][0]=INFINITY;
    check(b->run(&v.req,&out)==FR_NUMERIC,"nonfinite gate refused"); v.req.gate[0][0]=gate;
    float x=v.req.x[0][0]; v.req.x[0][0]=NAN;
    check(b->run(&v.req,&out)==FR_NUMERIC,"nonfinite input refused"); v.req.x[0][0]=x;
    fesetround(FE_UPWARD); check(b->run(&v.req,&out)==FR_INVALID,"run requires nearest rounding"); fesetround(FE_TONEAREST);
    check(!memcmp(&out,&before,sizeof(out)),"all invalid requests preserve output");
    check(!mprotect(v.data[0],FR_WEIGHT_BLOCK_BYTES,PROT_READ|PROT_WRITE),"restore fixture write access");
    unsigned char *p=v.data[0]+fr_slice_offset(0,id); unsigned char hi=p[1]; p[1]=0x7c;
    check(b->run(&v.req,&out)==FR_NUMERIC && !memcmp(&out,&before,sizeof(out)),"nonfinite weight scale refused atomically"); p[1]=hi;
    for (int d=0; d<FR_HIDDEN; ++d) v.req.x[0][d]=FLT_MAX;
    check(b->run(&v.req,&out)==FR_NUMERIC && !memcmp(&out,&before,sizeof(out)),"finite input causing fp32 overflow refused");
    check(!b->unload(0) && b->unload(0)==FR_STATE && b->unload(20)==FR_INVALID,"unload lifecycle");
    fr_vector_close(&v);
}
static void bursts(void) {
    fr_vector v; int rc=fr_vector_open(3,&v); check(!rc,"open reused-expert burst"); if (rc) return;
    for (int variant=0; variant<2; ++variant) {
        const fr_backend *b=variant?&fr_ref_q8:&fr_ref;
        check(!b->load(7,v.data[7],FR_WEIGHT_BLOCK_BYTES),"load repeated-expert weights");
        fr_part batch,single; fr_req req=v.req, original=v.req;
        check(!b->run(&req,&batch),"run reused-expert burst");
        check(!memcmp(&req,&original,sizeof(req)),"run preserves its request");
        int same=1;
        for (unsigned t=0; t<v.req.burst; ++t) {
            req.burst=1; memcpy(req.x[0],v.req.x[t],sizeof(req.x[0]));
            memcpy(req.gate[0],v.req.gate[t],sizeof(req.gate[0]));
            memcpy(req.expert[0],v.req.expert[t],sizeof(req.expert[0]));
            if (b->run(&req,&single) || memcmp(single.y[0],batch.y[t],sizeof(single.y[0]))) same=0;
        }
        check(same,"burst equals separate token runs bit for bit");
        check(batch.layer==v.req.layer && batch.burst==4,"partial identifies layer and burst");
        check(!b->unload(7),"release burst weights");
    }
    fr_vector_close(&v);
}
static void analytic(void) {
    fr_vector v; int rc=fr_vector_open(0,&v); check(!rc,"open analytic fixture"); if (rc) return;
    /* One nonzero hidden neuron: gate/up dot = 1, down weight = 1.
     * Shared gate 2 and routed gate 3 are deliberately not normalized. */
    unsigned char *p=v.data[0]+fr_slice_offset(0,FR_SHARED_EXPERT);
    memset(p,0,FR_SLICE_BYTES);
    p[1]=0x3c; p[2]=9; p[FR_PROJECTION_BYTES+1]=0x3c; p[FR_PROJECTION_BYTES+2]=9;
    for (int d=0; d<FR_HIDDEN; ++d) {
        unsigned char *row=p+2*FR_PROJECTION_BYTES+(size_t)d*FR_Q4_BYTES;
        row[1]=0x3c; row[2]=9;
    }
    memcpy(v.data[0]+fr_slice_offset(0,0),p,FR_SLICE_BYTES);
    memset(v.req.gate,0,sizeof(v.req.gate)); v.req.gate[0][10]=2; v.req.x[0][0]=1;
    float h=(float)(1/(1+exp(-1.0)));
    for (int variant=0; variant<2; ++variant) {
        const fr_backend *b=variant?&fr_ref_q8:&fr_ref; fr_part out;
        v.req.gate[0][0]=0; v.req.burst=1;
        check(!b->load(0,v.data[0],FR_WEIGHT_BLOCK_BYTES),"load analytic block");
        int ok=!b->run(&v.req,&out);
        for (int d=0; d<FR_HIDDEN; ++d) if (fabs(out.y[0][d]-2*h)>1e-6) ok=0;
        check(ok,"shared gate applied exactly once to SiLU(1)");
        v.req.gate[0][0]=3;
        ok=!b->run(&v.req,&out);
        for (int d=0; d<FR_HIDDEN; ++d) if (fabs(out.y[0][d]-5.0*h)>1e-6) ok=0;
        check(ok,"routed and shared gates add without renormalization");
        check(!b->load(19,v.data[0],FR_WEIGHT_BLOCK_BYTES),"noncontiguous last block can share immutable backing");
        ok=!b->run(&v.req,&out);
        for (int d=0; d<FR_HIDDEN; ++d) if (fabs(out.y[0][d]-10.0*h)>2e-6) ok=0;
        check(ok,"block partials add, including ownership boundary 19");
        check(!b->unload(0),"remove first block while last remains");
        ok=!b->run(&v.req,&out);
        for (int d=0; d<FR_HIDDEN; ++d) if (fabs(out.y[0][d]-5.0*h)>1e-6) ok=0;
        check(ok,"unload removes only its contribution");
        for (unsigned t=1; t<FR_MAX_BURST; ++t) {
            memcpy(v.req.x[t],v.req.x[0],sizeof(v.req.x[0]));
            memcpy(v.req.expert[t],v.req.expert[0],sizeof(v.req.expert[0]));
            memcpy(v.req.gate[t],v.req.gate[0],sizeof(v.req.gate[0]));
        }
        ok=1;
        for (unsigned n=1; n<=FR_MAX_BURST; ++n) {
            v.req.burst=n; if (b->run(&v.req,&out)) ok=0;
            for (unsigned t=0; t<FR_MAX_BURST; ++t) for (int d=0; d<FR_HIDDEN; ++d)
                if (t<n?fabs(out.y[t][d]-5.0*h)>1e-6:out.y[t][d]!=0) ok=0;
        }
        check(ok,"every burst length 1..8, including inactive output zeroing");
        check(!b->unload(19),"release analytic weights");
    }
    fr_vector_close(&v);
}
int main(void) {
    test_begin("backend");
    check(fr_backend_find("ref")==&fr_ref && fr_backend_find("ref-q8")==&fr_ref_q8 &&
          !fr_backend_find("unknown") && !fr_backend_find(NULL),"implemented references registered; unknown backend refused");
    float a[]={1,2,-3}, same[]={1,2,-3}, zero[3]={0}, bad[]={1,NAN,3}; fr_error e;
    check(!fr_compare(a,same,3,&e) && e.relative_l2==0 && e.normalized_max==0 && e.cosine>0.999999999999,"identical partial metrics");
    float opposite[]={-1,-2,3};
    check(!fr_compare(a,opposite,3,&e) && e.relative_l2==2 && e.normalized_max==2 && e.cosine<-.9999 && !fr_conforms(&e,&fr_backend_tolerance),"opposite outputs fail");
    check(!fr_compare(zero,zero,3,&e) && fr_conforms(&e,&fr_backend_tolerance),"exact zero reference passes");
    same[0]=0x1p-149f;
    check(!fr_compare(zero,same,3,&e) && !fr_conforms(&e,&fr_backend_tolerance),"nonzero result cannot pass a zero reference");
    check(!fr_compare(a,zero,3,&e) && e.cosine==0 && !fr_conforms(&e,&fr_backend_tolerance),"missing nonzero output fails");
    check(fr_compare(a,bad,3,&e)==FR_NUMERIC && fr_compare(a,a,0,&e)==FR_INVALID,"metrics reject nonfinite or empty arrays");
    e=(fr_error){fr_backend_tolerance.relative_l2,fr_backend_tolerance.normalized_max,fr_backend_tolerance.cosine_min};
    check(fr_conforms(&e,&fr_backend_tolerance),"limits are inclusive");
    e.relative_l2=nextafter(e.relative_l2,INFINITY);
    check(!fr_conforms(&e,&fr_backend_tolerance),"L2 just above limit fails"); e.relative_l2=0;
    e.normalized_max=nextafter(e.normalized_max,INFINITY);
    check(!fr_conforms(&e,&fr_backend_tolerance),"normalized max just above limit fails"); e.normalized_max=0;
    e.cosine=nextafter(e.cosine,-INFINITY);
    check(!fr_conforms(&e,&fr_backend_tolerance),"cosine just below floor fails");
    lifecycle(&fr_ref); lifecycle(&fr_ref_q8); bursts(); analytic();
    return test_end();
}
