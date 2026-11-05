/* Real GPU conformance. Host sanitizers and Metal API/shader validation are
 * complementary: the former do not instrument shader device memory access. */
#include "test.h"
#include "../src/metal.h"
#include "../src/vectors.h"
#include "../src/native.h"
#include <math.h>
#include <string.h>
#include <sys/mman.h>

int main(void) {
    test_begin("metal");
    check(fr_backend_find("metal")==&fr_metal,"compiled Apple Silicon backend registered");
    fr_hwinfo hw; int rc=fr_metal.probe(&hw);
    check(!rc && hw.metal && hw.metal_limit,"real Metal device and working-set limit");
    if (rc) return test_end();
    check(!fr_metal.self_test("tests/vectors"),"GPU passes frozen golden corpus and tolerances");
    fr_vector v; rc=fr_vector_open(3,&v); check(!rc,"create reused-expert fixture");
    if (rc) return test_end();
    fr_part batch, single, reference; memset(&batch,0x5a,sizeof(batch)); single=batch;
    check(fr_metal.run(&v.req,&batch)==FR_STATE && !memcmp(&batch,&single,sizeof(batch)),"unloaded run preserves output");
    check(fr_metal.load(-1,v.data[7],FR_WEIGHT_BLOCK_BYTES)==FR_INVALID &&
          fr_metal.load(7,v.data[7],1)==FR_INVALID,"invalid load shape/index");
    check(!fr_metal.load(7,v.data[7],FR_WEIGHT_BLOCK_BYTES),"borrow weight block");
    check(fr_metal.load(7,v.data[7],FR_WEIGHT_BLOCK_BYTES)==FR_STATE &&
          fr_metal.self_test("tests/vectors")==FR_STATE && fr_metal_close()==FR_STATE,"live state is not replaced");
    check(!fr_ref.load(7,v.data[7],FR_WEIGHT_BLOCK_BYTES) && !fr_ref.run(&v.req,&reference),"independent reference result");
    check(!mprotect(v.data[7],FR_WEIGHT_BLOCK_BYTES,PROT_READ),"protect immutable weight storage");
    fr_metal_sample before, after;
    check(!fr_metal_sample_get(&before) && !fr_metal.run(&v.req,&batch) && !fr_metal_sample_get(&after),"reused-expert GPU burst");
    /* Binding user exception: framework command buffers/compute encoders are
     * transient. Fronk malloc/calloc/realloc and resource creations still count
     * and must remain zero per barrier. No Metal 4 path or custom wait. */
    check(after.hot_allocations==0 && before.resources==after.resources,"zero fronk-owned allocations in the hot path");
    check(after.unique_experts==11 && after.weight_bytes==11*FR_SLICE_BYTES,"weights read once per unique expert per burst");
    check(after.ns[FR_METAL_GPU]>0 && after.ns[FR_METAL_TOTAL]>=after.ns[FR_METAL_GPU] &&
          after.ns[FR_METAL_OVERHEAD]>=0,"submission timings cover GPU and CPU work");
    int correct=1, identical=1;
    for (unsigned t=0; t<v.req.burst; ++t) {
        fr_error error;
        if (fr_compare(reference.y[t],batch.y[t],FR_HIDDEN,&error) || !fr_conforms(&error,&fr_backend_tolerance)) correct=0;
        fr_req req=v.req; req.burst=1;
        memcpy(req.x[0],v.req.x[t],sizeof(req.x[0])); memcpy(req.gate[0],v.req.gate[t],sizeof(req.gate[0]));
        memcpy(req.expert[0],v.req.expert[t],sizeof(req.expert[0]));
        if (fr_metal.run(&req,&single) || memcmp(single.y[0],batch.y[t],sizeof(single.y[0]))) identical=0;
    }
    check(correct,"each burst token conforms to double-accumulation ref");
    check(identical,"reused-expert burst equals separate GPU tokens");
    single=batch; v.req.layer=48;
    check(fr_metal.run(&v.req,&batch)==FR_INVALID && !memcmp(&single,&batch,sizeof(batch)),"bad request is atomic");
    v.req.layer=17; v.req.x[0][0]=NAN;
    check(fr_metal.run(&v.req,&batch)==FR_NUMERIC && !memcmp(&single,&batch,sizeof(batch)),"nonfinite CPU activation refused");
    v.req.x[0][0]=0;
    check(!mprotect(v.data[7],FR_WEIGHT_BLOCK_BYTES,PROT_READ|PROT_WRITE),"restore fixture permissions");
    unsigned char *p=v.data[7]+fr_slice_offset(17,FR_SHARED_EXPERT); p[0]=0; p[1]=0x7c;
    check(fr_metal.run(&v.req,&batch)==FR_NUMERIC && !memcmp(&single,&batch,sizeof(batch)),"shader rejects nonfinite Q4 scale without partial output");
    check(!fr_metal.unload(7) && fr_metal.unload(7)==FR_STATE && !fr_ref.unload(7),"unload lifecycle");
    fr_vector_close(&v);
    check(!fr_metal_close(),"release idle Metal resources");
    check(!fr_native_self_test(&fr_metal),"native Q4_K/MXFP4 and Q8_0 shared expert, every burst, vs ref");
    check(!fr_metal_close(),"release native Metal resources");
    return test_end();
}
