/* CPU port conformance uses sparse fixtures only for correctness, never timing. */
#include "test.h"
#include "../src/neon.h"
#include "../src/vectors.h"
#include "../src/native.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

int main(void) {
    test_begin("neon");
    check(fr_backend_find("neon")==&fr_neon,"compiled backend registered");
    check(fr_neon_features()&FR_ARM_DOTPROD,"native DotProd available");
    int rc=fr_vectors_check(&fr_neon,"tests/vectors",&fr_backend_tolerance,stdout);
    check(!rc,"Q8 port passes every frozen golden token and tolerance");
    if (rc) { fr_neon_close(); return test_end(); } /* Binding Q8 failure stop. */
    fr_vector v; rc=fr_vector_open(3,&v); check(!rc,"reused-expert fixture");
    if (rc) return test_end();
    fr_part actual, saved; memset(&actual,0x5a,sizeof(actual)); saved=actual;
    check(fr_neon.run(&v.req,&actual)==FR_STATE && !memcmp(&actual,&saved,sizeof(actual)),"unloaded run is atomic");
    check(fr_neon.load(-1,v.data[7],FR_WEIGHT_BLOCK_BYTES)==FR_INVALID && fr_neon.load(7,v.data[7],1)==FR_INVALID,"invalid load shape/index");
    check(!fr_neon.load(7,v.data[7],FR_WEIGHT_BLOCK_BYTES),"borrow canonical weight storage");
    check(fr_neon.load(7,v.data[7],FR_WEIGHT_BLOCK_BYTES)==FR_STATE && fr_neon_close()==FR_STATE &&
          fr_neon.self_test("tests/vectors")==FR_STATE,"live state is not replaced");
    check(!mprotect(v.data[7],FR_WEIGHT_BLOCK_BYTES,PROT_READ),"immutable weights protected");
    check(!fr_neon.run(&v.req,&actual),"run reused burst");
    struct timespec pause={.tv_nsec=60000000}; nanosleep(&pause,NULL);
    saved=actual;
    check(!fr_neon.run(&v.req,&actual) && !memcmp(&saved,&actual,sizeof(actual)),"parked pool wakes and deterministic reduction repeats");
    fr_neon_sample sample;
    check(!fr_neon_sample_get(&sample) && sample.hot_allocations==0,"zero fronk allocations during run");
    check(sample.unique_experts==11 && sample.weight_bytes==11*FR_SLICE_BYTES,"read weights once per unique expert per burst");
    check(sample.workers && sample.total_ns>0,"worker/timing diagnostics");
    saved=actual; v.req.layer=48;
    check(fr_neon.run(&v.req,&actual)==FR_INVALID && !memcmp(&actual,&saved,sizeof(actual)),"invalid layer preserves output");
    v.req.layer=17; v.req.x[0][0]=NAN;
    check(fr_neon.run(&v.req,&actual)==FR_NUMERIC && !memcmp(&actual,&saved,sizeof(actual)),"nonfinite activation rejected atomically");
    for (unsigned d=0;d<FR_HIDDEN;++d) v.req.x[0][d]=FLT_MAX;
    check(fr_neon.run(&v.req,&actual)==FR_NUMERIC && !memcmp(&actual,&saved,sizeof(actual)),"finite activation overflow rejected atomically");
    for (unsigned d=0;d<FR_HIDDEN;++d) v.req.x[0][d]=0;
    v.req.x[0][0]=0; v.req.expert[0][1]=v.req.expert[0][0];
    check(fr_neon.run(&v.req,&actual)==FR_INVALID,"duplicate routing rejected");
    v.req.expert[0][1]=511; v.req.gate[0][0]=-1;
    check(fr_neon.run(&v.req,&actual)==FR_INVALID,"negative gate rejected");
    v.req.gate[0][0]=0; fesetround(FE_DOWNWARD);
    check(fr_neon.run(&v.req,&actual)==FR_INVALID,"non-nearest rounding rejected"); fesetround(FE_TONEAREST);
    check(!mprotect(v.data[7],FR_WEIGHT_BLOCK_BYTES,PROT_READ|PROT_WRITE),"restore fixture permissions");
    unsigned char *p=v.data[7]+fr_slice_offset(17,FR_SHARED_EXPERT); p[0]=0; p[1]=0x7c;
    check(fr_neon.run(&v.req,&actual)==FR_NUMERIC && !memcmp(&actual,&saved,sizeof(actual)),"bad weight scale preserves output");
    check(!fr_neon.unload(7) && fr_neon.unload(7)==FR_STATE,"unload lifecycle");
    fr_vector_close(&v); check(!fr_neon_close(),"release idle pool");
    check(!fr_native_self_test(&fr_neon),"native Q4_K/MXFP4 and Q8_0 shared expert, every burst, vs ref");
    check(!fr_neon_close(),"release native pool");
    return test_end();
}
