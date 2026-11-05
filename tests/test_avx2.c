/* Shuttle port conformance: sparse fixtures are correctness-only, never timing. */
#include "test.h"
#include "../src/avx2.h"
#include "../src/vectors.h"
#include "../src/native.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

int main(void) {
    test_begin("avx2");
    check(fr_backend_find("avx2")==&fr_avx2,"compiled backend registered");
    fr_hwinfo hw; int eligibility=fr_avx2.probe(&hw);
    if (eligibility==FR_UNSUPPORTED) {
        check(fr_avx2.load(0,(const void *)1,FR_WEIGHT_BLOCK_BYTES)==FR_UNSUPPORTED,"unsupported ISA refuses setup before using weights");
        puts("[skip] AVX2/FMA/F16C or OS YMM support unavailable; golden execution not claimed");
        return test_end();
    }
    check(!eligibility && (hw.isa&FR_ISA_USABLE)==FR_ISA_USABLE,"runtime AVX2/FMA/F16C and OS YMM support");
    int rc=fr_vectors_check(&fr_avx2,"tests/vectors",&fr_backend_tolerance,stdout);
    check(!rc,"canonical Q4_0 adapter and Q8 microkernel pass every frozen golden token");
    if (rc) { fr_avx2_close(); return test_end(); } /* Binding Q8 failure stop. */
    fr_vector v; rc=fr_vector_open(3,&v); check(!rc,"reused-expert fixture");
    if (rc) return test_end();
    fr_part actual, saved; memset(&actual,0x5a,sizeof(actual)); saved=actual;
    check(fr_avx2.run(&v.req,&actual)==FR_STATE && !memcmp(&actual,&saved,sizeof(actual)),"unloaded run is atomic");
    check(fr_avx2.load(-1,v.data[7],FR_WEIGHT_BLOCK_BYTES)==FR_INVALID && fr_avx2.load(7,v.data[7],1)==FR_INVALID,"invalid load shape/index");
    check(!fr_avx2.load(7,v.data[7],FR_WEIGHT_BLOCK_BYTES),"borrow canonical weight storage");
    check(fr_avx2.load(7,v.data[7],FR_WEIGHT_BLOCK_BYTES)==FR_STATE && fr_avx2_close()==FR_STATE &&
          fr_avx2.self_test("tests/vectors")==FR_STATE,"live state is not replaced");
    check(!mprotect(v.data[7],FR_WEIGHT_BLOCK_BYTES,PROT_READ),"immutable weights protected");
    check(!fr_avx2.run(&v.req,&actual),"run reused burst");
    struct timespec pause={.tv_nsec=60000000}; nanosleep(&pause,NULL);
    saved=actual;
    check(!fr_avx2.run(&v.req,&actual) && !memcmp(&saved,&actual,sizeof(actual)),"parked pool wakes and deterministic reduction repeats");
    fr_avx2_sample sample;
    check(!fr_avx2_sample_get(&sample) && sample.hot_allocations==0,"zero fronk allocations during run");
    check(sample.unique_experts==11 && sample.weight_bytes==11*FR_SLICE_BYTES,"read weights once per unique expert per burst");
    check(sample.workers && sample.total_ns>0,"worker/timing diagnostics");
    saved=actual; v.req.layer=48;
    check(fr_avx2.run(&v.req,&actual)==FR_INVALID && !memcmp(&actual,&saved,sizeof(actual)),"invalid layer preserves output");
    v.req.layer=17; v.req.x[0][0]=NAN;
    check(fr_avx2.run(&v.req,&actual)==FR_NUMERIC && !memcmp(&actual,&saved,sizeof(actual)),"nonfinite activation rejected atomically");
    for (unsigned d=0;d<FR_HIDDEN;++d) v.req.x[0][d]=FLT_MAX;
    check(fr_avx2.run(&v.req,&actual)==FR_NUMERIC && !memcmp(&actual,&saved,sizeof(actual)),"finite activation overflow rejected atomically");
    for (unsigned d=0;d<FR_HIDDEN;++d) v.req.x[0][d]=0;
    v.req.x[0][0]=0; v.req.expert[0][1]=v.req.expert[0][0];
    check(fr_avx2.run(&v.req,&actual)==FR_INVALID,"duplicate routing rejected");
    v.req.expert[0][1]=511; v.req.gate[0][0]=-1;
    check(fr_avx2.run(&v.req,&actual)==FR_INVALID,"negative gate rejected");
    v.req.gate[0][0]=0; fesetround(FE_DOWNWARD);
    check(fr_avx2.run(&v.req,&actual)==FR_INVALID,"non-nearest rounding rejected"); fesetround(FE_TONEAREST);
    check(!mprotect(v.data[7],FR_WEIGHT_BLOCK_BYTES,PROT_READ|PROT_WRITE),"restore fixture permissions");
    unsigned char *p=v.data[7]+fr_slice_offset(17,FR_SHARED_EXPERT); p[0]=0; p[1]=0x7c;
    check(fr_avx2.run(&v.req,&actual)==FR_NUMERIC && !memcmp(&actual,&saved,sizeof(actual)),"bad weight scale preserves output");
    check(!fr_avx2.unload(7) && fr_avx2.unload(7)==FR_STATE,"unload lifecycle");
    fr_vector_close(&v); check(!fr_avx2_close(),"release idle pool");
    check(!fr_native_self_test(&fr_avx2),"lossless Q4_K/MXFP4 and Q8_0 shared expert vs ref; distinct/reused bursts 1..8");
    check(!fr_avx2_close(),"release native self-test pool");
    return test_end();
}
