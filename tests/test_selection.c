/* Synthetic backend curves exercise policy without allocating benchmark weights. */
#include "test.h"
#include "../src/selection.h"
#include "../src/native.h"
#include <math.h>
#include <string.h>
int main(void) {
    test_begin("selection");
    fr_hwinfo h={.available=15*FR_NATIVE_BLOCK_BYTES,.metal_limit=4*FR_NATIVE_BLOCK_BYTES,.metal=1,.margin=FR_NATIVE_BLOCK_BYTES,.head_reserve=2*FR_NATIVE_BLOCK_BYTES};
    check(selection_capacity(&h,"neon",0)==14 && selection_capacity(&h,"metal",0)==3,"Metal limit applies only to Metal");
    check(selection_capacity(&h,"neon",1)==13 && selection_capacity(&h,"metal",1)==2,"head reserve applied to each backend cap");
    h.translated=1; check(!selection_capacity(&h,"neon",0),"translated host excluded");
    fr_selection s={.burst=1};
    check(!selection_add(&s,"metal",3) && !selection_add(&s,"neon",10),"backend-dependent capacity offers");
    check(selection_add(&s,"neon",1)<0 && selection_pick(&s,1)<0,"duplicates and unmeasured choices refused");
    { fr_selection u={.burst=1};
      check(!selection_add(&u,"mine",2) && !selection_add(&u,"m1-max_v2",2) && selection_add(&u,"Mine",2)<0 &&
            selection_add(&u,"a b",2)<0 && selection_add(&u,"",2)<0 && selection_add(&u,"sixteen-letters1",2)<0,
            "a node's own backend name is accepted when well formed");
      check(selection_capacity(&h,"mine",0)==selection_capacity(&h,"neon",0),"own backends get the CPU memory cap"); }
    for (unsigned b=1;b<=3;++b) check(!selection_record(&s,0,b,100*b,120*b,b*FR_WEIGHT_BLOCK_BYTES),"record resident Metal point");
    for (unsigned b=1;b<=5;++b) check(!selection_record(&s,1,b,50+50*b,70+60*b,b*FR_WEIGHT_BLOCK_BYTES),"record resident CPU point");
    check(selection_complete(&s) && selection_pick(&s,1)==1 && selection_pick(&s,3)==1,"CPU wins exact tie and lower measured cost");
    check(selection_pick(&s,4)==1 && selection_pick(&s,11)<0,"candidate excluded beyond its own cap");
    check(fabs(selection_cost(&s.candidate[1],10)-550)<1e-8,"larger share uses sweep fit within backend capacity");
    char text[FR_SELECTION_TEXT]; s.assigned=3; fr_selection parsed={0};
    int n=selection_format(&s,text,sizeof(text));
    check(n>0 && !selection_parse(text,(size_t)n,&parsed) && !memcmp(&s,&parsed,sizeof(s)),"bounded curve wire round trip");
    fr_selection before=parsed;
    check(selection_parse(text,(size_t)n-1,&parsed)<0 && !memcmp(&before,&parsed,sizeof(parsed)),"truncated curve is atomic");
    check(selection_record(&s,1,1,1,2,0)<0 && selection_record(&s,1,1,NAN,2,FR_WEIGHT_BLOCK_BYTES)<0,"invalid residency and nonfinite cost rejected");
    check(selection_reason(&s,3,text,sizeof(text))>0 && strstr(text,"neon") && strstr(text,"metal") && strstr(text,"burst 1"),"assigned-share explanation lists every candidate");
    check(selection_reason(&s,10,text,sizeof(text))>0 && strstr(text,"estimated") && strstr(text,"capacity"),"extrapolation and ineligible alternative explicit");
    fr_selection capped=s; capped.assigned=0;
    check(!selection_limit(&capped,0,2) && capped.candidate[0].capacity==2 && capped.candidate[0].measured==3,"capacity limit drops points above it");
    n=selection_format(&capped,text,sizeof(text));
    check(n>0 && !selection_parse(text,(size_t)n,&parsed) && parsed.candidate[0].capacity==2,"capped curve still saves and parses");
    check(!selection_limit(&capped,0,0) && !capped.candidate[0].measured && selection_limit(&capped,2,1)<0,"limit to zero and bad candidate");
    fr_selection native={.burst=4};
    check(!selection_add(&native,"avx2",1) && !selection_record(&native,0,1,100,150,FR_NATIVE_BLOCK_BYTES),"native unit residency accepted for measured selection");
    check(!selection_settings(&native,0,1,4,2),"runtime settings saved with their measured share");
    n=selection_format(&native,text,sizeof(text));
    check(n>0 && !selection_parse(text,(size_t)n,&parsed) && parsed.candidate[0].resident[0]==FR_NATIVE_BLOCK_BYTES && parsed.candidate[0].workers[0]==4 && parsed.candidate[0].prefetch[0]==2,"native residency and runtime settings survive selection wire round trip");
    fr_selection hybrid={.burst=3}; double cold[2]={900,1600},warm[2]={250,300},tail[2]={1500,2400};
    check(!selection_add(&hybrid,"metal",2) && !selection_record(&hybrid,0,1,100,130,FR_NATIVE_BLOCK_BYTES) &&
          !selection_record(&hybrid,0,2,200,260,2*FR_NATIVE_BLOCK_BYTES) &&
          !selection_hybrid(&hybrid,2,2,FR_SSD_DEFAULT_CACHE,cold,warm,tail),"measured RAM plus SSD candidate");
    check(selection_complete(&hybrid) && selection_cost(&hybrid.candidate[1],3)==900 &&
          selection_cost(&hybrid.candidate[1],4)==1600 && isinf(selection_cost(&hybrid.candidate[1],5)),
          "SSD uses complete cold timings, never optimistic warm costs or extrapolation");
    n=selection_format(&hybrid,text,sizeof(text));
    check(n>0 && !strncmp(text,"FRCHOICE2",9) && !selection_parse(text,(size_t)n,&parsed) &&
          parsed.candidate[1].ram_blocks==2 && parsed.candidate[1].cache_bytes==FR_SSD_DEFAULT_CACHE &&
          parsed.candidate[1].warm_ns[1]==300,"hybrid calibration survives node-to-head transport");
    check(selection_paged(&hybrid.candidate[1],(1u<<1)|(1u<<6)|(1u<<12))==(1u<<12),"streamed mask covers only extra local slices");
    fr_hwinfo fit={.metal=1,.total=24ull<<30,.available=24ull<<30,.metal_limit=24ull<<30,.head_reserve=10ull<<30};
    check(selection_fits(&hybrid.candidate[1],&fit,1,4),"resident bytes, bounded cache and staging fit");
    fit.available=fit.head_reserve+2*FR_NATIVE_BLOCK_BYTES;
    check(!selection_fits(&hybrid.candidate[1],&fit,1,3),"cache and staging cannot consume reserved dense memory");
    return test_end();
}
