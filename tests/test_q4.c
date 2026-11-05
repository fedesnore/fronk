#include "test.h"
#include "../src/backend.h"
#include "../src/q4.h"
#include <math.h>
#include <fenv.h>
#include <float.h>
#include <string.h>

int main(void) {
    test_begin("q4/fp16/q8");
    int all=1;
    for (unsigned i=0; i<65536; ++i) {
        unsigned e=(i>>10)&31, m=i&1023;
        double value=e?ldexp(1.0+(double)m/1024,(int)e-15):ldexp((double)m,-24);
        if (i&32768) value=-value;
        float got=fr_f16((uint16_t)i);
        if (e==31) { if (m?!isnan(got):(!isinf(got) || !!signbit(got)!=!!(i&32768))) all=0; }
        else if (got!=value || !!signbit(got)!=!!(i&32768)) all=0;
    }
    check(all,"all 65536 fp16 encodings, including signed zero and specials");
    unsigned char storage[37]={0}, *q=storage+1; /* deliberately unaligned */
    q[1]=0x3c; /* d = 1 */
    float x[64]={0}; double result=123;
    for (int i=0; i<16; ++i) q[2+i]=(unsigned char)(i|((15-i)<<4));
    for (int i=0; i<32; ++i) {
        memset(x,0,sizeof(x)); x[i]=1;
        check(!fr_q4_dot(q,x,32,&result) && result==(i<16?i-8:23-i),"low/high nibble column order and signed levels");
    }
    q[1]=0xbc; x[31]=1;
    check(!fr_q4_dot(q,x,32,&result) && result==8,"negative fp16 scale");
    q[0]=1; q[1]=0;
    check(!fr_q4_dot(q,x,32,&result) && result==-8*0x1p-24,"subnormal scale");
    q[0]=0; q[1]=0x7c; result=123;
    check(fr_q4_dot(q,x,32,&result)==FR_NUMERIC && result==123,"nonfinite scale rejected without output write");
    check(fr_q4_dot(q,x,31,&result)==FR_INVALID && fr_q4_dot(NULL,x,32,&result)==FR_INVALID,"dot shape/pointer validation");
    memset(q,0,36); memset(x,0,sizeof(x)); q[1]=q[19]=0x3c; q[2]=9; q[20]=7;
    x[0]=2; x[32]=3;
    check(!fr_q4_dot(q,x,64,&result) && result==-1,"multiple Q4 groups use distinct columns");
    float in[96]={127,0.5f,1.5f,2.5f,-0.5f,-1.5f,-2.5f}, out[96];
    in[64]=254; in[65]=3;
    check(!fr_q8_activation(in,out,96) && out[0]==127 && out[1]==0 && out[2]==2 && out[3]==2 &&
          out[4]==0 && out[5]==-2 && out[6]==-2 && out[64]==254 && out[65]==4,"Q8 nearest-even ties, signed symmetry, 64+32 groups");
    check(!fr_q8_activation(in,in,96) && !memcmp(in,out,sizeof(in)),"in-place Q8 is identical");
    memset(in,0,sizeof(in));
    check(!fr_q8_activation(in,out,96) && out[0]==0 && out[95]==0,"all-zero Q8 groups");
    in[0]=FLT_MAX;
    check(!fr_q8_activation(in,out,32) && isfinite(out[0]) && out[0]>FLT_MAX*0.99f,"largest finite Q8 activation");
    in[0]=0x1p-149f;
    check(!fr_q8_activation(in,out,32) && out[0]==0,"underflowing Q8 scale yields zero");
    in[0]=0x1p-125f;
    check(!fr_q8_activation(in,out,32) && isfinite(out[0]) && out[0]>0,"overflowing reciprocal uses safe division");
    in[0]=NAN;
    check(fr_q8_activation(in,out,32)==FR_NUMERIC,"Q8 rejects NaN");
    check(fr_q8_activation(in,out,31)==FR_INVALID,"Q8 rejects non-group shape");
    fesetround(FE_DOWNWARD);
    check(fr_q8_activation(in,out,32)==FR_INVALID,"Q8 refuses altered rounding mode");
    fesetround(FE_TONEAREST);
    return test_end();
}
