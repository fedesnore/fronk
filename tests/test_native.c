#include "test.h"
#include "../src/native.h"
#include <math.h>
#include <string.h>
int main(void) {
    test_begin("native");
    unsigned char q[144]={0}; float x[256]={0}; double y;
    unsigned sc[8]={1,2,3,4,17,34,51,63},mi[8]={63,51,34,17,4,3,2,1};
    q[1]=0x3c; q[3]=0x38; /* d=1, dmin=.5 */
    for (unsigned g=0;g<4;++g) { q[4+g]=(unsigned char)(sc[g]|((sc[g+4]>>4)<<6)); q[8+g]=(unsigned char)(mi[g]|((mi[g+4]>>4)<<6)); q[12+g]=(unsigned char)((sc[g+4]&15)|((mi[g+4]&15)<<4)); }
    for (unsigned j=0;j<128;++j) q[16+j]=(unsigned char)((j%16)|((15-j%16)<<4));
    int ok=1;
    for (unsigned j=0;j<256;++j) {
        memset(x,0,sizeof(x)); x[j]=1; unsigned g=j/32,v=(g%2)?15-j%16:j%16;
        if (fr_native_dot(q,FR_Q4_K,x,256,&y) || y!=(double)sc[g]*v-.5*mi[g]) ok=0;
    }
    check(ok,"Q4_K all subgroup scales/minima, both nibbles and all 256 lanes");
    q[1]=0x7c; check(fr_native_dot(q,FR_Q4_K,x,256,&y)==FR_NUMERIC,"nonfinite Q4_K scale rejected");
    unsigned char mx[17]={127};
    const float values[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
    for (unsigned j=0;j<16;++j) mx[j+1]=(unsigned char)(j|((15-j)<<4));
    ok=1;
    for (unsigned j=0;j<32;++j) { memset(x,0,sizeof(x)); x[j]=1; if (fr_native_dot(mx,FR_MXFP4,x,32,&y) || y!=values[j<16?j:31-j]) ok=0; }
    check(ok,"MXFP4 lookup, signs and nibble order");
    memset(x,0,sizeof(x)); x[2]=1; mx[0]=0;
    check(!fr_native_dot(mx,FR_MXFP4,x,32,&y) && y==ldexp(1.,-127),"MXFP4 zero exponent is subnormal 2^-127");
    mx[0]=255; check(fr_native_dot(mx,FR_MXFP4,x,32,&y)==FR_NUMERIC,"reserved MXFP4 exponent rejected");
    unsigned char shared[34]={0,0x38}; shared[4]=128;
    check(!fr_native_dot(shared,FR_Q8_0,x,32,&y) && y==-64,"shared Q8_0 signed-byte decode");
    check(fr_native_dot(q,FR_Q4_K,x,32,&y)==FR_INVALID,"partial Q4_K group refused");
    check(fr_native_offset(47,512)+FR_NATIVE_SHARED_BYTES==FR_NATIVE_BLOCK_BYTES,"lossless ownership layout includes shared expert");
    check(!fr_native_self_test(&fr_ref_q8),"native mixed-format fixture, every burst, Q8 envelope vs ref");
    return test_end();
}
