/* Canonical scalar Q4_0/Q8 primitives define the independent oracle.
 * Half zeros/subnormals and activation rounding must remain explicit. */
#include "q4.h"
#include "backend.h"
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <string.h>
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,
               "backend requires IEEE binary32");

float fr_f16(uint16_t h) {
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exponent=(h>>10)&31, mantissa=h&1023;
    if (!exponent) {
        float f=ldexpf((float)mantissa,-24);
        return sign?-f:f;
    }
    uint32_t bits=sign | (exponent==31?0x7f800000:(exponent+112)<<23) | mantissa<<13;
    float f; memcpy(&f,&bits,sizeof(f)); return f;
}

int fr_q4_dot(const unsigned char *q, const float *x, size_t n, double *out) {
    if (!q || !x || !out || !n || n%FR_Q4_WEIGHTS) return FR_INVALID;
    double sum=0;
    for (size_t k=0; k<n; k+=FR_Q4_WEIGHTS,q+=FR_Q4_BYTES) {
        float scale=fr_f16((uint16_t)(q[0]|(uint16_t)q[1]<<8));
        if (!isfinite(scale)) return FR_NUMERIC;
        for (unsigned j=0; j<FR_Q4_WEIGHTS; ++j) {
            int value=(j<16?q[2+j]&15:q[2+j-16]>>4)-8;
            sum+=((double)value*scale)*x[k+j];
        }
    }
    if (!isfinite(sum)) return FR_NUMERIC;
    *out=sum; return FR_OK;
}

int fr_q8_activation(const float *x, float *out, size_t n) {
    if (!x || !out || !n || n%32 || fegetround()!=FE_TONEAREST) return FR_INVALID;
    for (size_t k=0; k<n; k+=64) {
        size_t count=n-k<64?n-k:64; float max=0;
        for (size_t j=0; j<count; ++j) {
            if (!isfinite(x[k+j])) return FR_NUMERIC;
            if (fabsf(x[k+j])>max) max=fabsf(x[k+j]);
        }
        float scale=max/127.0f;
        if (!scale) { for (size_t j=0; j<count; ++j) out[k+j]=0; continue; }
        float inv=127.0f/max;
        for (size_t j=0; j<count; ++j) {
            float v=isfinite(inv)?x[k+j]*inv:(x[k+j]/max)*127.0f;
            v=nearbyintf(v);
            if (v>127) v=127;
            if (v< -127) v=-127;
            double dequant=(double)v*scale;
            /* Rounded max/127 can otherwise overshoot FLT_MAX by one ulp. */
            if (dequant>FLT_MAX) dequant=FLT_MAX;
            if (dequant< -FLT_MAX) dequant=-FLT_MAX;
            out[k+j]=(float)dequant;
        }
    }
    return FR_OK;
}
