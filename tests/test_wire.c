#include "test.h"
#include "../src/wire.h"
#include <string.h>
#include <math.h>
static unsigned u16(const unsigned char *p) { return (unsigned)p[0]*256+p[1]; }
static uint32_t u32(const unsigned char *p) { return (uint32_t)p[0]<<24|(uint32_t)p[1]<<16|(uint32_t)p[2]<<8|p[3]; }
int main(void) {
    test_begin("wire"); fr_req req={.layer=47}; uint16_t values[FR_MAX_BURST][FR_HIDDEN];
    for (unsigned t=0;t<FR_MAX_BURST;++t) {
        for (unsigned d=0;d<FR_HIDDEN;++d) values[t][d]=(uint16_t)(0x3000+t*100+d%100);
        for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) req.expert[t][e]=(uint16_t)(t*10+e);
        for (unsigned e=0;e<FR_ACTIVE_EXPERTS;++e) req.gate[t][e]=(t+1)*0.125f+e/64.0f;
    }
    unsigned maximum=0,routing_records=0;
    for (unsigned burst=1;burst<=8;++burst) {
        req.burst=burst;
        for (unsigned frag=0;frag<4*burst;++frag) {
            unsigned char bytes[FR_UDP_PAYLOAD_MAX+2]; memset(bytes,0xa5,sizeof(bytes));
            unsigned char *p=bytes+1; unsigned token=frag/4,range=frag%4;
            size_t n=fr_req_encode(p,FR_UDP_PAYLOAD_MAX,&req,values,0x01020304,0x10203040,frag);
            int ok=n==(range?1312u:1376u) && n<=1472 && bytes[0]==0xa5 && bytes[n+1]==0xa5 && bytes[sizeof(bytes)-1]==0xa5 &&
                !memcmp(p,"FRK1",4) && p[4]==1 && p[5]==255 && p[6]==burst && !p[7] &&
                u32(p+8)==0x01020304 && u32(p+12)==0x10203040 && u16(p+16)==47 && u16(p+18)==frag &&
                u32(p+20)==n-24 && u16(p+24)==token && u16(p+26)==range*640 && u16(p+28)==640 && !u16(p+30);
            for (unsigned d=0;d<640;++d) if (u16(p+32+2*d)!=values[token][range*640+d]) ok=0;
            if (!range) {
                ++routing_records;
                for (unsigned e=0;e<10;++e) if (u16(p+1312+2*e)!=req.expert[token][e]) ok=0;
                for (unsigned e=0;e<11;++e) {
                    uint32_t bits; memcpy(&bits,&req.gate[token][e],4);
                    if (u32(p+1332+4*e)!=bits) ok=0;
                }
            }
            if (n>maximum) maximum=(unsigned)n;
            char label[100]; snprintf(label,sizeof(label),"burst %u fragment %u: correct token/routing, network bytes, MTU and bounds",burst,frag);
            check(ok,label);
        }
    }
    check(maximum==1376 && routing_records==36,"all 144 fragments fit and exactly one routing record exists per token");
    unsigned char small[1376]; memset(small,0xa5,sizeof(small));
    check(!fr_req_encode(small,1375,&req,values,1,1,0) && small[0]==0xa5 && small[1375]==0xa5,"insufficient capacity rejected before writing");
    req.burst=0; check(!fr_req_encode(small,sizeof(small),&req,values,1,1,0),"zero burst rejected");
    req.burst=9; check(!fr_req_encode(small,sizeof(small),&req,values,1,1,0),"oversized burst rejected");
    req.burst=8; check(!fr_req_encode(small,sizeof(small),&req,values,1,1,32),"fragment beyond burst rejected");
    check(fr_req_encode(small,sizeof(small),&req,values,1,0,0)==1376 && !u32(small+12),"decode step zero is a valid wire counter");
    int roundtrip=1;
    for (unsigned i=0;i<65536;++i) if ((i&0x7c00)!=0x7c00) {
        uint16_t h=0;
        if (fr_half(fr_unhalf((uint16_t)i),&h) || h!=i) roundtrip=0;
    }
    check(roundtrip,"every finite binary16 value round trips exactly, including subnormals and signed zero");
    uint16_t h=0;
    check(!fr_half(1.00048828125f,&h) && h==0x3c00 && !fr_half(1.00146484375f,&h) && h==0x3c02,"round-to-nearest ties-to-even");
    check(fr_half(NAN,&h) && fr_half(INFINITY,&h) && fr_half(65520,&h),"nonfinite values and fp16 overflow rejected");
    fr_fragment decoded={0};
    size_t n=fr_req_encode(small,sizeof(small),&req,values,1,2,0);
    check(!fr_fragment_read(small,n,&decoded) && decoded.expert[9]==req.expert[0][9] && decoded.values[639]==fr_unhalf(values[0][639]),"REQ decoding includes routing and activations");
    fr_fragment before=decoded; small[30]=1;
    check(fr_fragment_read(small,n,&decoded) && !memcmp(&before,&decoded,sizeof(before)),"invalid range reserved byte rejected atomically");
    small[30]=0; small[32]=0x7c; small[33]=0;
    check(fr_fragment_read(small,n,&decoded),"nonfinite wire activation rejected");
    fr_part part={.layer=47,.burst=8};
    for (unsigned t=0;t<8;++t) for (unsigned d=0;d<2560;++d) part.y[t][d]=(float)((int)d-1000)/1024;
    n=fr_part_encode(small,sizeof(small),&part,1,2,3,31);
    check(n==1312 && !fr_fragment_read(small,n,&decoded) && decoded.node==3 && decoded.fragment==31 && decoded.values[639]==part.y[7][2559],"last PART range for burst eight round trips");
    small[18]=0; small[19]=30;
    check(fr_fragment_read(small,n,&decoded),"fragment/range mismatch rejected");
    return test_end();
}
