/* test_probe.c - wire byte order, ranges, MTU bounds and invalid packet cases. */
#include "test.h"
#include "../src/probe.h"
#include <string.h>
int main(void) {
    test_begin("probe"); unsigned char bytes[FR_PROBE_BYTES], cookie[16], got[16];
    memset(cookie,0xa5,sizeof(cookie)); fr_identity id={.id=UINT64_C(0x0123456789abcdef),.name="local-1",.control_port=17700,.data_port=17701}, parsed;
    check(probe_hello(bytes,cookie,NULL)==24 && probe_discovery_read(bytes,24,got,&parsed)==1 && !memcmp(cookie,got,16),"HELLO cookie roundtrip");
    check(probe_hello(bytes,cookie,&id)==100 && probe_discovery_read(bytes,100,got,&parsed)==2 &&
          parsed.id==id.id && bytes[92]==1 && bytes[99]==239 && !strcmp(parsed.name,id.name) && parsed.control_port==17700 && parsed.data_port==17701,"identity roundtrip");
    bytes[28]='/'; check(probe_discovery_read(bytes,100,got,&parsed)<0,"unsafe node name rejected");
    probe_hello(bytes,cookie,&id); bytes[6]=1;
    check(probe_discovery_read(bytes,100,got,&parsed)<0 && probe_discovery_read(bytes,23,got,&parsed)<0,"reserved bits and truncation rejected");
    fr_probe p={.session=0x01020304,.step=0x05060708,.node=255}, q;
    int complete=1;
    for (unsigned k=0; k<3; ++k) for (unsigned f=0; f<probe_count(k); ++f) {
        p.kind=k; p.frag=f; size_t n=probe_make(bytes,&p);
        if (n!=(k?(f%4?1312u:1376u):64u) || n+28>1500 || probe_read(bytes,n,&q) || q.kind!=k || q.frag!=f || q.session!=p.session || q.step!=p.step) complete=0;
    }
    check(complete,"all 21 probe fragments have canonical ranges and fit MTU 1500");
    check(probe_train_bytes(0)==64 && probe_train_bytes(1)==5312 && probe_train_bytes(2)==21248 && !probe_train_bytes(3),"large probes match corrected burst 1/4 REQ byte counts");
    check(bytes[8]==1 && bytes[9]==2 && bytes[12]==5 && bytes[15]==8,"network byte order");
    p=(fr_probe){.session=1,.step=1,.node=2,.reply=1};
    check(probe_make(bytes,&p)==64 && !probe_read(bytes,64,&q) && q.reply && q.node==2,"PONG identity");
    unsigned offsets[]={0,7,16,20,24,26,28,30,32}; int rejected=1;
    for (size_t i=0; i<sizeof(offsets)/sizeof(offsets[0]); ++i) {
        probe_make(bytes,&p); bytes[offsets[i]]^=1;
        if (!probe_read(bytes,64,&q)) rejected=0;
    }
    check(rejected,"malformed headers/ranges/payload rejected");
    check(probe_read(bytes,0,&q)<0 && probe_read(bytes,65,&q)<0,"invalid sizes");
    p.kind=3; check(!probe_make(bytes,&p) && !probe_count(3),"unknown probe kind");
    return test_end();
}
