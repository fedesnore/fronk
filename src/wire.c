/* Explicit discovery, probe and layer codecs avoid native layout and padding.
 * Logical fragments fit MTU 1500. Inference batches them into a larger UDP
 * datagram (wire.h), so IP fragmentation still occurs. See docs/PROTOCOL.md. */
#include "wire.h"
#include "probe.h"
#include <math.h>
#include <string.h>
_Static_assert(FR_TOKEN_FRAGMENTS*FR_FRAGMENT_DIMS==FR_HIDDEN,"four ranges cover one hidden state");
_Static_assert(FR_ROUTING_BYTES==2*FR_ACTIVE_ROUTED+4*FR_ACTIVE_EXPERTS,"one routing record per token");
_Static_assert(FR_WIRE_HEADER+FR_RANGE_HEADER+2*FR_FRAGMENT_DIMS+FR_ROUTING_BYTES==FR_REQ_FRAGMENT_MAX &&
               FR_REQ_FRAGMENT_MAX<=FR_UDP_PAYLOAD_MAX,"each logical REQ fragment fits IPv4 MTU 1500");
static void put16(unsigned char *p,unsigned v) { p[0]=(unsigned char)(v>>8); p[1]=(unsigned char)v; }
static void put32(unsigned char *p,uint32_t v) { for (unsigned i=0;i<4;++i) p[i]=(unsigned char)(v>>(24-8*i)); }
static uint32_t get32(const unsigned char *p) { return (uint32_t)p[0]<<24|(uint32_t)p[1]<<16|(uint32_t)p[2]<<8|p[3]; }
static unsigned get16(const unsigned char *p) { return (unsigned)p[0]<<8|p[1]; }
static void header(unsigned char *out,size_t size) { memset(out,0,size); memcpy(out,"FRK1",4); }
size_t fr_req_encode(unsigned char *out,size_t capacity,const fr_req *req,
                     const uint16_t activations[FR_MAX_BURST][FR_HIDDEN],
                     uint32_t session,uint32_t step,unsigned fragment) {
    if (!out || !req || !activations || !session || req->layer>=FR_LAYERS ||
        req->burst<1 || req->burst>FR_MAX_BURST || fragment>=FR_TOKEN_FRAGMENTS*req->burst) return 0;
    unsigned token=fragment/FR_TOKEN_FRAGMENTS,range=fragment%FR_TOKEN_FRAGMENTS;
    size_t size=FR_WIRE_HEADER+FR_RANGE_HEADER+2*FR_FRAGMENT_DIMS+(range?0:FR_ROUTING_BYTES);
    if (capacity<size) return 0;
    if (!range) {
        for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) {
            if (req->expert[token][e]>=FR_ROUTED_EXPERTS) return 0;
            for (unsigned j=0;j<e;++j) if (req->expert[token][e]==req->expert[token][j]) return 0;
        }
        for (unsigned e=0;e<FR_ACTIVE_EXPERTS;++e) if (!isfinite(req->gate[token][e]) || req->gate[token][e]<0) return 0;
    }
    header(out,FR_WIRE_HEADER+FR_RANGE_HEADER);
    out[4]=1; out[5]=255; out[6]=(unsigned char)req->burst;
    put32(out+8,session); put32(out+12,step); put16(out+16,req->layer); put16(out+18,fragment); put32(out+20,(uint32_t)size-FR_WIRE_HEADER);
    put16(out+24,token); put16(out+26,range*FR_FRAGMENT_DIMS); put16(out+28,FR_FRAGMENT_DIMS);
    for (unsigned d=0;d<FR_FRAGMENT_DIMS;++d) put16(out+32+2*d,activations[token][range*FR_FRAGMENT_DIMS+d]);
    if (!range) {
        unsigned char *routing=out+FR_WIRE_HEADER+FR_RANGE_HEADER+2*FR_FRAGMENT_DIMS;
        for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) put16(routing+2*e,req->expert[token][e]);
        for (unsigned e=0;e<FR_ACTIVE_EXPERTS;++e) {
            uint32_t bits; memcpy(&bits,&req->gate[token][e],sizeof(bits)); put32(routing+2*FR_ACTIVE_ROUTED+4*e,bits);
        }
    }
    return size;
}
int fr_half(float value,uint16_t *out) {
    uint32_t bits; memcpy(&bits,&value,4);
    unsigned sign=bits>>16&0x8000,exponent=bits>>23&255,mantissa=bits&0x7fffff;
    if (!out || exponent==255) return -1;
    int e=(int)exponent-127+15;
    if (e>=31) return -1;
    uint32_t rounded;
    if (e<=0) {
        if (e < -10) { *out=(uint16_t)sign; return 0; }
        mantissa|=0x800000; unsigned shift=(unsigned)(14-e);
        rounded=(mantissa+((1u<<(shift-1))-1)+((mantissa>>shift)&1))>>shift;
    } else {
        rounded=((unsigned)e<<10)+((mantissa+0xfff+((mantissa>>13)&1))>>13);
        if (rounded>=0x7c00) return -1;
    }
    *out=(uint16_t)(sign|rounded); return 0;
}
float fr_unhalf(uint16_t value) {
    unsigned sign=value>>15,e=value>>10&31,m=value&1023;
    float f=e?ldexpf((float)(1024+m),(int)e-25):ldexpf((float)m,-24);
    return sign?-f:f;
}
size_t fr_frame_next(const unsigned char *p,size_t left) {
    if (left<32 || memcmp(p,"FRK1",4)) return 0;
    size_t n=24+((size_t)p[20]<<24|(size_t)p[21]<<16|(size_t)p[22]<<8|p[23]);
    return n<=left?n:0;
}
int fr_fragment_read(const void *data,size_t len,fr_fragment *out) {
    if (!data || !out || len<32) return -1;
    const unsigned char *p=data;
    fr_fragment f={.type=p[4],.node=p[5],.burst=p[6],.session=get32(p+8),
        .step=get32(p+12),.layer=get16(p+16),.fragment=get16(p+18),
        .token=get16(p+24),.start=get16(p+26)};
    if (memcmp(p,"FRK1",4) || (f.type!=1 && f.type!=2) || p[7] || !f.session ||
        !f.burst || f.burst>FR_MAX_BURST || f.layer>=FR_LAYERS || f.fragment>=4*f.burst ||
        f.token!=f.fragment/4 || f.start!=f.fragment%4*640 || get16(p+28)!=640 || get16(p+30) ||
        (f.type==1?f.node!=255:f.node>=FR_CONNECTION_NODES)) return -1;
    size_t expected=1312+(f.type==1 && f.fragment%4==0?64:0);
    if (len!=expected || get32(p+20)!=len-24) return -1;
    for (unsigned d=0;d<640;++d) {
        unsigned h=get16(p+32+2*d); if ((h&0x7c00)==0x7c00) return -1;
        f.values[d]=fr_unhalf((uint16_t)h);
    }
    if (expected==1376) {
        for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) {
            f.expert[e]=(uint16_t)get16(p+1312+2*e);
            if (f.expert[e]>=FR_ROUTED_EXPERTS) return -1;
            for (unsigned j=0;j<e;++j) if (f.expert[j]==f.expert[e]) return -1;
        }
        for (unsigned e=0;e<FR_ACTIVE_EXPERTS;++e) {
            uint32_t bits=get32(p+1332+4*e); memcpy(&f.gate[e],&bits,4);
            if (!isfinite(f.gate[e]) || f.gate[e]<0) return -1;
        }
    }
    *out=f; return 0;
}
size_t fr_part_encode(unsigned char *out,size_t capacity,const fr_part *part,
                      uint32_t session,uint32_t step,unsigned node,unsigned fragment) {
    if (!out || !part || capacity<1312 || !session || node>=FR_CONNECTION_NODES ||
        part->layer>=FR_LAYERS || !part->burst || part->burst>FR_MAX_BURST || fragment>=4*part->burst) return 0;
    unsigned token=fragment/4,start=fragment%4*640; uint16_t h[640];
    for (unsigned d=0;d<640;++d) if (fr_half(part->y[token][start+d],&h[d])) return 0;
    header(out,32); out[4]=2; out[5]=(unsigned char)node; out[6]=(unsigned char)part->burst;
    put32(out+8,session); put32(out+12,step); put16(out+16,part->layer); put16(out+18,fragment); put32(out+20,1288);
    put16(out+24,token); put16(out+26,start); put16(out+28,640);
    for (unsigned d=0;d<640;++d) put16(out+32+2*d,h[d]);
    return 1312;
}
unsigned probe_count(unsigned kind) { return kind==0?1:kind==1?4:kind==2?16:0; }
unsigned probe_train_bytes(unsigned kind) { return kind==0?64:kind==1?5312:kind==2?21248:0; }
size_t probe_make(unsigned char *out, const fr_probe *p) {
    if (!p->session || !p->step || p->frag>=probe_count(p->kind) || p->reply>1 ||
        (p->reply?p->node>=FR_CONNECTION_NODES:p->node!=255)) return 0;
    size_t n=p->kind?(p->frag%4?1312:1376):64;
    header(out,n); out[4]=(unsigned char)(p->reply?4:3);
    out[5]=(unsigned char)p->node; out[6]=(unsigned char)(p->kind==2?4:1);
    put32(out+8,p->session); put32(out+12,p->step); put16(out+18,p->frag); put32(out+20,(uint32_t)n-24);
    put16(out+24,p->frag/4); put16(out+26,(p->frag%4)*640); put16(out+28,p->kind?640:16);
    for (size_t i=32; i<n; ++i) out[i]=(unsigned char)(i+p->frag);
    return n;
}
int probe_read(const void *data, size_t len, fr_probe *p) {
    if (len!=64 && len!=1312 && len!=1376) return -1;
    const unsigned char *in=data;
    if (memcmp(in,"FRK1",4) || (in[4]!=3 && in[4]!=4) || (in[6]!=1 && in[6]!=4)) return -1;
    *p=(fr_probe){.session=get32(in+8),.step=get32(in+12),.reply=in[4]==4,
        .node=in[5],.kind=len==64?0:in[6]==1?1:2,.frag=get16(in+18)};
    unsigned char expected[FR_PROBE_BYTES]; size_t n=probe_make(expected,p);
    return n==len && !memcmp(in,expected,len)?0:-1;
}
size_t probe_hello(unsigned char *out, const unsigned char cookie[16], const fr_identity *id) {
    size_t n=id?FR_ANNOUNCE_BYTES:FR_HELLO_BYTES;
    memset(out,0,n); memcpy(out,"FRH2",4); out[4]=(unsigned char)(id?2:1); memcpy(out+8,cookie,16);
    if (id) { put16(out+24,id->control_port); put16(out+26,id->data_port); memcpy(out+28,id->name,64); put32(out+92,(uint32_t)(id->id>>32)); put32(out+96,(uint32_t)id->id); }
    return n;
}
int probe_discovery_read(const void *data, size_t len, unsigned char cookie[16], fr_identity *id) {
    if (len!=FR_HELLO_BYTES && len!=FR_ANNOUNCE_BYTES) return -1;
    const unsigned char *in=data;
    if (memcmp(in,"FRH2",4) || in[4]!=(len==24?1:2) || in[5] || in[6] || in[7]) return -1;
    memcpy(cookie,in+8,16); memset(id,0,sizeof(*id));
    if (len==24) return 1;
    id->id=(uint64_t)get32(in+92)<<32|get32(in+96);
    if (!id->id) return -1;
    id->control_port=get16(in+24); id->data_port=get16(in+26);
    if (!id->control_port || !id->data_port || !in[28]) return -1;
    int ended=0;
    for (unsigned i=0; i<64; ++i) {
        unsigned char c=in[28+i];
        if (!c) ended=1;
        else if (ended || !((c>='a' && c<='z') || (c>='A' && c<='Z') ||
                 (c>='0' && c<='9') || c=='_' || c=='-' || c=='.')) return -1;
        id->name[i]=(char)c;
    }
    return ended?2:-1;
}
