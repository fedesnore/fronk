/* Encode and validate the bounded fragments inside FRK1 layer messages.
 * Activations are supplied as fp16 bit patterns, already converted by the
 * caller into preallocated storage. All wire integers/float bits are big
 * endian. Fragment 4*token alone carries that token's 64-byte routing record.
 */
#ifndef FR_WIRE_H
#define FR_WIRE_H
#include "backend.h"
#define FR_WIRE_HEADER 24
#define FR_RANGE_HEADER 8
#define FR_FRAGMENT_DIMS 640
#define FR_TOKEN_FRAGMENTS 4
#define FR_ROUTING_BYTES 64
#define FR_REQ_FRAGMENT_MAX 1376
#define FR_UDP_PAYLOAD_MAX 1472
/* One message (all REQ fragments of a barrier, or one node's PART) travels as a
 * single UDP datagram of concatenated fragments; the kernel IP-fragments it.
 * macOS charges ~40 us per sendto, so one send replaces 4*burst (BENCH.md). */
#define FR_FRAME_MAX (4*FR_MAX_BURST*FR_REQ_FRAGMENT_MAX)
/* Length of the fragment at p within a frame of `left` bytes, 0 if malformed. */
size_t fr_frame_next(const unsigned char *p,size_t left);
/* Returns encoded UDP bytes; zero rejects invalid input/capacity before any
 * output write. No allocation or copying into temporary packet storage. */
size_t fr_req_encode(unsigned char *out,size_t capacity,const fr_req *req,
                     const uint16_t activations[FR_MAX_BURST][FR_HIDDEN],
                     uint32_t session,uint32_t step,unsigned fragment);
typedef struct {
    uint32_t session,step;
    unsigned type,node,burst,layer,fragment,token,start;
    float values[FR_FRAGMENT_DIMS];
    uint16_t expert[FR_ACTIVE_ROUTED];
    float gate[FR_ACTIVE_EXPERTS];
} fr_fragment;
/* Finite IEEE binary16, round-to-nearest/ties-to-even. Overflow is an error. */
int fr_half(float value,uint16_t *out);
float fr_unhalf(uint16_t value);
int fr_fragment_read(const void *data,size_t len,fr_fragment *out);
size_t fr_part_encode(unsigned char *out,size_t capacity,const fr_part *part,
                      uint32_t session,uint32_t step,unsigned node,unsigned fragment);
#endif
