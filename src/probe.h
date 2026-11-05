/* probe.h - fixed CONNECTION discovery messages and MTU-sized RTT probes.
 * Discovery uses FRH2 plus a 128-bit run cookie. Probes use the binding FRK1
 * 24-byte header and 8-byte range header, PING/PONG types, no inference data.
 */
#ifndef FR_PROBE_H
#define FR_PROBE_H
#include <stddef.h>
#include <stdint.h>
#define FR_CONNECTION_NODES 20
#define FR_PROBE_BYTES 1376
#define FR_HELLO_BYTES 24
#define FR_ANNOUNCE_BYTES 100
typedef struct { uint64_t id; char name[64]; unsigned control_port, data_port; } fr_identity;
typedef struct { uint32_t session, step; unsigned kind, frag, node, reply; } fr_probe;
/* kind 0: 64B. Kinds 1/2 match burst 1/4 REQ sizes: four fragments per
 * token, first 1,376B and remaining three 1,312B (routing bytes included). */
unsigned probe_count(unsigned kind);
unsigned probe_train_bytes(unsigned kind);
size_t probe_make(unsigned char *out, const fr_probe *probe);
int probe_read(const void *data, size_t len, fr_probe *probe);
/* NULL identity encodes HELLO. Read returns 1 HELLO, 2 announcement, -1 invalid. */
size_t probe_hello(unsigned char *out, const unsigned char cookie[16], const fr_identity *identity);
int probe_discovery_read(const void *data, size_t len, unsigned char cookie[16], fr_identity *identity);
#endif
