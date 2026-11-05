/* resp.h - bounded RESP2 values and wire encoding for the control plane.
 *
 * Parsing borrows input bytes and uses the caller's flat value array, in
 * preorder. next skips an entire subtree. Values are valid only on success.
 * No allocation, NUL termination, or alignment of input is required.
 */
#ifndef FR_RESP_H
#define FR_RESP_H
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#define RESP_MAX_MESSAGE 65536
#define RESP_MAX_VALUES 128
#define RESP_MAX_DEPTH 8

typedef enum { RESP_STRING, RESP_ERROR, RESP_INTEGER, RESP_BULK, RESP_ARRAY } resp_type;
typedef struct {
    resp_type type;
    const char *data;
    size_t len;                 /* String bytes, or array element count. */
    int64_t integer;
    size_t next;
    int is_null;
} resp_value;

/* Bytes consumed, 0 for incomplete input, -1 for invalid/over-limit input.
 * Also accepts plain whitespace-separated inline commands for nc debugging;
 * inline quoting/escaping is deliberately unsupported. */
ssize_t resp_parse(const char *input, size_t len, resp_value *values, size_t cap);

typedef struct { char *data; size_t cap, len; int error; } resp_buf;
void resp_init(resp_buf *out, char *data, size_t cap);
void resp_simple(resp_buf *out, const char *text);
void resp_error(resp_buf *out, const char *text);
void resp_integer(resp_buf *out, int64_t value);
/* NULL data with zero length encodes a null bulk; "" encodes an empty bulk. */
void resp_bulk(resp_buf *out, const void *data, size_t len);
void resp_array(resp_buf *out, int64_t count);
/* Error is sticky; callers must discard an incomplete encoding on error. */
#endif
