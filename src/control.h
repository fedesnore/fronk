/* control.h - bounded synchronous RESP exchanges used by CLI and CONNECTION.
 * The caller owns the connected socket and buffers. deadline is an absolute
 * monotonic millisecond value shared by authentication and subsequent calls.
 * NULL means success; errors are static diagnostic strings without secrets.
 * On transport/framing failure values[0] is cleared; no response is valid.
 */
#ifndef FR_CONTROL_H
#define FR_CONTROL_H
#include "resp.h"
#include <stdint.h>
const char *control_exchange(int fd, const char *request, size_t len,
                            char response[RESP_MAX_MESSAGE], resp_value *values, int64_t deadline);
const char *control_authenticate(int fd, const char *password,
                                char response[RESP_MAX_MESSAGE], resp_value *values, int64_t deadline);
#endif
