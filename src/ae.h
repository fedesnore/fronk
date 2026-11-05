/* ae.h - small level-triggered readiness interface for the control thread.
 *
 * The caller owns descriptors and supplies generation tags. Returned events
 * may predate a close/reuse: compare tags before touching a connection slot.
 * This module never owns callbacks, timers, threads, or descriptor lifetimes.
 */
#ifndef FR_AE_H
#define FR_AE_H
#include <stdint.h>
#define AE_READ 1
#define AE_WRITE 2
#define AE_MAX_EVENTS 64
typedef struct { uint64_t tag; int mask; } ae_event;
int ae_open(void);
/* old_mask is the current registration, zero for a new fd. new_mask zero
 * removes it. On update failure the caller must close the watched fd. */
int ae_set(int loop, int fd, int old_mask, int new_mask, uint64_t tag);
/* timeout_ms: -1 indefinitely, 0 poll, positive bounded wait. EINTR yields 0. */
int ae_wait(int loop, ae_event *events, int cap, int timeout_ms);
const char *ae_name(void);
#endif
