/* The terminal, web console and installer must start the same engine policy.
 * These were previously shell settings on the development head. Keep this
 * list small: experimental kernels/schedulers are opt-in at their call sites.
 * This function runs before threads start. setenv(..., 0) preserves overrides,
 * including an explicitly empty value used by some diagnostic switches. */
#include "runtime.h"
#include <stdlib.h>

int fr_head_defaults(void) {
    /* Missing slices must fail generation, never silently omit expert work. */
    if (setenv("FRONK_REQUIRE_ALL", "1", 0)) return -1;
    /* Event waits allow the local expert GPU queue to run. GPU polling can
     * hold the GPU while the very work needed to resume it is queued. */
    if (setenv("FRONK_EVENT_RESUME", "1", 0)) return -1;
    /* Synthetic GPU keepalive competes with dense and local expert work on
     * a shared head GPU. Nodes keep their separately calibrated policy. */
    if (setenv("FRONK_METAL_KEEPALIVE", "0", 0)) return -1;
    return 0;
}
