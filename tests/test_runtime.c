/* Startup policy must be identical through the GUI, CLI and installer while
 * explicit diagnostic settings (including empty values) remain reproducible. */
#include "test.h"
#include "../src/runtime.h"
#include <stdlib.h>
#include <string.h>

int main(void) {
    test_begin("runtime");
    unsetenv("FRONK_REQUIRE_ALL");
    unsetenv("FRONK_EVENT_RESUME");
    unsetenv("FRONK_METAL_KEEPALIVE");
    check(!fr_head_defaults(), "initialize ordinary head policy");
    check(!strcmp(getenv("FRONK_REQUIRE_ALL"), "1"), "require every planned slice");
    check(!strcmp(getenv("FRONK_EVENT_RESUME"), "1"), "event resume leaves GPU available");
    check(!strcmp(getenv("FRONK_METAL_KEEPALIVE"), "0"), "head does not compete with keepalive");
    setenv("FRONK_REQUIRE_ALL", "0", 1);
    setenv("FRONK_EVENT_RESUME", "", 1);
    setenv("FRONK_METAL_KEEPALIVE", "1000", 1);
    check(!fr_head_defaults(), "repeat initialization");
    check(!strcmp(getenv("FRONK_REQUIRE_ALL"), "0") &&
          !strcmp(getenv("FRONK_EVENT_RESUME"), "") &&
          !strcmp(getenv("FRONK_METAL_KEEPALIVE"), "1000"), "retain explicit diagnostic overrides");
    return test_end();
}
