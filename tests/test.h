/* test.h - small, allocation-free checks for standalone C test programs.
 *
 * A failed check records its source location and allows later checks to run.
 * Empty suites fail too, so an accidentally removed test cannot pass silently.
 */
#ifndef FR_TEST_H
#define FR_TEST_H

#include <stdio.h>
#include <stdlib.h>

static const char *test_name;
static unsigned test_passed, test_failed;

static inline void test_begin(const char *name) {
    test_name = name;
    test_passed = test_failed = 0;
}

static inline void test_check(int ok, const char *message,
                              const char *file, int line) {
    if (ok) {
        ++test_passed;
    } else {
        ++test_failed;
        fprintf(stderr, "%s:%d: %s: %s\n", file, line, test_name, message);
    }
}

#define check(condition, message) \
    test_check(!!(condition), (message), __FILE__, __LINE__)

static inline int test_end(void) {
    if (test_passed + test_failed == 0)
        test_check(0, "empty suite", __FILE__, __LINE__);
    printf("[%s] %s: %u passed, %u failed\n",
           test_failed ? "fail" : "ok", test_name, test_passed, test_failed);
    return test_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

#endif
