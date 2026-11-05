/* test_test.c - ensure the tiny test runner cannot report false success.
 *
 * Expected failures run in a child with output suppressed; only their status
 * is inspected, keeping the normal suite output unambiguous.
 */
#include "test.h"
#include <sys/wait.h>
#include <unistd.h>

static int child_status(int empty) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stdout) ||
            !freopen("/dev/null", "w", stderr)) _exit(2);
        test_begin("expected failure");
        if (!empty) check(0, "deliberate failed check");
        exit(test_end());
    }
    int status;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
}

int main(void) {
    test_begin("test");
    int evaluated = 0;
    check(++evaluated == 1, "check evaluates its expression once");
    check(evaluated == 1, "no repeated evaluation");
    check(child_status(0) == EXIT_FAILURE, "a failed check produces failure status");
    check(child_status(1) == EXIT_FAILURE, "an empty suite produces failure status");
    return test_end();
}
