/* test_main.c - check executable options and mandatory node credentials.
 *
 * Run the sanitizer-built executable, including usage errors and unavailable
 * modes. No successful invocation may suggest that a server is already running.
 */
#include "test.h"
#include <errno.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int run_args(const char *binary, const char *arg, const char *extra, const char *last,
               int expected, const char *message) {
    int pipefd[2];
    if (pipe(pipefd) < 0) return 0;
    fflush(NULL);
    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return 0; }
    if (pid == 0) {
        close(pipefd[0]);
        if (dup2(pipefd[1], STDOUT_FILENO) < 0 ||
            dup2(pipefd[1], STDERR_FILENO) < 0) _exit(126);
        close(pipefd[1]);
        execl(binary, binary, arg, extra, last, (char *)NULL);
        _exit(127);
    }
    close(pipefd[1]);
    char output[4096] = {0}, chunk[512];
    size_t used = 0;
    int readable = 1;
    for (;;) {
        ssize_t n = read(pipefd[0], chunk, sizeof(chunk));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { readable = n == 0; break; }
        size_t keep = (size_t)n;
        if (keep > sizeof(output) - 1 - used) keep = sizeof(output) - 1 - used;
        memcpy(output + used, chunk, keep);
        used += keep;
    }
    close(pipefd[0]);
    int status;
    if (waitpid(pid, &status, 0) != pid) return 0;
    return readable && WIFEXITED(status) && WEXITSTATUS(status) == expected &&
           strstr(output, message) != NULL;
}

static int run(const char *binary, const char *arg, const char *extra, int expected, const char *message) {
    return run_args(binary,arg,extra,NULL,expected,message);
}

int main(int argc, char **argv) {
    test_begin("main");
    if (argc != 2) { check(0, "test needs the fronk executable path"); return test_end(); }
    const char *binary = argv[1];
    check(run(binary, "--help", NULL, 0, "Usage:"), "help succeeds");
    check(run(binary, "--version", NULL, 0, "fronk " FR_VERSION), "version succeeds");
    check(run(binary, "--version", NULL, 0, "debug " FR_BUILD_OS " " FR_BUILD_ARCH),
          "build reports debug/platform identity");
    check(run(binary, NULL, NULL, 2, "Usage:"), "missing command fails");
    check(run(binary, "other", NULL, 2, "Usage:"), "unknown mode fails");
    check(run(binary, "--invalid", NULL, 2, "Usage:"), "unknown option fails");
    check(run(binary, "--version", "extra", 2, "Usage:"), "extra version argument fails");
    check(run(binary, "head", "--invalid", 2, "Usage:"), "invalid head option fails");
    check(run(binary, "node", "--invalid", 2, "Usage:"), "invalid node option fails");
    check(run(binary, "node", "--config", 2, "Usage:"), "missing config path fails");
    check(run_args(binary,"demo","--passes","-1",2,"Usage:"), "negative demo count rejected");
    check(run_args(binary,"demo","--passes","4oops",2,"Usage:"), "invalid demo count rejected");
    char path[]="/tmp/fronk-bind-test.XXXXXX";
    int fd=mkstemp(path), ok=fd>=0;
    if (fd>=0) {
        const char *config="bind 0.0.0.0\n";
        ok=write(fd,config,strlen(config))==(ssize_t)strlen(config);
        if (close(fd)<0) ok=0;
        if (ok) ok=run_args(binary,"node","--config",path,1,"password is required");
        if (unlink(path)<0) ok=0;
    }
    check(ok,"node refuses to start without a password");
    return test_end();
}
