/* test_local_cluster.c - exercise launcher ownership without a network server.
 *
 * This executable also supplies disposable node and client fixtures. The real
 * RESP node arrives in milestone 1; these fixtures test process cleanup, config
 * isolation and failure propagation without claiming network coverage.
 */
#include "test.h"
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int node_fixture(int argc, char **argv) {
    if (argc != 5 || strcmp(argv[2], "--config") || strcmp(argv[4], "--debug")) return 2;
    if (getenv("FRONK_FIXTURE_FAIL") && strstr(argv[3], "node-2.conf")) return 7;
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s.ready", argv[3]) >= (int)sizeof(path)) return 2;
    FILE *ready = fopen(path, "w");
    if (!ready) return 2;
    if (fclose(ready) != 0) return 2;
    for (;;) pause();
}

static int inspect(const char *mode) {
    const char *dir = getenv("FRONK_CLUSTER_DIR");
    const char *record = getenv("FRONK_FIXTURE_RECORD");
    if (!dir || !record) return 2;
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/pids", dir) >= (int)sizeof(path)) return 2;
    FILE *pids = fopen(path, "r"), *out = fopen(record, "w");
    if (!pids || !out) return 2;
    int count = 0;
    long pid;
    if (fprintf(out, "%s\n", dir) < 0) return 2;
    while (fscanf(pids, "%ld", &pid) == 1) {
        if (pid <= 0 || fprintf(out, "%ld\n", pid) < 0) return 2;
        ++count;
    }
    if (ferror(pids) || fclose(pids) != 0 || fclose(out) != 0) return 2;
    const char *expected = getenv("FRONK_EXPECTED_N");
    if (!expected || count != atoi(expected) || count < 1 || count > 20) return 2;

    for (int i = 1; i <= count; ++i) {
        if (snprintf(path, sizeof(path), "%s/node-%d.conf", dir, i) >= (int)sizeof(path)) return 2;
        FILE *config = fopen(path, "r");
        if (!config) return 2;
        char key[64], value[128];
        int loopback = 0, control = 0, data = 0;
        while (fscanf(config, "%63s %127s", key, value) == 2) {
            if (!strcmp(key, "bind")) loopback = !strcmp(value, "127.0.0.1");
            if (!strcmp(key, "control-port")) control = atoi(value);
            if (!strcmp(key, "data-port")) data = atoi(value);
        }
        if (ferror(config) || fclose(config) != 0) return 2;
        if (!loopback || control != 19400 + 2 * (i - 1) || data != control + 1) return 3;
        if (snprintf(path, sizeof(path), "%s/node-%d.conf.ready", dir, i) >= (int)sizeof(path)) return 2;
        int tries;
        for (tries = 0; tries < 200 && access(path, F_OK) != 0; ++tries) {
            struct timespec delay = {0, 10000000};
            if (nanosleep(&delay, NULL) != 0) return 2;
        }
        if (tries == 200) return 4;
    }
    if (!strcmp(mode, "signal")) return kill(getppid(), SIGTERM) == 0 ? 0 : 2;
    return !strcmp(mode, "command-fail") ? 7 : 0;
}

static int run(const char *self, const char *count, const char *mode, int expected) {
    char record[] = "/tmp/fronk-launch-test.XXXXXX";
    int fd = mkstemp(record);
    if (fd < 0) return 0;
    close(fd);
    FILE *log = tmpfile();
    if (!log) { unlink(record); return 0; }
    fflush(NULL);
    pid_t child = fork();
    if (child == 0) {
        if (dup2(fileno(log), STDOUT_FILENO) < 0 || dup2(fileno(log), STDERR_FILENO) < 0 ||
            setenv("FRONK_BIN", !strcmp(mode, "missing") ? "/nonexistent/fronk" : self, 1) ||
            setenv("FRONK_FIXTURE_RECORD", record, 1) ||
            setenv("FRONK_EXPECTED_N", count, 1) ||
            setenv("FRONK_BASE_PORT", !strcmp(mode, "bad-port") ? "65535" : "19400", 1)) _exit(126);
        if (!strcmp(mode, "node-fail") && setenv("FRONK_FIXTURE_FAIL", "1", 1)) _exit(126);
        execl("/bin/sh", "sh", "tests/local_cluster.sh", count, self,
              "--inspect", mode, (char *)NULL);
        _exit(127);
    }
    int status = 0;
    int ok = child > 0 && waitpid(child, &status, 0) == child &&
             WIFEXITED(status) && WEXITSTATUS(status) == expected;
    FILE *saved = fopen(record, "r");
    char dir[PATH_MAX];
    if (!saved) ok = 0;
    else {
        if (fgets(dir, sizeof(dir), saved)) {
            dir[strcspn(dir, "\n")] = '\0';
            if (access(dir, F_OK) == 0 || errno != ENOENT) ok = 0;
            long pid;
            while (fscanf(saved, "%ld", &pid) == 1)
                if (pid <= 0 || kill((pid_t)pid, 0) == 0 || errno != ESRCH) ok = 0;
        } else if (expected == 0 || !strcmp(mode, "command-fail") ||
                   !strcmp(mode, "node-fail") || !strcmp(mode, "signal")) ok = 0;
        if (ferror(saved) || fclose(saved) != 0) ok = 0;
    }
    if (!ok) {
        rewind(log);
        int c;
        while ((c = fgetc(log)) != EOF) fputc(c, stderr);
    }
    if (fclose(log) != 0 || unlink(record) != 0) ok = 0;
    return ok;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "node")) return node_fixture(argc, argv);
    if (argc == 3 && !strcmp(argv[1], "--inspect")) return inspect(argv[2]);
    test_begin("local_cluster");
    check(run(argv[0], "1", "success", 0), "one node, isolated config and complete cleanup");
    check(run(argv[0], "3", "success", 0), "three nodes, distinct ports and complete cleanup");
    check(run(argv[0], "20", "success", 0), "maximum local node count");
    check(run(argv[0], "0", "invalid", 2), "reject zero nodes");
    check(run(argv[0], "21", "invalid", 2), "reject excessive node count");
    check(run(argv[0], "3x", "invalid", 2), "reject malformed node count");
    check(run(argv[0], "3", "missing", 1), "reject missing node executable");
    check(run(argv[0], "3", "bad-port", 2), "reject overflowing port range");
    check(run(argv[0], "3", "command-fail", 7), "client failure propagates and cleans nodes");
    check(run(argv[0], "3", "node-fail", 4), "failed startup is detected by client; siblings cleaned");
    check(run(argv[0], "3", "signal", 143), "SIGTERM cleans nodes and temporary files");
    return test_end();
}
