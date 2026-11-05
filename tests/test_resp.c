/* test_resp.c - wire fixtures, partial reads, nesting and hostile lengths.
 * Every prefix of a valid frame must wait, never expose a partial value tree.
 */
#include "test.h"
#include "../src/resp.h"
#include <limits.h>
#include <string.h>

static resp_value v[RESP_MAX_VALUES];
static int parses(const char *s) { return resp_parse(s, strlen(s), v, RESP_MAX_VALUES) == (ssize_t)strlen(s); }
static int invalid(const char *s) { return resp_parse(s, strlen(s), v, RESP_MAX_VALUES) == -1; }

int main(void) {
    test_begin("resp");
    check(parses("+OK\r\n") && v[0].type == RESP_STRING && v[0].len == 2, "simple string");
    check(parses("-ERR bad\r\n") && v[0].type == RESP_ERROR, "error");
    check(parses(":-9223372036854775808\r\n") && v[0].integer == INT64_MIN, "minimum integer");
    check(parses(":9223372036854775807\r\n") && v[0].integer == INT64_MAX, "maximum integer");
    check(parses("$0\r\n\r\n") && v[0].len == 0 && !v[0].is_null, "empty bulk");
    check(parses("$-1\r\n") && v[0].is_null, "null bulk");
    check(parses("*-1\r\n") && v[0].is_null, "null array");
    check(parses("*0\r\n") && !v[0].is_null && v[0].next == 1, "empty array");
    const char binary[] = "$3\r\na\0b\r\n";
    check(resp_parse(binary, sizeof(binary)-1, v, RESP_MAX_VALUES) == sizeof(binary)-1 &&
          v[0].len == 3 && !memcmp(v[0].data, "a\0b", 3), "binary-safe bulk");
    check(parses("*2\r\n*1\r\n:7\r\n+end\r\n") && v[0].next == 4 &&
          v[1].next == 3 && v[2].integer == 7 && v[3].type == RESP_STRING, "nested preorder tree");
    const char *frame = "*2\r\n$4\r\nINFO\r\n$6\r\nserver\r\n";
    int prefixes = 1;
    for (size_t i = 0; i < strlen(frame); ++i)
        if (resp_parse(frame, i, v, RESP_MAX_VALUES) != 0) prefixes = 0;
    check(prefixes && parses(frame), "every split boundary waits then completes");
    check(resp_parse("+OK\r\n:3\r\n", 9, v, RESP_MAX_VALUES) == 5, "one frame consumed from pipeline");
    check(parses("PING\r\n") && v[0].type == RESP_ARRAY && v[0].len == 1 && v[1].len == 4, "nc inline command");
    check(parses("  INFO\tserver\n") && v[0].len == 2 && v[2].len == 6, "inline whitespace and LF");
    check(invalid("?bad\r\n"), "unknown type");
    check(invalid("+bad\n"), "RESP requires CRLF");
    check(invalid("+bad\rx"), "invalid CRLF");
    check(invalid(":\r\n") && invalid(":+1\r\n") && invalid(":1x\r\n"), "strict signed decimal");
    check(invalid(":9223372036854775808\r\n") && invalid(":-9223372036854775809\r\n"), "integer overflow");
    check(invalid("$-2\r\n") && invalid("*-2\r\n"), "only minus one is null");
    check(invalid("$999999999999999999999\r\n") && invalid("$65536\r\n"), "oversized bulk rejected early");
    check(invalid("*128\r\n"), "array exceeds value budget");
    check(invalid("$1\r\nxZZ"), "bulk terminator checked");
    check(resp_parse(frame, strlen(frame), v, 2) == -1, "caller value budget enforced");
    check(resp_parse(NULL, 0, v, RESP_MAX_VALUES) == 0, "empty input");
    check(resp_parse("x", 1, NULL, 1) == -1, "invalid output rejected");
    check(invalid("*1\r\n*1\r\n*1\r\n*1\r\n*1\r\n*1\r\n*1\r\n*1\r\n*1\r\n:0\r\n"), "bounded nesting");
    char buf[256]; resp_buf out;
    resp_init(&out, buf, sizeof(buf));
    resp_array(&out, 3); resp_simple(&out, "OK"); resp_integer(&out, INT64_MIN); resp_bulk(&out, "a\0b", 3);
    check(!out.error && resp_parse(buf, out.len, v, RESP_MAX_VALUES) == (ssize_t)out.len &&
          v[0].len == 3 && v[2].integer == INT64_MIN && !memcmp(v[3].data,"a\0b",3), "encode/decode every non-error scalar");
    resp_init(&out, buf, sizeof(buf)); resp_error(&out, "ERR nope");
    check(!out.error && resp_parse(buf,out.len,v,RESP_MAX_VALUES)>0 && v[0].type == RESP_ERROR, "encode error");
    resp_init(&out, buf, 5); resp_simple(&out, "OK");
    check(!out.error && out.len == 5, "exact buffer capacity");
    resp_bulk(&out, "x", 1); size_t stopped = out.len; resp_simple(&out, "again");
    check(out.error && out.len == stopped, "overflow is sticky");
    resp_init(&out, buf, sizeof(buf)); resp_simple(&out, "bad\r\nvalue");
    check(out.error, "no response injection in simple strings");
    resp_init(&out, buf, sizeof(buf)); resp_array(&out, -2);
    check(out.error, "invalid array encoder length");
    static char large[RESP_MAX_MESSAGE];
    memset(large, '0', sizeof(large)); large[0] = '$';
    large[sizeof(large)-2] = '\r'; large[sizeof(large)-1] = '\n';
    check(resp_parse(large, sizeof(large), v, RESP_MAX_VALUES) == -1,
          "maximum-length bulk header leaves no room for terminator");
    uint32_t random = 1; int bounded = 1;
    for (int trial = 0; trial < 5000; ++trial) {
        char input[128]; size_t n = (size_t)trial % sizeof(input);
        for (size_t i = 0; i < n; ++i) {
            random ^= random << 13; random ^= random >> 17; random ^= random << 5;
            input[i] = (char)random;
        }
        ssize_t got = resp_parse(input, n, v, 16);
        if (got < -1 || got > (ssize_t)n || (got > 0 && v[0].next > 16)) bounded = 0;
    }
    check(bounded, "deterministic malformed-byte corpus stays within budgets");
    return test_end();
}
