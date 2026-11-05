/* resp.c - bounded RESP2 parsing and encoding, without heap allocation.
 *
 * TCP boundaries have no protocol meaning: incomplete frames return zero.
 * Lengths are validated before arithmetic, and arrays consume a fixed caller
 * budget. Inline requests are only a convenience for simple nc commands.
 */
#include "resp.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int line(const char *s, size_t len, size_t start, size_t *end) {
    for (size_t i = start; i < len; ++i) {
        if (s[i] == '\n') return -1;
        if (s[i] != '\r') continue;
        if (i + 1 == len) return 0;
        if (s[i+1] != '\n') return -1;
        *end = i;
        return 1;
    }
    return 0;
}

static int number(const char *s, size_t n, int64_t *out) {
    if (!n) return -1;
    int negative = s[0] == '-';
    size_t i = negative ? 1 : 0;
    if (i == n) return -1;
    uint64_t value = 0, limit = (uint64_t)INT64_MAX + (unsigned)negative;
    for (; i < n; ++i) {
        if (s[i] < '0' || s[i] > '9') return -1;
        unsigned digit = (unsigned)(s[i] - '0');
        if (value > (limit - digit) / 10) return -1;
        value = value * 10 + digit;
    }
    *out = negative && value ? -(int64_t)(value - 1) - 1 : (int64_t)value;
    return 0;
}

static int value(const char *s, size_t len, size_t *pos, resp_value *v,
                 size_t cap, size_t *used, unsigned depth) {
    if (*pos == len) return 0;
    if (*used == cap || depth >= RESP_MAX_DEPTH) return -1;
    size_t index = (*used)++, start = (*pos)++, end;
    resp_value *out = &v[index];
    memset(out, 0, sizeof(*out));
    char type = s[start];
    if (!strchr("+-:$*", type) || type == '\0') return -1;
    int rc = line(s, len, *pos, &end);
    if (rc <= 0) return rc;
    out->data = s + *pos;
    out->len = end - *pos;
    *pos = end + 2;
    if (type == '+' || type == '-') out->type = type == '+' ? RESP_STRING : RESP_ERROR;
    else {
        int64_t n;
        if (number(out->data, out->len, &n) < 0) return -1;
        out->data = NULL; out->len = 0;
        if (type == ':') { out->type = RESP_INTEGER; out->integer = n; }
        else {
            if (n < -1) return -1;
            out->type = type == '$' ? RESP_BULK : RESP_ARRAY;
            out->is_null = n == -1;
            if (n >= 0 && type == '$') {
                if (*pos > RESP_MAX_MESSAGE - 2 ||
                    (uint64_t)n > RESP_MAX_MESSAGE - *pos - 2) return -1;
                if ((uint64_t)n + 2 > len - *pos) return 0;
                out->data = s + *pos; out->len = (size_t)n;
                *pos += (size_t)n;
                if (s[*pos] != '\r' || s[*pos+1] != '\n') return -1;
                *pos += 2;
            } else if (n >= 0) {
                if ((uint64_t)n > cap - *used) return -1;
                out->len = (size_t)n;
                for (int64_t i = 0; i < n; ++i) {
                    rc = value(s, len, pos, v, cap, used, depth + 1);
                    if (rc <= 0) return rc;
                }
            }
        }
    }
    out->next = *used;
    return 1;
}

static ssize_t inline_command(const char *s, size_t len, resp_value *v, size_t cap) {
    size_t end = 0;
    while (end < len && s[end] != '\n') ++end;
    if (end == len) return 0;
    size_t text_end = end;
    if (text_end && s[text_end-1] == '\r') --text_end;
    size_t used = 1, i = 0;
    memset(&v[0], 0, sizeof(v[0])); v[0].type = RESP_ARRAY;
    while (i < text_end) {
        if (s[i] == ' ' || s[i] == '\t') { ++i; continue; }
        size_t start = i;
        while (i < text_end && s[i] != ' ' && s[i] != '\t') {
            if ((unsigned char)s[i] < 32 || (unsigned char)s[i] > 126) return -1;
            ++i;
        }
        if (used == cap) return -1;
        v[used] = (resp_value){.type=RESP_BULK, .data=s+start, .len=i-start, .next=used+1};
        ++used;
    }
    v[0].len = used - 1; v[0].next = used;
    return (ssize_t)(end + 1);
}

ssize_t resp_parse(const char *input, size_t len, resp_value *values, size_t cap) {
    if (!values || !cap || (!input && len) || len > RESP_MAX_MESSAGE) return -1;
    if (cap > RESP_MAX_VALUES) cap = RESP_MAX_VALUES;
    if (!len) return 0;
    if ((input[0] >= 'A' && input[0] <= 'Z') || (input[0] >= 'a' && input[0] <= 'z') ||
        input[0] == ' ' || input[0] == '\t') return inline_command(input, len, values, cap);
    size_t pos = 0, used = 0;
    int rc = value(input, len, &pos, values, cap, &used, 0);
    return rc > 0 ? (ssize_t)pos : rc;
}

void resp_init(resp_buf *out, char *data, size_t cap) {
    *out = (resp_buf){.data=data, .cap=cap, .error=!data && cap != 0};
}
static void append(resp_buf *out, const void *data, size_t len) {
    if (out->error) return;
    if (len > out->cap - out->len) { out->error = 1; return; }
    if (len) memcpy(out->data + out->len, data, len);
    out->len += len;
}
static void text(resp_buf *out, char prefix, const char *s) {
    if (!s || strpbrk(s, "\r\n")) { out->error = 1; return; }
    append(out, &prefix, 1); append(out, s, strlen(s)); append(out, "\r\n", 2);
}
void resp_simple(resp_buf *out, const char *s) { text(out, '+', s); }
void resp_error(resp_buf *out, const char *s) { text(out, '-', s); }
static void header(resp_buf *out, char prefix, int64_t n) {
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%c%" PRId64 "\r\n", prefix, n);
    if (len < 0 || (size_t)len >= sizeof(buf)) { out->error = 1; return; }
    append(out, buf, (size_t)len);
}
void resp_integer(resp_buf *out, int64_t n) { header(out, ':', n); }
void resp_bulk(resp_buf *out, const void *data, size_t len) {
    if (len > RESP_MAX_MESSAGE || (!data && len)) { out->error = 1; return; }
    if (!data) { header(out, '$', -1); return; }
    header(out, '$', (int64_t)len); append(out, data, len); append(out, "\r\n", 2);
}
void resp_array(resp_buf *out, int64_t count) {
    if (count < -1 || count >= RESP_MAX_VALUES) { out->error = 1; return; }
    header(out, '*', count);
}
