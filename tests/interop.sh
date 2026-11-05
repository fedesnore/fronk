#!/bin/sh
# interop.sh - actual redis-cli and nc requests against three local nodes.
# Invoked inside local_cluster.sh, which owns nodes and checks their exits.
set -eu
: "${FRONK_CLUSTER_DIR:?run through local_cluster.sh}"
: "${FRONK_CLI:?select the built CLI}"
: "${FRONK_REDIS_TEST:?select the redis-cli driver}"
command -v redis-cli >/dev/null
command -v nc >/dev/null
for config in "$FRONK_CLUSTER_DIR"/node-*.conf; do
    port=$(awk '$1 == "control-port" {print $2}' "$config")
    tries=0
    until "$FRONK_CLI" -p "$port" PING >/dev/null 2>&1; do
        tries=$((tries + 1))
        if [ "$tries" -ge 100 ]; then echo "interop: node on $port did not start" >&2; exit 1; fi
        sleep 0.02
    done
    test "$(redis-cli --raw -h 127.0.0.1 -p "$port" PING)" = PONG
    "$FRONK_REDIS_TEST" "$port"
    hw=$("$FRONK_CLI" -p "$port" --config "$config" HW | tr -d '\r')
    printf '%s\n' "$hw" | grep -q '^cpu_model:'
    printf '%s\n' "$hw" | grep -q '^mem_cap_bytes:'
    info=$("$FRONK_CLI" -p "$port" --config "$config" INFO | tr -d '\r')
    for section in server backend memory session dataplane system; do
        printf '%s\n' "$info" | grep -qx "# $section"
    done
    printf '%s\n' "$info" | grep -qx 'mode:node'
    printf '%s\n' "$info" | grep -qx 'auth:hmac-sha256'
    test "$(printf '*1\r\n$4\r\nPING\r\n' | nc -w 1 127.0.0.1 "$port" | tr -d '\r')" = +PONG
    test "$(printf 'PING\r\n' | nc -w 1 127.0.0.1 "$port" | tr -d '\r')" = +PONG
    test "$(printf 'INFO\r\n' | nc -w 1 127.0.0.1 "$port" | tr -d '\r')" = '-NOAUTH authentication required'
    printf '[ok] interop: port %s, authenticated redis-cli/fronk-cli HW/INFO, nc PING/NOAUTH\n' "$port"
done
