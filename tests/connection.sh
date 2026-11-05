#!/bin/sh
# connection.sh - wait for three real nodes before testing CONNECTION.
set -eu
: "${FRONK_CLUSTER_DIR:?run through local_cluster.sh}"
: "${FRONK_CLI:?select CLI}"
for config in "$FRONK_CLUSTER_DIR"/node-*.conf; do
    port=$(awk '$1 == "control-port" {print $2}' "$config")
    tries=0
    until "$FRONK_CLI" -p "$port" PING >/dev/null 2>&1; do
        tries=$((tries + 1))
        if [ "$tries" -ge 100 ]; then echo "connection: node did not start" >&2; exit 1; fi
        sleep 0.02
    done
done
if [ -n "${FRONK_CONNECTION_TEST:-}" ]; then
    exec "$FRONK_CONNECTION_TEST" "$FRONK_CLUSTER_DIR/head.conf"
fi
exec "$FRONK_BIN" head --config "$FRONK_CLUSTER_DIR/head.conf"
