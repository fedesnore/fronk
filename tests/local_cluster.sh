#!/bin/sh
# local_cluster.sh - own N disposable localhost nodes around one test command.
#
# The command receives FRONK_CLUSTER_DIR with configs, logs and child PIDs.
# It must wait for protocol readiness and detect startup failures: a live PID
# alone is not proof of readiness. Cleanup runs on success, error and signals.
# Lifecycle unit tests use fixtures; interop tests launch real RESP nodes.

set -eu
if [ "$#" -lt 2 ]; then
    echo "Usage: tests/local_cluster.sh N command [args ...]" >&2
    exit 2
fi
count=$1
shift
case "$count" in
    [1-9]|1[0-9]|20) ;;
    *) echo "local_cluster: N must be an integer from 1 to 20" >&2; exit 2 ;;
esac
base=${FRONK_BASE_PORT:-17700}
case "$base" in
    ''|0*|*[!0-9]*) echo "local_cluster: invalid base port" >&2; exit 2 ;;
esac
if [ "${#base}" -gt 5 ] || [ "$base" -lt 1024 ] ||
   [ "$base" -gt "$((65535 - 2 * count))" ]; then
    echo "local_cluster: port range must fit within 1024..65535" >&2
    exit 2
fi
root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
binary=${FRONK_BIN:-$root/fronk}
test_interfaces=${FRONK_TEST_INTERFACES:-}
test_bind=127.0.0.1
if [ -n "$test_interfaces" ]; then
    case "$test_interfaces" in *[!0-9.,]*) echo "local_cluster: invalid test interface list" >&2; exit 2 ;; esac
    test_bind=0.0.0.0
fi
if [ ! -x "$binary" ]; then
    echo "local_cluster: node executable not found: $binary" >&2
    exit 1
fi
umask 077
FRONK_CLUSTER_DIR=$(mktemp -d "${TMPDIR:-/tmp}/fronk-local.XXXXXX")
export FRONK_CLUSTER_DIR
pids=
cleanup() {
    status=$?
    trap - 0 HUP INT TERM
    for pid in $pids; do kill -TERM "$pid" 2>/dev/null || :; done
    for pid in $pids; do
        child_status=0
        wait "$pid" 2>/dev/null || child_status=$?
        case "$child_status" in
            0|143) ;;
            *) if [ "$status" -eq 0 ]; then status=1; fi ;;
        esac
    done
    if [ "$status" -ne 0 ]; then
        for log in "$FRONK_CLUSTER_DIR"/*.log; do
            if [ -f "$log" ]; then cat "$log" >&2; fi
        done
    fi
    rm -rf "$FRONK_CLUSTER_DIR"
    exit "$status"
}
trap cleanup 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
: > "$FRONK_CLUSTER_DIR/pids"
discovery=$((base + 2 * count))
cat > "$FRONK_CLUSTER_DIR/head.conf" <<EOF
name local-head
state-dir $FRONK_CLUSTER_DIR/head-state
profile-file $FRONK_CLUSTER_DIR/profile.txt
interface 127.0.0.1
discovery-port $discovery
multicast 239.7.7.7
password fronk-local-test
EOF
if [ -n "$test_interfaces" ]; then echo "interfaces $test_interfaces" >> "$FRONK_CLUSTER_DIR/head.conf"; fi
i=1
while [ "$i" -le "$count" ]; do
    control=$((base + 2 * (i - 1)))
    config=$FRONK_CLUSTER_DIR/node-$i.conf
    cat > "$config" <<EOF
name local-$i
state-dir $FRONK_CLUSTER_DIR/state-$i
bind $test_bind
control-port $control
data-port $((control + 1))
interface 127.0.0.1
discovery-port $discovery
multicast 239.7.7.7
password fronk-local-test
EOF
    if [ -n "$test_interfaces" ]; then
        if [ "${FRONK_TEST_NODE_AUTO:-0}" = 1 ]; then echo "interfaces auto" >> "$config"
        else echo "interfaces $test_interfaces" >> "$config"; fi
    fi
    "$binary" node --config "$config" --debug > "$FRONK_CLUSTER_DIR/node-$i.log" 2>&1 &
    pid=$!
    pids="$pids $pid"
    echo "$pid" >> "$FRONK_CLUSTER_DIR/pids"
    i=$((i + 1))
done
status=0
"$@" || status=$?
for pid in $pids; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "local_cluster: node $pid exited before the test completed" >&2
        if [ "$status" -eq 0 ]; then status=1; fi
    fi
done
exit "$status"
