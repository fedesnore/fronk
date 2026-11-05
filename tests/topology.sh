#!/bin/sh
# Run only when explicitly requested: uses a second existing local interface.
set -eu
root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
FRONK_TEST_INTERFACES=$("$root/build/test/test_topology" --interfaces)
export FRONK_TEST_INTERFACES
# This test intentionally includes localhost; auto now excludes it.
FRONK_TEST_NODE_AUTO=0
export FRONK_TEST_NODE_AUTO
FRONK_BASE_PORT=25700
FRONK_BIN=$root/build/test/fronk
FRONK_CLI=$root/build/test/fronk-cli
FRONK_CONNECTION_TEST=$root/build/test/test_topology
export FRONK_BASE_PORT FRONK_BIN FRONK_CLI FRONK_CONNECTION_TEST
"$root/tests/local_cluster.sh" 3 sh "$root/tests/connection.sh"
"$root/build/test/test_fanout" "${FRONK_TEST_INTERFACES#*,}"
