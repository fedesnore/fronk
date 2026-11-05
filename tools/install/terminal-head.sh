#!/usr/bin/env bash
# Keep stdin on the user's terminal. exec preserves the PID recorded by Fronk.
set -eu
state=$1; shift
umask 077
log="$state/terminal-head.$$.log"
echo "To watch this head in your browser, open the head's installer manager and choose web-console."
exec ./fronk head "$@" > >(tee "$log") 2>&1
