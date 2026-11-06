#!/usr/bin/env bash
# Read-only node checks, sent over SSH before the head or any node is installed.
set -euo pipefail
prefix=$1
fail() { printf 'Node prerequisite: %s\n' "$*" >&2; exit 1; }
[[ $prefix != "$HOME" ]] || fail 'choose a dedicated installation directory, not the home directory'
for tool in bash python3 cc make; do
    command -v "$tool" >/dev/null || fail "$tool is missing from the non-interactive SSH PATH"
done
python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' || fail 'Python 3.9 or newer is required in the SSH PATH'
case $(uname -s) in
    Darwin)
        xcode-select -p >/dev/null 2>&1 || fail 'install Xcode command line tools on this Mac first (xcode-select --install)'
        launchctl print "gui/$(id -u)" >/dev/null 2>&1 || fail 'log in graphically as this user on the Mac before installing its LaunchAgent'
        service=$HOME/Library/LaunchAgents/ai.fronk.node.plist
        ;;
    Linux)
        systemctl --user show-environment >/dev/null 2>&1 || fail 'a working systemd user session is required; use the documented manual --no-service procedure instead'
        service=$HOME/.config/systemd/user/fronk-node.service
        ;;
    *) fail 'only macOS and Linux nodes are supported' ;;
esac
if [[ -f $service ]] && ! grep -Fq "$prefix/src/fronk" "$service"; then
    fail 'another prefix owns the node service; reuse it or stop/uninstall it with its manager before migrating'
fi
printf 'Node prerequisites: ready (%s)\n' "$(uname -sm)"
