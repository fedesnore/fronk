#!/bin/sh
# caffeinate-plus for Linux laptops - keep this machine awake after the lid closes,
# like caffeinate+ (github.com/clashman/caffeinate-plus) on a MacBook.
#
#   sudo ./caffeinate-plus-linux.sh        # Ctrl+C or SIGTERM to stop
#
# While it runs it holds a systemd-logind block on lid-switch handling, sleep and
# idle suspend (systemd-inhibit), so closing the lid does nothing and GNOME's
# automatic suspend cannot fire. No setting is changed: the block ends with the
# process. On exit, if the lid is still closed, the machine is suspended, as the
# lid asked. logind grants the lid block only to root, hence sudo.
#
# Mind the heat: a node computing in a closed laptop on a soft surface overheats.
set -u
[ "$(id -u)" = 0 ] || { echo "caffeinate-plus: run it with sudo" >&2; exit 1; }
command -v systemd-inhibit >/dev/null || { echo "caffeinate-plus: needs systemd-logind (systemd-inhibit)" >&2; exit 1; }

lid_closed() {
    for f in /proc/acpi/button/lid/*/state; do
        [ -r "$f" ] && grep -q closed "$f" && return 0
    done
    return 1
}

systemd-inhibit --what=handle-lid-switch:sleep:idle --who=caffeinate-plus \
    --why="keep running with the lid closed" --mode=block sleep infinity &
child=$!

stop() {
    kill "$child" 2>/dev/null; wait "$child" 2>/dev/null
    if lid_closed; then echo "caffeinate-plus: lid closed, suspending"; systemctl suspend
    else echo "caffeinate-plus: stopped"; fi
    exit 0
}
trap stop INT TERM HUP

sleep 1
kill -0 "$child" 2>/dev/null || { echo "caffeinate-plus: logind refused the block" >&2; exit 1; }
echo "caffeinate-plus: awake with the lid closed (pid $$); Ctrl+C or kill $$ to stop"
wait "$child"
