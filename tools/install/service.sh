# Per-user node service lifecycle. Sourced by fronk-install.sh after its
# configuration helpers. The caller supplies OS, WORK, STATE, CONF, PREFIX,
# service_file and label. Never stop a process solely because a PID file exists.

service_owned() { [[ -f $service_file ]] && grep -Fq "$WORK/fronk" "$service_file"; }
stop_service() {
    # v0.2 on Linux could start a detached worker alongside a disabled user
    # service. Its pid file may name the parent shell or a reused pid. Ask the
    # authenticated worker for its pid and verify its executable before stopping.
    if [[ $OS == Linux && -f $STATE/node.pid && -x $WORK/fronk-cli && -f $CONF ]]; then
        local legacy_pid
        legacy_pid=$(cd "$WORK" && node_info 2>/dev/null | awk -F: '$1=="pid" {sub(/\r$/,"",$2);print $2}') || legacy_pid=""
        if [[ $legacy_pid =~ ^[1-9][0-9]*$ && $(readlink "/proc/$legacy_pid/exe") == "$WORK/fronk" ]]; then
            kill -TERM "$legacy_pid"
            for ((i=0;i<50;i++)); do [[ -e /proc/$legacy_pid/exe ]] || break; sleep 0.1; done
            [[ ! -e /proc/$legacy_pid/exe ]] || die "the previous node is still stopping; retry after it exits"
        fi
        rm -f "$STATE/node.pid"
    fi
    service_owned || return 0
    if [[ $OS == Darwin ]]; then
        if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then launchctl bootout "gui/$(id -u)/$label"; fi
    else systemctl --user disable --now fronk-node.service; fi
    rm -f "$service_file"
    if [[ $OS == Linux ]]; then systemctl --user daemon-reload; fi
}
node_info() {
    local address port
    address=$(config_value bind); port=$(config_value control-port)
    [[ -n $address && $address != 0.0.0.0 ]] || address=127.0.0.1
    "$WORK/fronk-cli" -h "$address" -p "${port:-7700}" --config "$CONF" INFO server
}
wait_for_node() {
    local attempt service_pid info node_name
    node_name=$(config_value name)
    for ((attempt=0;attempt<12;attempt++)); do
        if [[ $OS == Darwin ]]; then
            service_pid=$(launchctl print "gui/$(id -u)/$label" | awk '$1=="pid" && $2=="=" {print $3}')
        else service_pid=$(systemctl --user show fronk-node.service --property=MainPID --value); fi
        if [[ $service_pid =~ ^[1-9][0-9]*$ && -s $STATE/node.id ]] && info=$(node_info 2>/dev/null) &&
            awk -v n="$node_name" -v id="$(cat "$STATE/node.id")" -v pid="$service_pid" '
                {sub(/\r$/,"")} $0=="name:"n {a=1} $0=="id:"id {b=1} $0=="pid:"pid {c=1}
                END {exit !(a && b && c)}' <<< "$info"; then
            say "Node $node_name service started and authenticated. Return to the HEAD and choose Verify, then run the demo to check its data connection."
            return 0
        fi
        sleep 1
    done
    die "the service did not authenticate as $node_name; check $STATE/node.log on this node, then retry"
}
