# Shared head workflow, sourced by the terminal and graphical manager.
valid_name() { [[ $1 =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,62}$ ]]; }

# Check the actual state filesystem before compiling or receiving any weights.
# Linux /tmp may be RAM-backed: enough free bytes there are not disk capacity.
check_storage_directory() {
    local filesystem free required=1048576
    if [[ $OS == Linux ]]; then
        filesystem=$(stat -f -c %T "$STATE") || die "cannot inspect storage at $STATE"
        case $filesystem in tmpfs|ramfs)
            die "$STATE is on $filesystem (RAM, not persistent disk). Choose a disk-backed prefix, for example --prefix \"\$HOME/fronk\"." ;;
        esac
    fi
    free=$(df -Pk "$STATE" | awk 'END {print $4}')
    [[ $free =~ ^[0-9]+$ ]] || die "cannot measure free disk space at $STATE"
    # A new node must receive one unit and keep 1 GiB free. Upgrades may reuse
    # their existing units; RECV checks capacity again for each missing unit.
    [[ $role != node || -n ${installed:-} ]] || required=$((required + 3268560))
    ((free >= required)) || die "too little disk space at $STATE: need at least $required KiB free ($free KiB available). Free space or choose a prefix on another disk."
}

# The same choices feed the terminal's numbered menu and Tk's read-only list.
# Explicit CLI names remain available for scripts and existing installations.
choose_node_name() {
    local previous=$1 options="" n choice default=node-1
    for ((n=1;n<=20;n++)); do options+="${options:+|}node-$n"; done
    if [[ -n $previous ]]; then
        default=$previous
        [[ "|$options|" == *"|$previous|"* ]] || options="$previous|$options"
    fi
    choice=$(ask "On this node, select the name shown on the head" "$default" "$options|custom-name")
    if [[ $choice == custom-name ]]; then choice=$(ask "Custom node name, exactly as shown on the head" "$previous"); fi
    printf '%s\n' "$choice"
}

choose_roster() {
    local previous=$1 options="" n choice default=1 planned=""
    # --yes without an explicit --nodes must not invent machines or start a scan.
    if ((yes)); then printf '%s\n' "$previous"; return; fi
    for ((n=1;n<=20;n++)); do options+="${options:+|}$n"; done
    if [[ -n $previous ]]; then options="keep-current|$options"; default=keep-current; fi
    choice=$(ask "How many node computers will you connect? The head is not counted." "$default" "$options|custom-names|later")
    case $choice in
        keep-current) planned=$previous ;;
        later) ;;
        custom-names) planned=$(ask "Custom node names, separated by spaces" "$previous") ;;
        *)
            [[ $choice =~ ^([1-9]|1[0-9]|20)$ ]] || die "choose 1 to 20 nodes, custom-names or later"
            for ((n=1;n<=choice;n++)); do planned+="${planned:+ }node-$n"; done ;;
    esac
    printf '%s\n' "$planned"
}

choose_backend() {
    local listing=$1 saved=$2 options recommended
    options=$(awk '$1 ~ /^[0-9]+\)$/ && $3=="runs" && $4=="here" {printf "%s%s", sep, $2; sep="|"}' <<< "$listing")
    recommended=$(awk '$1 ~ /^[0-9]+\)$/ && $3=="runs" && $4=="here" && /recommended/ {print $2; exit}' <<< "$listing")
    options="${options:+$options|}auto"
    [[ -n $saved && "|$options|" == *"|$saved|"* ]] || saved=${recommended:-auto}
    ask "Select a compute backend. Continue keeps the saved choice or recommendation." "$saved" "$options"
}

show_secret() {
    local burst code
    if ((yes)); then say "The pairing secret is in $STATE/cluster.secret on the head."; return; fi
    burst=$(config_value burst); code="fronk-v1:${burst:-1}:$(cat "$STATE/cluster.secret")"
    if ((dialog)); then printf '@secret\t%s\n' "$code"
    else say "Node pairing code (enter it privately on each node): $code"; fi
}

# A report belongs to this build, roster and network configuration. Only a
# successful, complete fronk demo can unlock a model download.
demo_key() {
    [[ -s $STATE/install-nodes ]] || return 1
    { cksum ./fronk; cat "$STATE/install-nodes"; awk '$1!="model" {print}' "$CONF"; } | cksum
}
demo_done() {
    [[ -s $STATE/install-demo.key && -s $STATE/install-demo.log && -s $STATE/install-nodes ]] &&
        [[ $(cat "$STATE/install-demo.key") == "$(demo_key)" ]]
}
run_demo() {
    step "Head · Try the cluster before getting the model"
    say "Stay on this head and leave the nodes running. Fronk will time synthetic blocks on these machines and send real requests over your cables."
    say "No model file is needed. You will see the memory fit, estimated tokens per second, compute time on each machine and communication time."
    say "The model has 20 expert blocks (about 66.9 GB total). Each node's compute and network times are per layer. The slowest reply determines when the head can continue."
    local report=$STATE/install-demo.log node
    rm -f "$STATE/install-demo.key"
    if ! ./fronk demo --passes 40 --config "$CONF" 2>&1 | tee "$report.tmp" |
        awk '/^(TIMING|BENCH (node=|backend=|waiting)|CONNECTION complete|PLAN |  .* link=|CAPACITY |DEMO |REPORT (node=|split |bottleneck=)|fronk:|connection:)/ {print; fflush()}' ; then
        mv "$report.tmp" "$STATE/install-demo-failed.log"
        die "the demo could not verify this cluster. Check the affected node and its local network permission, then choose join-nodes on the head. Details: $STATE/install-demo-failed.log"
    fi
    while IFS= read -r node; do
        # A node may have disconnected between the pairing check and the demo.
        awk -v name="node=$node" '$1=="path" && $2==name && /chosen=yes status=measured/ {found++} END {exit found!=1}' "$report.tmp" ||
            die "$node was missing during the demo. Check it and rerun join-nodes."
    done < <(tr ' ' '\n' < "$STATE/install-nodes" | awk 'NF')
    grep -q '^DEMO verdict:' "$report.tmp" || die "incomplete demo; see $report.tmp"
    mv "$report.tmp" "$report"
    demo_key > "$STATE/install-demo.key"
    say "The full timing report is saved in $report."
}

guide_nodes() {
    step "Head · Plan your nodes"
    say "Quit any running Fronk head before verification. Keep each node running."
    local planned name seen=" " index=0 answer entry snapshot previous=""
    local -a roster
    [[ -f $STATE/install-nodes ]] && previous=$(cat "$STATE/install-nodes")
    planned=${nodes:-$(choose_roster "$previous")}
    read -r -a roster <<< "$planned"
    if [[ -z $planned ]]; then say "Node setup pending. Resume on the head with $PREFIX/bin/fronk-manager --join-nodes."; return; fi
    for name in "${roster[@]}"; do
        valid_name "$name" && [[ $seen != *" $name "* ]] || die "invalid or duplicate node name: $name"
        seen+="$name "
    done
    (( ${#roster[@]} > 0 && ${#roster[@]} <= 20 )) || die "enter 1 to 20 distinct node names"
    rm -f "$STATE/install-demo.key"
    printf '%s\n' "${roster[*]}" > "$STATE/install-nodes.tmp"; mv "$STATE/install-nodes.tmp" "$STATE/install-nodes"
    show_secret
    local online=""
    if [[ -n $previous ]]; then
        step "Head · Recheck previously configured nodes"
        online=$(./fronk peers --config "$CONF") || online=""
    fi
    for name in "${roster[@]}"; do
        index=$((index+1))
        step "Node $index of ${#roster[@]} · $name"
        if [[ $(awk -v n="$name" '$1=="PEER" && $2==n {count++} END {print count+0}' <<< "$online") == 1 && $index != ${#roster[@]} ]]; then
            say "$name is already connected and authenticated. Continue on the head."
            continue
        fi
        say "If $name is already running, stay on the head and choose Verify. For a new node:"
        say "Go to $name. Connect it to the head's LAN and open the same Fronk Installer Manager there."
        say "Choose Node, select $name (or custom-name if needed) and paste the head's pairing code. Start the service and allow local network access."
        say "Terminal alternative on $name (from the download folder): bash fronk-manager.sh --role node --name $name"
        say "When $name reports that its service has started (or the manual worker is listening), return to THIS HEAD and choose Verify."
        while :; do
            answer=$(ask "Back on the head: verify $name now or finish node setup later?" verify "verify|later")
            if [[ $answer == later ]]; then say "Node setup pending at $name. Resume with $PREFIX/bin/fronk-manager --join-nodes."; return; fi
            [[ $answer == verify ]] || die "choose verify or later"
            step "Head · Checking $name"
            if snapshot=$(./fronk peers --config "$CONF"); then
                # On the last pass every planned node must still be present,
                # exactly once, on the same authenticated discovery pass.
                local missing=""
                for entry in "${roster[@]}"; do
                    if [[ $entry == "$name" || $index == ${#roster[@]} ]]; then
                        if [[ $(awk -v n="$entry" '$1=="PEER" && $2==n {count++} END {print count+0}' <<< "$snapshot") != 1 ]]; then
                            missing+=" $entry"
                        fi
                    fi
                done
                if [[ -z $missing ]]; then say "Head verified $name."; break; fi
                say "Missing or duplicate node names:$missing"
            fi
            say "On the affected node, check its name, pairing secret, running service and network permission. Then retry here."
            ((yes)) && die "node verification failed"
        done
    done
    say "All ${#roster[@]} planned nodes authenticated in the same head check."
    if ((demo_later)) || [[ $(ask "All nodes are connected. Run the weight-free demo now or finish later?" demo "demo|later") == later ]]; then
        say "Demo pending. Choose join-nodes on the head when the machines are ready."
        return
    fi
    run_demo
}

head_setup() {
    while :; do
        guide_nodes
        if ! demo_done; then
            # An explicit local GGUF can be verified offline. Downloads are
            # always gated, including non-interactive and resumed installs.
            if [[ -n $model ]] || ((download)); then configure_model; fi
            return
        fi
        case $(ask "After the demo, choose model weights, add nodes and remeasure, or finish later" weights "weights|add-nodes|later") in
            weights) configure_model; return ;;
            add-nodes) nodes="" ;;
            later) say "On the head, choose configure-model when ready."; return ;;
            *) die "choose weights, add-nodes or later" ;;
        esac
    done
}

configure_model() {
    step "Head · Model weights"
    say "Supported file: $MODEL_NAME ($MODEL_BYTES bytes). Other GGUF variants are not compatible."
    say "Model license (separate from Fronk): $MODEL_LICENSE_URL"
    local existing choice path found="" candidate
    existing=$(awk '$1=="model" {print $2;exit}' "$CONF")
    if ! demo_done && [[ -z $model && $download == 0 && ! -f $existing ]]; then
        say "First choose join-nodes on the head and complete the synthetic demo. You can choose the weights after seeing its report."
        return
    fi
    if [[ -z $model ]] && ((!download)); then
        if [[ -n $existing && -f $existing ]]; then model=$existing
        else
            for candidate in "$PREFIX/models/$MODEL_NAME" "$HOME/Downloads/$MODEL_NAME" /Volumes/*/"$MODEL_NAME" /media/*/*/"$MODEL_NAME" /mnt/*/"$MODEL_NAME"; do
                if [[ -f $candidate ]]; then found=$candidate; say "Found a local model at $found"; break; fi
            done
            choice=$(ask "Use a file or folder on this computer or an external drive, download 177 GB, or finish later?" local "local|download|later")
            if ((yes)) && [[ -z $found ]]; then choice=later; fi
            case $choice in
                local) model=$(ask "Full path of $MODEL_NAME or the folder containing it" "$found") ;;
                download) download=1 ;;
                later) say "Model setup pending. Resume on the head with $PREFIX/bin/fronk-manager --configure-model."; return ;;
                *) die "choose local, download or later" ;;
            esac
        fi
    fi
    if ((download)); then
        demo_done || die "run join-nodes and finish the synthetic demo before downloading the weights"
        command -v curl >/dev/null || die "curl is required to download weights; install curl or choose an existing local file"
        say "Downloading the pinned Q4 GGUF from antirez/qwen3.8-flash-next-gguf on Hugging Face."
        say "Source: $MODEL_URL"
        [[ -n $download_dir ]] || download_dir=$(ask "Download folder on this head or an external drive" "$PREFIX/models")
        [[ $download_dir == /* ]] || die "choose an absolute download folder"
        model=$download_dir/$MODEL_NAME; mkdir -p "$download_dir"
        local have=0 free
        [[ -f $model ]] && have=$(wc -c < "$model")
        free=$(df -Pk "$(dirname "$model")" | awk 'END {print $4}')
        ((free*1024 >= MODEL_BYTES-have)) || die "too little disk space for this download; choose a local model or another volume"
        if ((have != MODEL_BYTES)); then curl -fL -C - -o "$model" "$MODEL_URL"; fi
    fi
    [[ ! -d $model ]] || model=${model%/}/$MODEL_NAME
    [[ -f $model ]] || die "model file not found: $model"
    path=$(cd "$(dirname "$model")" && pwd -P)/$(basename "$model")
    python3 "$HERE/verify-model.py" "$path" "$STATE" "$MODEL_BYTES" "$MODEL_SHA256" ||
        die "model verification failed. On macOS, allow the manager to read the SSD on the head."
    # fronk.conf has one whitespace-delimited value per line. Link a path
    # containing whitespace or '#' without copying the weights.
    if [[ ! $path =~ ^/[A-Za-z0-9._/-]+$ || ${#path} -gt 480 ]]; then
        ln -sfn "$path" "$STATE/model.gguf"; path=$STATE/model.gguf
    fi
    set_conf model "$path"
    model_ready=1
    say "Verified weights configured. Start the head with $PREFIX/bin/fronk-head or open $PREFIX/bin/fronk-console."
}
