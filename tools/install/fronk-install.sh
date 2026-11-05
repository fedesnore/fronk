#!/usr/bin/env bash
# Fronk Installer Manager. Both interfaces drive this same local installation.
# This file owns arguments, prerequisites, builds and the installed launchers.
# onboard.sh owns pairing, calibration, the demo and model selection; service.sh
# owns service lifecycle/removal. ask() is the only question/answer boundary:
# terminal menus and manager.py therefore follow the same sequence.
set -euo pipefail
# Finder or an Intel Python can launch the shell under Rosetta on an arm64
# Mac. Build and probe the native platform in both the UI and its worker.
if [[ $(uname -s) == Darwin && $(uname -m) == x86_64 && $(/usr/sbin/sysctl -in sysctl.proc_translated 2>/dev/null) == 1 ]]; then
    exec /usr/bin/arch -arm64 /bin/bash "$0" "$@"
fi
HERE=$(cd "$(dirname "$0")" && pwd -P)
SRC_TREE=$(cd "$HERE/../.." && pwd -P)
VERSION=$(sed -n 's/^VERSION := //p' "$SRC_TREE/Makefile")
PREFIX=${FRONK_PREFIX:-$HOME/fronk}
source "$HERE/model.conf"
label=ai.fronk.node
role="" name="" nodes="" node_burst="" model="" download_dir="" action="" yes=0 secret_stdin=0 download=0 purge=0 no_service=0 dialog=0 gui=0 measure=1 demo_later=0 model_ready=0
module_paths=() custom_modules=() requested_backend=""
say() { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
# Every stop says where help is: the installed guides once they exist. The briefs
# in docs/llm are written for a frontier model, the guides for the person.
die() {
    printf 'Fronk setup: %s\n' "$*" >&2
    local docs=docs where=" in the Fronk source"
    [[ -f ${WORK:-}/docs/llm/DEBUG.md ]] && { docs=$WORK/docs where=""; }
    printf 'Need help? Reopen the manager here to continue. Read %s/TROUBLESHOOTING.md%s, or give a frontier model %s/llm/DEBUG.md and this error.\n' "$docs" "$where" "$docs" >&2
    exit 1
}
ask() { # question, default, optional choices separated by |, optional secret
    local answer="" index option
    local -a choices
    if ((yes)); then printf '%s\n' "$2"; return; fi
    if ((dialog)); then
        printf '@prompt\t%s\t%s\t%s\t%s\n' "$1" "$2" "${3:-}" "${4:-}" >&2
        IFS= read -r answer || exit 1
    else
        if [[ -n ${3:-} ]]; then
            IFS='|' read -r -a choices <<< "$3"
            printf '%s\n' "$1" >&2
            for ((index=0;index<${#choices[@]};index++)); do printf '  %d) %s\n' "$((index+1))" "${choices[index]}" >&2; done
            printf 'Choose an option number, or press Enter for %s: ' "$2" >&2
        else printf '%s [%s] ' "$1" "$2" >&2; fi
        if [[ ${4:-} == secret ]]; then IFS= read -rs answer </dev/tty || exit 1; echo >&2
        else IFS= read -r answer </dev/tty || exit 1; fi
        if [[ -n ${3:-} && $answer =~ ^[1-9][0-9]?$ ]] && ((answer <= ${#choices[@]})); then
            answer=${choices[answer-1]}
        fi
    fi
    answer=${answer:-$2}
    if [[ -n ${3:-} ]]; then
        IFS='|' read -r -a choices <<< "$3"
        for option in "${choices[@]}"; do [[ $answer != "$option" ]] || { printf '%s\n' "$answer"; return; }; done
        die "choose one of the displayed options"
    fi
    printf '%s\n' "$answer"
}
usage() { cat <<HELP
Fronk Installer Manager $VERSION
  --gui | --terminal                 graphical wizard or terminal wizard (default)
  --role head|node [--prefix DIR]     install for this user (default ~/fronk)
  --name NAME                        name this node as shown on the head
  --nodes 'NAME NAME'                head's planned nodes, in installation order
  --secret-stdin                     read the head pairing code or legacy secret from stdin
  --burst N                          head burst (1–4) when pairing a node with a legacy secret
  --no-service                       launch this node manually with fronk-node
  --no-measure                       defer RAM benchmarks; skip SSD calibration
  --backend-module FOLDER            add a source module (repeat for more than one)
  --backend NAME                     select this compatible expert backend
  --demo-later                       pause after pairing; resume with --join-nodes
  --model FILE|FOLDER | --download   verify local GGUF or download after the measured demo
  --download-dir FOLDER              destination for a resumed or new download
  --join-nodes | --configure-model   resume setup on an installed head
  --web-console                      open the head and node dashboard in your browser
  --status                           check an installed node with authenticated INFO
  --run | --stop                     run the installed program or stop a registered node
  --uninstall [--purge]              remove programs; purge also erases managed state
  --yes                              use defaults without prompts (new nodes need --secret-stdin)
HELP
}
args=("$@")
while (($#)); do
    case $1 in
        --role|--name|--nodes|--model|--prefix|--download-dir|--burst|--backend-module|--backend)
            [[ $# -ge 2 && -n $2 && $2 != --* ]] || die "$1 needs a value"
            case $1 in
                --backend-module) module_paths+=("$2") ;; --backend) requested_backend=$2 ;;
                --burst) node_burst=$2 ;; --role) role=$2 ;; --name) name=$2 ;; --nodes) nodes=$2 ;;
                --model) model=$2 ;; --download-dir) download_dir=$2 ;; --prefix) PREFIX=$2 ;;
            esac; shift ;;
        --gui) gui=1 ;; --terminal) gui=0 ;; --dialog) dialog=1 ;;
        --yes|-y) yes=1 ;; --secret-stdin) secret_stdin=1 ;; --download) download=1 ;;
        --demo-later) demo_later=1 ;; --no-service) no_service=1 ;; --no-measure) measure=0 ;; --purge) purge=1 ;;
        --uninstall|--join-nodes|--configure-model|--status|--run|--stop|--web-console)
            [[ -z $action ]] || die "choose one manager action"; action=$1 ;;
        --help|-h) usage; exit 0 ;; *) die "unknown option $1 (see --help)" ;;
    esac; shift
done
[[ -z $role || $role == head || $role == node ]] || die "role must be head or node"
[[ -z $node_burst || $node_burst =~ ^[1-4]$ ]] || die "--burst must be 1 to 4"
[[ -z $model || $download == 0 ]] || die "choose --model or --download"
[[ $purge == 0 || $action == --uninstall ]] || die "--purge requires --uninstall"
[[ -z $action ]] || { ((${#module_paths[@]} == 0)) && [[ -z $requested_backend ]]; } || die "backend options apply to install/upgrade, not maintenance actions"
if ((gui && !dialog)); then
    (( !secret_stdin )) || die "--secret-stdin requires --terminal"
    # Finder and a terminal can see different Pythons. Homebrew splits Tk
    # into a separate package, even when another installed Python includes it.
    while IFS= read -r python; do
        if "$python" -c 'import sys, tkinter; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
            export PATH="$(dirname "$python"):$PATH"
            exec "$python" "$HERE/manager.py" "${args[@]}"
        fi
    done < <(type -a -p python3 || true)
    message="The graphical manager needs Python 3.9 or newer with Tk. Install Python with Tk, or run bash fronk-manager.sh --terminal."
    if [[ $(uname -s) == Darwin ]]; then
        osascript -e 'display alert "Fronk Installer Manager" message "Install Python 3.9 or newer with Tk, or run the terminal installer with bash fronk-manager.sh --terminal."'
    elif command -v zenity >/dev/null; then zenity --error --text="$message"; fi
    die "$message"
fi
command -v python3 >/dev/null || die "Python 3.9 or newer is required; install it and reopen the manager"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' || die "Python 3.9 or newer is required; upgrade Python and reopen the manager"
OS=$(uname -s); ARCH=$(uname -m)
[[ $OS == Darwin || $OS == Linux ]] || die "supported systems are macOS and Linux"
[[ $PREFIX =~ ^/[a-zA-Z0-9._/-]+$ && $PREFIX != / && $PREFIX != "$HOME" && $PREFIX != */ && $PREFIX != *//* && $PREFIX/ != */../* && $PREFIX/ != */./* ]] || die "choose a dedicated absolute prefix using letters, digits, dots, hyphens, underscores or slashes"
STATE=$PREFIX/state WORK=$PREFIX/src CONF=$PREFIX/src/fronk.conf
source "$HERE/onboard.sh"
[[ -z $name ]] || valid_name "$name" || die "invalid node name"
[[ $PREFIX != "$SRC_TREE" && $PREFIX != "$SRC_TREE/"* ]] || die "choose a prefix outside the source folder"
plist=$HOME/Library/LaunchAgents/$label.plist
unit=$HOME/.config/systemd/user/fronk-node.service
if [[ $OS == Darwin ]]; then service_file=$plist; else service_file=$unit; fi
config_value() { awk -v k="$1" '$1==k {print $2;exit}' "$CONF"; }
set_conf() {
    awk -v k="$1" -v v="$2" '$1!=k {print} END {print k " " v}' "$CONF" > "$CONF.new"
    mv "$CONF.new" "$CONF"
}
source "$HERE/service.sh"
source "$HERE/modules.sh"
step "Fronk Installer Manager $VERSION · $(hostname -s) · $OS $ARCH"
installed=""
if [[ -x $WORK/fronk && -f $CONF && -f $WORK/tools/install/fronk-install.sh ]]; then
    if [[ -f $STATE/install-role ]]; then installed=$(cat "$STATE/install-role")
    elif [[ -n $(config_value ds4) || -n $(config_value model) ]]; then installed=head
    else installed=node; fi
    [[ $installed == head || $installed == node ]] || die "unrecognized installed role"
elif [[ -d $PREFIX && -n $(ls -A "$PREFIX") && ! -f $PREFIX/.fronk-manager && ! -f $PREFIX/fronk.conf.saved && ! -f $STATE/install-role ]]; then
    die "$PREFIX contains files from another installation; choose a dedicated --prefix"
fi
if [[ -n $installed && -z $action && -z $role ]] && { ((${#module_paths[@]})) || [[ -n $requested_backend ]]; }; then role=$installed; fi
if [[ -n $installed && -z $action && -z $role ]] && ((!yes)); then
    options="status|run|upgrade|uninstall|cancel"; default=status
    if [[ $installed == node ]] && service_owned; then options="status|run|stop|upgrade|uninstall|cancel"; fi
    if [[ $installed == head ]]; then
        options="web-console|run|join-nodes|configure-model|upgrade|uninstall|cancel"; default=web-console
        [[ -f $(config_value model) ]] || default=join-nodes
    fi
    if [[ $("$WORK/fronk" --version) != "fronk $VERSION ("* ]]; then
        say "The installed version differs from this manager. Upgrade to use its new setup steps."
        options="upgrade|uninstall|cancel"; default=upgrade
    fi
    menu=$(ask "Installed $installed in $PREFIX. Choose an action." "$default" "$options")
    case $menu in upgrade) role=$installed ;; cancel) exit 0 ;; web-console|run|stop|status|join-nodes|configure-model|uninstall) action=--$menu ;; *) die "unknown action" ;; esac
fi
# Config loading persists a supplied FRONK_SECRET. Manager operations always
# use the explicitly paired secret, independent of the launching environment.
unset FRONK_SECRET
if [[ $action == --web-console ]]; then
    [[ $installed == head ]] || die "open web-console on the installed head"
    cd "$WORK"
    (( !dialog )) || printf '@running\n'
    exec python3 tools/fronk-web.py --open
fi
if [[ $action == --run ]]; then
    [[ -n $installed ]] || die "install Fronk on this machine first"
    cd "$WORK"
    if [[ $installed == node ]] && service_owned; then
        if [[ $OS == Darwin ]]; then
            if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then launchctl kickstart "gui/$(id -u)/$label"
            else launchctl bootstrap "gui/$(id -u)" "$plist"; fi
        else systemctl --user start fronk-node.service; fi
        wait_for_node; exit
    fi
    if ((dialog)); then printf '@running\n'; fi
    if [[ $installed == node ]]; then exec ./fronk node --config "$CONF"
    elif ((dialog)); then exec python3 tools/fronk-web.py --open
    else exec bash tools/install/terminal-head.sh "$STATE" --config "$CONF"; fi
fi
if [[ $action == --stop ]]; then
    [[ $installed == node ]] && service_owned || die "stop requires a registered node service"
    if [[ $OS == Darwin ]]; then
        if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then launchctl bootout "gui/$(id -u)/$label"; fi
    else systemctl --user stop fronk-node.service; fi
    say "Local node stopped. Choose run to start it again."; exit
fi
if [[ $action == --status ]]; then
    [[ $installed == node ]] || die "status requires an installed node"
    cd "$WORK"; node_info; exit
fi
if [[ -n $action && $action != --uninstall && $installed != head ]]; then die "$action requires an installed head"; fi
if [[ $action == --uninstall && -z $installed && ! -f $PREFIX/fronk.conf.saved ]]; then die "no managed installation in $PREFIX"; fi
if [[ $installed == head && -f $STATE/head.lock ]]; then
    python3 - "$STATE/head.lock" <<'HEAD_LOCK' || die "stop the running head in its console or terminal before changing this installation"
import fcntl
import sys
with open(sys.argv[1]) as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(1)
HEAD_LOCK
fi
mkdir -p "$PREFIX"
lock=$PREFIX/.manager-lock
mkdir "$lock" 2>/dev/null || die "another setup is active in $PREFIX (if interrupted, remove the empty $lock directory and retry)"
stage=""
cleanup() { [[ -z $stage ]] || rm -rf "$stage"; rmdir "$lock" 2>/dev/null || true; }
finish_head() {
    ((model_ready)) || return 0
    say "Keep the named nodes running. Review the real placement on the head, choose Proceed to load the verified units, and wait for the generation check before chatting."
    if ((yes)) || [[ $(ask "Start the configured head now, or finish and run it later?" start "start|later") != start ]]; then return 0; fi
    cleanup; trap - EXIT HUP INT TERM
    cd "$WORK"
    if ((dialog)); then printf '@running\n'; exec python3 tools/fronk-web.py --open --start; fi
    exec bash tools/install/terminal-head.sh "$STATE" --config "$CONF"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM
if [[ $action == --uninstall ]]; then
    if ((!yes)); then
        if ((!purge)); then
            choice=$(ask "Remove programs and keep your cluster settings, or erase the managed cluster state for a fresh install? Model GGUFs are kept." keep-state "keep-state|fresh-install|cancel")
            case $choice in keep-state) ;; fresh-install) purge=1 ;; *) exit 0 ;; esac
        fi
        [[ $(ask "Confirm removal from $PREFIX (including managed identity, secret and block cache if fresh install was selected)?" no "yes|no") == yes ]] || exit 0
    fi
    stop_service
    if [[ -f $CONF ]]; then cp "$CONF" "$PREFIX/fronk.conf.saved"; fi
    # Keep files that a user may have placed in the source tree, including models
    # and chat attachments. Older distributions did not record a file manifest.
    if [[ -f $STATE/install-files ]]; then
        while IFS= read -r file; do
            [[ $file == ./* && $file != */../* ]] || die "invalid package manifest"
            rm -f "$WORK/$file"
        done < "$STATE/install-files"
    fi
    rm -f "$WORK/fronk" "$WORK/fronk-cli" "$CONF"
    [[ -d $WORK ]] && find "$WORK" -depth -type d -empty -delete
    for exe in fronk fronk-cli fronk-manager fronk-uninstall fronk-node fronk-head fronk-console; do rm -f "$PREFIX/bin/$exe"; done
    if ((purge)); then
        if [[ -d $STATE ]]; then
            find "$STATE" -depth ! -type d ! -iname '*.gguf' -delete
            find "$STATE" -depth -type d -empty -delete
        fi
        rm -f "$PREFIX/fronk.conf.saved"
        printf 'fresh\n' > "$PREFIX/.fronk-manager"
    fi
    if ((purge)); then say "Removed programs and managed state. Reopen the downloaded manager to start a new cluster."
    else say "Removed the programs. Reopen the downloaded manager to reuse the saved settings."; fi
    exit
fi
if [[ $action == --join-nodes || $action == --configure-model ]]; then
    cd "$WORK"
    if [[ $action == --join-nodes ]] && ! ./fronk --help | grep -q 'fronk peers'; then die "upgrade the installed head before pairing nodes"; fi
    if [[ $action == --join-nodes ]]; then head_setup; else configure_model; fi
    finish_head; exit
fi

suggest=node; [[ $OS == Darwin ]] && suggest=head
step "Choose this machine's role"
say "Start from the head. Tested head: Apple Silicon with Metal. Choose node on computers lending memory and compute for expert slices. Intel Mac and Linux GPU heads are experimental."
[[ -n $role ]] || role=$(ask "Install this machine as head or node?" "${installed:-$suggest}" "head|node")
[[ $role == head || $role == node ]] || die "choose head or node"
gpu_head=0
if [[ $role == head && ( $OS != Darwin || $ARCH != arm64 ) ]]; then gpu_head=1; fi
if ((gpu_head)); then say "This platform uses the experimental Vulkan head. A GPU check is required, but full installer-to-chat qualification is still pending. The tested head is Apple Silicon with Metal."; fi
[[ -z $installed || $installed == "$role" ]] || die "this prefix contains a $installed; use another prefix for a different role"
if [[ $role == node ]]; then
    default_name=""
    if [[ -n $installed ]]; then
        default_name=$(config_value name)
        # Legacy configs used the short hostname when no name was written.
        # An upgrade must retain that identity in the head's existing roster.
        [[ -n $default_name ]] || default_name=$(hostname -s)
    fi
    [[ -n $name ]] || name=$(choose_node_name "$default_name")
    valid_name "$name" || die "invalid node name"
    if ((!no_service)); then
        [[ ! -f $service_file ]] || service_owned || die "another prefix owns the node service; use --no-service or manage that installation first"
        if [[ $OS == Linux ]]; then systemctl --user show-environment >/dev/null || die "a systemd user session is required; use --no-service to run manually"; fi
    fi
elif [[ -n $node_burst || -n $name || $secret_stdin == 1 || $no_service == 1 ]]; then die "--burst, --name, --no-service and --secret-stdin apply to nodes"; fi
if [[ -d $HOME/fronk-cluster/source && $role == node ]]; then
    say "A development node exists in ~/fronk-cluster/source. Stop it before starting this service if it uses the same ports."
fi
for tool in cc make python3; do command -v "$tool" >/dev/null || die "$tool is required (macOS: install the command line tools and Python 3; Debian/Ubuntu: install build-essential python3)"; done
if [[ $OS == Darwin ]]; then xcode-select -p >/dev/null || die "install Xcode command line tools with xcode-select --install, then reopen the manager"; fi
if [[ $role == head ]]; then
    ((gpu_head)) || [[ -d $SRC_TREE/ds4 ]] || die "use the downloadable manager built by make dist; it includes the patched ds4 engine"
    command -v openssl >/dev/null || die "openssl is required for head pairing and model verification"
    command -v python3 >/dev/null || die "the head console requires Python 3"
fi
if [[ -z $installed && -f $PREFIX/fronk.conf.saved ]]; then
    saved_role=node
    if awk '$1=="ds4" || $1=="model" {found=1} END {exit !found}' "$PREFIX/fronk.conf.saved"; then saved_role=head; fi
    [[ $saved_role == "$role" ]] || die "saved $saved_role configuration in this prefix; use another prefix for a $role"
fi
if [[ -n $installed && $(config_value state-dir) != "$STATE" ]]; then die "this installation uses a custom state-dir; restore its managed state-dir before upgrading"; fi
if [[ -f $PREFIX/.fronk-manager && $(cat "$PREFIX/.fronk-manager") != fresh && $(cat "$PREFIX/.fronk-manager") != pending && $(cat "$PREFIX/.fronk-manager") != "$role" ]]; then
    die "this prefix belongs to a different role; choose another prefix"
fi
[[ -f $PREFIX/.fronk-manager ]] || echo pending > "$PREFIX/.fronk-manager"
mkdir -p "$STATE" "$PREFIX/bin"; chmod 700 "$STATE"
check_storage_directory
if [[ $role == node ]]; then
    secret=""
    if ((secret_stdin)); then IFS= read -r secret || [[ -n $secret ]] || die "missing secret on stdin"
    elif [[ ! -s $STATE/cluster.secret ]]; then
        ((!yes)) || die "supply the head's pairing secret with --secret-stdin"
        secret=$(ask "On $name, paste the pairing code shown on the head" "" "" secret)
    elif ((!yes)) && [[ $(ask "Keep this node's saved pairing secret or pair with another head?" keep "keep|replace") == replace ]]; then
        secret=$(ask "Paste the new head's pairing secret" "" "" secret); [[ -n $secret ]] || die "missing secret"
    fi
    if [[ $secret =~ ^fronk-v1:([1-4]):(.+)$ ]]; then
        paired_burst=${BASH_REMATCH[1]}; secret=${BASH_REMATCH[2]}
        [[ -z $node_burst || $node_burst == "$paired_burst" ]] || die "--burst disagrees with the head pairing code"
        node_burst=$paired_burst
    fi
    if [[ -n $secret ]]; then [[ ${#secret} -le 128 && $secret != *$'\r'* ]] || die "invalid secret"
    elif ((secret_stdin)) || [[ ! -s $STATE/cluster.secret ]]; then die "missing pairing secret"; fi
fi
step "Build $role on this machine"
stage=$(mktemp -d "$PREFIX/.build.XXXXXX")
export COPYFILE_DISABLE=1
(cd "$SRC_TREE" && tar --exclude __pycache__ --exclude .DS_Store -cf - -T tools/install/runtime-files) | tar -xf - -C "$stage"
prepare_backend_modules
ds4=""
if ((gpu_head)); then
    step "Head · Check the GPU before pairing"
    bundle=$SRC_TREE/gpu-runtime/$OS-$ARCH
    [[ ! -x $SRC_TREE/gpu/fronk-gpu-probe ]] || bundle=$SRC_TREE/gpu
    if [[ -x $bundle/fronk-gpu-probe && -x $bundle/fronk-llama-server ]]; then
        cp -R "$bundle" "$stage/gpu"
    elif [[ $OS == Linux ]]; then
        say "Building the pinned llama.cpp Vulkan head. See $STATE/build-gpu.log."
        for tool in cmake git glslc; do command -v "$tool" >/dev/null || die "Linux GPU builds need cmake, git, glslc, Vulkan development headers and SPIRV-Headers"; done
        "$stage/tools/gpu-head/build.sh" "$stage/gpu" > "$STATE/build-gpu.log" 2>&1 || die "GPU build failed; see $STATE/build-gpu.log"
    else die "this download is missing its Intel GPU runtime; rebuild the distribution with make gpu-head first"; fi
    gpu_profile=$(MVK_CONFIG_LOG_LEVEL=1 "$stage/gpu/fronk-gpu-probe" --profile) || die "this GPU did not pass the Vulkan compute check; see the driver message above"
    read -r device gpu_memory gpu_bytes gpu_free <<< "$gpu_profile"
    [[ $device =~ ^Vulkan[0-9]+$ ]] || die "the GPU probe returned no usable device"
    # Pinned GGUF tensor inventory: trunk plus shared experts 4,711,247,360
    # bytes, output projection 688,578,560 bytes. KV and graph scratch are extra.
    ((gpu_free >= 5399825920)) || die "the dense layers and output alone need 5.40 GB of free GPU memory, plus context buffers. Free GPU memory and retry, or use this machine as a node."
    say "The dense layers and output need 5.40 GB of GPU memory before context buffers."
    say "The dense layers will load on $device ($((gpu_bytes/1048576)) MiB, $gpu_memory memory). Fronk can assign expert blocks to this machine's CPU and system RAM alongside the other nodes."
elif [[ $role == head ]]; then
    mkdir "$stage/ds4"
    (cd "$SRC_TREE/ds4" && tar --exclude '*.o' --exclude '*.a' -cf - .) | tar -xf - -C "$stage/ds4"
    say "Building the model engine. Progress is in $STATE/build-ds4.log"
    make -C "$stage/ds4" libds4-fronk.a >"$STATE/build-ds4.log" 2>&1 || die "ds4 build failed; see $STATE/build-ds4.log"
    ds4=ds4
fi
say "Building Fronk. Progress is in $STATE/build.log"
make -C "$stage" fronk fronk-cli DS4="$ds4" >"$STATE/build.log" 2>&1 || die "build failed; see $STATE/build.log"
check_backend_modules
if [[ $role == head && $gpu_head == 0 ]]; then
    step "Head · Verify Metal on this Mac"
    (cd "$stage" && ./fronk backends --test metal) > "$STATE/head-gpu.log" 2>&1 ||
        die "the Metal compute test failed. Check $STATE/head-gpu.log before setting up nodes."
    say "Metal passed the GPU self-test. The dense model will run on this GPU."
fi
if [[ $role == head && $OS == Darwin ]]; then
    make -C "$stage" fronk-extract >>"$STATE/build.log" 2>&1 || say "The optional document extractor could not be built; see $STATE/build.log"
fi
save_backend_modules
if [[ -n $installed && $role == node ]]; then stop_service; fi
mkdir -p "$WORK"
remove_old_module_copies
# Overlay only shipped files. Preserve existing runs, local files and cached weights.
(cd "$stage" && tar --exclude ./fronk --exclude ./fronk-cli -cf - .) | tar -xf - -C "$WORK"
mv -f "$stage/fronk" "$WORK/fronk"; mv -f "$stage/fronk-cli" "$WORK/fronk-cli"
for exe in fronk fronk-cli; do ln -sf "$WORK/$exe" "$PREFIX/bin/$exe"; done
launcher() {
    local file=$PREFIX/bin/$1; shift
    { echo '#!/usr/bin/env bash'; printf 'cd %q || exit 1\n' "$WORK"; printf 'exec '; printf '%q ' "$@"; echo '"$@"'; } > "$file"
    chmod +x "$file"
}
launcher fronk-manager bash tools/install/fronk-install.sh --prefix "$PREFIX"
launcher fronk-uninstall bash tools/install/fronk-install.sh --prefix "$PREFIX" --uninstall
if [[ ! -f $CONF && -f $PREFIX/fronk.conf.saved ]]; then cp "$PREFIX/fronk.conf.saved" "$CONF"; fi
if [[ ! -f $CONF ]]; then
    { echo "bind 0.0.0.0"; echo "interfaces auto"; echo "control-port 7700"; echo "data-port 7701"
      echo "discovery-port 7702"; echo "multicast 239.7.7.7"; echo "password-env FRONK_SECRET"
      echo "state-dir $STATE"; echo "profile-file $STATE/profile-$role.txt"
      if [[ $role == head ]]; then echo "name head"; echo "burst 3"; echo "ds4 builtin"; else echo "burst 1"; fi
    } > "$CONF"
fi
if ((gpu_head)); then
    set_conf ds4 tools/gpu-head/run.sh
    set_conf burst 1
    printf '%s\n' "$device" > "$STATE/gpu-device"
    [[ -f $STATE/ds4-ctx ]] || echo 4096 > "$STATE/ds4-ctx"
    if [[ $gpu_memory == dedicated && -z $(config_value head-reserve-gb) ]]; then
        if [[ $OS == Darwin ]]; then ram=$(sysctl -n hw.memsize)
        else ram=$(awk '/^MemTotal:/ {printf "%.0f", $2*1024}' /proc/meminfo); fi
        # Dedicated VRAM is separate. Leave a host margin for the OS, engine
        # and transfer buffers; the real plan probes RAM again after model load.
        reserve=$(awk -v ram="$ram" 'BEGIN {gb=ram/1e10; if(gb<3)gb=3; printf "%.1f", gb+1}')
        set_conf head-reserve-gb "$reserve"
    fi
fi
if [[ $role == node ]]; then
    set_conf name "$name"
    [[ -z $node_burst ]] || set_conf burst "$node_burst"
    if [[ -n $secret ]]; then (umask 077; printf '%s\n' "$secret" > "$STATE/cluster.secret.new"); mv "$STATE/cluster.secret.new" "$STATE/cluster.secret"; unset secret; fi
    launcher fronk-node ./fronk node --config "$CONF"
else
    if [[ ! -s $STATE/cluster.secret ]]; then (umask 077; openssl rand -hex 16 > "$STATE/cluster.secret.new"); mv "$STATE/cluster.secret.new" "$STATE/cluster.secret"; fi
    launcher fronk-head bash tools/install/terminal-head.sh "$STATE" --config "$CONF"
    launcher fronk-console python3 tools/fronk-web.py --open
fi
(cd "$stage" && find . -type f -print) > "$STATE/install-files.new"
mv "$STATE/install-files.new" "$STATE/install-files"
printf '%s\n' "$role" > "$STATE/install-role"
printf '%s\n' "$role" > "$PREFIX/.fronk-manager"
if [[ $role == head ]]; then
    if ((!gpu_head)); then say "Dense head engine: ds4 Metal on the Apple GPU."; fi
    step "Head · Backend for local expert blocks"
    say "This choice controls expert blocks assigned to this computer. The dense model runs on the GPU."
else step "Node · Expert compute backend"; fi
cd "$WORK"
backend_listing=$(./fronk backends --config "$CONF")
say "$backend_listing"
saved_backend=""
[[ ! -s $STATE/backend ]] || saved_backend=$(cat "$STATE/backend")
if [[ -n $requested_backend ]]; then choice=$requested_backend
else choice=$(choose_backend "$backend_listing" "$saved_backend"); fi
configured_burst=$(config_value burst)
flags=(--burst "${configured_burst:-1}")
if [[ $measure == 0 ]]; then flags+=(--no-measure); fi
if ((measure)); then
    say "Calibration measures compute per 32-neuron slice and RAM + SSD timings on Metal. Reconnecting reuses compatible saved profiles."
else
    say "RAM calibration is deferred until the first head connection (--no-measure)."
    say "Before using Metal SSD capacity, run: $WORK/fronk backends --measure --config $CONF"
fi
printf '%s\n' "$choice" | ./fronk backends --choose "${flags[@]}" --config "$CONF"
if [[ $role == head ]]; then
    head_setup
    finish_head
    say "Head terminal: $PREFIX/bin/fronk-head"
    say "Head GUI: $PREFIX/bin/fronk-console"
    exit
fi
if ((no_service)); then
    step "Node $name · manual launch"
    say "On $name, run $PREFIX/bin/fronk-node and leave it running. Then return to the head and choose Verify for $name."
    exit
fi
step "Node $name · start the service"
if [[ $OS == Darwin ]]; then
    mkdir -p "$(dirname "$plist")"
    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
<key>Label</key><string>$label</string>
<key>ProgramArguments</key><array><string>$WORK/fronk</string><string>node</string><string>--config</string><string>$CONF</string></array>
<key>WorkingDirectory</key><string>$WORK</string>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
<key>StandardOutPath</key><string>$STATE/node.log</string><key>StandardErrorPath</key><string>$STATE/node.log</string>
</dict></plist>
PLIST
    launchctl bootstrap "gui/$(id -u)" "$plist" || die "could not load the LaunchAgent; log in on this Mac and retry"
    say "On $name, allow Fronk to find devices on the local network if macOS asks."
else
    mkdir -p "$(dirname "$unit")"
    cat > "$unit" <<UNIT
[Unit]
Description=Fronk node
After=network-online.target
[Service]
WorkingDirectory=$WORK
ExecStart=$WORK/fronk node --config $CONF
Restart=on-failure
StandardOutput=append:$STATE/node.log
StandardError=append:$STATE/node.log
[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload
    systemctl --user enable --now fronk-node.service
    say "On Linux, to keep this user service running after logout, enable lingering with loginctl enable-linger."
fi
wait_for_node
