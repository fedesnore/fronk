#!/usr/bin/env bash
# Orchestrate the documented manager over working SSH; no alternate installer.
set -euo pipefail
umask 077
manager= checksums= head_prefix= nodes_file= model= ssh_config= through=run resume=0
usage() {
    cat <<'HELP'
Usage: setup-cluster.sh --manager FILE --checksums SHA256SUMS --head-prefix DIR
       --nodes FILE [--model GGUF] [--ssh-config FILE] [--through install|demo|model|run]
       [--resume]
Node file: one "NAME SSH_HOST ABSOLUTE_PREFIX" row per node; # comments allowed.
SSH must already work without prompts. Network configuration, OS permissions and
cleanup are prerequisites. --resume skips installation, retaining completed work.
--model is required for the model and run stages. Run this from the head.
The final run leaves node services and the head console running after SSH closes.
HELP
}
while (($#)); do
    case $1 in
        --manager|--checksums|--head-prefix|--nodes|--model|--ssh-config|--through)
            [[ $# -ge 2 && -n $2 ]] || { usage >&2; exit 2; }
            case $1 in
                --manager) manager=$2;; --checksums) checksums=$2;; --head-prefix) head_prefix=$2;;
                --nodes) nodes_file=$2;; --model) model=$2;; --ssh-config) ssh_config=$2;; --through) through=$2;;
            esac; shift 2;;
        --resume) resume=1; shift;;
        --help|-h) usage; exit 0;;
        *) usage >&2; exit 2;;
    esac
done
fail() { printf 'Cluster setup: %s\n' "$*" >&2; exit 1; }
valid_prefix() { [[ $1 =~ ^/[A-Za-z0-9._/-]+$ && $1 != / && $1 != */ && $1 != *//* && $1/ != */../* && $1/ != */./* ]]; }
script_dir=$(cd "$(dirname "$0")" && pwd)
[[ -f $manager && -f $checksums && -f $nodes_file ]] || fail 'manager, checksums and nodes files are required'
valid_prefix "$head_prefix" && [[ $head_prefix != "$HOME" ]] || fail 'head prefix must be a dedicated absolute path without spaces, dot components or a trailing slash'
for tool in bash python3 ssh scp; do command -v "$tool" >/dev/null || fail "$tool is required on the head"; done
python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' || fail 'Python 3.9 or newer is required on the head'
[[ -f $script_dir/ssh-prerequisites.sh && -f $script_dir/../api-check.py ]] || fail 'keep this script with the other tools in the source checkout'
case $through in install|demo|model|run) ;; *) fail 'invalid --through value';; esac
case $through in model|run) [[ -f $model ]] || fail 'select a local model with --model, or first use --through demo';; esac
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10)
[[ -z $ssh_config ]] || { [[ -f $ssh_config ]] || fail 'SSH config missing'; ssh_args+=(-F "$ssh_config"); }
python3 - "$manager" "$checksums" <<'PY'
import hashlib, pathlib, sys
p=pathlib.Path(sys.argv[1]); matches=[]
for line in pathlib.Path(sys.argv[2]).read_text().splitlines():
    fields=line.split()
    if len(fields)==2 and fields[1].lstrip('*')==p.name: matches.append(fields[0])
if len(matches)!=1 or hashlib.sha256(p.read_bytes()).hexdigest()!=matches[0]:
    sys.exit('Manager checksum mismatch or missing manifest entry')
print('Manager checksum: verified')
PY
LC_ALL=C grep -aq '^__FRONK_PAYLOAD__$' "$manager" || fail 'use the portable release fronk-manager.sh, not tools/install/fronk-install.sh (which needs its source tree)'
names=() hosts=() prefixes=()
while read -r name host prefix extra || [[ -n ${name:-} ]]; do
    [[ -n $name && $name != \#* ]] || continue
    [[ -z ${extra:-} && $name =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,62}$ && $host =~ ^[A-Za-z0-9_][A-Za-z0-9_.@-]*$ ]] && valid_prefix "$prefix" || fail 'invalid node row (name, SSH destination or dedicated absolute prefix)'
    for old in "${names[@]:-}"; do [[ $old != "$name" ]] || fail "duplicate node name: $name"; done
    for old in "${hosts[@]:-}"; do [[ $old != "$host" ]] || fail "duplicate SSH host: $host"; done
    names+=("$name"); hosts+=("$host"); prefixes+=("$prefix")
done < "$nodes_file"
((${#names[@]})) || fail 'empty node roster'
((${#names[@]} <= 20)) || fail 'at most 20 nodes are supported'
for ((i=0;i<${#hosts[@]};i++)); do
    host=${hosts[i]}; prefix=${prefixes[i]}
    printf '\nChecking SSH prerequisites: %s\n' "$host"
    ssh -T "${ssh_args[@]}" "$host" "bash -s -- '$prefix'" < "$script_dir/ssh-prerequisites.sh" || fail "SSH/prerequisites failed: $host (check the non-interactive SSH PATH and user login)"
done
# Check local ports before making changes, even on a fresh prefix. A different
# installation can own them despite having a different state-directory lock.
if [[ $through != install ]]; then
    python3 - "$head_prefix" <<'PY'
import pathlib, socket, sys
p=pathlib.Path(sys.argv[1]); fields={}; config=p/'src/fronk.conf'
for line in config.read_text().splitlines() if config.exists() else []:
    parts=line.split('#',1)[0].split()
    if len(parts)==2: fields[parts[0]]=parts[1]
for port in {7730, int(fields.get('control-port','7700'))+10}:
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex(('127.0.0.1',port))==0:
            sys.exit(f'Port {port} is already in use. Use the running console, or stop its head/console before setup.')
PY
fi
if ((!resume)); then
    bash "$manager" --yes --role head --prefix "$head_prefix" --demo-later
    for ((i=0;i<${#names[@]};i++)); do
        host=${hosts[i]}; prefix=${prefixes[i]}; name=${names[i]}
        node_root=$(ssh -T "${ssh_args[@]}" "$host" 'printf "%s" "$HOME"')
        [[ $node_root =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "unsupported remote home: $host"
        remote_manager=$node_root/fronk-manager.sh
        printf '\nInstalling %s on %s\n' "$name" "$host"
        scp "${ssh_args[@]}" "$manager" "$host:$remote_manager"
        # The pairing code flows only through this pipe, never argv or a log.
        "$head_prefix/bin/fronk-manager" --pairing-code |
            ssh -T "${ssh_args[@]}" "$host" "bash '$remote_manager' --yes --role node --name '$name' --prefix '$prefix' --secret-stdin"
        ssh -T "${ssh_args[@]}" "$host" "'$prefix/bin/fronk-manager' --prefix '$prefix' --status"
    done
fi
[[ $through != install ]] || exit 0
"$head_prefix/bin/fronk-manager" --join-nodes --yes --nodes "${names[*]}"
[[ $through != demo ]] || exit 0
"$head_prefix/bin/fronk-manager" --configure-model --yes --model "$model"
[[ $through != model ]] || exit 0
# setsid is unavailable on stock macOS. A new session prevents --run from
# choosing its interactive terminal path when this script was run in Terminal.
python3 - "$head_prefix" "$script_dir/../api-check.py" <<'PY'
import json,pathlib,subprocess,sys,time,urllib.request
p=pathlib.Path(sys.argv[1]); fields={}
for line in (p/'src/fronk.conf').read_text().splitlines():
    parts=line.split('#',1)[0].split()
    if len(parts)==2: fields[parts[0]]=parts[1]
port=int(fields.get('control-port','7700')); console='http://127.0.0.1:7730'
log=p/'state/setup-head.log'
with log.open('ab') as output:
    proc=subprocess.Popen([str(p/'bin/fronk-manager'),'--run'],stdin=subprocess.DEVNULL,
                          stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
print('Head console starting:',console,'; log:',log,flush=True)
print('Waiting for the complete generation check. On failure, services remain available for diagnosis.',flush=True)
deadline=time.monotonic()+1800
last_phase=None
while time.monotonic()<deadline:
    if proc.poll() is not None:
        sys.exit(f'Head console exited ({proc.returncode}). Inspect {log}.')
    try:
        with urllib.request.urlopen(console+'/state',timeout=5) as r: state=json.load(r)
        phase=state.get('phase')
        if phase!=last_phase:
            print('Head phase:',phase,flush=True); last_phase=phase
        if state.get('waiting') in ('failed','exited'):
            reason=state.get('selftest','')
            if 'failed' not in reason and 'mismatch' not in reason:
                reason=state.get('failure','')
            sys.exit('Head stopped: '+reason+'; inspect the console and log.')
        if state.get('running') and state.get('phase')=='REPORT' and state.get('waiting')=='after': break
    except (OSError,ValueError): pass
    time.sleep(2)
else: sys.exit('Head not ready after 30 minutes. Inspect setup-head.log and console; do not bypass checks.')
# Use this checkout's public API check, including its llama.cpp thinking flag.
subprocess.run([sys.executable,sys.argv[2],'--base-url',f'http://127.0.0.1:{port+10}/v1'],check=True)
print('Cluster verified. Console:',console)
PY
