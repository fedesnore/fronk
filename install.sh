#!/usr/bin/env bash
# install.sh - the simple way to put Fronk on a computer.
#
#   bash install.sh head      on the Mac with the best GPU, first
#   bash install.sh node      on every other computer, when the head says so
#   bash install.sh check     only report what this computer has and lacks
#
# It does three things and then hands over to the installer manager
# (tools/install/fronk-install.sh), which asks the questions and builds:
#   1. looks at this computer: system, processor, disk, memory;
#   2. finds what is missing to build Fronk and offers to install it, showing
#      every command first. Nothing is installed without a yes;
#   3. gets the Fronk source if this file was downloaded alone.
#
# The head it installs is the llama.cpp one with the 3-bit weights (docs/GSQ.md):
# an Apple Silicon Mac. Nodes are macOS or Linux, x86 with AVX2 or ARM.
# Written for the bash 3.2 that macOS ships: no associative arrays, no mapfile.
set -euo pipefail

FRONK_REF=${FRONK_REF:-v0.7.0}          # tag or branch to download when run alone
FRONK_REPO=${FRONK_REPO:-fedesnore/fronk}
PREFIX=${FRONK_PREFIX:-$HOME/fronk}
role="" assume_yes=0 accept_tools=0 pass=()

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die()  { printf '\nFronk install stopped: %s\n' "$*" >&2
         printf 'Nothing was left half done: fix the point above and run the same command again.\n' >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
# Questions are read from the terminal, so "curl ... | bash" can still ask.
confirm() { # question -> 0 for yes
    local answer=""
    if ((accept_tools)); then say "$1 yes (--install-tools)"; return 0; fi
    if ((assume_yes)); then return 1; fi
    { : </dev/tty; } 2>/dev/null || return 1
    printf '%s [y/N] ' "$1" >/dev/tty
    IFS= read -r answer </dev/tty || return 1
    case $answer in y|Y|yes|YES|s|S|si|SI) return 0 ;; *) return 1 ;; esac
}

usage() { cat <<'HELP'
install.sh - the simple way to put Fronk on a computer.

  bash install.sh head      on the Mac with the best GPU, first
  bash install.sh node      on every other computer, when the head says so
  bash install.sh check     only report what this computer has and lacks

Options: --prefix DIR (default ~/fronk), --yes (never ask; stop if something is missing),
--install-tools (install the missing build tools it lists, without asking).
Anything after -- goes to the installer manager (see its --help).
HELP
}

while (($#)); do
    case $1 in
        head|node|check) role=$1 ;;
        --prefix) [[ $# -ge 2 ]] || die "--prefix needs a folder"; PREFIX=$2; shift ;;
        --yes|-y) assume_yes=1 ;;
        --install-tools) accept_tools=1 ;;
        --help|-h) usage; exit 0 ;;
        --) shift; while (($#)); do pass+=("$1"); shift; done; break ;;
        *) die "unknown word: $1 (use head, node or check; --help explains)" ;;
    esac
    shift
done

OS=$(uname -s); ARCH=$(uname -m)
[[ $OS == Darwin || $OS == Linux ]] || die "Fronk runs on macOS and Linux; this is $OS."
# A terminal opened under Rosetta reports an Intel Mac. Look at the real processor.
if [[ $OS == Darwin && $ARCH == x86_64 && $(/usr/sbin/sysctl -in sysctl.proc_translated 2>/dev/null || echo 0) == 1 ]]; then ARCH=arm64; fi

# A shell started by ssh or by another program may not have Homebrew's folders in
# its PATH, and would not see a CMake that is installed.
if [[ $OS == Darwin ]]; then
    for dir in /opt/homebrew/bin /usr/local/bin; do
        [[ ! -d $dir || ":$PATH:" == *":$dir:"* ]] || PATH=$PATH:$dir
    done
    export PATH
fi

step "Fronk install · $(hostname -s 2>/dev/null || hostname) · $OS $ARCH"
if [[ -z $role ]]; then
    say "Start on the head: the Apple Silicon Mac with the best GPU. Then run this on each node."
    if { : </dev/tty; } 2>/dev/null && ((!assume_yes)); then
        printf 'Is this computer the head or a node? [head/node] ' >/dev/tty
        IFS= read -r role </dev/tty || role=""
    fi
    [[ $role == head || $role == node ]] || die "say which part this computer is: bash install.sh head, or bash install.sh node"
fi

# ---- 1. this computer -------------------------------------------------------
problems=0 fixable=0     # problems stop the install; fixable ones this script can install
note() { say "  ok    $*"; }
miss() { say "  MISSING  $*"; problems=$((problems+1)); }
lack() { say "  MISSING  $*  (this script can install it)"; fixable=$((fixable+1)); }

# A bare system can lack even awk: then the sizes are unknown until it is installed.
num() { [[ $1 =~ ^[0-9]+$ ]]; }
memory_gb() {
    have awk || { echo "?"; return; }
    if [[ $OS == Darwin ]]; then /usr/sbin/sysctl -n hw.memsize | awk '{printf "%.0f", $1/1e9}'
    else awk '/^MemTotal:/ {printf "%.0f", $2/1e6}' /proc/meminfo; fi
}
free_gb() { # free space where PREFIX will live, in GB
    have awk && have df || { echo "?"; return; }
    local dir=$PREFIX
    while [[ ! -d $dir ]]; do dir=$(dirname "$dir"); done
    df -Pk "$dir" | awk 'END {printf "%.0f", $4/1e6}'
}
cpu_ok() { # can this processor run the expert kernels?
    case $ARCH in
        arm64|aarch64) return 0 ;;
        x86_64)
            local flags
            if [[ $OS == Linux ]]; then flags=$(grep -m1 '^flags' /proc/cpuinfo || true)
            else flags=$( { /usr/sbin/sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null || true; } | tr 'A-Z\n' 'a-z ' ); fi
            flags=" $(printf '%s' "$flags" | tr 'A-Z' 'a-z') "
            # Linux writes avx2 fma f16c bmi2; macOS writes AVX2 FMA F16C BMI2.
            for f in avx2 fma f16c bmi2; do [[ $flags == *" $f "* ]] || return 1; done ;;
        *) return 1 ;;
    esac
}

step "1 of 3 · This computer"
ram=$(memory_gb); disk=$(free_gb)
say "  memory $ram GB, free disk for $PREFIX: $disk GB"
if [[ $role == head ]]; then
    if [[ $OS == Darwin && $ARCH == arm64 ]]; then note "Apple Silicon Mac: it can be the head"
    else
        say "  This computer is $OS $ARCH. The head installed here needs an Apple Silicon Mac."
        say "  Use this computer as a node (bash install.sh node), or see docs/INSTALL.md for the"
        say "  experimental heads on other hardware."
        problems=$((problems+1))
    fi
    # Two model files (75.8 GB), the expert units cut from them (43.7 GB), the engine build (2 GB).
    if num "$disk" && ((disk >= 125)); then note "disk: enough for the model, its units and the build (about 122 GB)"
    else say "  NOTE  the model (76 GB) and its units (44 GB) need about 122 GB; $disk GB are free here."
         say "        The model can live on an external SSD: you choose the folder later."; fi
else
    if cpu_ok; then note "processor: has what the expert kernels need"
    else miss "this processor lacks AVX2, FMA, F16C or BMI2 (x86), or is not x86/ARM: it cannot compute expert slices"; fi
    # One unit is 2.2 GB on disk and in memory; a node is useful from one unit up.
    if ! num "$disk"; then say "  NOTE  free disk space not measured yet (awk is missing): a node keeps 2.2 GB per unit"
    elif ((disk >= 4)); then note "disk: room for at least one unit (2.2 GB each)"
    else miss "less than 4 GB free in $(dirname "$PREFIX"): a node keeps 2.2 GB per unit it is given"; fi
fi

# ---- 2. what is needed to build ---------------------------------------------
step "2 of 3 · Tools to build Fronk"
need_pkgs=""            # Linux packages to ask for
need_clt=0 need_cmake=0
python_ok() { have python3 && python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; }

if [[ $OS == Darwin ]]; then
    # Without the command line tools /usr/bin/cc, make, git and python3 are stubs
    # that open an install dialog instead of running.
    if xcode-select -p >/dev/null 2>&1 && cc --version >/dev/null 2>&1; then note "Apple command line tools (compiler, make, git, Python)"
    else lack "Apple command line tools (compiler, make, git, Python)"; need_clt=1; fi
    if ((!need_clt)); then
        if python_ok; then note "Python $(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        else miss "Python 3.9 or newer"; fi
    fi
else
    # awk and find are part of every installed system; a bare container can lack them.
    for tool in cc make tar awk find; do
        if have "$tool"; then note "$tool"; else lack "$tool"; need_pkgs="$need_pkgs $tool"; fi
    done
    # A compiler without the C library headers cannot build anything.
    if have cc && ! printf '#include <stdio.h>\nint main(void){return 0;}\n' | cc -x c - -o /dev/null 2>/dev/null; then
        lack "C library headers"; need_pkgs="$need_pkgs headers"
    fi
    if python_ok; then note "Python $(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    elif have python3; then miss "Python 3.9 or newer (this system has $(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null): upgrade the distribution or install a newer python3)"
    else lack "Python 3"; need_pkgs="$need_pkgs python3"; fi
    if have curl || have wget || python_ok; then :; else lack "curl or wget"; need_pkgs="$need_pkgs curl"; fi
fi
if [[ $role == head ]]; then
    if have cmake; then note "CMake (builds the llama.cpp engine)"; else lack "CMake (builds the llama.cpp engine)"; need_cmake=1; fi
    if have openssl; then note "openssl (makes the pairing code)"; else miss "openssl"; fi
fi

linux_install_command() { # only what is missing, by this distribution's package names
    local sudo="sudo " tool list="" compiler make=make python=python3 awk=gawk find=findutils
    [[ $(id -u) != 0 ]] || sudo=""
    if have apt-get; then compiler="build-essential"
    elif have dnf || have yum || have zypper; then compiler="gcc glibc-devel"
    elif have pacman; then compiler="gcc"; python=python
    elif have apk; then compiler="build-base"; awk=gawk
    else return 1; fi
    for tool in $need_pkgs; do
        case $tool in
            cc|headers) [[ " $list " == *" $compiler "* ]] || list="$list $compiler" ;;
            make) list="$list $make" ;; python3) list="$list $python" ;; awk) list="$list $awk" ;;
            find) list="$list $find" ;; tar) list="$list tar" ;; curl) list="$list curl" ;;
        esac
    done
    if have apt-get; then echo "${sudo}apt-get update && ${sudo}apt-get install -y$list"
    elif have dnf; then echo "${sudo}dnf install -y$list"
    elif have yum; then echo "${sudo}yum install -y$list"
    elif have zypper; then echo "${sudo}zypper --non-interactive install$list"
    elif have pacman; then echo "${sudo}pacman -Sy --needed --noconfirm$list"
    else echo "${sudo}apk add --no-cache$list"; fi
}

if [[ $role == check ]]; then
    step "Result"
    if ((problems)); then say "This computer cannot be a node as it is: see the lines marked MISSING."; exit 1; fi
    if ((fixable)); then say "$fixable tool(s) missing. Run bash install.sh node (or head): it offers to install them."; exit 1; fi
    say "This computer has everything a Fronk node needs."; exit 0
fi

((problems == 0)) || die "this computer cannot take this role as it is: see the lines marked MISSING above"

if ((need_clt)); then
    say ""
    say "The Apple command line tools are needed. macOS installs them itself:"
    say "    xcode-select --install"
    confirm "Open Apple's installer now?" || die "install the command line tools (xcode-select --install), then run this command again"
    xcode-select --install >/dev/null 2>&1 || true
    say "A window asks to install the tools: choose Install there. Waiting for it to finish (5 to 15 minutes)..."
    waited=0
    until xcode-select -p >/dev/null 2>&1 && cc --version >/dev/null 2>&1; do
        sleep 10; waited=$((waited+10))
        ((waited < 3600)) || die "the command line tools did not finish installing in an hour; complete Apple's installer, then run this command again"
    done
    say "Command line tools installed."
    python_ok || die "this macOS has no Python 3.9 or newer; install Python 3 from python.org, then run this command again"
fi
if [[ -n $need_pkgs ]]; then
    command=$(linux_install_command) || die "missing:$need_pkgs. This distribution's package manager is not one this script knows: install a C compiler, make, Python 3.9+ and curl, then run this command again"
    say ""
    if [[ $(id -u) == 0 ]]; then say "Missing:$need_pkgs. This command installs them:"
    else say "Missing:$need_pkgs. This command installs them (it asks for your password):"; fi
    say "    $command"
    confirm "Run it now?" || die "run the command above (or ask who administers this computer), then run this command again"
    # The pairing code may be waiting on standard input for the manager: the package
    # manager must not swallow it. sudo asks for its password on the terminal itself.
    sh -c "$command" </dev/null || die "the package installation failed; read its message above"
    for tool in cc make awk find tar; do have "$tool" || die "$tool is still missing after the installation"; done
    python_ok || die "this system's Python is $(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo missing) and Fronk needs 3.9 or newer: the distribution is too old for its own package, so upgrade it or install a newer python3 first"
fi
if ((need_cmake)); then
    say ""
    if have brew; then
        say "CMake is missing. Homebrew can install it:"
        say "    brew install cmake"
        confirm "Run it now?" || die "install CMake (brew install cmake), then run this command again"
        brew install cmake </dev/null || die "Homebrew could not install CMake; read its message above"
    else
        # No Homebrew: a private copy from the Python package index, inside the
        # Fronk tools folder. Nothing system-wide changes and no password is asked.
        tools=${XDG_DATA_HOME:-$HOME/.local/share}/fronk-tools
        say "CMake is missing. A private copy can go in $tools (no password, nothing else changes):"
        say "    python3 -m venv $tools && $tools/bin/pip install cmake"
        confirm "Install it there now?" || die "install CMake (from cmake.org, or with Homebrew), then run this command again"
        python3 -m venv "$tools" </dev/null || die "could not create $tools"
        "$tools/bin/pip" install --quiet --disable-pip-version-check 'cmake>=3.21,<5' </dev/null || die "could not download CMake; check the internet connection and run this command again"
        export PATH="$tools/bin:$PATH"
    fi
    have cmake || die "CMake is still missing after the installation"
    say "CMake $(cmake --version | awk 'NR==1 {print $3}') ready."
fi
# A private CMake installed by an earlier run of this script.
if [[ $role == head ]] && ! have cmake && [[ -x ${XDG_DATA_HOME:-$HOME/.local/share}/fronk-tools/bin/cmake ]]; then
    export PATH="${XDG_DATA_HOME:-$HOME/.local/share}/fronk-tools/bin:$PATH"
fi

# ---- 3. the Fronk source ------------------------------------------------------
step "3 of 3 · Fronk source"
here=$(cd "$(dirname "$0")" 2>/dev/null && pwd -P || true)
if [[ -n $here && -f $here/tools/install/fronk-install.sh && -f $here/Makefile ]]; then
    source_dir=$here
    say "  Using the source this file is in: $source_dir"
else
    cache=${XDG_CACHE_HOME:-$HOME/.cache}/fronk
    name=$(printf '%s' "$FRONK_REF" | tr -c 'A-Za-z0-9._-' '-')
    source_dir=$cache/source-$name
    if [[ -f $source_dir/tools/install/fronk-install.sh ]]; then
        say "  Using the source downloaded before: $source_dir"
    else
        url=https://github.com/$FRONK_REPO/archive/$FRONK_REF.tar.gz
        say "  Downloading $url"
        mkdir -p "$cache"; work=$(mktemp -d "$cache/download.XXXXXX")
        trap 'rm -rf "$work"' EXIT
        if have curl; then curl -fsSL --retry 3 -o "$work/source.tar.gz" "$url"
        elif have wget; then wget -q -O "$work/source.tar.gz" "$url"
        else python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$url" "$work/source.tar.gz"; fi ||
            die "could not download the source; check the internet connection and run this command again"
        mkdir "$work/tree"
        tar -xzf "$work/source.tar.gz" -C "$work/tree" --strip-components 1 || die "the downloaded source archive is damaged; run this command again"
        [[ -f $work/tree/tools/install/fronk-install.sh ]] || die "the downloaded archive is not a Fronk source tree"
        rm -rf "$source_dir"; mv "$work/tree" "$source_dir"
        rm -rf "$work"; trap - EXIT
        say "  Source ready: $source_dir"
    fi
fi

# ---- hand over ----------------------------------------------------------------
manager=$source_dir/tools/install/fronk-install.sh
# --no-measure: timings are taken at the first connection, on the units the head
# really assigns, not on a trial of the other unit format. --backend auto: each
# machine uses the kernel made for its processor.
args=(--terminal --prefix "$PREFIX" --role "$role" --backend auto --no-measure)
if [[ $role == head ]]; then args+=(--engine gsq); fi
if ((assume_yes)); then args+=(--yes); fi
if ((${#pass[@]})); then args+=("${pass[@]}"); fi
if [[ $role == node && $OS == Linux ]] && ! systemctl --user show-environment >/dev/null 2>&1; then
    say "  This Linux has no user service manager here: the node will be started by hand (the manager says how)."
    args+=(--no-service)
fi
step "Installing the $role"
exec bash "$manager" "${args[@]}"
