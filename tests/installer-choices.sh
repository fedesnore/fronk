#!/usr/bin/env bash
# Check the shared GUI/terminal decisions without a build, service or network.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/tools/install/onboard.sh"
die() { echo "$*" >&2; exit 1; }
yes=0
answer=""
ask() {
    # A selected default must actually be present in a read-only choice list.
    [[ -z ${3:-} || "|$3|" == *"|$2|"* ]] || die "default missing from menu: $2"
    printf '%s\n' "${answer:-$2}"
}
[[ $(choose_roster '') == node-1 ]]
answer=3
[[ $(choose_roster '') == 'node-1 node-2 node-3' ]]
answer=20
roster=$(choose_roster '')
read -r -a names <<< "$roster"
[[ ${#names[@]} == 20 && ${names[19]} == node-20 ]]
answer=later
[[ -z $(choose_roster 'desk spare') ]]
answer=""
[[ $(choose_roster 'desk spare') == 'desk spare' ]]
yes=1
[[ -z $(choose_roster '') && $(choose_roster 'desk spare') == 'desk spare' ]]
yes=0
[[ $(choose_node_name '') == node-1 && $(choose_node_name desk) == desk ]]
listing='1) ref runs here
2) avx2 not supported here
3) metal runs here <- recommended
4) neon runs here'
[[ $(choose_backend "$listing" '') == metal ]]
[[ $(choose_backend "$listing" avx2) == metal ]]
[[ $(choose_backend "$listing" neon) == neon ]]
[[ $(choose_backend "$listing" auto) == auto ]]
ask() {
    case "$1" in
        'How many'*) [[ $3 == *custom-names* ]]; echo custom-names ;;
        'Custom node names'*) echo 'desk spare' ;;
        'On this node'*) [[ $3 == *custom-name* ]]; echo custom-name ;;
        'Custom node name,'*) echo desk ;;
        *) die "unexpected question: $1" ;;
    esac
}
[[ $(choose_roster '') == 'desk spare' && $(choose_node_name '') == desk ]]
# Resuming with a successful old demo must still honor an explicit pause. In
# particular, the install-only SSH stage must not jump into model verification.
guide_nodes() { :; }
demo_done() { return 0; }
ask() { die 'unexpected model question after --demo-later'; }
configure_model() { die 'unexpected model verification after --demo-later'; }
demo_later=1
head_setup
echo '[ok] installer choices: assigned names, counts, existing roster, custom names, compatible backends and unattended defaults'
