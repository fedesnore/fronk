#!/usr/bin/env bash
# Storage failures are checked before a build or model transfer, without
# filling a real filesystem or depending on this test host's free space.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/tools/install/onboard.sh"
die() { echo "$*" >&2; exit 1; }
OS=Linux STATE=/fixture/state role=node installed=""
fixture_fs=ext2/ext3 fixture_free=10000000
stat() { printf '%s\n' "$fixture_fs"; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted\nfixture 20000000 0 %s 0%% /fixture\n' "$fixture_free"; }
check_storage_directory
fixture_fs=tmpfs
if result=$(check_storage_directory 2>&1); then die 'RAM filesystem accepted'; fi
[[ $result == *'RAM, not persistent disk'* ]]
fixture_fs=ramfs
if (check_storage_directory) 2>/dev/null; then die 'ramfs accepted'; fi
fixture_fs=ext2/ext3 fixture_free=4317135
if result=$(check_storage_directory 2>&1); then die 'insufficient unit space accepted'; fi
[[ $result == *'4317136 KiB'* && $result == *'4317135 KiB available'* ]]
fixture_free=4317136
check_storage_directory
fixture_free=1048576 installed=node
check_storage_directory # An upgrade can reuse existing unit files.
installed="" role=head
check_storage_directory
fixture_free=1048575
if (check_storage_directory) 2>/dev/null; then die 'insufficient build reserve accepted'; fi
fixture_free=unknown
if (check_storage_directory) 2>/dev/null; then die 'unreadable free space accepted'; fi
OS=Darwin fixture_free=1048576
stat() { die 'Linux-only filesystem probe used on macOS'; }
check_storage_directory
echo '[ok] installer storage: persistent filesystem, free space, upgrade reuse and platform checks'
