#!/usr/bin/env bash
# Work on a private source copy. The original ds4 checkout is also used by
# the baseline and must remain available if a release or an experiment fails.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source_dir=$(cd "${1:?ds4-fronk checkout required}" && pwd -P)
mkdir -p "${2:?staging directory required}"
dest=$(cd "$2" && pwd -P)
[[ $source_dir != "$dest" && -f $source_dir/Makefile ]]
(cd "$source_dir" && tar -cf - -T "$here/../engine/source-files") | tar -xf - -C "$dest"
(cd "$dest" && patch -f -p1 < "$here/ds4-tree.patch")
cp "$here/fronk_mtp_tree.inc" "$here/fronk_mtp_worker.inc" "$dest/"
python3 "$here/patch-head.py" "$dest"
