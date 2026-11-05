#!/usr/bin/env bash
# Reconstruct the tested Fronk engine from a public commit and our readable
# patch. No developer checkout, installed engine or running binary is modified.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
revision=8db1d1d155cb0400a86a86b9c62d0defb3a6148b
destination=${1:?usage: prepare.sh DESTINATION}
key="$revision $(cksum < "$here/ds4-fronk.patch")"
complete_source() {
    local name
    while IFS= read -r name; do
        [[ -f $1/$name ]] || { echo "Missing engine source: $name" >&2; return 1; }
    done < "$here/source-files"
}
if [[ -f $destination/.fronk-source && $(cat "$destination/.fronk-source") == "$key" ]] && complete_source "$destination"; then
    printf 'Engine source already prepared: %s\n' "$destination"
    exit
fi
[[ ! -e $destination ]] || { echo "Refusing to replace $destination; choose a new destination." >&2; exit 1; }
mkdir -p "$(dirname "$destination")"
work=$(mktemp -d "${destination}.tmp.XXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir "$work/source"
if [[ -n ${DS4_UPSTREAM:-} ]]; then
    git -C "$DS4_UPSTREAM" archive "$revision" | tar -xf - -C "$work/source"
else
    git -C "$work/source" init -q
    git -C "$work/source" fetch --depth=1 https://github.com/antirez/ds4.git "$revision"
    git -C "$work/source" checkout -q --detach FETCH_HEAD
fi
# An archived offline source has no .git. Give it its own repository boundary:
# otherwise git apply can discover Fronk's parent repository and skip every
# patch path because this working directory is below that repository's root.
git -C "$work/source" init -q
(cd "$work/source" && git apply --check "$here/ds4-fronk.patch" && git apply "$here/ds4-fronk.patch")
complete_source "$work/source"
printf '%s\n' "$key" > "$work/source/.fronk-source"
mv "$work/source" "$destination"
printf 'Prepared engine source: %s\n' "$destination"
