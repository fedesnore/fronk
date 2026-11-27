#!/usr/bin/env bash
# First settings of a llama.cpp (GSQ) head: one-line files in its state directory,
# read by tools/engine/gsq/run.sh at every start.
#
#   gsq-head-settings.sh STATE_DIR MEMORY_BYTES
#
# The engine's own defaults stay as they are. A cluster installed with them wrote
# 2 to 5 tokens per second (docs/GSQ.md, "The engine's first settings"); these are
# the values the author's head uses. None of them changes what the model writes:
# drafts are still verified token by token (gsq-lenience is not touched). The last
# three trade context and chat-switching speed for memory, so they are written
# only on a head with less than 32 GiB.
#
# Written once. A file that is already there is kept, and after the first run a
# record (gsq-settings.installed) stops later runs, so a file the person removed
# to get the engine's default stays removed.
set -euo pipefail
state=${1:?state directory} memory=${2:?memory in bytes}
[[ -d $state ]] || { echo "gsq-head-settings: no such directory: $state" >&2; exit 1; }
[[ $memory =~ ^[0-9]+$ ]] || { echo "gsq-head-settings: memory is not a number: $memory" >&2; exit 1; }
record=$state/gsq-settings.installed
[[ ! -e $record ]] || exit 0
: > "$record.new"
put() {  # NAME VALUE
    if [[ -e $state/$1 ]]; then
        printf '  %-20s kept as it was\n' "$1"
    else
        printf '%s\n' "$2" > "$state/$1"
        printf '%s %s\n' "$1" "$2" >> "$record.new"
        printf '  %-20s %s\n' "$1" "$2"
    fi
}
put gsq-threads 2          # more threads compete with the sum of the experts
put gsq-draft 12           # MTP draft tokens per pass
put gsq-draft-p-min 0.05   # do not stop a draft at the first uncertain token
put gsq-draft-chain 0.75   # chain threshold of the drafts
put gsq-copy-drafts 31     # drafts copied from the context fill one 32-token call
if ((memory < 32 * 1024 * 1024 * 1024)); then
    put ds4-ctx 32768            # context length in tokens
    put gsq-cache-ram-mb 0       # no memory for chats other than the current one
    put gsq-ctx-checkpoints 4    # rollback points per chat, about 112 MiB each
else
    put ds4-ctx 65536
fi
mv "$record.new" "$record"
