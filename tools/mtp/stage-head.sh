#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
"$here/stage-ds4.sh" "${1:?ds4 source required}" "${2:?private destination required}"
