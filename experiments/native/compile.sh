#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: C550_ARCH=<observed compiler arch> [C550_COPY_LAUNCH_BOUND=0|1024] compile.sh OUTPUT_BINARY" >&2
    exit 2
fi
probe_copy_launch_bound="${C550_COPY_LAUNCH_BOUND-0}"
case "$probe_copy_launch_bound" in
    0|1024) ;;
    *) echo "C550_COPY_LAUNCH_BOUND must be 0 or 1024" >&2; exit 2 ;;
esac
: "${C550_ARCH:?Set C550_ARCH from compiler/device evidence before CPU-only compilation}"
probe_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${MXCC:-mxcc}" -O3 -std=c++17 -x maca -offload-arch="$C550_ARCH" \
    --maca-path="${MACA_PATH:-/opt/maca}" "-DC550_COPY_LAUNCH_BOUND=$probe_copy_launch_bound" \
    "$probe_source_dir/probe.cpp" -o "$1"
