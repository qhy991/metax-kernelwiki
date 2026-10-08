#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: C550_ARCH=<observed compiler arch> [C550_WMMA_CONTROL=0|1] compile.sh OUTPUT_BINARY" >&2
    exit 2
fi
probe_control="${C550_WMMA_CONTROL-0}"
case "$probe_control" in
    0|1) ;;
    *) echo "C550_WMMA_CONTROL must be 0 or 1" >&2; exit 2 ;;
esac
: "${C550_ARCH:?Set C550_ARCH from compiler/device evidence before CPU-only compilation}"
probe_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${MXCC:-mxcc}" -O3 -std=c++17 -x maca -offload-arch="$C550_ARCH" \
    --maca-path="${MACA_PATH:-/opt/maca}" "-DC550_WMMA_CONTROL=$probe_control" \
    "$probe_source_dir/probe.cpp" -o "$1"
