#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: C550_ARCH=<observed compiler arch> [C550_WMMA_CONTROL=0|1] [C550_WMMA_PREFIX=0|1] [C550_WMMA_WITNESS=0|1|2|3|4] compile.sh OUTPUT_BINARY" >&2
    exit 2
fi
probe_control="${C550_WMMA_CONTROL-0}"
case "$probe_control" in
    0|1) ;;
    *) echo "C550_WMMA_CONTROL must be 0 or 1" >&2; exit 2 ;;
esac
probe_prefix="${C550_WMMA_PREFIX-0}"
case "$probe_prefix" in
    0|1) ;;
    *) echo "C550_WMMA_PREFIX must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$probe_prefix" == 1 && "$probe_control" != 1 ]]; then
    echo "C550_WMMA_PREFIX=1 requires C550_WMMA_CONTROL=1" >&2
    exit 2
fi
probe_witness="${C550_WMMA_WITNESS-0}"
case "$probe_witness" in
    0|1|2|3|4) ;;
    *) echo "C550_WMMA_WITNESS must be 0, 1, 2, 3 or 4" >&2; exit 2 ;;
esac
if [[ "$probe_witness" != 0 && ( "$probe_control" != 1 || "$probe_prefix" != 0 ) ]]; then
    echo "C550_WMMA_WITNESS=$probe_witness requires C550_WMMA_CONTROL=1 and C550_WMMA_PREFIX=0" >&2
    exit 2
fi
: "${C550_ARCH:?Set C550_ARCH from compiler/device evidence before CPU-only compilation}"
probe_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${MXCC:-mxcc}" -O3 -std=c++17 -x maca -offload-arch="$C550_ARCH" \
    --maca-path="${MACA_PATH:-/opt/maca}" "-DC550_WMMA_CONTROL=$probe_control" \
    "-DC550_WMMA_PREFIX=$probe_prefix" "-DC550_WMMA_WITNESS=$probe_witness" "$probe_source_dir/probe.cpp" -o "$1"
