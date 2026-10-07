#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: C550_ARCH=<observed compiler arch> compile.sh OUTPUT_BINARY" >&2
    exit 2
fi
: "${C550_ARCH:?Set C550_ARCH from compiler/device evidence before CPU-only compilation}"
probe_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${MXCC:-mxcc}" -O3 -std=c++17 -x maca -offload-arch="$C550_ARCH" --maca-path="${MACA_PATH:-/opt/maca}" "$probe_source_dir/probe.cpp" -o "$1"
