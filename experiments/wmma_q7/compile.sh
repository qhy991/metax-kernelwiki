#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: compile.sh FRESH_OUTPUT_BINARY" >&2
    exit 2
fi
if [[ "${C550_ARCH-xcore1000}" != xcore1000 ]]; then
    echo "This collector requires C550_ARCH=xcore1000" >&2
    exit 2
fi
if [[ -e "$1" || -L "$1" ]]; then
    echo "Output binary must not already exist" >&2
    exit 2
fi
probe_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${MXCC:-mxcc}" -O3 -std=c++17 -x maca -offload-arch=xcore1000 \
    --maca-path="${MACA_PATH:-/opt/maca}" "$probe_source_dir/repro.cpp" -o "$1"
