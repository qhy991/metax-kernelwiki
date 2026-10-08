#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: compile.sh FRESH_OUTPUT_BINARY" >&2
    exit 2
fi
if [[ -e "$1" || -L "$1" ]]; then
    echo "Output binary must not already exist" >&2
    exit 2
fi
producer_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
producer_sdk_root="${MACA_PATH:-/opt/maca}"
# Installed public MCRTC functions are exported by libmcruntime.
producer_compile_command=("${CXX:-c++}" -O2 -std=c++17
    -I "$producer_sdk_root/include" "$producer_source_dir/producer.cpp"
    -L "$producer_sdk_root/lib" "-Wl,-rpath,$producer_sdk_root/lib"
    -lmcruntime -o "$1")
printf 'Host-only compile:'
printf ' %q' "${producer_compile_command[@]}"
printf '\n'
"${producer_compile_command[@]}"
