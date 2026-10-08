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
probe_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_sdk_root="${MACA_PATH:-/opt/maca}"
# Host compilation only: the retained native ELF is supplied at execution time.
module_compile_command=("${CXX:-c++}" -O2 -std=c++17
    -I "$module_sdk_root/include" "$probe_source_dir/repro.cpp"
    -L "$module_sdk_root/lib" "-Wl,-rpath,$module_sdk_root/lib"
    -lmcruntime -o "$1")
printf 'Host-only compile:'
printf ' %q' "${module_compile_command[@]}"
printf '\n'
"${module_compile_command[@]}"
