#!/usr/bin/env bash
set -euo pipefail

source_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cpu_sparse_tool_executor.cpp"
json_include="${JSON_INCLUDE:-}"
output="${1:-$(dirname "$source_file")/cpu_sparse_tool_executor}"

if [[ -z "$json_include" ]]; then
    printf 'JSON_INCLUDE must point to the directory containing nlohmann/json.hpp\n' >&2
    exit 2
fi
if [[ ! -f "$json_include/nlohmann/json.hpp" ]]; then
    printf 'missing JSON header: %s/nlohmann/json.hpp\n' "$json_include" >&2
    exit 2
fi

g++ -std=c++17 -O3 -DNDEBUG -pthread \
    -I"$json_include" "$source_file" -o "$output"
