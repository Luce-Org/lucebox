#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test_binary=$(mktemp /tmp/lucebox-memory-test.XXXXXX)
trap 'rm -f "$test_binary"' EXIT
${CXX:-g++} -std=c++17 -Wall -Wextra -Werror -I server/src/common server/test/test_memory_admission.cpp -o "$test_binary"
"$test_binary"
