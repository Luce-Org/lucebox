#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
exec uv run --frozen --extra dev python -m harness.qualification.qwen36.qualify "$@"
