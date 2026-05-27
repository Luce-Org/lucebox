#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

uv lock --check

read -r -a EXTRA_SYNC_ARGS <<< "${UV_WORKSPACE_SYNC_ARGS:---no-install-package torch}"
uv sync --frozen "${EXTRA_SYNC_ARGS[@]}"

uv run --frozen --no-sync python - <<'PY'
import sys

import datasets
import gguf
import huggingface_hub
import jinja2
import numpy
import transformers

print("workspace import OK from repo root")
PY

(
  cd server
  uv run --frozen --no-sync python - <<'PY'
from pathlib import Path
import sys

root_venv = Path("..").resolve() / ".venv"
assert Path(sys.prefix).resolve() == root_venv, (sys.prefix, root_venv)

print("workspace discovery OK from server/")
PY
)
