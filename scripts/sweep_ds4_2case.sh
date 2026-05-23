#!/usr/bin/env bash
# scripts/sweep_ds4_2case.sh
#
# Local 5-config ds4-eval sweep on the two hard cases (1 GPQA Diamond,
# 2 SuperGPQA) used as the rebench corpus while bragi/sindri tune
# thinking-budget + KV-quant + DFlash knobs. Designed to run on any
# host with `dflash_server` built and the standard Qwen3.6-27B target +
# GGUF DFlash draft in dflash/models/.
#
# Output: one tuning-snapshot pair (.txt + .json) per config under
# dflash/docs/tuning-snapshots/<host>-rtx*-qwen36-<date>-postmerge-<cfg>.{txt,json}
# Each pair is auto-committed and pushed to the current branch's
# remote. The repo-tracked snapshot makes cross-host diffs trivial.
#
# Env overrides:
#   LUCEBOX_HOST           default: $(hostname -s)
#   LUCEBOX_GPU            default: rtx3090ti  (free-form; for snapshot filename)
#   LUCEBOX_DATE_TAG       default: $(date +%Y-%m-%d)-postmerge
#   LUCEBOX_TARGET_GGUF    default: dflash/models/Qwen3.6-27B-Q4_K_M.gguf
#   LUCEBOX_DRAFT_GGUF     default: dflash/models/draft/dflash-draft-3.6-q8_0.gguf
#   LUCEBOX_DFLASH_BIN     default: dflash/build/dflash_server
#   LUCEBOX_PORT           default: 1236
#   LUCEBOX_MAX_CTX        default: 98304
#   LUCEBOX_DDTREE_BUDGET  default: 22
#   LUCEBOX_NO_PUSH=1      skip the git push (commit locally only)
#   LUCEBOX_NO_COMMIT=1    skip both commit + push (just write files)
#
# Quitting:
#   Ctrl-C kills the current child server cleanly; partially-completed
#   configs leave a half-snapshot but no commit (atomic per config).

set -uo pipefail

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT"

HOST=${LUCEBOX_HOST:-$(hostname -s)}
GPU=${LUCEBOX_GPU:-rtx3090ti}
DATE_TAG=${LUCEBOX_DATE_TAG:-$(date +%Y-%m-%d)-postmerge}
TARGET=${LUCEBOX_TARGET_GGUF:-dflash/models/Qwen3.6-27B-Q4_K_M.gguf}
DRAFT=${LUCEBOX_DRAFT_GGUF:-dflash/models/draft/dflash-draft-3.6-q8_0.gguf}
BIN=${LUCEBOX_DFLASH_BIN:-dflash/build/dflash_server}
PORT=${LUCEBOX_PORT:-1236}
MAX_CTX=${LUCEBOX_MAX_CTX:-98304}
BUDGET=${LUCEBOX_DDTREE_BUDGET:-22}
SNAP_DIR=dflash/docs/tuning-snapshots

# Sanity
[ -f "$TARGET" ] || { echo "FATAL: target GGUF missing: $TARGET" >&2; exit 1; }
[ -f "$DRAFT"  ] || { echo "FATAL: draft GGUF missing: $DRAFT"  >&2; exit 1; }
[ -x "$BIN"    ] || { echo "FATAL: dflash_server binary missing: $BIN — run cmake --build dflash/build --target dflash_server" >&2; exit 1; }

mkdir -p "$SNAP_DIR"
WORK=$(mktemp -d -t lucebox-sweep.XXXX)
echo "work=$WORK"
echo "host=$HOST gpu=$GPU date_tag=$DATE_TAG"

# Reap stale dflash_server
pkill -9 -f "dflash_server.*--port $PORT" 2>/dev/null
sleep 2

run_one() {
  local NAME=$1 CTK=$2 CTV=$3
  shift 3
  local DRAFT_ARGS=("$@")
  local OUT="$WORK/$NAME"
  mkdir -p "$OUT"
  local SERVER_LOG="$OUT/server.log"
  local SPID

  echo "=== $NAME (ctk=$CTK ctv=$CTV draft_args=${DRAFT_ARGS[*]:-<none>}) at $(date '+%H:%M:%S') ==="

  nohup env DFLASH27B_DRAFT_SWA=2048 \
    "$BIN" "$TARGET" \
    --host 127.0.0.1 --port "$PORT" \
    --max-ctx "$MAX_CTX" \
    --prefix-cache-slots 0 \
    --think-max-tokens 15488 \
    --cache-type-k "$CTK" --cache-type-v "$CTV" \
    "${DRAFT_ARGS[@]}" \
    > "$SERVER_LOG" 2>&1 &
  SPID=$!
  trap "kill -9 $SPID 2>/dev/null" RETURN
  disown $SPID 2>/dev/null || true

  echo "  waiting for /health (pid=$SPID)..."
  local START=$(date +%s) ELAPSED=0
  while ! curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    kill -0 $SPID 2>/dev/null || { echo "  SERVER DIED during boot"; tail -30 "$SERVER_LOG"; return 1; }
    sleep 3
    ELAPSED=$(( $(date +%s) - START ))
    if [ "$ELAPSED" -gt 240 ]; then
      echo "  TIMEOUT waiting for /health"
      kill -9 $SPID 2>/dev/null
      return 1
    fi
  done
  echo "  /health up after ${ELAPSED}s"

  for c in 1 2; do
    if [ "$c" = "1" ]; then ID="recNu3MXkvWUzHZr9"; else ID="001b51d76b4d422988f2c11f104a2c6c"; fi
    echo "  case $c: $ID"
    python3 dflash/scripts/bench_http_capability.py \
      --url "http://127.0.0.1:$PORT" \
      --area ds4-eval \
      --case-id "$ID" \
      --model dflash \
      --min-pass-rate 0.0 \
      --json-out "$OUT/case${c}.json" \
      --trace "$OUT/case${c}-trace.txt" \
      --max-tokens 16000 \
      --timeout 1800 \
      --think 2>&1 | tail -3
  done

  kill $SPID 2>/dev/null; sleep 2; kill -9 $SPID 2>/dev/null || true
  trap - RETURN

  # Write snapshot pair
  local SHA=$(git rev-parse --short HEAD)
  local TXT="$SNAP_DIR/${HOST}-${GPU}-qwen36-${DATE_TAG}-${NAME}.txt"
  local JSON="$SNAP_DIR/${HOST}-${GPU}-qwen36-${DATE_TAG}-${NAME}.json"
  HOST="$HOST" CFG="$NAME" SHA="$SHA" CDIR="$OUT" TXT_OUT="$TXT" JSON_OUT="$JSON" python3 - <<'PY'
import json, os, datetime, pathlib, re
cfg = os.environ['CFG']; cdir = os.environ['CDIR']
def load(p):
    try: return json.load(open(p))
    except Exception as e: return {"_error": str(e)}
c1 = load(f"{cdir}/case1.json"); c2 = load(f"{cdir}/case2.json")
try: serverlog = open(f"{cdir}/server.log").read()
except: serverlog = ""
def grep_kv(pat):
    m = re.search(pat, serverlog)
    return m.group(1).strip() if m else None
def row(c):
    if not c.get("rows"): return {}
    r = c["rows"][0]
    keep = ('id','source','given','correct','graded_pass','strict_pass','format_pass',
            'semantic_hint','wall_s','prompt_tokens','completion_tokens','thinking_tokens',
            'content_tokens','close_kind','finish_reason','http_status')
    out = {k: r.get(k) for k in keep}
    out['correct'] = (r.get('correct') or [None])[0]
    out['result'] = 'PASS' if r.get('graded_pass') else 'FAIL'
    return out
r1, r2 = row(c1), row(c2)
data = {
    "schema": "sweep-2case-v1",
    "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    "host": os.environ['HOST'],
    "binary_sha": os.environ['SHA'],
    "config_name": cfg,
    "server_knobs": {
        "cache_type_k": grep_kv(r'cache_type_k\s*=\s*(\S+)'),
        "cache_type_v": grep_kv(r'cache_type_v\s*=\s*(\S+)'),
        "max_ctx":      grep_kv(r'max_ctx\s*=\s*(\d+)'),
        "think_max_tokens": grep_kv(r'think_max_tokens=\s*(\d+)'),
        "ddtree":       grep_kv(r'ddtree\s*=\s*(\S+)'),
        "draft":        grep_kv(r'draft\s*=\s*(\S+)'),
    },
    "cases": {
        "case1.GPQA_Diamond.recNu3MXkvWUzHZr9": r1,
        "case2.SuperGPQA.001b51d76b4d422988f2c11f104a2c6c": r2,
    },
}
pathlib.Path(os.environ['JSON_OUT']).write_text(json.dumps(data, indent=2)+"\n")
lines = [f"# ds4-eval 2-case sweep snapshot — {data['host']} {cfg}",
         f"# schema={data['schema']}", "",
         "[snapshot]",
         f"schema={data['schema']}", f"captured_at={data['captured_at']}",
         f"host={data['host']}", f"binary_sha={data['binary_sha']}",
         f"config_name={data['config_name']}", "", "[server_knobs]"]
for k,v in data['server_knobs'].items(): lines.append(f"{k}={v}")
lines.append("")
for label, r in (("case.1", r1), ("case.2", r2)):
    lines.append(f"[{label}]")
    for k,v in r.items(): lines.append(f"{k}={v}")
    lines.append("")
pathlib.Path(os.environ['TXT_OUT']).write_text("\n".join(lines))
print(f"snapshot: {os.environ['TXT_OUT']}")
PY

  if [ -z "${LUCEBOX_NO_COMMIT:-}" ]; then
    git add "$TXT" "$JSON"
    git commit -m "data(ds4-eval): ${HOST} post-merge sweep — config ${NAME}" 2>&1 | tail -2
    if [ -z "${LUCEBOX_NO_PUSH:-}" ]; then
      local BRANCH=$(git rev-parse --abbrev-ref HEAD)
      local REMOTE=$(git config --get "branch.${BRANCH}.remote" || echo origin)
      # Fetch + rebase before push — multiple hosts may be pushing
      # snapshots on the same branch concurrently.
      git fetch "$REMOTE" "$BRANCH" 2>/dev/null || true
      git rebase "$REMOTE/$BRANCH" 2>&1 | tail -2 || true
      git push "$REMOTE" "$BRANCH" 2>&1 | tail -2 || true
    fi
  fi
  echo "=== $NAME done at $(date '+%H:%M:%S') ==="
}

run_one "A-baseline-rebuild" q4_0 q4_0 --draft "$DRAFT" --ddtree --ddtree-budget "$BUDGET"
run_one "B-kv-mix"           q8_0 q4_0 --draft "$DRAFT" --ddtree --ddtree-budget "$BUDGET"
run_one "C-kv-q8"            q8_0 q8_0 --draft "$DRAFT" --ddtree --ddtree-budget "$BUDGET"
run_one "D-ar-q4kv"          q4_0 q4_0
run_one "E-ar-kv-mix"        q8_0 q4_0
echo "=== SWEEP DONE at $(date '+%Y-%m-%d %H:%M:%S') — work=$WORK ==="
