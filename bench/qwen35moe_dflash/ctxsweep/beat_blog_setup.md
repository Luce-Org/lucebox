# Reproduce-and-Beat: dFlash 27B Blog Numbers (setup)

Target: Qwen3.6-27B-Q4_K_M + dflash-draft-3.6 on RTX 3090. No GPU runs done here — this is the located harness + config + ready commands. All flag names are verbatim with file:line.

## Models (confirmed on disk)
- Target: `/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf` — EXISTS
- Draft (pick one in `/home/peppi/models/qwen3.6-27b-dflash/`):
  - `dflash-draft-3.6-bf16-reconverted.gguf` (default / highest accept)
  - `dflash-draft-3.6-q8_0.gguf`
  - `dflash-draft-3.6-q4_k_m.gguf`
- Server binary: `/home/peppi/Dev/lucebox-hub/server/build/dflash_server`
- Chat templates (both EXIST): `/home/peppi/models/qwen3.6-27b-chat-template.jinja`, `/home/peppi/models/qwen3-coder-chat-template.jinja`

## Verbatim flags (file:line where parsed)
- `--ddtree` enables tree-verify → `bargs.ddtree_mode=true` + `fast_rollback=true` — `server/src/server/server_main.cpp:431-432`. Default OFF (`backend_factory.h:53`).
- `--ddtree-budget <N>` → `bargs.ddtree_budget` — `server/src/server/server_main.cpp:434-435`. Default 22 (`backend_factory.h:54`). So budget=22 is `--ddtree --ddtree-budget 22` (or just `--ddtree`, since 22 is default).
- `--draft <path>` — `server_main.cpp:354`
- `--max-tokens <N>` (server default gen length; per-request `max_tokens` in JSON overrides) — `server_main.cpp:364`
- `--fa-window <N>` — `server_main.cpp:421`
- `--cache-type-k <t>` / `--cache-type-v <t>` (use `q4_0`) — `server_main.cpp:610-613`
- `--lazy-draft` — `server_main.cpp:557`
- KV-quant env (legacy equivalents): `DFLASH27B_KV_Q4` forces both K,V to q4_0 — `server/src/kv_quant.cpp:168`; per-tensor `DFLASH27B_KV_K`/`DFLASH27B_KV_V` — `kv_quant.cpp:176,184`. The `--cache-type-k/v q4_0` flags are the preferred way; `DFLASH27B_KV_Q4=1` is the env shorthand the blog cited.
- `DFLASH_FEAT_RING_CAP=<N>` = rolling target_feat ring-buffer capacity (the blog's "rolling 4096-slot target_feat buffer"; bench scripts set it to max_ctx) — `server/src/qwen35/qwen35_target_graph.cpp:199-202` and `qwen35_backend.cpp:374`.
- `DFLASH_DRAFT_CTX_MAX=<N>` caps drafter self-attn window for long ctx — `qwen35_backend.cpp:2158-2160` (default 2048).

## 1. HumanEval reproduce/beat 129.52 tok/s (DDTree budget 22, n_gen=256, greedy)

GAP — see "Gaps" below. The HumanEval harness `server/scripts/quality_humaneval_plus.py` does NOT pass `--ddtree`, hardcodes `MAX_TOKENS=512` (line 78), and does not measure decode tok/s. It cannot reproduce 129.52 as-is.

Closest working decode-tok/s + ddtree path = launch the dflash server WITH `--ddtree`, then drive it. Server launch (budget 22, CUDA-graph N/A see Gaps):

```bash
/home/peppi/Dev/lucebox-hub/server/build/dflash_server \
  /home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf \
  --draft /home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf \
  --host 127.0.0.1 --port 18081 \
  --max-ctx 8192 --max-tokens 256 \
  --ddtree --ddtree-budget 22 \
  --fa-window 0 \
  --chat-template-file /home/peppi/models/qwen3.6-27b-chat-template.jinja \
  --model-name luce-dflash-27b --lazy-draft
```

Then drive HumanEval with per-request `max_tokens=256`, temperature 0 (greedy). To use the existing harness you must (a) set `PFLASH_TARGET` + `PFLASH_DRAFT` env to the paths above, (b) patch `MAX_TOKENS=256` and add `--ddtree --ddtree-budget 22` into a CONFIGS flags entry in `server/scripts/quality_humaneval_plus.py`, and (c) parse decode tok/s from the server log (the harness currently only times wall-clock). Dataset: `/home/peppi/Dev/lucebox-hub/dflash/eval/humanevalplus.jsonl` (humaneval_plus dir).

Verified-working ddtree wiring to copy from: `harness/clients/common.sh:150-153` (`ddtree_args=(--ddtree --ddtree-budget "$BUDGET")`).

## 2. 128K-decode beat 134.78 tok/s (Q4_0 KV + rolling target_feat)

Template = `bench/qwen35moe_dflash/ctxsweep/run_dense27b_rebaseline.py` (server arg array, lines 69-83) / `launch_arm.sh`. For 128K set `--max-ctx 131072` and `DFLASH_FEAT_RING_CAP` to match:

```bash
DFLASH_FEAT_RING_CAP=131072 DFLASH_DRAFT_CTX_MAX=2048 \
/home/peppi/Dev/lucebox-hub/server/build/dflash_server \
  /home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf \
  --draft /home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf \
  --host 127.0.0.1 --port 18081 \
  --max-ctx 131072 --max-tokens 200 \
  --ddtree --ddtree-budget 22 \
  --fa-window 0 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --chat-template-file /home/peppi/models/qwen3.6-27b-chat-template.jinja \
  --model-name luce-dflash-27b --lazy-draft
```

Drive via `bench/qwen35moe_dflash/ctxsweep/run_dense27b_rebaseline.py` (POSTs to :18081, parses `prefill=Xs decode=Ys(Ztok/s)` from server log). The blog's `DFLASH27B_KV_Q4=1` ≡ `--cache-type-k q4_0 --cache-type-v q4_0`. The "rolling 4096-slot buffer" wording = `DFLASH_FEAT_RING_CAP` (blog ran a 4096 cap; bench scripts default it to max_ctx — set 4096 to match blog exactly, or larger to beat it).

## 3. HumanEval DDTree driver (ready to run)

Script: `bench/qwen35moe_dflash/ctxsweep/run_humaneval_ddtree.py`

Prompt source: REAL HumanEval+ dataset (`server/eval/humaneval_plus/humanevalplus.jsonl`, 164 tasks). Driver uses first 10 (HumanEval/0-9). NOT an embedded stand-in.

Usage — launch server manually first, redirect output to a log, then run driver:

```bash
# Terminal 1: launch server (redirect logs for metric parsing)
DFLASH_FEAT_RING_CAP=4096 \
  flock /tmp/lucebox_gpu.lock \
  server/build/dflash_server \
    /home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf \
    --draft /home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf \
    --host 127.0.0.1 --port 18081 \
    --max-ctx 8192 --max-tokens 256 \
    --ddtree --ddtree-budget 22 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --chat-template-file /home/peppi/models/qwen3.6-27b-chat-template.jinja \
    --model-name luce-dflash-27b --lazy-draft \
    > /tmp/dflash_he_ddtree.log 2>&1 &

# Terminal 2: run bench (wait for server to be healthy first)
python3 bench/qwen35moe_dflash/ctxsweep/run_humaneval_ddtree.py \
    --server-log /tmp/dflash_he_ddtree.log

# Or all-in-one (script handles server launch + kill):
python3 bench/qwen35moe_dflash/ctxsweep/run_humaneval_ddtree.py --run-server \
    --server-log /tmp/dflash_he_ddtree.log
```

Results saved to: `bench/qwen35moe_dflash/ctxsweep/humaneval_ddtree_results.json`
Beat target: mean decode_tps > 129.52 AND mean AL >= 8.31

## 4. 128K decode beat 134.78 tok/s (ready to run — DO NOT RUN YET)

One-liner launch (beat blog 134.78 tok/s / AL 8.33 at 128K):

```bash
DFLASH_FEAT_RING_CAP=4096 DFLASH_DRAFT_CTX_MAX=2048 \
  flock /tmp/lucebox_gpu.lock \
  server/build/dflash_server \
    /home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf \
    --draft /home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf \
    --host 127.0.0.1 --port 18081 \
    --max-ctx 131072 --max-tokens 256 \
    --ddtree --ddtree-budget 22 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --chat-template-file /home/peppi/models/qwen3.6-27b-chat-template.jinja \
    --model-name luce-dflash-27b --lazy-draft \
    > /tmp/dflash_128k_ddtree.log 2>&1 &
```

Then drive with an existing 128K context prompt (e.g. `bench/qwen35moe_dflash/ctxsweep/ctx_131072.json` if generated, or adapt `run_dense27b_rebaseline.py` with `MAX_CTX=131072`). Parse `decode=Ys(Ztok/s)` and `avg_commit=C` from `/tmp/dflash_128k_ddtree.log`.

Notes:
- `DFLASH_FEAT_RING_CAP=4096` matches the blog's "rolling 4096-slot buffer" (NOT max_ctx-sized ring).
- `DFLASH_DRAFT_CTX_MAX=2048` caps drafter self-attn at 2048 for long-ctx speed.
- `DFLASH27B_KV_Q4=1` env is equivalent to `--cache-type-k q4_0 --cache-type-v q4_0`; explicit flags preferred.
- Beat threshold: decode_tps > 134.78 AND AL > 8.33.

## 5. GAPS (uncertain / MISSING)
- **CUDA graphs flag: MISSING.** No runtime `--cuda-graph` / `CUDA_GRAPH` / `GGML_CUDA_GRAPH` in the http server. Only compile-time refs in `server/test/test_dflash.cpp`. The blog's "~1.6x lever that was OFF" has NO runtime toggle in this tree — would need to be wired (build-time or new flag). Cannot enable from a run command today.
- **HumanEval decode-tok/s: the harness can't produce 129.52 as-shipped.** `quality_humaneval_plus.py` hardcodes MAX_TOKENS=512, lacks `--ddtree`, and measures only wall-clock — not decode tok/s. Needs 3 edits (above) or build a small decode-tok/s driver that reads the server log like `run_ctxsweep.py` does. The blog number most plausibly came from a ddtree-enabled launch driven by a log-parsing bench, NOT this quality script.
- **128K AL/peak metrics** (AL 8.33, peak) are not emitted by `run_dense27b_rebaseline.py` (it parses decode tok/s only); acceptance-length reporting would need the server to log AL or a parser addition.
- **n_gen=256 vs 512:** blog used n_gen=256; harness default 512 — must patch.
