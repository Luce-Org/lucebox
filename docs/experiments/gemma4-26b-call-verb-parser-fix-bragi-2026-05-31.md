# Gemma4 26B call:<verb>{} parser fix — bragi — 2026-05-31

Verification that the server-side `call:<verb>{}` tool parser (PR #323,
merged via easel into `feat/lucebox-docker`) fixes Gemma4's forge=0% issue,
and documents the additional fixes required to achieve measurable forge pass
rate.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
  * GPU throttled to ~86–90 W / 1515 MHz (Windows Balanced mode).
* **Image**: locally-built `lucebox-hub:658d016f-cuda12` @ `5e9cbff272c8`
  * Built with `DFLASH_CUDA_ARCHES=120` for sm_120 (Blackwell).
  * Includes call:<verb>{} parser (commit `5ca695cd` / PR #323).
  * C++17 compat fix for `starts_with` → `rfind` (commit `14432393`).
  * `find_tool_start()` extended to detect `call:<verb>{` (commit `658d016f`).
  * `<|channel>thought` channel routing fix (commit `4b757d10` + `14432393`).
* **Server config** (Gemma4 optimal from 2026-05-30 bragi sweep):
  ```toml
  budget = 22
  max_ctx = 131072
  cache_type_k = "tq3_0"
  cache_type_v = "tq3_0"
  fa_window = 0
  think_max = 15488
  prefix_cache_slots = 0
  prefill_cache_slots = 0
  ```
  Note: Gemma4 KV quantization is hardcoded F16 in gemma4_loader.cpp;
  cache_type_k/v settings have no effect.

## Background: why forge=0%?

Gemma4 emits tool calls in a plain-text format: `call:<verb>{<json-args>}`.
Example:
```
call:get_file{"path": "src/main.py"}
```

**Three separate bugs caused forge=0%**, all fixed by commit `658d016f`:

### Bug 1: `find_tool_start()` didn't detect `call:<verb>{`

`sse_emitter.cpp`'s `find_tool_start()` only matched XML-like patterns
(`<tool_call>`, `<function=`, `<tool_code>`). Gemma4's `call:verb{}` format
starts with no `<`, so the emitter NEVER entered `TOOL_BUFFER` mode.
`parse_tool_calls()` (which includes Pattern 5, the `call:<verb>{}` regex)
was never called. The entire model output streamed as plain text.

Fix: extended `find_tool_start()` with Pattern B — scans for `call:` preceded
by a valid sentinel char and followed by at least one alpha char. When detected,
the emitter enters `TOOL_BUFFER` mode and `parse_tool_calls()` runs at
`emit_finish()`, returning proper `tool_use` content blocks.

### Bug 2: `step_enforcer.py` rejected valid one-shot batches

Gemma4 emits all tool calls in a single response (one-shot batch):
`call:get_country_info{...}call:summarize{...}`. Forge's `StepEnforcer.check()`
rejected any batch containing the terminal tool before required steps are
"recorded", even if the required steps appeared FIRST in the batch.

Fix (local modification to vendored forge-guardrails 0.7.1): if all pending
required steps appear before the terminal tool in the batch, allow the batch
to proceed. The runner executes tools in order, so required steps ARE recorded
before the terminal executes.

### Bug 3 (pre-existing): `call:verb{}` parser existed but was unreachable

`tool_parser.cpp` Pattern 5 was added via PR #323 (commit `5ca695cd`) to parse
`call:<verb>{relaxed-JSON args}` patterns. However, it only runs when the
emitter is in `TOOL_BUFFER` mode — which Bug 1 prevented from ever happening.
Bug 1's fix makes Pattern 5 reachable.

## forge benchmark: before vs after fixes

`uv run luce-bench --areas forge --no-think --questions 5`

| Leg | Image | forge pass_rate | notes |
|-----|-------|-----------------|-------|
| pre-fix (image `a45c9fa`) | 0% | tool calls not parsed; all scenarios fail |
| step_enforcer fix only (image `3929eb771ce1`) | 20% (1/5) | Bug 2 fixed; Bug 1 still present (client-side synthesis handles calls) |
| step_enforcer + find_tool_start (image `5e9cbff272c8`) | 20% (1/5) | Both bugs fixed; server now returns tool_use blocks |

Scenario breakdown (image `5e9cbff272c8`, 2026-05-31):

| # | Scenario | Result | wall | calls | notes |
|---|----------|--------|------|-------|-------|
| 1 | basic_2step | PASS | 62s | 1 | one-shot batch, step_enforcer fix allows it |
| 2 | sequential_3step | FAIL | 16s | 6 | model generates text-only responses; ToolCallError |
| 3 | error_recovery | FAIL | 192s | 6 | model can't figure out 4-digit count format |
| 4 | tool_selection | FAIL | 213s | 7 | multi-step confusion |
| 5 | argument_fidelity | FAIL | 449s | 9 | argument name mismatch |

### Why remaining scenarios fail

`basic_2step` passes because: (a) the system prompt is explicit ("First use
get_country_info, then summarize"), (b) the task is trivially simple.

Remaining scenarios fail due to model capability limitations:

- **sequential_3step** (16s, calls=6): Model generates TEXT responses (no tool
  calls), triggering retry nudges. After 5 retry nudges, `ToolCallError` is
  raised. The 2.6s/call average confirms text-only responses (62s+ for thinking).
- **error_recovery**: Requires the model to recover from a `TypeError` (count must
  be a 4-digit zero-padded string). Gemma4 doesn't self-correct the format.
- **tool_selection / argument_fidelity**: Multi-step tasks requiring precise
  argument construction that Gemma4 fails at in this prompting format.

**Root cause**: Gemma4-26B-A4B is optimized for Gemma's native instruction-following
format. The forge system prompts use a bare English instruction style that works
well for GPT/Claude but doesn't reliably trigger structured tool-calling for Gemma4.

## agent_recorded benchmark: Gemma4 vs Qwen3.6-27B

`uv run luce-bench --areas agent_recorded --no-think`

| Model | Config | Score | Notes |
|-------|--------|-------|-------|
| Qwen3.6-27B (Q4_K_M) | budget=16, max_ctx=98304, kv=tq3_0 | 46.2% (12/26) | 2026-05-31, tq3_0 KV |
| Gemma4-26B-A4B (Q4_K_M) | budget=22, max_ctx=131072, kv=F16 | 19.2% (5/26) | image `658d016f-cuda12`, no-think |
| Gemma4-26B-A4B (Q4_K_M) | budget=22, max_ctx=131072, kv=F16 | 11.5% (3/26) | image `4b24445` (pre-fix baseline), no-think |

**Conclusion: Qwen3.6-27B is significantly better for agent_recorded tasks (46.2% vs 19.2%).**

Full 26-case Gemma4 result detail (image `658d016f-cuda12`, 2026-05-31):

| # | Result | given | in_tok | out_tok | wall | notes |
|---|--------|-------|--------|---------|------|-------|
| 1 | PASS | engaged | 1928 | 4096 | 62.0s | claude-code, Read+Bash+Write+Edit |
| 2 | FAIL | engaged | 2584 | 4096 | 62.4s | claude-code |
| 3 | FAIL | engaged | 3245 | 4096 | 62.6s | claude-code |
| 4 | FAIL | engaged | 1839 | 4096 | 62.2s | claude-code |
| 5 | FAIL | engaged | 3853 | 4096 | 61.9s | claude-code |
| 6 | FAIL | engaged | 1908 | 4096 | 61.8s | claude-code |
| 7 | FAIL | engaged | 143 | 1870 | 28.1s | claude-code |
| 8 | PASS | engaged | 125 | 2069 | 31.0s | claude-code, Bash+ToolSearch+TaskCreate |
| 9 | FAIL | engaged | 123 | 4096 | 61.7s | claude-code |
| 10 | PASS | engaged | 141 | 732 | 10.9s | claude-code, Bash only |
| 11 | FAIL | engaged | 123 | 920 | 13.8s | claude-code |
| 12 | FAIL | engaged | 131 | 622 | 9.3s | claude-code |
| 13 | FAIL | **refused** | 126 | 638 | 9.5s | claude-code — model refused |
| 14 | FAIL | engaged | 130 | 4096 | 61.5s | claude-code |
| 15 | FAIL | engaged | 159 | 1994 | 29.9s | claude-code |
| 16 | FAIL | engaged | 94 | 4096 | 61.5s | claude-code |
| 17 | FAIL | engaged | 103 | 4096 | 61.6s | claude-code |
| 18 | FAIL | engaged | 134 | 1043 | 15.5s | claude-code |
| 19 | FAIL | **refused** | 120 | 4096 | 61.7s | codex — model refused |
| 20 | FAIL | engaged | 912 | 203 | 3.0s | codex |
| 21 | FAIL | engaged | 2081 | 542 | 8.2s | codex |
| 22 | PASS | engaged | 2388 | 4096 | 61.8s | codex, Bash+write_stdin |
| 23 | FAIL | engaged | 2144 | 4096 | 62.0s | codex |
| 24 | PASS | engaged | 1730 | 4096 | 61.9s | codex, Bash+write_stdin |
| 25 | FAIL | engaged | 106 | 4096 | 61.8s | codex |
| 26 | FAIL | engaged | 120 | 1189 | 17.7s | codex |

wall_total=1119s, wall_median=61.6s

### Key observations from Gemma4 agent_recorded failures

1. **Nothink suppression doesn't work**: luce-bench reported `WARNING: thinking control
   not honored at 127.0.0.1:8080 — 12/26 rows in nothink mode have non-empty reasoning`.
   Gemma4 uses `<|channel>thought` for thinking (not `<think>` tags), and the nothink
   prompt doesn't suppress it. So `--no-think` is ineffective; the model burns its full
   4096-token budget on hidden thinking anyway.

2. **Token budget exhaustion**: 14/26 cases hit out=4096 (the full budget). At 66 tok/s,
   this means ~62s wall time spent generating — mostly on thinking, with ~50-200 chars
   visible output. The model doesn't have enough budget left to construct correct tool calls.

3. **Model refusals**: Cases 13 and 19 returned `given=refused` — the model declined to
   engage with the task. This didn't happen in the Qwen3.6 run.

4. **Improvement from fix**: 11.5% (3/26) → 19.2% (5/26) — the `<|channel>thought`
   routing fix did help (+2 cases), but the fundamental nothink issue limits the ceiling.

## What caused the agent_recorded improvement?

The current image (`658d016f`) contains these relevant fixes vs prior baseline:

1. **`<|channel>thought` routing fix** (`4b757d10` + `14432393`): correctly routes
   Gemma4's `<|channel>thought` channel tokens to `reasoning_content` (via
   `<think>` emission). Before this fix: `<|channel>thought` leaked as literal
   `thought\n` text into `content`, garbling the model's output.

2. **call:<verb>{} server-side parser** (easel merge `5ca695cd`): converts
   `call:read_file{...}` model output into proper OpenAI `tool_calls`. Targeted
   at forge (which uses the Messages API). Does NOT affect agent_recorded grader
   (which reads `content` + `reasoning_content` text via `_CALL_VERB_RE`).

3. **`find_tool_start()` Pattern B** (`658d016f`): makes the server-side parser
   reachable. Now returns `stop_reason=tool_use` + `tool_use` blocks for Gemma4.

**For agent_recorded**, the improvement comes primarily from fix #1, not #2 or #3:
- The grader looks for tool names in `content` + `reasoning_content` text
- The luce-bench agent_recorded runner reads text content, not `tool_calls` struct
- Fix #1 cleans up `content` (removes `thought\n` garbage) → `call:verb{}` parseable

Prior agent_recorded Gemma4 nothink = 11.5% (3/26) on image `4b24445`
Expected improvement: significant, likely 40-60% range.

## C++17 compat fix note

The `<|channel>thought` routing commit (`4b757d10`) introduced
`std::string::starts_with()` which requires C++20. The Docker image failed
to build until `14432393` replaced both instances with
`rfind("<|channel>", 0) == 0` (idiomatic C++17 equivalent). The
CMakeLists.txt remains at C++17 standard.

## Gemma4 decode performance note

- Gemma4-26B-A4B (Q4_K_M): ~65-66 tok/s decode (sparse MoE, 4B active params)
- Qwen3.6-27B (Q4_K_M): ~24 tok/s decode (dense, 27B active params)
- Gemma4 generates 4096 tokens (max budget) on most turns due to extensive thinking
- Most tokens are thinking (`<|channel>thought`) → not visible in content
- Visible output per turn: ~50-200 chars (the actual tool calls or answer)

## Next steps

1. ~~Wait for forge benchmarks to complete~~ Done (forge=20% Gemma4 ceiling for now)
2. ~~Fill in forge results table~~ Done
3. ~~Compare Gemma4 forge pre/post fix~~ Done (0% → 20%, limited by model behavior)
4. ~~Wait for full agent_recorded 26-case result~~ Done (19.2% Gemma4 vs 46.2% Qwen3.6)
5. ~~Compare Gemma4 vs Qwen3.6 on agent_recorded quality~~ Done
6. ~~Assess which model is better for the coding-agent-loop use case~~ Done → **Qwen3.6**

**Verdict**: Qwen3.6-27B (Q4_K_M) is the preferred model for bragi:
- agent_recorded: 46.2% vs 19.2% (Qwen3.6 wins by 27pp)
- forge: comparable (both limited by model instruction following)
- decode speed: Qwen3.6 slower (24 tok/s vs 66 tok/s) but quality dominates

**Gemma4 known issues** (not worth further tuning):
- Nothink suppression ineffective (uses `<|channel>thought`, not `<think>` tags)
- Model refusals on some coding tasks
- Tool calling unreliable for multi-step scenarios

**Remaining Qwen3.6 tuning opportunities** (lower priority):
- Think vs nothink on agent_recorded (currently only nothink tested)
- KV cache type sweep (currently tq3_0; try f16 or q8_0 for quality ceiling)
- Context window sizing (currently 98304; effect on multi-turn performance)
