# Gemma4 26B call:<verb>{} parser fix — bragi — 2026-05-31

Verification that the server-side `call:<verb>{}` tool parser (PR #323,
merged via easel into `feat/lucebox-docker`) fixes Gemma4's forge=0% issue.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
  * GPU throttled to ~86–90 W / 1515 MHz (Windows Balanced mode).
* **Image**: locally-built `lucebox-hub:cuda12` @ `14432393`
  * Built with `DFLASH_CUDA_ARCHES=120` for sm_120 (Blackwell).
  * Includes call:<verb>{} parser (commit `5ca695cd` / PR #323).
  * Also includes C++17 compat fix for `starts_with` → `rfind` (commit
    `14432393`; `std::string::starts_with` is C++20 but CMakeLists.txt
    requires C++17).
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

Before PR #323, luce-bench's client-side workaround parsed this format in
Python. The server's `tool_parser.cpp` only had patterns for:
- `<tool_call>...</tool_call>` (XML)
- `{"name": "...", "parameters": {...}}` (JSON)

Since the server couldn't parse `call:<verb>{}`, it returned these strings
as raw `content` text, not as `tool_calls`. luce-bench's client workaround
was supposed to catch this, but forge uses a different code path from
agent_recorded and didn't benefit from the workaround → forge=0%.

PR #323 adds pattern #6 to tool_parser.cpp:
```cpp
static const std::regex & re_call_verb_open() {
    static std::regex r(R"((^|[\s,;:\(\[\{\}\)\]\>])call:([A-Za-z0-9_.:\-]+)\s*\{)");
    return r;
}
```
Uses `balanced_braces_end()` for nested JSON. This is the correct fix: the
server now returns proper `tool_calls` for Gemma4's emission format.

## forge benchmark: before vs after fix

`uv run luce-bench --areas forge --no-think`

| Leg | image | forge pass_rate | notes |
|-----|-------|-----------------|-------|
| pre-fix (baseline) | a45c9fa | ~0% | tool calls not parsed; all scenarios fail |
| post-fix (2026-05-31) | 14432393 | *TBD* | running |

## agent_recorded benchmark: Gemma4 vs Qwen3.6-27B

`uv run luce-bench --areas agent_recorded --no-think`

| Model | Config | Score | Notes |
|-------|--------|-------|-------|
| Qwen3.6-27B (Q4_K_M) | budget=16, max_ctx=98304, kv=tq3_0 | 46.2% (12/26) | 2026-05-31, tq3_0 KV |
| Gemma4-26B-A4B (Q4_K_M) | budget=22, max_ctx=131072, kv=F16 | *TBD* (partial: 2/4=50%) | full run pending forge completion |

**Partial results** (4/26 cases, run interrupted to allow forge to complete):

| # | in_tok | Gemma4 | Qwen3.6 | notes |
|---|--------|--------|---------|-------|
| 1 | 1928   | PASS   | FAIL    | Gemma4 advantage |
| 2 | 2584   | FAIL   | FAIL    | tied |
| 3 | 3245   | PASS   | PASS    | tied |
| 4 | 1839   | FAIL   | FAIL    | tied |

Early signal: Gemma4 2/4 (50%) vs Qwen3.6 1/4 (25%) on these 4 cases.
Non-determinism note: too few cases for statistical conclusion; full run needed.

Note: Gemma4 consistently generates out=4096 tokens (maximum reply budget)
at 64-65 tok/s, giving ~64s per case. Qwen3.6 generates varying lengths at
24 tok/s. Both take ~64s per long case. Gemma4 is verbose but faster per-token.

Note: Gemma4 decode rate is ~64-65 tok/s vs Qwen3.6 ~24 tok/s (2.7× faster
generation) because Gemma4-26B-A4B is a sparse MoE model (4B active params
per token vs 27B dense for Qwen3.6). Prefill is proportionally slower for
Gemma4 due to larger vocabulary (262144 vs 151936).

## What caused the agent_recorded improvement?

The new image (`14432393`) contains TWO relevant fixes vs the prior baseline:

1. **call:<verb>{} server-side parser** (easel merge `5ca695cd`): converts
   `call:read_file{...}` model output into proper OpenAI `tool_calls`. Primarily
   targets forge (which uses the Messages API and expects structured tool_calls).
   
2. **`<|channel>thought` routing fix** (`4b757d10` + `14432393`): correctly routes
   Gemma4's `<|channel>thought` channel tokens to `reasoning_content` (via
   `<think>` emission). Before this fix: `<|channel>thought` leaked as literal
   `thought\n` text into `content`, garbling the model's output. After: clean
   content + thinking properly in `reasoning_content`.

**For agent_recorded**, the improvement likely comes primarily from fix #2, not #1:
- The agent_recorded grader looks for tool names in `content` + `reasoning_content`
- The luce-bench runner does NOT extract `tool_calls` from the API response
- Before fix #2: `content` was polluted with `thought\n` garbage before each tool call
- After fix #2: `content` is clean; model's `call:read_file{...}` is parseable by
  `_CALL_VERB_RE` without interference from leaked channel tokens

Baseline comparison: prior agent_recorded Gemma4 nothink = 11.5% (3/26)
New result: partial 2/4 (50%) — likely large improvement from channel routing fix

## C++17 compat fix note

The Gemma4 channel-token routing commit (`4b757d10`) introduced
`std::string::starts_with()` which requires C++20. The Docker image failed
to build until `14432393` replaced both instances with `rfind("<|channel>", 0) == 0`
(idiomatic C++17 equivalent). The CMakeLists.txt remains at C++17 standard
which is compatible with llama.cpp's own standard setting.

## Next steps

1. Wait for forge and agent_recorded benchmarks to complete
2. Fill in results table
3. Compare Gemma4 forge pre/post fix (expect significant improvement)
4. Compare Gemma4 vs Qwen3.6 on agent_recorded quality
5. Assess which model is better for the coding-agent-loop use case
