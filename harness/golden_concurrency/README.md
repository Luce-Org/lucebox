# Golden Concurrency Gate (`harness/golden_concurrency`)

Single-source-of-truth golden concurrency pack and stdlib-only gate runner (`gate.py`) shared between `lucebox` engine CI and `luce_box` Manage/release validation.

All checked-in fixtures are **100% public-safe synthetic data** (zero private sessions or proprietary benchmarks).

## 4-Tier Concurrency Pyramid

| Tier | Scope | Budget | Trigger |
| :--- | :--- | :--- | :--- |
| **Tier 0 (CPU DST + Static Contracts)** | Pure-C++ deterministic simulation (`test_concurrency_dst.cpp` in `check-cpu`) + launcher static checks | `< 30 s` | Every PR (no GPU) |
| **Tier 1 (`tier1_pr_smoke`)** | Live `luce_server` at `C=3, 4, 10` (clamped only by explicit model `max_supported_concurrency`): canary isolation, 2-turn prefix reuse, tool-call edge cases, mid-prefill/mid-decode socket resets, oversubscription, oversized `400` probe | `< 3 min` | Engine / launcher PRs on `lucebox3` |
| **Tier 2 (`tier2_nightly_agentic`)** | Multi-turn public OSS coding trajectories (`C=1, 4, 10`), `default_spec_prefix` + `kvflash_pflash_ssd` modes, 25% chaos aborts, 1.5x oversubscription | `20-40 min` | Nightly on `main` |
| **Tier 3 (`tier3_release_soak`)** | 3-model flagship release matrix (`qwen3.8-27b`, `qwen3.8-flash-next`, `inkling-2-mini`) launched via customer Manage launchers, 4-tier memory HWM audit | `2-4 hr` | Release candidate / `EXPECTED_COMMIT` bump |

## The 7 Mandatory Invariants

1. **I1 — Isolation & Zero Cross-Talk**: Every lane carries a unique `CANARY-<lane>-<hex>` token; asserts `own_canary_recall >= threshold` AND `sibling_canary_mentions == 0` across both `content` and `reasoning_content`.
2. **I2 — Serial-vs-Concurrent Equivalence**: Compares `C=1` (or baseline scorecard) against `C=3/4/10` recall; fails if accuracy drops beyond `serial_vs_concurrent_accuracy_drop_max_pp`.
3. **I3 — Quality Floor**: Enforces exact NIAH recall and `100%` valid structured JSON tool calls (including hyphenated parameters like `-i` and nested JSON schemas).
4. **I4 — Prefix Cache Hit-Rate Floor**: Replays 2-turn coding sessions (`P1` -> `P2` with `>= 90%` shared prefix) and enforces `cached_tokens / shared_prefix >= threshold` and `TTFT(P2) / TTFT(P1) <= ratio_max`.
5. **I5 — Graceful Backpressure**: Runs an oversubscription burst (`1.25x`/`1.5x` configured concurrency) + an oversized prompt probe; requires `0` HTTP 5xx responses and HTTP `400` on the oversized prompt.
6. **I6 — Cancellation & Disconnect Safety**: Injects abrupt TCP closes mid-prefill and mid-decode while verifying surviving lanes achieve `100%` recall and `/props` reports `0` leaked slots and `0` leaked KV pool tokens after wave drain.
7. **I7 — Spec-Decode / Metrics Coexistence**: Parses `usage.timings` and `[concurrency-metrics]` server log records when speculative decoding is enabled, verifying coverage and `mean_accepted_len >= spec_mean_accepted_min`.

## Pack Layout (`v1/`)

- [`v1/pack.json`](v1/pack.json): Hardware profiles (`r9700_32g`, `strix_halo_98g`), model capability contracts (`max_supported_concurrency`, SLO floors), tier matrices, and public OSS trajectory contract. Uses JSON so `gate.py` requires zero third-party packages.
- [`v1/workloads/canary_niah_4k.jsonl`](v1/workloads/canary_niah_4k.jsonl): 12 distinct lanes with unique per-lane `CANARY-L<nn>-<hex>` tokens embedded in synthetic code context.
- [`v1/workloads/agentic_2turn_smoke.jsonl`](v1/workloads/agentic_2turn_smoke.jsonl): 12 distinct 2-turn coding-agent sessions (`P1` repo context -> `P2` follow-up sharing `>= 90%` prefix and expecting a valid JSON tool call).
- [`v1/workloads/tool_calls_golden.jsonl`](v1/workloads/tool_calls_golden.jsonl): Edge-case tool schemas including `-i` flag parameters, nested JSON objects/arrays, and strict type checks.

## CLI Usage

```bash
# 1) Attach to an already-running luce_server
python3 harness/golden_concurrency/gate.py \
  --pack harness/golden_concurrency/v1/pack.json \
  --tier tier1_pr_smoke \
  --model-profile qwen3.8-27b \
  --base-url http://127.0.0.1:8080/v1 \
  --output-dir artifacts/tier1-attach

# 2) Engine PR CI in lucebox (boots and manages luce_server)
python3 harness/golden_concurrency/gate.py \
  --pack harness/golden_concurrency/v1/pack.json \
  --tier tier1_pr_smoke \
  --model-profile qwen3.8-27b \
  --server-bin ./server/build/luce_server \
  --model-path /opt/models/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft-path /opt/models/Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --output-dir artifacts/tier1-qwen38

# 3) Customer Manage / Release validation in luce_box (boots actual launcher script)
python3 harness/golden_concurrency/gate.py \
  --pack harness/golden_concurrency/v1/pack.json \
  --tier tier3_release_soak \
  --model-profile qwen3.8-27b \
  --launcher ./lucebox_dashboard/launchers/run-qwen38-server.sh \
  --baseline-scorecard release-baselines/qwen3.8-27b.json \
  --output-dir artifacts/release-qwen38
```

## Deterministic Exit Codes

| Exit Code | Meaning |
| :--- | :--- |
| `0` | **PASS** — All 7 invariants, quality floors, memory ceilings, and SLO thresholds met. |
| `1` | **INVARIANT_FAIL** — Cross-talk leak, recall/tool-call quality drop, HTTP 5xx / non-400 oversized response, slot/KV leak, or missing spec-decode metrics. |
| `2` | **SLO_OR_MEM_FAIL** — Correctness passed, but prefix cache hit rate, P2/P1 TTFT ratio, goodput/latency SLO, spec acceptance floor, or VRAM/UMA/RAM/SSD HWM breached threshold. |
| `3` | **INFRA_INVALID** — Invalid pack/profile, server failed to boot/become healthy, missing launcher/binary, or infrastructure error. |

## Public OSS Tier-2 Trajectory Policy

Tier 2 nightly agentic runs must use **100% public-safe open-source trajectories** (e.g., Terminal-Bench 4.0 and Hugging Face SWE-PolyBench / OpenHands Apache-2.0 / MIT tasks) extracted into deterministic P1+P2 JSONL form. Never commit private customer sessions or proprietary logs into `harness/golden_concurrency/`.

## Running Self-Tests (CPU-Only, `< 3s`)

```bash
python3 -m unittest discover -s harness/golden_concurrency -v
```
