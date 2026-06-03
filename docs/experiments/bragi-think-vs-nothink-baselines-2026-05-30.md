# Think vs nothink baselines — bragi — 2026-05-30

Full luce-bench sweeps (all areas) for qwen3.6-27b and gemma-4-26b in
both think and nothink modes, using the optimal autotune configs from
the 2026-05-30 coding-agent-loop sweeps.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
  * GPU throttled to ~86–90 W / 1515 MHz (Windows Balanced mode).
    All numbers are ~40–60% of full-performance potential.
* **Image**: locally-built `lucebox-hub:cuda12` @ `4b24445-dirty`
  (DFLASH_CUDA_ARCHES=120)
* **luce-bench**: v0.2.7.dev0

## Optimal configs used

**Qwen3.6-27B:**
```toml
budget = 16
max_ctx = 98304
cache_type_k = "tq3_0"
cache_type_v = "tq3_0"
fa_window = 0
think_max = 15488
```

**Gemma-4-26B:**
```toml
budget = 22
max_ctx = 131072
fa_window = 0
think_max = 15488
```
(KV cache: tq3_0 auto-selected by server; F16 also fits at 131K)

## Results: Qwen3.6-27B

| area | nothink | think | delta |
|------|---------|-------|-------|
| smoke | 100% | 66.7% | −33 pp |
| ds4-eval | 70.7% | **81.5%** | +10.8 pp ✓ |
| gsm8k | **89.0%** | 82.0% | −7 pp |
| truthfulqa-mc1 | 80.0% | **84.0%** | +4 pp ✓ |
| hellaswag | **90.0%** | 73.0% | −17 pp |
| code | **80.0%** | 20.0% | −60 pp ⚠ |
| longctx | 100% | 100% | = |
| agent | **75.0%** | 50.0% | −25 pp |
| agent_recorded | 42.3% | **46.2%** | +3.9 pp ✓ |
| forge | 0% | 0% | = |

**Wall time (ds4-eval):** 12352s (68.1s median) nothink → 37882s (552.1s median) think — **3.1× slower**.

## Results: Gemma-4-26B

| area | nothink | think | delta |
|------|---------|-------|-------|
| smoke | 100% | 100% | = |
| ds4-eval | 77.2% | **81.5%** | +4.3 pp ✓ |
| gsm8k | 91.0% | 91.0% | = |
| truthfulqa-mc1 | **77.0%** | 68.0% | −9 pp |
| hellaswag | **73.0%** | 42.0% | −31 pp ⚠ |
| code | 0% | 0% | = (server bug) |
| longctx | **100%** | 83.3% | −16.7 pp |
| agent | 25.0% | **50.0%** | +25 pp ✓ |
| agent_recorded | 11.5% | **23.1%** | +11.6 pp ✓ |
| forge | 0% | 0% | = |

**Wall time (ds4-eval):** 4225s (15.1s median) nothink → 12834s (153.7s median) think — **3.0× slower**.

## Cross-model comparison (nothink)

| area | qwen nothink | gemma nothink | winner |
|------|-------------|---------------|--------|
| ds4-eval | 70.7% | **77.2%** | gemma +6.5 pp |
| gsm8k | 89.0% | **91.0%** | gemma +2 pp |
| truthfulqa-mc1 | **80.0%** | 77.0% | qwen +3 pp |
| hellaswag | **90.0%** | 73.0% | qwen +17 pp |
| code | **80.0%** | 0% | qwen (gemma bug) |
| longctx | 100% | 100% | = |
| agent | **75.0%** | 25.0% | qwen +50 pp |
| agent_recorded | **42.3%** | 11.5% | qwen +30.8 pp |

**Speed:** Gemma decodes at ~67 tok/s vs qwen's ~24 tok/s (2.8× faster). Gemma's ds4-eval
median wall time is 15.1s vs 68.1s (4.5× faster per case).

## Findings

### 1. Think mode is task-class dependent for both models

**Benefits hard multi-step reasoning, hurts pattern-matching and structured output.**

- ds4-eval (AIME/GPQA/SuperGPQA): qwen +10.8 pp, gemma +4.3 pp — consistent win.
- hellaswag (common-sense completion): qwen −17 pp, gemma −31 pp — consistent loss.
  Hellaswag rewards first-instinct token prediction; extra reasoning overrides it.
- code (HumanEval completion): qwen −60 pp (format breaks), gemma 0% both (server bug).

### 2. Gemma's thinking is emergent, not instruction-gated

Gemma 4 26B reasons in `content` regardless of think/nothink mode. The
`<|channel>thought` channel only controls *where* reasoning appears
(`reasoning_content` vs `content`), not *whether* the model reasons.
Evidence:

- gsm8k: identical 91% in both modes (model always reasons through math)
- ds4-eval nothink median: 15.1s (slow for direct retrieval — model is reasoning
  in content even without the thinking channel)
- Hellaswag collapse (−31 pp): more permissive context (thinking channel) lets
  model reason more freely, hurting fast pattern tasks

This confirms the 2026-05-25 thinking-control characterization experiment:
"thinking budget isn't a knob for Gemma 4 the way it is for Qwen3."

### 3. Qwen think benefits are genuine, not just channel routing

Qwen's ds4-eval jump (+10.8 pp, 70.7% → 81.5%) accompanied a 8× wall-time
increase per case (68s → 552s). Qwen is actually *doing more work* in think mode
— the `</think>` token triggers a genuine wrap-up behavior that produces a cleaner
final answer. Gemma's 3× time increase with only +4.3 pp gain reflects the overhead
of the reasoning channel with less behavioral benefit.

### 4. Gemma is faster but weaker at tool-calling

Gemma's 2.8× decode speed advantage makes it attractive for high-throughput
workloads, but qwen dominates on all agent/tool-calling areas (agent, agent_recorded,
code). Gemma's low agent scores stem from:
- Outputting narrative instead of structured tool calls
- A server-side token leak bug that corrupts code completions

### 5. Gemma code=0% is a fixable server bug

The `<|channel>thought` token (id 100) leaks as `thought\n` text because
`http_server.cpp` checks `raw == "<|channel>"` but the raw vocab string is
`<|channel>thought`. Fix: change to `raw.starts_with("<|channel>")` (lines 1534,
1711). Requires image rebuild. Tracked as follow-up #3 in
`docs/experiments/gemma4-26b-thinking-control-2026-05-25.md`.

## Recommended mode by task

| task | qwen | gemma |
|------|------|-------|
| hard reasoning (AIME, GPQA) | think | think |
| code generation / tool calling | nothink | nothink (but subpar) |
| common-sense / MC | nothink | nothink |
| long-context retrieval | either | nothink |
| agentic coding | nothink | nothink |

For general-purpose use on bragi: **qwen nothink** is the most reliable
default (strong across all areas); switch to think for tasks you know
require deep multi-step reasoning.

## Baseline data

Raw results in luce-bench-baselines repo:
- `bragi-rtx5090laptop-qwen36-27b-autotune-nothink-2026-05-30/`
- `bragi-rtx5090laptop-qwen36-27b-autotune-think-2026-05-30/`
- `bragi-rtx5090laptop-gemma4-26b-autotune-nothink-2026-05-30/`
- `bragi-rtx5090laptop-gemma4-26b-autotune-think-2026-05-31/`
