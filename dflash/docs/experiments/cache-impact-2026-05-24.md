# Cache impact + sampling variance experiment — 2026-05-24

## Question

Does the prefix cache or full-compress cache change benchmark *outputs*,
or only *wall time*? And how much pass-rate variance comes from temp=1.0
sampling alone?

## Hypothesis

- Caches reuse KV state. Given the same prompt, cache HIT produces the
  same logits as a fresh prefill (deterministic).
- With temp=1.0 sampling, the sampler RNG picks different tokens on each
  trial regardless of cache state. So **outputs vary across trials, cache
  or no cache**.
- Cache should reduce **wall time** on repeated prompts (no re-prefill).
- Cache should NOT change **pass rate**, output content distribution, or
  the unique-outputs-per-trials ratio.
- At temp=0.0 (greedy), outputs should be **bit-identical** across trials
  whether cache is on or off. If they're not, we have a cache bug.

## Experimental design

**3 cases × 3 modes × 5 trials = 45 trials.** Wall estimate: ~2.5-4h on
RTX 3090 Ti.

### Cases

| Source | case_id | Difficulty | Why |
|---|---|---|---|
| SuperGPQA | `001b51d76b4d422988f2c11f104a2c6c` | easy | Consistent ~100s PASS; baseline |
| GPQA Diamond | `recNu3MXkvWUzHZr9` | medium | LMC astronaut — borderline at temp=1.0 |
| AIME2025 | `aime2025-02` | hard | Per bragi data: only passes at 65k --think; --no-think misses |

### Modes

| Mode | Temp | Cache | What it measures |
|---|---|---|---|
| A | 1.0 | OFF (`--prefix-cache-slots 0`) | Pure sampling variance baseline |
| B | 1.0 | ON  (`--prefix-cache-slots 32`) | Cache speedup + transparency to outputs |
| C | 0.0 | ON  (`--prefix-cache-slots 32`) | Determinism check — outputs bit-identical |

All modes use `--no-think` (established as the winning config) and
`--max-tokens 16000`. Per-trial output capture:

```json
{
  "case_id":      "...",
  "trial":        3,
  "mode":         "B",
  "wall_ms":      82456,
  "prefill_ms":   1234.5,   // from usage.timings (needs post-#37 binary)
  "decode_ms":    81211.2,
  "decode_tok_per_sec": 41.6,
  "given":        "B",
  "correct":      "B",
  "pass":         true,
  "output_hash":  "sha256:..."   // hash of the reply content for diversity stats
}
```

## Pre-flight: blockers to clear before running

1. **`--temperature` flag in `bench_http_capability.py`** — needed for
   mode C (temp=0.0). Currently bench inherits server defaults. Add the
   flag (one-line argparse + pass-through) before kicking the orchestrator.

2. **Server binary with usage.timings** — current sindri binary (May 23
   build) predates the #37 timings work. Rebuild dflash_server from
   integration tip before the experiment so we have prefill/decode
   timings per trial. (The orchestrator at `/tmp/cache-impact/run.sh`
   does this in its boot helper.)

3. **Port 1236 must be free** — all prior orchestrators (sindri ds4-eval,
   AIME sweep, --think 92 sweep, Gemma/Laguna local runs, forge vs
   sindri) need to have released the port. The cache-impact orchestrator
   polls and waits.

## Mode A → B comparison: what we expect

| Metric | Mode A (cache OFF) | Mode B (cache ON) | If they differ |
|---|---|---|---|
| First-trial wall | full prefill + decode | full prefill + decode (no warm cache yet) | Should match. If they don't, the cache infra has overhead. |
| Trials 2-5 wall | full prefill + decode | ~ decode only | Cache speedup = (A_mean - B_mean) / A_mean |
| Pass rate | sampling-noise floor | should match | If they differ ≥10pp at N=5, cache changes correctness — bug. |
| Output hash diversity | high (5 unique) | high (5 unique) | If B gives identical hashes, cache is freezing logits — bug. |

## Mode C: determinism check

Pass rate at temp=0.0 should be **100% or 0% per case** — greedy is
deterministic. All 5 trials should:
- Land the same `given` answer
- Have identical `output_hash`
- Have near-identical decode tokens (variation only from GPU non-
  determinism, which is usually <1 token).

If trials disagree at temp=0.0, that's a real determinism bug worth
investigating (cuBLAS variance? thread-race in sampler?).

## Orchestrator

Script at `/tmp/cache-impact/run.sh` (created when this experiment is
about to run — placeholder for now). Will poll for port 1236 to be
free, then run modes A → B → C sequentially, snapshotting per-trial
JSON to `dflash/docs/tuning-snapshots/sindri-2026-05-24-cache-impact/`.

## Out of scope

- **Disk prefix cache** (`--disk-cache-dir`). Adds restore-from-disk
  speedup but the experiment is the same shape; defer to a follow-up.
- **PFlash compression effects**. Separate axis.
- **Cross-host validation** (running same experiment on bragi). Would
  triangulate hardware determinism. Defer.

## When this runs

Lowest priority in the current bench queue. Estimated start: after the
RTX 3090 Ti pipeline completes — sindri ds4-eval (running) → AIME budget
sweep → --think 92-case sweep → Gemma/Laguna local runs → forge vs
sindri. Realistically Tuesday-Wednesday EDT.
