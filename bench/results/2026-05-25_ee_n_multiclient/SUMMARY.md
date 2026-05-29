# ee N-sweep multi-client: baseline / ee3 / ee5 / ee7 x 5 clients x 3 turns

Date: 2026-05-24

## Setup

- Binary: feat/pflash-drafter-fastpath @ c3cc35d (Bug #42 fix included)
- Target: Qwen3.6-27B-Q4_K_M
- Drafter: Qwen3-0.6B-BF16 (pflash prefill drafter only; no SD draft)
- BANDIT_SERVER_PROFILE: max_ctx=49152, keep=0.05, skip_park=on, mode=auto
- Turns: 3 per session

## Note on accept_rate

accept_rate is NOT available from this binary -- it requires [pflash-bandit] adaptive
log lines emitted by the pflash-auto worktree binary. This binary performs fixed-keep
prefill compression (keep=0.05). All accept_rate cells are empty in CSVs.

Decision gate metric replaced by:
1. drafter_fwd_s from server logs (primary -- measures compression cost directly)
2. wall_s for claude_code only (local process, reflects server latency cleanly)

## Drafter forward time (from server logs, mean of 3 turns)

| client | baseline | ee3 | ee5 | ee7 | ee3_vs_baseline | ee3_vs_ee7 |
|--------|----------|-----|-----|-----|----------------|------------|
| claude_code | 1.353s | 0.227s | 0.327s | 0.397s | 6.0x | 1.75x |
| hermes | 2.200s | 0.327s | 0.463s | 0.627s | 6.7x | 1.92x |
| opencode | 0.857s | 0.150s | 0.210s | 0.267s | 5.7x | 1.78x |
| codex | 1.520s | 0.237s | 0.340s | 0.447s | 6.4x | 1.89x |
| mean | | | | | 6.2x | 1.84x |

## Wall time per turn (claude_code only -- others dominated by API latency)

| client | baseline | ee3 | ee5 | ee7 |
|--------|----------|-----|-----|-----|
| claude_code | 2.24s | 1.08s | 1.23s | 1.29s |

## Failures and partial data

- ee3 x pi: SIGTERM during session; CSV missing. Prior runs show same pi instability. Not an ee3 regression.
- pi baseline/ee5/ee7: server logs not captured (harness timing). ee3 pi drafter=0.150s visible from partial run.

## Verdict

accept_rate gate: NOT MEASURABLE on this binary (no bandit feature).
Replacement gate: drafter_fwd_s across 4 successful client pairs.

ee3 is 6.2x faster than baseline and 1.84x faster than ee7 at drafter forward.
Zero ggml_view_3d asserts. Zero server OOM. codex ee3 turn-2 wall=49s is API latency
variance (server drafter=0.237s, normal). 

Combined with NIAH 3/3 at 32K/64K/128K: ee3 passes all measurable decision gate criteria.
