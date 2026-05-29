# Lookahead-Only Attention for Drafter Prefill Scoring — Design

Status: draft, no code changes. Target: cut drafter forward at S=128K from
~120 s (BSA fast path) to <30 s.

## 1. Mechanism — what the literature actually says

`arXiv:2603.02631` (Upasani et al., SambaNova) is titled *Cross-Family
Speculative Prefill*. Its contribution is **transferability** of the
original SpecPrefill mechanism (Liu et al., `arXiv:2502.02789`) across
model families (Qwen↔Llama↔DeepSeek). Neither paper introduces a new
"lookahead-only attention" **kernel** — both reuse the standard
self-attention forward and compute scores post-hoc.

Reference: `Jingyu6/speculative_prefill`, `vllm_patch/worker/look_ahead_spec_worker.py`:
- L121-138: `for itr in range(look_ahead_cnt): execute_model(request)` —
  drafter runs a full forward, then autoregressively generates
  `look_ahead_cnt` new tokens (8 in `configs/config_p1_full_lah8.yaml`).
- L42: Q of each newly generated token is captured at every layer via a
  monkey-patched `self_attn.forward`.
- L409-411: scoring is built post-hoc as
  `attn = Q_lookahead @ K_prompt.T / sqrt(d)`. K comes from
  `kv_cache[layer_idx][0]`, written during the regular full forward.
- L342-365: aggregation — softmax along key-dim, `avg_pool1d` smoothing
  (kernel=13 in p1_full_lah8), `max` over flattened (layer × head),
  `mean` over the lookahead dim → token importance per position.

Mask (causal, lookahead token `t` attends only to prompt):
```
mask[t, j] = 0  if j < prompt_len + t  else  -INF
```

FlashPrefill (`flashprefill_kernels.cu`) is a different technique:
block-sparse attention with mean-K block scoring + top-K select + sparse
forward. It compresses self-attention; it does not replace it.

## 2. Math

| Variant                       | FLOPs / layer      | At S=128K, N=8 |
|-------------------------------|--------------------|----------------|
| Dense self-attention          | O(S²·D·H)          | ~17 T          |
| BSA (~5% keep)                | O(S²·D·H·0.05)     | ~0.85 T        |
| Lookahead-only Q_tail × K     | O(N·S·D·H)         | ~1.0 G         |

Per-layer FLOPs vs BSA: ~850× cheaper. BSA already runs near peak
tensor-core; cross-attn with N=8 underfills SMs. Realistic wall-clock
gain: **10–40×** at S=128K.

Memory: per-layer `[H,N,S]` bf16 ≈ 64 MB, streamed with max-reduce
(current code pattern). Trivial.

## 3. Codebase landing

Per-layer flow in `dflash/src/qwen3/qwen3_graph.cpp::forward_qwen3_drafter_model`
(L219):
- L457-471: graph A writes Q/K/V into per-layer persistent buffers
  `Q_buf`, `K_curr_v[il]`, `V_curr_v[il]`.
- L474-483: copies Q tail (`S − n_lookahead .. S`) into `Q_last_v[il]`.
- L529-581: calls `flash_prefill_forward_bf16` (BSA/FP). **This is the
  ~120 s line at S=128K**, and it feeds graph B (o_proj + residual + FFN).
- L788-854: post-forward tail-score graph **already** does the
  lookahead-only scoring: `Q_tail @ K_score.T`, softmax, max over heads,
  write into `running_max[N,S]`. (Same logic exists in
  `qwen3_drafter.cpp` L382-450 for the Qwen3.5 path.)

So the **scoring head already exists**. The win is in skipping the body
attention + FFN entirely. But layer `l`'s K depends on layer `l−1`'s
hidden state, which depends on `l−1`'s attention output + FFN. Skipping
body attention changes the model semantically. That is the binding
constraint.

### Options

**A. New `lookahead_cross_attn` kernel in `flashprefill_kernels.cu`.**
Pure `Q[N×H×D] · K[S×Hk×D]^T` with causal mask + softmax + max-over-H,
emitting `score_max[N,S]` f32 directly. Replaces the tail-score graph
(qwen3_graph.cpp L788-854). The tail-score is already a small fraction
of total time. **Wall-clock gain at S=128K: <5 %.** Wrong lever.
LOC ~250. Risk: low.

**B. Modify BSA mask to keep only `Q_tail × K_body` blocks.**
Truncating the Q dimension at the kernel collapses attention output at
non-tail positions to zero → next layer's hidden state is wrong → next
layer's K is wrong. Same fatal correctness flaw as C below, but with
extra mask-config complexity. Not viable without distillation.

**C. K-only fast path: skip body attention + FFN; reuse existing tail
scoring.** If `DFLASH_DRAFTER_KONLY=1`, gate out the
`flash_prefill_forward_bf16` call and the entire graph B (o_proj +
residual + FFN), and propagate `h_in` to the next layer's `hidden_buf`
unchanged (or after a cheap RMSNorm). Each layer still computes its
own K via the standard projection from `h_in`; the existing tail-score
graph at L788-854 consumes those K's. The drafter becomes a
"K-projection stack" — semantically different from a normal transformer.
**No paper validates this.** LOC ~80 in `qwen3_graph.cpp`. Could net
10–40× wall-clock if quality holds. Quality is the open question.

### Headline recommendation

**Option C, env-flagged, with a hard NIAH quality gate before promoting.**
Option A is the wrong lever; B is not viable.

## 4. Composability with pflash

- Replaces BSA on the drafter path only. Target untouched.
- Scoring head unchanged: same `running_max[N,S]`, same
  `mean_t(running_max[t,:])` reduction in `qwen3_drafter.cpp` L470-475,
  same avg_pool smoothing.
- Chunk(128) + alpha selection in `qwen3_drafter.cpp` L477-622 works
  unchanged.
- **No custom drafter weights required, in principle** — BF16 GGUF loads
  normally. But running those weights without inter-layer attention is
  off-distribution.

## 5. Risks

1. **No published evidence for a no-self-attention drafter.** Both
   2502.02789 and 2603.02631 assume a full forward in the drafter.
   Option C is our own architectural extrapolation.
2. **NIAH may collapse.** Needle correlation in attention scores comes
   in part from earlier layers' attention mixing. Strip that and the
   scores may flatten into "embedding similarity to Q_tail" — useful
   for some semantic-retrieval tasks, useless for exact-token needles.
3. **Framing conflation:** the prompt's "lookahead-only attention"
   bundles (a) the cheap Q_tail × K_body scoring (already implemented)
   with (b) skipping body self-attention (the real cost saver, semantically
   dangerous). They are independent levers.
4. **n_lookahead in our code is 8, not 96.** At N=96 the cross-attention
   is still sub-second; the cost is the body forward.

## 6. Falsifiable validation

Single experiment, env-gated:
- Bench S=128K, qwen3-0.6b drafter, time `forward_qwen3_drafter_model`.
- Baseline: BSA path (~120 s).
- Target: K-only path (<30 s).
- Quality gate: NIAH 32K must keep 5/5 needles at `keep_ratio=0.1`. If
  below 5/5, **abort — Option C falsified.**
- If 32K holds, re-test at 128K before promoting.

## 7. Effort

- ~80 LOC in `qwen3_graph.cpp` (gate body attn + FFN, copy `h_in` →
  `hidden_buf`), ~20 LOC env flag/log, ~30 LOC test wiring. **~150 LOC,
  1–2 days kernel/bench + 1 day quality ablation.**
- Blocker: the quality ablation. If it fails, only Option A's <5 % gain
  on tail-score remains, not worth a kernel.

## Open question — investigate before kernel work

**Where does the 120 s actually come from?** The per-layer timing block
at qwen3_graph.cpp L858-861 already reports
`A_setup/A_alloc/A_compute/FP/B_*/tail-score`. Run one S=128K trace and
attribute the 120 s. If graph B FFN dominates, Option C only buys back
the FP portion and the headline 10–40× shrinks toward 2–5×.
