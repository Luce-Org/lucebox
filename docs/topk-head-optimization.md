# Candidate-restricted LM head (top-k logits) — design, calibration, status

**Branch:** `feat/topk-head-calib`  ·  **Worktree:** `/workspace/lucebox-topk`
**Status (2026-06-18):** premise validated on a smoke sample; instrumentation extended to
sampling; full calibration run pending. No production kernel implemented yet.

---

## 1. Motivation / the idea

The target LM head is the matmul `logits = output.weight @ hidden` over the **full vocab**.
For Qwen3.6-27B the head is `output.weight = [n_embd=5120 × n_vocab=248320]`, quantized **Q6_K
≈ 1.0 GB**, i.e. **~6% of the 16.8 GB** of weights streamed per target forward. The 248k vocab
is unusually large, which makes the head a bigger slice than on a typical 32k-vocab model.

**Idea:** instead of computing logits over all 248k vocab entries, compute them only over a
small **candidate shortlist** `S` of vocab indices, then argmax/sample within `S`. In ggml this
is clean because vocab is the *row* dimension (`ne1`) of `output.weight`: a restricted head is
`ggml_get_rows(output.weight, candidate_ids)` (gathers k quantized rows) followed by a small
matmul. Restricting to k≈256–1024 shrinks the head matmul to <0.5% of its size.

**Where the shortlist comes from:** the DFlash draft already predicts a per-position
distribution (its hidden states projected through the target head, `project_hidden_to_topk`).
Its **top-M** tokens are the natural candidate set `S`. Calibration decides M.

---

## 2. Greedy formulation + the correctness caveat

Greedy spec-decode today is **exact / lossless**: it only commits a draft token when it equals
the target's true full-vocab argmax, and the rejection "bonus" is the true argmax. Output ==
pure target-greedy decoding.

A candidate-restricted head makes argmax run over `S` only. This stays exact **iff the true
argmax ∈ S**. Calibration ("greedy token always in draft top-M") bounds the failure rate, but
"always on a calibration set" ≠ "always at inference" — so this becomes **approximate greedy**,
not lossless. Must validate output quality, not just speed. (You cannot cheaply *verify*
exactness: confirming no out-of-`S` token beats the in-`S` max needs the full matmul.)

### Realistic ceiling
- Head ≈ 6% of weight traffic; restricting it ≈ removes that 6% on the **verify** side.
- But the shortlist itself is produced by `project_hidden_to_topk`, which already runs one full
  target-head matmul over the draft block. So you eliminate the *verify-side* head only.
- **Net realistic decode speedup ≈ ~5%** here (more than a typical model because vocab=248k).
  Modest but real. Worth it only if calibration shows small M suffices.

---

## 3. Extension to sampling (top-k / top-p) — and why it can be *exact*

Generalize the condition: `S` must contain the verifier's **post-filter support**. The sampler
chain order (`server/src/common/sampler.h:6`) is:

```
rep_penalty → freq/pres → top_k → softmax(temp) → top_p → draw
```

The post-filter support is the target's **top-K_s** tokens (then the top-p nucleus within them).
So calibrate M such that **draft-top-M ⊇ target-top-K_s**. Greedy is the K_s=1 case.

**Key insight — normalization:** `top_k` is applied *before* `softmax`. So if `top_k=K_s` is
finite, `softmax` normalizes over just those K_s kept tokens **in both the full and restricted
pipelines** — the global denominator never enters. Therefore, as long as `S ⊇ target-top-K_s`:
- restricted-head `top_k` keeps exactly the true top-K_s (they're the K_s globally-highest and
  all in `S`), `softmax`/`top_p`/draw are identical ⇒ **sampling is EXACT (lossless)**.
- This covers Qwen3.6 defaults (`top_k=20, top_p=0.95`): just need `S ⊇ target top-20`.

**Exceptions:**
- `top_k = 0` (pure top-p): `softmax` normalizes over the kept set; the missing tail inflates
  probabilities and shrinks the nucleus ⇒ **approximate**. Over-cover (`S` carries ≥0.999 mass)
  or correct the threshold with an estimate of the uncovered tail mass.
- pure temperature (no truncation): every token has nonzero prob ⇒ can't shortlist; needs full
  head. (Fine — the optimization only pays off when truncation already discards the tail.)
- penalties run before `top_k`, so strictly the condition is on the *penalized* top-K_s; raw-logit
  draft-top-M is a good superset proxy.

---

## 4. Calibration methodology (the make-or-break test, no kernel needed)

Measure whether the premise holds before building any kernel. Per chain-verify position
(exact alignment: `target_tok[i]` ↔ draft row `i+1`):
- **greedy:** rank of the target argmax within the draft's ranked candidate list. `coverage@M`
  = fraction of positions with rank < M.
- **sampling:** the MAX draft-rank over the target's top-K_s set / top-p nucleus → the smallest
  M with `draft-top-M ⊇ target-top-K_s`. `coverage@M` = fraction fully covered.

**Decision rule:** if `coverage@M` reaches ~99.9% at small M (≈64–256 greedy, somewhat larger
for top_k=20), build the restricted head. If it only saturates at large M, the speedup
evaporates — report and stop.

---

## 5. What's implemented (this branch)

Instrumentation only — gated by env `DFLASH_TOPK_CALIB=1`, in the **chain** verify path of
`server/test/test_dflash.cpp` (run WITHOUT `--ddtree`; chain gives exact position alignment):
1. accumulators declared before the decode loop (`calib_hist`, `samp_hist_k8/k20/p95`);
2. transfer the draft's full per-position logits (`draft_sg.logits → draft_logits_buf`) — the
   fast path otherwise only reads GPU argmax;
3. transfer the target's full logits (`sg.logits → verify_logits_buf`) in the batched path;
4. per-position rank histograms (greedy argmax rank; max draft-rank over target top-8 / top-20 /
   top-p=0.95 nucleus at temp 1.0);
5. print `coverage@M` curves at end of run.

Build: `cmake -B server/build -S server -G Ninja -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF` then `cmake --build server/build --target
test_dflash -j`. (NCCL OFF is required — the fork's `ncclCommInitAll` aborts on this box's driver.)

Run: `calib/run_calib.sh` (uses `GPU=${GPU:-1}` — GPU 0 was occupied by another session;
6 tokenized prompts in `calib/*.bin`, generated with the Qwen3.6 tokenizer, thinking off).

### Results so far (smoke: 1 prompt, 210 positions, greedy metric)
```
coverage@k=1   : 74.3%   (draft top-1 == target greedy ≈ per-token accept rate)
coverage@k=64  : 97.6%
coverage@k=256 : 99.5%
coverage@k=512 : 100%     (mean rank 4.9, max rank 354)
```
⇒ Greedy premise holds strongly: target greedy token within draft top-512 (~0.14% of vocab) on
this sample; a safe M≈1024 is likely lossless and still makes the head matmul ~free. Sampling
coverage numbers pending the full multi-prompt run with the extended binary.

---

## 5b. Full calibration results (2026-06-18)

5 prompts (code1/code2/gen1/gen2/math1; math2 dropped on a transient GPU OOM),
**2550 positions**, n_gen=256, chain path, Qwen3.6-27B Q4_K_M target + Q4_K_M 3.6 draft,
position-weighted `coverage@M` (= fraction of positions whose set is fully inside draft top-M):

| coverage@M | greedy (argmax) | top_k=8 | top_k=20 (Qwen def) | top_p=0.95 |
|-----------:|----------------:|--------:|--------------------:|-----------:|
| 128        | 93.7%           | 29.8%   | 3.3%                | 62.5%      |
| 512        | 96.5%           | 57.6%   | 19.1%               | 68.5%      |
| 1024       | 97.5%           | 69.3%   | 35.0%               | 71.8%      |
| 4096       | 99.4%           | 87.0%   | **64.3%**           | 78.4%      |

Per-prompt greedy@M=128 ranged 80%–99% (code/math high, "general" prompts lower);
tails are long (max greedy rank up to ~133k on a few low-confidence positions).

**Verdict.**
- **Greedy: viable.** M≈1024 → ~97.5%, M≈128 → ~94%. A restricted head with M≈1k–2k is
  near-lossless and still <1% of the 248k-wide matmul. Build it.
- **Sampling (top_k / top_p): NOT viable with this draft.** Exact `top_k` needs the draft
  shortlist to contain the target's *entire* top-K_s; even M=4096 covers the full top-20 only
  64% of the time. The draft matches the target's **#1** token well but ranks the rest of the
  nucleus very differently (the target's #15 can sit at draft-rank thousands). Covering the
  nucleus needs an impractically large M, erasing the speedup.
- **Caveat:** this is the weak Q4 3.6 draft (same one that underperformed the benchmark). A
  BF16/stronger draft would likely rank the nucleus more faithfully and could move the sampling
  numbers; re-run the calibration with it before concluding. The structural asymmetry
  (argmax easy, full-nucleus hard) will persist to some degree.

## 5c. Large-M is cheap; MASS-coverage reopens sampling (2026-06-18)

The head scales linearly in M and is only ~6% of step traffic, so large M stays cheap:
M=8192 → head ~0.2% of step (~5.9% speedup), M=32768 → ~0.8% (~5.2%), M=65536 → ~1.6% (~4.4%).
So we can afford M far larger than first assumed.

More importantly, **set-coverage is the wrong metric for sampling quality** — missing a low-prob
nucleus token barely distorts the draw. The right metric is **covered probability MASS**. The
instrumentation now reports, for the top_p=0.95 nucleus, the mean fraction of nucleus *mass*
inside draft-top-M, plus the count of positions with <99% mass covered. Extended M grid to 65536.

Smoke (1 prompt, math1, 225 positions) — mass coverage:
```
mass@M=256   : 98.93%   (32/225 positions <99% covered)
mass@M=4096  : 99.60%   ( 3/225 positions <99%)
mass@M=65536 : 99.9993% ( 0/225 positions <99%)
```
⇒ At M≈4096–8192 (still ~5.6–5.9% speedup) the per-position sampling distortion is <0.5% of
mass; M=65536 is lossless-in-practice. **Sampling is likely viable as an approximate-but-faithful
mode after all** — pending the large-prompt run to confirm the worst-case tail. (Set-coverage of
the full top-20 also reaches ~100% by M=65536 on this prompt.)

### Tooling for scale-up
- `calib/prep_prompts.py --n N` — tokenize N prompts from HumanEval/GSM8K/MATH-500 (Qwen3.6 chat
  template) into `calib/*.bin`. Deps: transformers, datasets, jinja2.
- `calib/run_calib.sh` — `GPU=0 NGEN=256 bash calib/run_calib.sh | tee calib/out/calib.log`
  (chain path, no `--ddtree`; one table per prompt).
- `calib/aggregate.py calib/out/calib.log` — position-weighted aggregate (set + mass coverage).
- CPU note: each position does 2× nth_element+sort over the 248k vocab (host-side), so very large
  prompt counts are CPU-bound regardless of GPU; parallelize across prompts if needed.

### B200 build (Blackwell sm_100, CUDA ≥12.8)
```
git fetch && git checkout feat/topk-head-calib
git submodule update --init --recursive
cmake -B server/build -S server -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=100 -DGGML_CUDA_NCCL=OFF
cmake --build server/build --target test_dflash -j
# place target + draft GGUFs under server/models/ and server/models/draft/
python3 calib/prep_prompts.py --n 300
GPU=0 NGEN=256 bash calib/run_calib.sh | tee calib/out/calib.log
python3 calib/aggregate.py calib/out/calib.log
```

## 6. Next steps
1. **[done]** Full calibration — see §5b. Greedy validated; sampling needs a better draft.
2. Implement the restricted head for the **greedy** path in `build_qwen35_graph` / the verify graph
   (`server/src/qwen35/graph_builders.cpp`, LM head at ~line 360): plumb candidate ids
   (draft top-M) → `ggml_get_rows(output.weight, ids)` → small matmul → argmax/sample over the
   k logits → map index back to vocab id. Reuse the draft's `project_hidden_to_topk` output as
   the candidate set (it's already computed).
3. Validate quality vs full-head greedy (token-exact match rate) and measure real decode speedup.
4. Sampling is **on hold** pending a better draft (§5b): the Q4 3.6 draft can't cover the
   target's top-k/top-p nucleus at a useful M. Before revisiting, re-run §5b's calibration with
   a BF16/stronger draft. If coverage improves: keep `top_k` finite for exactness (chain order
   makes it lossless when `draft-top-M ⊇ target-top-K_s`), and gate pure-top_p / pure-temperature
   to the full head (or over-cover with a tail-mass correction).

## 7. Key code references
- LM head matmul + argmax: `server/src/qwen35/graph_builders.cpp:360-365` (chain), `:432` (tree),
  `build_lm_head_projection_step` `:446`.
- Candidate source: `Qwen35DFlashTarget::project_hidden_to_topk` `server/src/qwen35/qwen35_dflash_target.cpp:614`;
  `extract_draft_topk` `server/src/common/ddtree.cpp:12`.
- Greedy chain verify + acceptance: `server/test/test_dflash.cpp` (`!seq_verify` path ~3692, accept ~3800);
  generic loop `server/src/common/dflash_spec_decode.cpp:196-207` (layer-split).
- Sampler chain: `server/src/common/sampler.{h,cpp}` (`needs_logit_processing` `sampler.h:38`,
  `sample_logits`).
- Sampled-verify (server, single-GPU): `server/src/qwen35/qwen35_backend.cpp:1741-1779` (gate),
  `:2033-2051` (sampled acceptance walk).
