# HF Scorer Deep-Dive — Drafter Models for PFlash Tail-Score

**Date**: 2026-05-22  **Hardware**: RTX 3090 24 GB  **Target main model**: Qwen3.6-27B Q4_K_M (~15 GB)
**Drafter VRAM budget**: ≤ 1.5 GB  **Hard kernel constraint**: head_dim=128 (BSA kernel rejects others, see `project_cross_family_bsa_blocked.md`)
**Baseline drafter**: Qwen3-0.6B BF16 (~1.2 GB), `hidden=1024 n_head=16 n_kv=8 head_dim=128 n_layer=28`

PFlash uses only Q+K projections of the drafter for tail-score (97.86 s / 43.9 % of 128K wall). FFN/V/O/LM-head are dead weight (B_compute=0 % per `drafter_forward_profile_128k.md`). A purpose-built scorer would need only Q+K + a few layers.

## Section 1 — Top-10 candidate table

| Rank | Model | HF link | BF16 size | Architecture (hidden/heads/kv/head_dim/layers) | Tokenizer | Trained for | License | GGUF path |
|---:|---|---|---:|---|---|---|---|---|
| 1 | Qwen3-Reranker-0.6B | huggingface.co/Qwen/Qwen3-Reranker-0.6B | 1.19 GB | 1024/16/8/**128**/28 (qwen3) | qwen2 (vocab 151669) | Cross-encoder relevance | Apache-2.0 | `convert_hf_to_gguf.py` direct |
| 2 | Qwen3-Embedding-0.6B | huggingface.co/Qwen/Qwen3-Embedding-0.6B | 1.19 GB | 1024/16/8/**128**/28 | qwen2 (151669) | Dense retrieval | Apache-2.0 | Prebuilt GGUF Q8 = 639 MB |
| 3 | Qwen3-0.6B-Base | huggingface.co/Qwen/Qwen3-0.6B-Base | 1.19 GB | 1024/16/8/**128**/28 | qwen2 (151936) | Next-token (no SFT) | Apache-2.0 | direct |
| 4 | Qwen3-0.6B (current) | huggingface.co/Qwen/Qwen3-0.6B | 1.19 GB | 1024/16/8/**128**/28 | qwen2 (151936) | Next-token + chat SFT | Apache-2.0 | already loaded |
| 5 | mxbai-rerank-large-v2 | huggingface.co/mixedbread-ai/mxbai-rerank-large-v2 | ~3.1 GB | 1536/12/2/**128**/28 (qwen2) | qwen2 | Cross-encoder | Apache-2.0 | direct; oversized at BF16 |
| 6 | Qwen2.5-0.5B | huggingface.co/Qwen/Qwen2.5-0.5B | 0.99 GB | 896/14/2/64/24 | qwen2 | Next-token | Apache-2.0 | **BSA REJECTS head_dim=64** |
| 7 | Qwen2.5-Coder-0.5B | huggingface.co/Qwen/Qwen2.5-Coder-0.5B | 0.99 GB | 896/14/2/64/24 | qwen2 | Code LM | Apache-2.0 | rejected (head_dim=64) |
| 8 | bge-reranker-v2-gemma | huggingface.co/BAAI/bge-reranker-v2-gemma | ~5 GB | 2048/8/1/256/18 | gemma SP | Cross-encoder | Apache-2.0 | rejected (head_dim=256, 2B, max_pos=8192) |
| 9 | bge-reranker-v2-m3 | huggingface.co/BAAI/bge-reranker-v2-m3 | ~2.2 GB | XLM-Roberta 1024/16/64/24 | XLM-R SP | Cross-encoder | MIT | rejected (BERT, max_pos=8194, head_dim=64) |
| 10 | colbertv2.0 / jina-colbert-v2 | huggingface.co/colbert-ir/colbertv2.0 | 0.45 GB | BERT 768/12/64/12 | wordpiece | MaxSim per-token | MIT | rejected (max_pos=512, head_dim=64) |

Eliminated outright (one-line reason): all BERT-family rerankers (bge-m3, ms-marco-MiniLM, jina v2, ColBERT) have head_dim ∈ {32,64} **and** max_pos ≤ 8194 → unusable above 8K context, and the BSA kernel rejects them. Gemma reranker is head_dim=256 (also wrong) and too big. Mamba/SSM excluded per existing memory. Qwen2.5/SmolLM2 family is loader-ready but kernel-rejected.

## Section 2 — Risk analysis (top-3)

### #1 Qwen3-Reranker-0.6B (ship-it candidate)
- **Architectural compatibility**: identical to Qwen3-0.6B-Base (its declared `base_model`). Q+K projections accessible at every layer; `head_dim=128`; 28 layers.
- **BSA kernel acceptance**: shape-identical to current drafter → kernel passes without modification. Zero CUDA work.
- **Trained as a drafter/scorer?** No published work uses it as a PFlash drafter. But it is trained as a single-document cross-encoder that emits a yes/no relevance token given a (query, doc) pair (see HF README + blog). Its attention pattern is shaped by query-doc relevance loss — exactly the structure PFlash exploits.
- **Tokenizer cost**: vocab=151669 vs Qwen3.6's 151936. Same qwen2 BPE family + qwen2 pre-tokenizer; the 267 extra tokens in Qwen3.6 are reserved/special. Body tokens encode identically → ~0 ms re-tokenization. Sanity test: encode a 32K body with both tokenizers, diff IDs. (One-shot check.)
- **Published as drafter?** No. New territory.

### #2 Qwen3-Embedding-0.6B
- **Architectural compatibility**: same as #1.
- **BSA kernel**: passes.
- **Trained for**: dense retrieval (last-token pooling → embedding). Last-layer attention is shaped by retrieval contrast loss — less directly query-token-attention oriented than the reranker, but still relevance-driven.
- **Tokenizer cost**: same as #1.
- **Bonus**: prebuilt **Q8_0 GGUF (639 MB)** already published by Qwen → instant test path, no `convert_hf_to_gguf.py` round-trip needed.

### #3 Qwen3-0.6B-Base
- Same shape; pretrained checkpoint (no chat SFT). Useful as an ablation control to isolate "did reranker training help?" vs "did SFT hurt scoring?". This is the *scientific* third pick, not a ship candidate.

### Risk shared by all three
- All inherit Qwen3.6 tokenizer parity by construction.
- All ride the existing BSA kernel.
- All ~1.2 GB BF16; Q8_0 ~600 MB, Q4_0 ~360 MB. With Qwen3.6 27B Q4 (~15 GB) + KV cache (~5 GB) on a 24 GB 3090, ≤ 600 MB drafter is comfortable.
- **All three have zero published evidence as PFlash drafters.** This is a 1-day spike, not a months-long bet.

## Section 3 — Ship-it pick: **Qwen3-Reranker-0.6B**

**Why**: highest theoretical fit (trained for query-doc relevance scoring, which is structurally what PFlash tail-score computes), zero kernel work (head_dim=128 + qwen3 arch unchanged), zero loader work (the cross-arch adapter already handles qwen3), tokenizer is byte-identical for body tokens, Apache-2.0.

**Single validation experiment**: NIAH at 32K with `keep=0.20`, anchor_radius default, compare retrieval accuracy and TTFT vs `Qwen3-0.6B-BF16 + ee14` (the current shipping configuration on RTX 3090). Pass gate: ≥ 5/5 needles preserved AND wall-time ≤ current ee14 baseline. Run from `dflash/scripts/test_combined_long_prompt.py` analog.

**Predicted speedup**: zero direct latency win (same layer count, same shapes, same kernel). The bet is on **quality at lower keep ratios** — if the reranker's per-token attention selects more relevant tokens, `keep` can drop from 0.20 → 0.10 without NIAH loss, halving the kept-KV pass downstream and pushing 1.43–2.16× ee14 closer to 2.5–3×. If that bet fails, this experiment is a 1-day write-off; no architectural regret.

**VRAM**: Q8_0 = 639 MB vs current Qwen3-0.6B BF16 ~1.2 GB → ~600 MB savings, frees room for larger KV or longer context.

## Section 4 — Wildcard pick: **mxbai-rerank-large-v2**, distilled to head_dim=128 layer subset

mxbai-rerank-large-v2 is Qwen2.5-1.5B-shape with head_dim=128, n_head=12, n_layer=28. **BF16 3.1 GB is over budget**, but Q4_K_M ≈ 850 MB is in range. Trained as state-of-the-art cross-encoder (mxbai v2 outperforms BGE on BEIR per HF card). Bigger model → better relevance signal per token, potentially allowing `keep ≤ 0.10` with quality intact.

**Bold bet**: a stronger reranker may push `keep` so low that even at 1.5–2× the drafter forward cost, end-to-end TTFT drops because the kept-KV pass on the 27B target shrinks proportionally. Math: at S=128K, target prefill on Q4_K_M is ~80 % of wall; cutting kept tokens 2× saves more than a 0.5 s drafter delta costs. Worth a 1-day spike if Qwen3-Reranker NIAH passes but `keep` plateaus.

**One catch**: GQA ratio is 12:2 (KV groups of 6) vs Qwen3-0.6B's 16:8 (groups of 2). The BSA kernel's allocator may dispatch differently — verify before declaring kernel-clean.

## Section 5 — Genuine unknown

**Has anyone trained a model EXPLICITLY for PFlash-style prefill compression scoring?** Reading the SpecPrefill paper (Liu et al., arXiv 2502.02789, ICML 2025) the speculator is `Llama-3.1-8B-Instruct` BF16 — an off-the-shelf chat model, **not** trained for scoring. GemFilter (Shi et al. 2024), MInference (Jiang et al. 2024), H2O (Zhang et al. 2023), and SwiftKV (Qiao et al. 2024) all use the target model itself or a chopped sub-block, never a purpose-trained scorer. SnapKV / PyramidKV / Quest / FlexPrefill — same pattern. The Cross-Family Speculative Prefill work (SambaNova ICLR 2026) reuses pretrained drafters with cross-tokenizer alignment, no scoring-specific training.

**No published model is trained specifically to "produce per-token K vectors of a body given a query window for max-attention scoring."** Cross-encoders (Qwen3-Reranker, mxbai) are the closest analog — but they emit a *scalar* relevance, not per-token K weights. Embedding models (Qwen3-Embedding) optimize last-token pooling, again not per-token attention weights.

**This is the research opportunity.** A 6-layer head_dim=128 model trained with a "match the 27B target's tail-attention pattern" distillation loss (KL between drafter's Q·K^T and target's Q·K^T over a held-out body, per layer) would be the first purpose-built PFlash scorer. **Months** as a research project: needs distillation infra + a teacher signal extraction pipeline + held-out long-context corpus + ablations on layer count, head_dim, and loss form. **Days** to test if any existing model accidentally works: that's what Section 3 buys you.

---

## Honest engineering verdict

- **Days project**: swap to Qwen3-Reranker-0.6B Q8_0, run one NIAH-32K experiment, ship-or-shelf in 24 h.
- **Months project**: distill a custom scorer — has clear novelty (no prior art), clear training recipe, and clear win condition (lower keep at preserved quality), but it's a paper, not a sprint.

**Recommended next move**: 1-day spike on the ship-it pick. If it lifts quality at keep ≤ 0.10, file the distillation project as the follow-up moat.
