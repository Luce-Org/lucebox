# DS4V 80–85 GB candidate

The requested target is an 80–85 GB model and at least 35 tokens/s for short chat, while retaining image input and the previously qualified 131072-token context capacity. **The 83.62 GB conversion is complete, but the candidate has failed the quality gate and reached only 20.1 tokens/s median. It is not qualified for daily service.** The original c76 service has been restored and verified at `http://127.0.0.1:8016/v1` on soulf; neither reduced model nor experimental runtime was installed.

Conversion on `soulf` started 2026-09-06 at 05:07:59 UTC and completed successfully in 1 hour 41 minutes, including wrapper verification. The completed file is **83,619,648,416 bytes**, SHA256 `954433dcb2e64ce6082f4ea8c1478428198fd81b6e7ec0729e9eefa3d56f8497`. Full conversion receipts, completed-header checks and raw trial evidence are stored beside this document.

## Candidate and conversion evidence

- Verified complete file: **83,619,648,416 bytes / 83.619648416 decimal GB**. Original file: 113,745,874,400 bytes. This saves 30.126225984 GB, approximately 26.5% of the original file size.
- Expert gate/up: IQ2_XXS; expert down: IQ2_XS. Selected dense matrices: Q8_0. Vision, aligner, routing, normalization and related control tensors retain their source representations.
- Conversion reads the original FP4/FP8/BF16 safetensors directly. It does not requantize the older MIX GGUF.
- Importance calibration is explicitly **transferred text calibration**, not DS4V-specific calibration. The smaller candidate needs actual regression and image tests; byte correctness alone does not establish model quality.
- Source commit: `7e851cb81937fa92081d1b2e988af82b07d72575`. Converter SHA256: `afa56b0a60e7f883091ed669f8bad01439f9789fa7835184f736563075f84533`.
- Small conversion pilots were byte-identical across serial/parallel execution. The 17-expert pilot also verified the partial final worker batch with 8 and 16 workers.
- Full job limits: 16 CPU workers, 6 GiB cgroup memory maximum, no cgroup swap, six-hour runtime maximum. No GPU benchmarks run during conversion.
- Completed-file header verification passed: all 11 tokenizer keys and 33 architecture keys are identical; all 1641 tensor names/dimensions match. The 129 expert tensors and 346 selected dense tensors follow the new quantization policy; 1166 tensors preserve their type and encoded byte count. The parser's 23 CPU tests passed on soulf. Header validation and the separate full checksum do not establish model quality.

Remote final target: `/home/marcelorm/ds4v-work/DeepSeek-V4-Flash-Vision-IQ85-v1.gguf`.
Remote conversion evidence: `/home/marcelorm/ds4v-work/image-integration/quant85-full-v1/`.

## Frozen comparison and runtime sequence

The existing model scored **15/16 for answer content and 7/16 for strict JSON formatting** on the fixed small task set. Raw responses are in `quant85-quality-baseline-v1/`; the content-scoring rubric was frozen before candidate inference. The candidate gate requires retaining every baseline-correct answer and at least the baseline strict-format score. This is a narrow regression check, not a broad quality benchmark.

Three completed trials used the same candidate and unchanged c76 runtime:

| Decoding | Expert cap | Actual hot experts/layer | Text tokens per sample | Median text decode | Median image decode | Content / strict score |
|---|---:|---:|---|---:|---:|---|
| AR | 1024 MiB | 3 | 243 / 245 / 261 | 14.7 t/s | 14.9 t/s | 12/16 / 5/16 |
| Fused AR | 1024 MiB | 3 | 239 / 251 / 263 | 20.1 t/s | 20.3 t/s | 12/16 / 5/16 |
| Fused AR | 4096 MiB | 14 | 267 / 247 / 253 | 20.1 t/s | 20.4 t/s | 12/16 / 5/16 |

All 18 original functional cases passed in each trial, including image ordering and follow-ups. The appended quality gate then failed: `python-copy` and `nested-json` answered incorrectly, and `python-slice` gave the correct array inside an unrequested Markdown fence. The existing failing `python-loop` case remained wrong. No rubric was relaxed. Cache-eviction, SSE and long-context stages were **not run**, because the functional quality gate stopped each trial first.

Each owned model exited cleanly with no safety breach, OOM or global swap-out growth recorded by the supervisor. Each post-stop idle-memory recovery timed out; the separately authorized bounded TTM cleanup and fresh admission were required between trials. These failed overall reports remain intact. No candidate was installed.

The second candidate restores only the original BF16 embedding and output matrices while retaining the other 1639 IQ85 payloads byte-for-byte. Assembly and independent verification completed: **84,612,519,168 bytes**, SHA256 `99a2260c862e270fa654a1f1e75fad88ec824c58963a2c30135ba91edaf9bb2b`. Its first fused-AR trial passed arithmetic and color-image ordering/follow-ups, then failed the corn/carrot response-format assertion: correct labels appeared inside prose and a Markdown fence. This preserved failure stopped the trial before sustained throughput and the 16-task quality comparison. No sustained speed or quality improvement is established for this variant. The failed trial exited without a safety breach; bounded idle TTM recovery was again needed and completed separately.

A separately versioned diagnostic probe captured benchmark and quality evidence without converting that failed functional result into a pass. The IOBF16 AR diagnostic completed with **19.9 tokens/s median text decode** (252/240/256 output tokens) and **20.1 tokens/s median image decode** (182/143/139 output tokens). Its content/strict scores remained **12/16 and 5/16**, with the same three baseline regressions. Restoring the two BF16 matrices therefore showed no quality or speed benefit on these checks. Raw evidence is in `quant85-candidate-iobf16-diagnostic-ar-v1/`. This new diagnostic uses identical fixed requests across its forthcoming AR/reference/batched comparisons; it is not paired with the older randomized benchmarks.

The existing 10.65 GB draft completed its controlled reference diagnostic capture: text median **8.3 tokens/s**, image median **20.1 tokens/s**, and the same **12/16 content, 5/16 strict** scores. All 22 visible answers and their reported completion counts matched the paired AR run on identical requests (`quant85-iobf16-ar-reference-visible-comparison.json`). This is visible-text agreement, not token-ID parity: the HTTP API does not expose token IDs. No speculative runtime qualification is established. The target plus draft would exceed 85 GB of combined model storage; the target model itself remains below 85 GB. Batched verification completed at **14.0 tokens/s text**, **20.0 tokens/s image**, and **13/16 content, 6/16 strict**. Its text answers differ from the paired AR/reference lane, so this is not a verified equivalent speedup.

A timing-enabled control reproduced all 22 visible answers and completion counts at the same 14.0 tokens/s text median. Increasing graph cache slots from two to four reduced measured graph build time from approximately 23–29 ms to 10–11 ms per verification step, while compute remained approximately 104–107 ms. Text throughput rose to **16.8 tokens/s**, still below AR and the 35 tokens/s target. All 16 quality answers matched the two-slot lane, but all three benchmark text answers changed. See `quant85-iobf16-cache-visible-comparison.json`; numerical or token-ID parity is not established.

An isolated scoped Q4 MMVQ diagnostic was built on soulf at source commit `7fa6ad3ee6892c9b60faabba8159a252c6aac704`. It reproduced the old release binary and original graph object before compiling the patch; CPU unit tests and nine preparer tests passed. The existing source, build and release were unchanged. Its guarded GPU diagnostic completed at **15.8 tokens/s text** and **20.3 tokens/s image**, with the same **13/16 content and 6/16 strict** scores. This did not improve performance. Policy activation was logged, but no direct kernel-dispatch trace was captured. The model exited cleanly with no safety breach or observed global swap-out growth; idle TTM recovery was performed separately. This build is not qualified or installed.

A source review found a separate correctness defect in cached speculative attention: preserved ring-row views used construction-time offsets while runtime write indices advanced. The isolated fix replaces those views with a gather driven by refreshed indices. Existing CPU units and ten preparer tests passed. Real guarded cache2/cache4 tests now produced **identical visible text and completion counts for all 22 identical requests**, compared with 19/22 before the fix. This verifies removal of the observed cache-size-dependent divergence on this set, not broad numerical parity. The fixed cache2/cache4 text medians were **14.1/16.2 tokens/s**; both retained the failing **13/16 content, 6/16 strict** scores. Only 16/22 visible replies match the earlier AR-equivalent reference lane. See `fused-preserved-ring-offset.patch`, its evidence notes, and `quant85-iobf16-ring-cache2-cache4-visible-comparison.json`. The fix remains isolated and is not installed in daily service.

**No tested 80–85 GB candidate met the 35 tokens/s and quality requirements.** Candidate cache-eviction, SSE and long-context qualification were not run after the failed quality gate. The original c76 service is restored and verified; the smaller files and all failure evidence are retained.

A future candidate must retain quality and meet the short-chat speed target, then complete cache-eviction, SSE, 8K/32K/64K/124K text/image qualification and actual daily-service acceptance. Every model trial retains strict admission and the original service fallback.

The fixed cache sequence is 2K, 4K, 8K, 16K, 2K in one server process; it is separate from short throughput medians and from 124K qualification. The validation wrapper and pinned configuration generator passed CPU tests on soulf; these tests do not establish GPU or model behavior.

See [PREPARATION.md](PREPARATION.md) for exact unchanged guard requirements and [runtime-performance-review.md](runtime-performance-review.md) for source-backed trial settings. [draft-compatibility.md](draft-compatibility.md) assesses an optional speculative draft, which was exercised only in unqualified candidate diagnostics and adds approximately 10.65 GB of model storage. Current image requests do not use that speculative path.

Reducing file size alone does not establish 35 tokens/s. Report sustained generated-token counts, actual decode timings, image and context results, and memory observations before adopting the candidate as the daily service.

## Final original-service restoration

Restoration passed on 2026-09-06 after the final diagnostic exited and bounded idle cleanup reached the existing startup threshold. Exact runtime, source/config pins, arguments, environment, both GPU devices, namespace isolation, loopback listener and process identity passed the existing acceptance checker before and after workloads. The service remains active as PID `201518`, invocation `896aae1d96fb4f958323470b9d1e4508`, with zero restarts.

All **14 text/image functional cases** and **four edge cases** passed. A fresh oversized request was rejected with HTTP 400; real SSE cancellation correlated with `finish=client_disconnect` in the same service invocation, followed by a successful request in **0.83 seconds**. The restored service generated **269 text tokens at 13.8 tokens/s** and **251 image-response tokens at 13.7 tokens/s** in this acceptance run. These are single samples, not paired medians against the smaller-model benchmarks.

The service advertises 131072 total context tokens and 4096 default output tokens. The prior 124K text/image qualification for this unchanged c76 runtime and original model remains the relevant long-context evidence; it was not repeated during restoration. Final snapshots show zero owned-process VmSwap, OOM kills and kernel taint. The system-wide swap-out counter grew by 26,681,344 bytes during restoration; that is not attributed to this process and is not reported as zero.

Full restoration evidence and hash-bound summary: `quant85-original-restoration-v1/acceptance.json`. The original model remains 113.75 GB. The size target was achieved experimentally at 83.62 GB, but **35 tokens/s with retained quality was not achieved**.
