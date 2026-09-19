# Recommended Unsloth baseline reproduction

Completed on linuxmacan. Standard target freshly downloaded from `unsloth/Qwen3.8-27B-GGUF` revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`. DFlash2 downloaded from `incoai/Qwen3.8-27B-DFlash2` revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, converted using Lucebox's `convert_dflash_to_gguf.py`, then `quantize_dflash_draft.py --scheme q8_0`.

Explicitly removed the old standard target and both old Qwen drafter GGUFs before downloading. Preserved Huihui abliteration and the separate 0.6B scoring models. Retained the F16 intermediate and source safetensors. Both replacement target and Q8 drafter hashes exactly match the old files; this is a provenance refresh, not a new quantization variant.

Configuration: block 16, context 131072, Q8_0 K/V, one stream. Standard model hybrid caching/compression disabled for the tutorial baseline; PFlash startup confirms off. Existing endpoint remains port 8216. Previous configuration backed up in `/opt/lucebox/work/unsloth-reproduction-20260915/models.before.json`.

## Measurement

Freshly restarted server; ten repository `bench_he.py` HumanEval-style prompts, OpenAI chat requests, maximum 256 tokens, temperature 0, no requested thinking. Actual total 1507 output tokens. Aggregate decode throughput is total output tokens divided by summed server decode time; end-to-end is total output tokens divided by summed HTTP wall time.

| Metric | This run | Published |
|---|---:|---:|
| Decode tokens/s | 204.98 | 208.1 |
| End-to-end tokens/s | 158.79 | 156.2 |

Earlier exploratory run used the abbreviated `bench_he_http.py` prompts and measured 169.91 decode /136.32 end-to-end tokens/s. Prompt choice changes acceptance and must not be mixed in the comparison. Each result is a single ten-prompt run, not a multi-run confidence estimate or a coding-quality evaluation.

The host uses ROCm core-10.0; publication reports ROCm 7.2. Engine revision also differs. Results are close, not byte-for-byte reproduction of the author's full environment. No generated code was executed by these speed measurements.

The prior allocator/watchdog fixes on GitHub were not deployed as part of this model replacement. Phase-specific peak-memory admission remains unfinished; the short speed benchmark does not establish long-output OOM safety.

Sources: https://www.lucebox.com/blog/qwen38-r9700 and https://huggingface.co/Lucebox/Qwen3.8-27B-IQ4_XS-fast-GGUF

Raw manifest, request results and benchmark script: `work/unsloth-reproduction/` in this workspace. Server is healthy following ten completed requests.


## Compression-enabled rerun

Re-enabled the deployed hybrid/PFlash configuration: 8K scoring windows, 20% retention, 0.6B BF16 scorer, skip-park, request-scoped decoding drafter residency. Standard model default remains exact; these benchmark requests explicitly selected auto. The PFlash model alias defaults to auto. No new executable or allocator/watchdog patch was deployed.

### Same ten short canonical prompts

| Configuration | Decode tokens/s | End-to-end tokens/s | Output tokens |
|---|---:|---:|---:|
| Tutorial baseline | 204.98 | 158.79 | 1507 |
| Hybrid/PFlash configuration enabled | 147.83 | 92.24 | 1507 |

These prompts are below the 32K eligibility threshold and do not compress. Decode throughput fell 27.9%; end-to-end throughput fell 41.9%. This compares the complete configurations, including request-scoped drafter loading, and does not isolate the cause. One ten-prompt sample per configuration.

### Eligible 32K bulk document and repeat

Same synthetic explicitly marked bulk ledger, three requested access values, temperature zero, reasoning none, output capacity 32768. One sample per path:

| Path | HTTP wall seconds | Effective input | Prefilled tokens | Scored regions |
|---|---:|---:|---:|---:|
| Cold dense | 48.436 | 32084 | 32084 | 0 |
| Cold compression | 22.207 | 6388 | 6388 | 1 |
| Frozen-prefix repeat | 0.991 | 6388 | 244 | 0 |

Compression reduced total latency by 54.2% (2.18x speedup). Scoring took 14.478s; prefill took 7.301s versus dense 47.882s. Repeat restored 6144 tokens and performed 0.435s suffix prefill. All responses returned the three correct values, `507831, 926104, 318762`. This simple retrieval probe is not a general semantic-quality or coding-quality evaluation. No 80K or three-chat stress tests were run.

The first eligible request chose dense because redownloading identical model bytes changed mtime, which participates in calibration identity. After verifying matching old/new target and drafter hashes in the replacement manifest, copied the four none-thinking timing buckets from old identity d5574b99919f7e3f4204207f1b49b403 to observed identity 8efbddf010838f6b89f98420ae5e8fe8. Backed up calibration before copying, then restarted while idle. Checkpoint identity and model timestamps were not altered. Other reasoning-mode calibration identities were not migrated. This manual repair does not fix timestamp-sensitive calibration invalidation generally.

Artifacts: compression-enabled-results.json, compression-enabled-summary.json, eligible-compression-results.json, uncalibrated-eligible-compression-results.json, calibration-migration.json and check-compression.py under work/unsloth-reproduction. Remote backups and logs are in /opt/lucebox/work/unsloth-reproduction-20260915. Test owner explicitly released; final health reports healthy, idle, no queued requests. Compression configuration remains enabled with the standard exact default preserved.


## Removed residency knob and deployed persistent decoding drafter

Commit `3000d2a` pushed to `serving-resilience`. Removed --draft-residency and --lazy-draft, request-boundary park/reload, startup parking, policy parsing and associated configuration. The decoding drafter stays resident across HTTP requests; the PFlash scorer releases after scoring. Backend explicit compression parking APIs remain for other configurations; deployed hybrid compression uses skip-park.

Deployed only this residency change on the existing production base, not the earlier allocator/watchdog changes from GitHub. Backups: /opt/lucebox/work/persistent-draft-backup. Removed obsolete flags from standard and PFlash model entries. Standard default remains exact.

Validation: GPU server build succeeded; 564 server unit tests passed; both removed flags rejected as unknown. Two short successive completions returned ready with valid finish reasons. One eligible compression request completed in 21.290s, its frozen repeat in 0.817s; correct three-fact answers, scored regions 1 then 0. No decoding-drafter park/reload in logs after the new startup. Healthy and idle afterward. These short smoke checks do not establish a new ten-prompt throughput result or long-output OOM safety.

GPU admission sees actual free memory with the decoding drafter resident. Existing fixed scratch allowance remains; phase-specific memory prediction is still separate unfinished work.


## Same ten-prompt benchmark after persistent-drafter deployment

Re-ran the identical canonical bench_he.py prompts with auto cache mode, max_tokens 256, temperature 0. Existing healthy process stayed loaded; no restart or extra warmup was performed. Ten responses generated 1507 tokens in 9.240s HTTP wall time and 7.3104s summed decode time: **206.14 decode tokens/s, 163.09 end-to-end tokens/s**. Previous request-scoped configuration: 147.83 / 92.24; tutorial baseline: 204.98 / 158.79.

This recovers baseline performance in this sample: 39.4% higher decode throughput and 76.8% higher end-to-end throughput than request-scoped residency. Small differences versus baseline are not established improvements; each configuration has one ten-prompt run. These short prompts do not trigger compression. Long compression behavior was verified separately in the preceding smoke check. Server healthy and idle after all ten requests. Raw artifacts: persistent-draft-summary.json and persistent-draft-results.json in work/unsloth-reproduction.
