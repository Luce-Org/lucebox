> **Latest staged refinement status:** see [September 15 audit](cache-refinement-audit-20260915.md). Updated and restarted after authorization; exact default retained and timing projections disabled. Earlier sections below record prior experiment stages.

# Hybrid caching implementation — September 15, 2026

## Current implementation status: deployed and restarted

The hybrid strategy is installed on linuxmacan, and the local Pi extension is updated. Lucebox was restarted and verified healthy with one active-generation slot. A real Pi call passed with medium thinking and ownership metadata. Exact remains the default; use `/cache auto` per chat, `/cache exact` to override, and `/cache release` to release protection. Existing Pi sessions need `/reload` to load the updated extension. Standard Qwen and its PFlash alias now share one loaded backend.

Automatic compression is calibrated for the measured 32K buckets with thinking off or medium and a 32,768-token output allowance. Unknown cost buckets use conservative fallback; larger-context latency projections do not certify memory capacity or semantic accuracy.

### Follow-up progress review

Production health and deployed cache budgets were rechecked. The server is healthy, single-stream, with exact default and the expected 9 GiB GPU / 8 GiB RAM checkpoint budgets.

Fixed a reproduced Pi branch-navigation bug: visiting an existing branch previously allocated another owner and inherited the departing branch's policy. Existing branches now recover their saved owner and mode. Rewinds, root navigation and new summaries retain distinct branch ownership. The regression uses Pi's actual SessionManager and passes 50 branch revisits, reload, no-op navigation, rewind, summary and root cases. The previous installed extension fails this test. Existing hook tests, the real Pi wire test and a live Pi request also passed. The updated extension is installed; `/reload` activates it in existing Pi sessions. No server restart was required. Rollback extension: `work/hybrid-cache/pi/prefix-cache-guard.ts.before-branch-fix`.

Small-test extrapolation: a quadratic description of the 8K/16K/32K exact timings predicts 199.3 s at 80K; the earlier measured median was 186.8 s (6.7% lower). The earlier run used a different output allowance and build, so this is a useful trend check rather than precise validation. No new large request was run. Details: `work/hybrid-cache/scaling-trend.json`.

Remaining optimization opportunities: broader calibrated request shapes; reusing a verified earlier checkpoint during a deliberate rebase (the current rebase conservatively rebuilds from zero). Normal frozen-prefix reuse already works. These opportunities have not been silently enabled based on extrapolated costs.

### Latest short-loop results

Latest build: 567 server tests and 5 gateway tests passed. Small GPU ownership/repeat/stream/alias checks passed.

| Input/path | Seconds (one run, Pi output capacity 32,768) |
|---|---:|
| 8K exact | 9.77 |
| 16K exact | 20.00 |
| 32K exact | 48.59 |
| 32K compression | 22.43 |
| Frozen repeat | 1.00 |
| Full repeat | 0.58 |

32K improvement: 53.8%. Backend swap stayed at zero; peak observed GPU use was 26.07 GiB. These runs are not yet a three-observation calibration.

Tool medians: current production executable 0.663 s / 0.482 s versus candidate 0.812 s / 0.407 s (tool-request / result turns). Both pass the agreed 250 ms-or-10% allowance.

**Measured semantic loss:** the 32,518-token registry yielded 6/6 original facts, but its frozen representation yielded 0/6 newly requested facts; exact processing yielded 6/6. No rescoring occurred on the frozen follow-up. A 41,041-token compressed coding fixture passed 5/5 functional cases. These examples do not establish general coding or retrieval accuracy.

Raw evidence: `work/hybrid-cache/small-sweep-results.jsonl`, `quality-results.jsonl`, `review-below-threshold-results.jsonl`, and `baseline-results.jsonl`.

### Actual Pi settings and final routing

Medium thinking, output allowance 32,768, three-run 32K medians: exact **50.39 s**, compression **20.39 s** (59.5% faster). The real Pi wire check confirms those request settings.

Automatic selection without a force flag passed with Pi-style text blocks and a tool definition. Frozen reuse and the same-chat exact override passed. Temporary compaction retained the existing primary; expired requests returned 408. The oversized-output test discovered legacy silent clamping; hybrid mode now rejects over-limit requests explicitly. The regression build and a private boundary check passed. Missing IDs did not acquire a protected owner.

Two approximately 6K checkpoints completed verified GPU/RAM round trips and returned correct answers. No further three-chat or 80K stress runs were performed.

### Earlier calibration candidate

Three-run medians from the private calibration candidate:

| Marked input | Exact | Compression | Improvement |
|---|---:|---:|---:|
| 32,084 tokens | 48.31 s | 19.59 s | 59.5% |
| 80,084 tokens | 186.81 s | 49.75 s | 73.4% |

Frozen-prefix repeats skipped scoring. Full-prompt repeats skipped both scoring and prefill (about 0.57 s at 32K and 0.77 s at 80K). All requested ledger values were correct. Unmarked 32K and 80K fixtures stayed uncompressed. These measurements use an 80-token output allowance; actual Pi wire testing found a 32,768-token allowance, and the matched Pi calibration is reported above.

Implemented components include three-owner protection and explicit release, immutable frozen regions, deliberate rebase gap partitioning, calibrated selection with rebuild hysteresis, memory simulation, verified GPU/RAM snapshot copying, primary-first capture with actual position tracking, an exact override, and bounded pre-output recovery. A fourth owner can inherit a released slot without copying an already usable optional snapshot.

The final server build passed 567 unit tests, including Pi text-block support, frozen-region replay after aging, and oversized-output rejection. The real installed Pi RPC client passed a synthetic HTTP wire test: normal calls retained the owner, two compaction requests carried that owner with temporary purpose, and the resulting summary was marked protected. The extension is installed in the live Pi extension directory. The wire test exposed Pi's text-block arrays, which are now supported by the installed client and server source.

**Revised test scope (user instruction):** no more three-chat or 80K stress runs. Development defaults to one pass at approximately 8K, 16K and 32K; repeat timing only for the final candidate. Compression still requires at least 32K. Larger-context projections will be labeled estimates, not memory/correctness guarantees. The old stress unit is stopped. Its partial evidence is preserved: protected entries survived, but the reduced 6 GiB GPU budget caused a restore candidate to be rejected and a dense rebuild. This run did not pass its cache-hit assertion.

**Validation boundary:** the requested shorter test scope replaced the three-chat/80K stress gate. All final targeted checks above passed. The old reduced-budget stress result remains a known large-context limitation; it preserved snapshots but fell back to dense processing. Semantic loss remains intentional in auto mode and was measured, not waived silently. No new large-context stress or physical reboot was performed.

Rollback: server executable/configuration/gateway in `/opt/lucebox/work/hybrid-cache-20260915/deployment-before`; shared libraries in `before/lib`; Pi copies in `work/hybrid-cache/pi/deployment-before`. Reproducible scripts and controls are documented in `work/hybrid-cache/README.md`.

SSH access has been restored and production health was verified before the shorter private tests. The remote runner has automatic production restoration even if the SSH connection is lost. Raw calibration evidence is in `work/hybrid-cache/calibration-results.jsonl`; the Pi wire result is in `work/hybrid-cache/pi/wire-results.json`.

---

# Historical: earlier performance experiments — September 15, 2026

## Deployed and restarted

Production is running with one stream. The ordinary Qwen profile remains the default with exact GPU prefix checkpoints. A selectable **Qwen3.8-27B PFlash (lossy)** profile (`qwen3.8-27b-pflash`) is installed in both the server and local Pi configuration. It uses the validated 8K scoring windows and 20% retention. Existing Pi session-affinity settings are preserved.

The production fast-profile 32K request passed in **22.77 seconds**, including model switching. Switching back to dense passed: a 4,026-token request took **6.86 seconds**, then **0.90 seconds** on repeat with **3,584 tokens reused**. A final restart returned the service to the default dense model and cleared synthetic checkpoints.

## Significant new result: bounded PFlash scoring

Scoring the prompt in 8,192-token windows, each with a 512-token left overlap and the question tail appended, reduced the measured 80,084-token cold request from **185.74 seconds to 49.45 seconds (3.76× faster)**. The 32,084-token case improved from **48.96 to 20.76 seconds (2.36×)**. Three requested facts were correct in both cases. A previous build with an unrelated attention-padding experiment measured 21.44 / 48.90 seconds.

The main target remains on GPU; the decoding drafter is released between requests. Windowing bounds the scoring attention work. Existing global chunk selection retains roughly 20% of prompt tokens, in original order. It does not implement exact APC or add retained attention history. Each window uses locally normalized scores and only its 512-token left overlap, so long-distance relationships can be lost. This is an experimental, lossy fast mode, not a substitute for the exact cached profile.

### Validation so far

- 32K followed by 80K in the same process succeeded, unlike the previous whole-prompt scorer that ran out of memory.
- A randomized 28K registry returned all six requested values correctly, including a growing-conversation follow-up. An early-history mutation did not reuse a stale prefix. Those turns took 14.58, 14.66 and 14.20 seconds.
- A 30K Python repair used constants and instructions from distant parts of the prompt and passed five numerical cases. The original harness rejected valid local-variable assignments; its corrected check validated the unchanged saved output.
- A two-turn tool lookup succeeded but took **3.71 + 5.95 seconds**, slower overall than the earlier dense 6.34 + 0.49 seconds. Compression is not universally faster.
- Cleaned build: **545 server tests**, **6 CPU checkpoint tests**, **6 GPU checkpoint tests** passed. Two new gateway tests validate model-specific options and prevent compression settings leaking into the dense model.
- The cleaned 8K build passed the complete sequence: 32K, 80K, repeated 80K, tool use, code repair, and growing/mutated history. The repeated 80K request took **50.37 seconds with no prefix reuse**, versus the exact dense cache's earlier **1.80 seconds**. This is why compression remains a separate model selection.
- A further 4K-window variant passed 32K, 80K, coding and conversation checks at **17.89 / 45.63 seconds** for the cold prompts. It remains experimental; the deployed 8K variant has the broader repeat/tool validation sequence.
- The temporary SSH interruption did not stop the remote tests; all completed successfully.

## Other approaches actually tested

| Approach | Observed outcome / decision |
|---|---|
| Prefill batches 1,024 / 2,048 / 4,096 | 32K: 49.36 / 50.42 / 49.11 s; no useful gain over 512 |
| Full-prefill cache | Exact 32K repeat 0.668 s, all 32,084 tokens reused; no proof of growing-chat benefit |
| ROCWMMA attention | 32K prefill ~38 s, then decode failure with Q8 and F16; independent attention oracle failed; reverted |
| Complete-prompt rolling inline checkpoint | Growing chat missed because rendered historical assistant tokens differ from the generation prefix; reverted |
| Pad scoring attention length to 256 | 32K 37.80 s; subsequent 80K OOM; no material gain; reverted |
| Native RDNA4 256-wide attention | Numerical oracle passed after fixing compilation guard, but 32K took 101.14 s; reverted |
| Chunked BLAS attention | 32K took 67.06 s; rejected |
| Windowed PFlash scoring | 32K 20.76 s, 80K 49.45 s; promising result retained for final verification |

The live code contains the windowed scoring experiment; failed kernel/cache experiments were removed. The rebuilt HIP shared library has the same SHA-256 as the known-good original: `10f0ef14c022c01852870ef6da27b56a8b8a9481c53252eb2fb785a9ece49ff6`.

## Reproduction and deployment boundary

Experiment files are under `/opt/lucebox/work/performance-search-20260915` on linuxmacan and `work/performance-search/` in this task. `run-variant.py` runs each configuration in a private service, sequentially. `windowed-final.config.json` selects the cleaned binary, known-good GPU libraries, 8K windows, 20% keep ratio, no target parking, and request-scoped decode-drafter residency. Probes cover 32K, 80K, repeated prompts, a tool call, a code repair, and growing/mutated history.

**Deployed executable SHA-256:** `6dfaba426be35d1ede3dbd06f0b078f2d7836546ef9adb08d469fc3d932792ad`.

The gateway accepts validated per-model `extra_args` and `compression_score_window` settings. It explicitly resets the scoring window for dense models, avoiding environment leakage across model switches. Only standard Qwen has a PFlash profile; abliterated Qwen and Gemma were not separately validated for compression. Selecting another model unloads the previous one and clears its checkpoints. The shared single-stream cache does not restore the earlier protected per-chat 80K policy.

To revert this round, stop Lucebox, restore `before/dflash_server`, `before/model_router.py`, and `before/models.json` to their production paths, and restart. The original GPU library was restored byte-for-byte; if libraries are changed later, restore the `before/lib` copies too. Pi's original configuration is backed up beside it as `~/.pi/agent/models.json.before-pflash-20260915`.

Backups in `before/` include the production executable, gateway, model configuration and **all four ggml shared libraries**. Binary-only rollback is insufficient after rebuilding GPU kernels: the executable dynamically loads the shared libraries. Restore or explicitly select the saved libraries as well. The failed ROCWMMA/native256 options must stay off.

## Inspiration checked

- [llama.cpp maintainer discussion on batch size](https://github.com/ggml-org/llama.cpp/discussions/2463) motivated actual batch-size measurements.
- [Lucebox implementation and PFlash documentation](https://github.com/Luce-Org/lucebox) identified scoring/parking paths and existing compression behavior. Its headline speed figures describe other models/hardware, not this host.
- [ROCm rocWMMA](https://github.com/ROCm/rocWMMA) motivated a matrix-attention kernel experiment. Hardware support did not establish numerical correctness or speed; both were tested independently.

The older deployment and benchmark history below predates this round of experiments.

<!-- earlier-results -->
# Lucebox: single-stream configuration and tests

## Current deployment — September 15, 2026

Lucebox on linuxmacan now accepts **one active generation at a time**. Both Qwen configurations use the single-stream backend, Q8 attention KV, and **GPU-resident dense prefix checkpoints**. Full context remains 131,072 tokens. Gemma remains single-stream with its existing checkpoint placement. Only standard Qwen was benchmarked.

Production was restarted and checked with a cold request and a cached repeat, both returning the expected answer. A final restart cleared synthetic checkpoints. PFlash and agent-turn caching are off. No provider billing or Pi client settings changed in this iteration.

**Cache policy change:** this path uses two shared, token-matched prefix slots. It does not provide the previous concurrent path's protected 80K checkpoint per chat, fixed 80K capture cap, or 9 GiB snapshot admission budget. Other chats can replace shared cache entries. Prefix reuse checks matching tokens; memory placement does not change the model's attention history. Restart/model switch clears the cache. Full 128K and multiple maximum-size checkpoint occupancy were not validated.

## Measured results

Times are end-to-end request durations, including short answers, from individual synthetic runs. They are not statistical estimates or general coding-quality results.

| Test | Result |
|---|---:|
| Dense CPU-checkpoint baseline, fresh 32,084 tokens | 49.42 s |
| Dense GPU checkpoint, fresh 32,084 tokens | 48.96 s |
| Same GPU-cached 32K request repeated | **1.47 s**, 31,744 tokens reused |
| Dense GPU checkpoint, fresh 80,084 tokens | **185.74 s** |
| Same GPU-cached 80K request repeated | **1.80 s**, 79,872 tokens reused |
| PFlash, 20% retained, target never parked, fresh 32K | 38.04 s; all three facts correct |
| Same PFlash process, next request 80K | **Out of GPU memory**, HTTP 502; gateway recycled backend |
| PFlash never-park, fresh process, 80K | 180.38 s; all three facts correct |

The large improvement on repeated prompts comes from prefix reuse. The small cold CPU/GPU timing difference does not establish a meaningful speedup. GPU storage was verified by backend allocation logs and a byte-exact snapshot test.

### Tool-call caching

With GPU checkpoints in both runs, the ordinary two-turn lookup took **6.34 + 0.49 = 6.83 seconds**. Enabling `--agent-turn-cache` took **8.00 + 0.37 = 8.37 seconds**. It correctly cached through the generated call (5,209 tokens versus 5,120), but replay overhead outweighed the next-turn saving in this case. It remains off; longer calls or different streaming workloads could behave differently.

### PFlash conclusion

Removing the concurrency restriction allowed further testing. Never parking the target saved about 23% at 32K versus the current dense GPU baseline, but only about 3% at 80K in a fresh process. The 32K-to-80K sequence failed with a ROCm allocation error. A fresh 80K process succeeded, suggesting occupancy from prior generation is material; the exact allocation responsible was not isolated. This mode is not deployed. Earlier ordinary parking-mode PFlash tests also passed at 32K/80K (see historical results), with only a small 80K benefit.

PFlash removes prompt tokens. Three-fact retrieval success does not establish coding/tool-use quality or equivalence to dense attention. No quality-changing compression or KV eviction is enabled by default.

## Implementation and verification

- The legacy snapshot selector ignored the existing GPU override on discrete cards. The Qwen single-device backend now opts into it; other architectures retain their existing behavior.
- Dense snapshot save/restore now use device-safe tensor-span copies instead of treating snapshot addresses as CPU pointers. Copies synchronize before returning.
- Added a dense checkpoint regression covering distinct head strips, a partial prefix, untouched live-cache tail, recurrent state, and target features. Six snapshot tests pass on CPU and GPU; 545 server tests pass.
- End-to-end 32K/80K cold and warm retrieval passed, as did tool-call/result handling and the production cold/warm smoke check.
- No new APC block-sharing allocator, KVFlash eviction, lower-precision KV, or disk tier was implemented. Existing dense prefix reuse, speculative decoding, and prefix pinning remain available. GPU checkpoint copies are not vLLM-style shared-block APC.

## Reproduction and rollback

Experiment directory: `/opt/lucebox/work/single-stream-20260915`. Local probes, configurations, source copies, unit logs, and raw results are under this task's `work/single-stream/`. `probe.py` accepts `80k`; `probe-tools.py` tests a two-turn lookup; both use private port 8217. Configurations select dense GPU, agent-turn caching, or PFlash never-park independently. Stop production before using a private GPU test instance, and restore it afterward.

Deployed executable SHA-256: `649f071c337436d1626fb4ed24dd0f272b1f14a89d12f70e12ee7d170ea63e82`.

The `before/` directory contains the previous two-stream GPU production binary, gateway, and configuration. To restore that setup, stop Lucebox, restore `before/dflash_server` to `/opt/lucebox-concurrent/server/build-hip/dflash_server`, restore `before/models.json` and `before/model_router.py` to `/opt/lucebox/`, and restart `lucebox.service`. Source edits preserve the checkout's existing changes; no commit was made.

## Previous two-stream results (historical)

The following describes the previous deployment and is superseded by the single-stream status above.

# Lucebox caching: deployed configuration and results

## Deployed and restarted

Production `lucebox.service` is active and healthy on port 8216. A post-restart generation returned `OK`; backend diagnostics confirmed GPU checkpoint storage, two cache slots, an 80,000-token cap, PFlash off, and zero occupied test slots.

- Two concurrent Qwen streams instead of three.
- One protected rolling checkpoint per chat, capped at 80,000 prefix tokens.
- GPU checkpoint storage, including both attention and recurrent state.
- 196,608 shared GPU token slots; 9 GiB checkpoint/replacement budget.
- Reclaim unused HIP allocator scratch under memory pressure before allocating a GPU checkpoint. Existing live data and other chats' checkpoints remain intact.
- PFlash remains off.

The Qwen and abliterated Qwen model configurations use this policy. Gemma retains its existing single-stream cache path. The standard Qwen model was benchmarked; the abliterated model was not separately benchmarked.

## Measurements

| Test | Observed result |
|---|---|
| Two simultaneous 78K cold prompts | Both correct, approximately 484 seconds each |
| Subsequent concurrent turns, adding approximately 750 tokens | Approximately 7–9 seconds on the final measured turn; checkpoints advanced to 80K |
| Full-size GPU checkpoint save | Maximum 95.54 ms, including scratch reclamation; later saves were around 12 ms |
| Full-size GPU checkpoint restore | Maximum 15.30 ms including cancellation-recovery testing |
| One fresh 77,988-token prompt plus one ongoing 80K-cached chat | Cold chat 243.67 seconds; cached chat 2.25 seconds; both correct |
| Final full-size cache residency | Two GPU checkpoints, 5,884,370,944 bytes (about 5.48 GiB) |
| Highest sampled VRAM in two-chat long test | 30,959,951,872 bytes (about 28.83 GiB); sampled once per second |
| Small two-stream cached turns | Approximately 0.6–0.7 seconds with GPU snapshots versus 0.7–0.9 seconds in the earlier CPU-snapshot run |

The long test had eight successful replies, eight captures, six cache restores, zero capture failures, zero budget skips, and zero restore invalidations. Full-size cancellation recovery and subsequent changed-history replacement also passed. Both chats restored exactly 80,000 tokens during cancellation recovery; a cold restart of one chat preserved the other's checkpoint.

These are synthetic, short-output tests, not a coding-quality suite or a long-generation benchmark. The full request context remains 131,072 tokens, but two simultaneous maximum-context/maximum-output requests were not tested and do not fit the shared token pool. The GPU-versus-CPU timing observations are individual runs, not statistical confidence intervals.

## Why the first GPU experiment failed, and what fixed it

Moving snapshots to GPU memory initially failed at the 78K capture boundary. Each snapshot needed approximately 2,739 MiB, while the HIP allocator retained unused temporaries from previous prefill shapes. The backend subsequently freed 6,883.6 MiB at idle.

The fix checks free device memory before allocating a GPU checkpoint. If the allocation would leave less than 512 MiB, it synchronizes the backend and trims unused operator-pool allocations first. This preserves active model/KV/graph buffers and existing checkpoints. With that change, both 80K GPU checkpoints, rolling replacement, cancellation and one-cold/one-cached operation passed.

## PFlash: tested, not enabled

The current backend refuses PFlash together with paged attention, which the two-stream scheduler requires. Separate single-stream tests used the BF16 Qwen3-0.6B drafter and a 20% keep ratio.

| Source tokens | Dense | PFlash | Correctly retrieved all three facts? |
|---:|---:|---:|---|
| 32,084 | 49.38 s | 40.44 s | Yes, both modes |
| 80,084 | 187.15 s | 183.56 s | Yes, both modes |

PFlash reduced effective input to 6,388 and 15,988 tokens respectively, but scoring overhead consumed most of the savings at 80K. The approximately 18% and 2% latency reductions are single-case observations. They do not establish general coding/tool-use quality. Given the small measured 80K benefit and the concurrency incompatibility, PFlash was left off.

## Other mechanisms considered

- **Shared-block APC/reference counting:** could remove duplicate attention storage and help shared histories. It needs shared block lifetimes, copy-on-write for partial blocks, compatible recurrent-state checkpoints and admission rules that protect each chat. This is not implemented in the deployed change. Snapshot transfer overhead is now small; this would be a larger memory-management change, not a configuration toggle.
- **Cross-chat prefix sharing:** helps identical opening histories, not unrelated cold prompts. No global eviction policy was introduced; each chat still replaces only its own checkpoint.
- **Incremental snapshot updates:** could reduce copied bytes further, but measured GPU copy overhead is now small relative to processing the new suffix. Not added in this iteration.
- **More aggressive KV quantization:** Q8 KV is already configured. Lower precision changes numerical behavior and needs separate quality validation; it was not enabled as an exact-cache optimization.
- **Disk/tiered persistence:** could help reloads or larger cache populations, but was not the measured steady-state bottleneck. No disk paging was added to inference.
- **Prefill/decode scheduling:** the existing bounded mixed-prefill policy remains in use. The mixed test confirms an ongoing chat can respond while another prompt processes.

## Configuration, lifecycle and rollback

Select checkpoint storage using `backend.prefix_cache_device: "gpu"` or `"cpu"` in `/opt/lucebox/models.json`. The gateway passes this to `DFLASH_PREFIX_CACHE_DEVICE`. `/props` reports the selected mode. The default remains CPU if neither configuration nor environment selects GPU.

No idle expiration was added. With two occupied session slots, a third distinct chat can run uncached; it does not evict another chat. Restarting or switching the model clears checkpoints. Anonymous requests remain uncached in session mode. Requests beyond 80K still reuse only the cached first 80K tokens and process the remaining suffix normally.

Production binary SHA-256:
`759732150fb93ce2375bdc2a3cbe516a92d1b52f75e8b3e74a0db7fbba3e7bc4`

The executable was compared with the running validated process before installation. Production gateway/configuration and the CPU-checkpoint executable are backed up under `/opt/lucebox/work/session-cache-20260914/before/` as `model_router.cpu-rolling.py`, `models.two-cpu.json`, and `dflash_server.cpu-rolling`. Restore those files to their production paths and restart `lucebox` to revert the GPU optimization while retaining two streams and rolling session caches.

Source changes preserve the checkout's pre-existing unrelated edits; no commit was made. The source mirror, isolated GPU patch, probes and raw logs remain in this task's `work/` folder and the remote experiment directory. Unit validation includes 545 server tests and five snapshot cases on CPU, GPU-to-CPU and GPU-to-GPU paths. Earlier transaction and gateway identity tests also passed.
