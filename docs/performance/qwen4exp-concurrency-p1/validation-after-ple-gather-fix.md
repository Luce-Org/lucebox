# P1 validation after the PLE gather serial-fallback fix

Date: 2026-09-25. No commit.

## Result

Validation is **blocked by loss of box connectivity**, not by a reported test
failure. The box accepted SSH at 14:17 UTC and had no inference process. After
the P1 sources were copied to `/tmp/qwen4exp-p1/server`, a fresh CMake configure
and `smoke_qwen4exp_batched` build was started with
`-DGGML_HIP_GRAPHS=OFF -DCMAKE_BUILD_TYPE=Release -j4`. SSH became unreachable
before the build result could be collected. Repeated bounded attempts to both
`100.115.193.112` and `lucebox4.tail97592a.ts.net` timed out. No remote process
was killed. The client-side SSH session was interrupted only.

The box state captured immediately before the build was `platform_profile=balanced`.
`rocm-smi --showclocks --showpower --showtemp` aborted in its power parser;
the working `--showclocks`/sysfs observations were recorded in the session, but
there is no during-run GPU measurement because no validation run began.

| Gate | Status | Evidence / limitation |
|---|---|---|
| Fresh gfx1151 HIP build, HIP graphs off | **Blocked / result unknown** | Fresh configure/build command launched in `/tmp/qwen4exp-p1/build`; SSH loss prevented exit status or binary verification. |
| Solo HE/GSM/Math/recall | **Not run after PLE fix** | E83 baseline was HE10/GSM10/Math9/recall2, but it predates this shared loader change and cannot establish unchanged results. |
| `QWEN4EXP_UPSTREAM=1` differential | **Not run after PLE fix** | Must recheck onset and expert-ID mismatches; loader fix is not excluded under the reference flag. |
| C=1024 N=3 snapshot equality | **Not run after PLE fix** | Historical pre-fix hash was `1f34fa0b724ae8ca93a1172077df1a02e8dc676b5d7b7c063a18b70a16bd926c`; no post-fix hash was collected, so output-change status is unknown. |
| Batched fixed-full probe and N=1 API identity | **Previously passed; not reconfirmed** | The immediately preceding P1 run reported fixed-full `exit=0`, `failures=0`, and N=1 bit-identical logits/state. The requested fresh rerun could not start. |

Local checks passed: `git diff --check`; docs describe
`QWEN4EXP_BATCHED_DECODE` as default `0`, and the entry point declines when
`QWEN4EXP_UPSTREAM=1`. These source checks do not substitute for the blocked
GPU gates.

The gather correction changes shared PLE loader behavior and may change solo
logits relative to the old partially-populated gather. Whether it changes
quality, upstream parity, or the deterministic snapshot remains **unknown**
until the box gates complete. Do not treat the earlier fixed-full batched
result as evidence for those solo properties.

Raw connection/build attempt context: this local report; the build tree and
partial output, if any, remain under box `/tmp/qwen4exp-p1/`.

## Completed validation rerun (supersedes E94's blocked status)

The recovered box passed the pre-run guard: no exact-name `luce_server` or
`dflash_server` process, card2 GTT 18,636,800 bytes, and 122 GiB available RAM.
The source was copied into an isolated `/tmp/qwen4exp-p1-src`; the original box
checkout was left untouched. A fresh Release build used
`LUCE_GPU_BACKEND=hip`, gfx1151/ROCm 7.2.2, and `GGML_HIP_GRAPHS=OFF`, with
`-j4`. `luce_server`, `smoke_qwen4exp_batched`, `smoke_qwen4exp_forward`, and
the adapted bridge snapshot probe built successfully. The main build log is
`raw/validation-2026-09-25/build.log`; probe build logs are alongside it.

GPU gates serialized under `/tmp/qwen-perf/gpu.lock`, one IQ4_NL model at a
time, with `HIP_VISIBLE_DEVICES=1` and `LUCE_HIP_NO_AUTO_UMA=1`. The platform
profile remained `balanced`; the sampled pp_dpm state and temperatures are in
the raw per-arm files. The `rocm-smi --showtemp` readings worked; the power
selector was not used. `quality/server-environ.txt` is empty because its
`/proc/PID/environ` read raced exec, but the run wrapper explicitly exported
the delivered variables and its exact script is preserved locally.

| Gate | Result |
|---|---|
| Solo quality | **PASS:** HE 10/10, GSM 10/10, Math 9/10, recall 2/2. The single Math miss is the known default-cap case; totals match the pre-fix baseline. |
| `QWEN4EXP_UPSTREAM=1` reference differential | **PASS:** exit 0, no onset; expert-ID mismatches 0 across all 48 layers (160 choices/layer, final layer 10/10). Aligned activation ratios were 1.0000. |
| C=1024 N=3 fresh-process snapshot | **PASS determinism; changed from pre-fix:** three 144,784,108-byte snapshots all hash to `05322458e34ee7b94be8cbdcb318f664a921bf699204426c6d8d20aec8a93596`. The pre-fix hash was `1f34fa0b724ae8ca93a1172077df1a02e8dc676b5d7b7c063a18b70a16bd926c`; same probe/format/config and size, so solo cache state changed, although it remains repeatable. |
| Batched fixed-full probe | **PASS:** exit 0, `failures=0`; N=1 API logits and state are bit-identical to the single path; four identical rows remain bit-identical; `{3,1}` isolation, row permutation, and reset/reuse pass. |
| Batched-vs-solo numerical margin | **Formal margin gate passes, but not suitable for serving:** max epsilon 1.96272445; distinct slot 2 changes 487→4876 at margin 0.163226 (<2ε=3.925449). Identical rows diverge from solo on decode step 3 (3710→13) at margin 0.146307 (<2ε=3.458492), while batch rows agree with each other. Keep the feature default-off. |

The PLE serial-fallback fix therefore **does change the solo recurrent snapshot**
relative to the old tree. It preserves N=3 byte repeatability and the quality
suite and reference differential both pass. This establishes behavioral
correction/compatibility for the measured gates, not byte identity to the old
buggy solo state.

Compact raw evidence (logs, environment, power samples, hashes; large `.lbsnap`
payloads remain on the box under `/tmp/qwen4exp-p1-validation/snapshot/`) is in
`raw/validation-2026-09-25/`. The adapted serializer probe source is under
`probes/bridge-snapshot/`.
