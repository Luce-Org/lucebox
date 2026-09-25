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
