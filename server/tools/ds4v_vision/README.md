# Native DS4V vision runtime

Reusable, backend-owned projector/tower implementation. `VisionRuntime` borrows
one caller-selected backend and owns its validated BF16 weight buffer plus a
reusable graph allocator. Load is transactional: a rejected reload preserves the
prior runtime. No HTTP, decoder, image decode, sentinel placement, or N-layout
changes are included. `Sentinel` exposes the four learned delimiter vectors;
image rows are the `VisionOutput.embeddings` result.

The loader checks all exporter semantic fields and all 267 names, BF16 dtypes,
shapes, alignment, and file bounds before backend allocation. It rejects unknown
schema, layout, activation, language dimension/vocabulary, missing/extra tensors,
and malformed GGUF metadata. Original source names and weight bytes are retained.
The accepted artifact has no source-repository or source-hash metadata; provenance
is verified externally against the accepted exporter hash, not invented by the
runtime.

## Arithmetic and resource rationale

F32 graph tensors hold BF16-rounded activations. `cast(BF16)` then `cast(F32)`
preserves every source BF16 boundary: biased linear, RMSNorm with F32 weights,
rotary Q/K, attention output, residual additions, SiLU, gated product, and each
aligner operation. Normalization and rotary arithmetic remain F32. GELU uses
`ggml_gelu_erf`. Q/K use explicit F32 cosine/sine tables with half-split channel
pairs and height frequencies before width frequencies. Im2col performs exact
channel-first, bottom/right padded 3x3 unfolding.

Actual Torch 2.10 CPU tracing shows the source's **3D** SDPA dispatches
`aten::_scaled_dot_product_attention_math`. That implementation scales both F32
Q and K by sqrt(1/sqrt(head_dimension)) before matrix multiplication, so this
runtime preserves the same operation order. Scaling scores afterwards changed
rounding enough to measurably worsen both full fixtures. See
[PyTorch Math SDPA source](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/transformers/attention.cpp#L807-L891).
Explicit full softmax attention has no causal mask. CPUFlash is not the source
fixture path: forced Math reproduces both original fixtures bit for bit.

One block graph at a time bounds quadratic scratch, with a hard 2 GiB graph
buffer limit measured before allocation. Input grids must fit the complete
384-token N-layout budget, including delimiters, row/odd-row/parity padding,
and three reserved leading alignment tokens. Diagnostic tensors are independent snapshots; a flag on
a view alone does not protect its backing allocation. No attention matrices are
retained across blocks. Caller diagnostics run synchronously. Each block currently
returns its F32 residual to host and uploads it into the following graph; HIP
transfer cost and GPU behavior are unqualified. `release_scratch()` frees graph
buffers while keeping weights/sentinels. Sequential calls only.

Alternatives considered: built-in VISION RoPE has different frequency recurrence
rounding; adjacent-pair text RoPE is mathematically wrong. Flash attention would
add an unqualified kernel/dtype path. An all-F32 tower removes required source
rounding; F16 weights are not a lossless BF16 substitute. Keeping all 32 graphs
or attention diagnostics would multiply quadratic scratch. The chosen explicit
primitive graph makes each stage inspectable at the cost of host transfers.

## Standalone CPU qualification

Run these commands on `soulf`, from its isolated candidate worktree. The Mac is
for authorship only. CPU builds use two jobs; probe and reference execution use
two threads. Original fixtures and reference environment remain read-only.

```sh
cmake -S server/tools/ds4v_vision -B /tmp/ds4v-tower-a-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ds4v-tower-a-build -j2
OMP_NUM_THREADS=2 ctest --test-dir /tmp/ds4v-tower-a-build --output-on-failure
python3 server/tools/ds4v_vision/loader_tests.py /tmp/ds4v-tower-a-build/ds4v_vision_probe /home/marcelorm/ds4v-work/ds4v-mmproj.gguf
```

Full probes (the last argument enables independent stage snapshots):

```sh
OMP_NUM_THREADS=2 /usr/bin/time -v /tmp/ds4v-tower-a-build/ds4v_vision_probe /home/marcelorm/ds4v-work/ds4v-mmproj.gguf /home/marcelorm/lucebox-ds4v-mix-fix/artifacts/vision-reference/corn-patches.f32 23 34 artifacts/vision-tower-a/native corn 1
OMP_NUM_THREADS=2 /usr/bin/time -v /tmp/ds4v-tower-a-build/ds4v_vision_probe /home/marcelorm/ds4v-work/ds4v-mmproj.gguf /home/marcelorm/lucebox-ds4v-mix-fix/artifacts/vision-reference/carrots-patches.f32 42 61 artifacts/vision-tower-a/native carrots 1
```

Use `/home/marcelorm/lucebox-ds4v-mix-fix/.venv-vision-reference/bin/python`
with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2` for these tools:

- `compare.py REFERENCE NATIVE --output comparison.json`: original fixture
  hashes, raster shapes, finite values, max absolute error, RMSE, cosine.
  It exits 3 when the unchanged Candidate B gates fail: feature maxabs <=0.25,
  RMSE <=0.03, cosine >=0.9995; embedding maxabs <=0.75, RMSE <=0.08,
  cosine >=0.9990. These gates were fixed before the candidate fixture runs.
- `reference_stages.py SOURCE REFERENCE NATIVE stages.json`: original source
  hooks reproduce both fixture finals bitwise, then each original block consumes
  the native incoming residual to isolate local arithmetic from accumulated drift.
  Includes BF16 ULP distances; large max ULP distances across zero should be read
  with absolute error and p99, not interpreted as uniform relative error.
- `reference_math.py SOURCE REFERENCE NATIVE OUTPUT_DIR`: labeled parent-only
  Math sensitivity. It preserves original fixtures and is not a new tolerance.
- `attention_dispatch.py NATIVE`: records operator dispatch with source Q/K/V
  shapes, strides, and BF16 dtype.

`SOURCE` is `/home/marcelorm/lucebox-ds4v-2/models/DeepSeek-V4-Flash-Vision-Uncensored`.
`REFERENCE` is `/home/marcelorm/lucebox-ds4v-mix-fix/artifacts/vision-reference`.
Evidence is `/home/marcelorm/lucebox-ds4v-tower-a/artifacts/vision-tower-a`.
The projector SHA256 is
`58eb6b63243df2db21261ced5568b385b04991f38d45d39b781309497abd4b1c`.

## Verdict: ISSUES (numerical acceptance remains open)

Geometry checks, strict loader rejection, finite shapes, transactional reload,
sentinel access, and scratch release pass. Six deterministic arithmetic/geometry
checks are in CTest; sixteen malformed projector/language cases are in the loader
script. Corn observer-off and observer-on finals match bitwise. All graphs execute
on CPU without GPU, decoder, or HTTP changes.

Final native graph versus **original** CPU fixtures:

| Image/output | Shape | Max absolute | RMSE | Cosine |
|---|---|---:|---:|---:|
| carrots features | 2562 x 1024 | 0.1376953125 | 0.00259378329 | 0.99967582518 |
| carrots embeddings | 294 x 4096 | 0.02642822266 | 0.00142200006 | 0.99981156934 |
| corn features | 782 x 1024 | 1.484375 | 0.00629541949 | 0.99822935535 |
| corn embeddings | 96 x 4096 | 0.09423828125 | 0.00319258993 | 0.99907754975 |

All four outputs are finite. Corn fails the fixed feature gate, so the comparison
command returns 3. Differences begin at patch embedding and accumulate through BF16
residuals. Stage diagnostics isolate sharp amplification at blocks 12 and 31.
Same-input unfolding is bitwise exact for both fixtures, and same-input final
norm/aligner comparisons isolate small local kernel errors. The source Math
attention check does not explain away remaining end-to-end drift. No numerical
tolerance was widened. Full decoder logits, GPU transfer cost, and HIP arithmetic
remain unqualified.

Final CPU resource measurements with stage snapshots enabled: weights 932,786,176
bytes; corn scratch 77,774,592 bytes and 3.30829 s encode; carrots scratch
546,669,312 bytes and 17.6707 s encode. Peak process RSS was 1,827,056 KiB
(about 1.74 GiB), including the transient read-only mapped weight source during
load. The configured 2 GiB graph scratch cap is distinct from total process RSS.
Largest permitted grids are bounded analytically and by allocation measurement;
only the original 782/2562-patch grids have full numerical reference qualification.

## Selected-base follow-up

The full N-layout budget regression fails at5845ed5 and passes at5bf705e.
The largest grid permitted by this budget (6x561=3366patches) runs with finite
outputs and bitwise observer invariance. It is an allocation boundary fixture,
not a source resize-aspect fixture. Scratch is822488832bytes without diagnostics
and891424512bytes with them; encode times26.74/27.14seconds, peak RSS1881796KiB.
Failure/reload checks precede each successful encode; scratch release follows it.

An optional HIP build can prepare the same probe for the later GPU window:

```sh
ROCM_PATH=/opt/rocm-7.2.4 cmake -S server/tools/ds4v_vision -B /tmp/ds4v-runtime-hip-build -DCMAKE_BUILD_TYPE=Release -DDS4V_VISION_HIP=ON '-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1151' -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/lib/llvm/bin/clang++
cmake --build /tmp/ds4v-runtime-hip-build -j2
```

Append `hip:0` or `hip:1` to an encode/load-only probe command to request that
device explicitly. It fails if unavailable and never falls back to CPU. Building
the target is not GPU qualification. Do not run it on GPU before the private text
load proof and the operator's GPU window permit it.


## Explicit HIP fused-bias source comparison

`ds4v_linear_source cpu|hip:0 tiny|patch|qkv FIXTURE_DIR REFERENCE_F32 NEW_OUTPUT_DIR`
runs the shared production vision-linear helper on exact BF16 values stored as
F32 fixture files. Frozen source fixtures, input/library hashes, actual device
identity, and GPU release must be checked by the external supervisor. The tool
never selects a different backend or modifies a reference. Exit 3 preserves a
numerical failure; it is not a successful qualification. HIP requires exactly
one explicit fused operation in the graph and exactly one actual Lt launch.
The unbiased lane separately retains its ordinary product and final BF16 cast.

The historical `ds4v_linear_rounding` mathematical RNE probe continues to test
its explicitly requested old helper mode (no HIP capability argument). Its
biased RNE oracle differs from the frozen original HIP fused-source oracle;
it does not qualify the new runtime operation. Use `ds4v_linear_source` and
then the unchanged full-image gates for the actual runtime path.

HIP biased linears now use a dedicated, source-configured BF16 hipBLASLt bias
epilogue operation. CPU/NVIDIA and unbiased graph construction retain the
previous implementation. The HIP backend retains one 76 MiB workspace outside
the graph allocator, charged against the 2 GiB scratch bound and included in
`scratch_bytes()` even after `release_scratch()`. It is freed when the borrowed
backend context is destroyed. Graphs containing the operation do not use HIP
graph capture. Only the pinned Radeon RX 7900 XT reference is the qualification
target; other HIP hardware has not been qualified.
