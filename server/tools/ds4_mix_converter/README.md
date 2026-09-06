# Original-source DS4V converters

The default `--recipe mix` retains the existing MIX2 gate/up, MIX3 down converter.
`--recipe iq85` is experimental: IQ2_XXS (2.0625 bits/weight) gate/up, IQ2_XS
(2.3125 bits/weight) down, and Q8_0 (8.5 bits/weight) for text attention
kv/output_a/output_b/q_a/q_b, shared FFN matrices, embedding and output matrices.
Vision/aligner/image, HC, router, indexer/compressor, vectors, norms and biases
remain unchanged. Both paths read the original safetensors; neither accepts a
previously quantized GGUF as input.

For the inventoried model the tensor-size projection is approximately **83.62
decimal GB**, including preserved vision. Exact metadata differs from MIX;
`--plan-only` prints JSON with actual serialized metadata, each tensor's type and
bytes, and exact final file bytes without reading/encoding weight payloads.
It still validates source headers and the imatrix. No 35 token/s performance or
quality claim follows from this storage estimate.

IQ85 requires an imatrix and an explicit operator-supplied label:
`--imatrix-provenance uniform-unvalidated`, `activation-derived`, or
`transferred-text-calibration` (text-model calibration transferred to DS4V). Uniform
weights are not calibration; all-zero entries and an all-uniform imatrix labeled
activation-derived are rejected. A nonuniform matrix alone does not prove its
provenance or quality. Preserve a hash and collection procedure in experiment
evidence. Each legacy imatrix entry may contain one input-width vector or
`input_width * source_expert_count` floats, with contiguous expert vectors.
Smaller expert-only pilots retain the original source expert count for this
validation and select the appropriate prefix of expert vectors. Each selected
expert must have positive total importance. `--absmax-only` and `--force` are
rejected for IQ85.

```sh
ds4_mix_converter --input /absolute/original-source --output /absolute/new.gguf \
  --recipe iq85 --imatrix /absolute/importance.dat \
  --imatrix-provenance uniform-unvalidated --plan-only > plan.json
```

Encoding uses canonical ggml IQ/Q8 encoders. `--encode-threads 1..16` (default 1)
parallelizes independent experts in ordered batches; lookup tables initialize
before workers launch. Each worker owns one encoded expert plus row scratch and
two source descriptors. With the actual 4096x2048 expert dimensions, encoded
payloads are 2.0625 MiB gate/up and 2.3125 MiB down, at most 37 MiB for sixteen
results. This excludes canonical IQ lookup/encoder scratch, source headers,
imatrix and thread stacks. Dense encoding uses one row plus its small FP8 scale
grid. No entire expert is expanded to F32. Worker failures drain launched jobs
before propagation. A fresh `.partial` path is exclusively created; complete
bytes/header and raw preserved tensors are verified before atomic no-overwrite
publication. Failed partial files remain for inspection and are never reused.

Build/test on the authorized remote Linux CPU host only:

```sh
cmake -S server/tools/ds4_mix_converter -B /absolute/fresh-build -DCMAKE_BUILD_TYPE=Release
cmake --build /absolute/fresh-build -j2
ctest --test-dir /absolute/fresh-build --output-on-failure
```

`test_ds4_iq_converter` covers dense preservation, canonical source-to-Q8 bytes,
an independent analytical FP8/scaling fixture crossing 128-row/column boundaries,
importance rejection/provenance, canonical IQ serial/8/16-worker identity with
nonuniform weights and a batch tail, and allocation/worker failure bounds.
`prove_iq85.py` runs exactly one layer and 1–17 experts (default 8), using 1, 8,
then 8 workers by default. `--reference-threads 1|8` and
`--parallel-threads 8|16` allow a bounded 8/16/repeat16 comparison after the
serial/8 qualification. It uses fresh output paths, timeouts, exact plan-size checks, hashes
and whole-file identity. It records binary/imatrix/source-index hashes and does
not build, run GPU code or convert a full model. Synthetic and bounded source
tests do not establish text/image quality or long-context performance; those
require separate runtime qualification against the existing model.
