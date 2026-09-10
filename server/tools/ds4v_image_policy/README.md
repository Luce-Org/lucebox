# DS4V image selection and raw visibility policy

This standalone C++17 unit is independent of the native tower. It adds no decoder,
loader, graph, image transport, or server wiring.

`select_image_experts` consumes finite, nonnegative F32 unbiased scores from
sqrt(softplus(router logits)) and finite image biases. It ranks score+bias, then
normalizes the selected **unbiased** scores in F32 and applies route scale 1.5.
The result owns bounded arrays for at most 256 experts. Invalid counts, pointers,
nonfinite values, overflowed corrected scores/sums, and zero selected sums fail
with a cleared result and an error. Input pointers must address `experts` readable
floats and must not alias the output. No router matmul or token/hash dispatch is
performed here. Equal corrected scores select lower indices first; this is an
explicit deterministic rule, not a claim of equal-score ordering parity with
Torch or every GGML kernel.

`raw_key_visible` tests one absolute query/key pair. An image range is
[IMAGE_START, IMAGE_END+1), excluding leading compression padding. Queries in that
range see the union of the complete range and ordinary causal sliding-window
positions. Queries outside it retain ordinary visibility. Pass (-1,-1) when no
range applies. Nonnegative query/key, positive window, and an absent or ordered
nonnegative range are required; invalid arguments return false with `visible`
cleared. The helper uses differences rather than query+1, including at INT64_MAX.
It owns no image block, allocates no mask, and changes no compressed-row policy.
The eventual request boundary must supply validated sequence/image spans; this
scalar predicate has no sequence length or source image-token budget to validate.

## Verification on soulf

Tests were committed first at e30c1b0. The remote Release build succeeded and
CTest failed with `valid selection rejected` before implementation. At 60ce979,
the same tests passed, covering unbiased weighting, selection bias, deterministic
ties, malformed contracts, causal/image boundaries, and near-INT64_MAX positions.
All build and fixture execution took place on soulf with at most two threads.

From the isolated `~/lucebox-ds4v-policy` checkout:

```sh
cmake -S server/tools/ds4v_image_policy -B /tmp/ds4v-image-policy-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ds4v-image-policy-build -j2
ctest --test-dir /tmp/ds4v-image-policy-build --output-on-failure
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ~/lucebox-ds4v-mix-fix/.venv-vision-reference/bin/python server/tools/ds4v_image_policy/verify_fixtures.py /tmp/ds4v-image-policy-build/ds4v_image_policy_probe ~/lucebox-ds4v-mix-fix/artifacts/routing-mask-reference ~/lucebox-ds4v-2/models/DeepSeek-V4-Flash-Vision-Uncensored artifacts/image-policy/fixtures
```

The fixture verifier checks every original fixture hash and the parent model.py
hash before use. It executes only the exact source `linear` function and the
score-producing AST statements from `Gate.forward`, using the saved original
F32 input/router fixtures. New scores and native outputs go to a separate
evidence directory; neither source fixtures nor the Python environment changes.
Production code does not depend on Python, Torch, NumPy, or GGML.

All 120 selected image expert indices match exactly across layers 0, 2, 3, 42
and all five image token kinds. Maximum absolute weight errors are 2.9802322e-08,
5.9604645e-08, 0, and 0 respectively, against the unchanged 1e-6 limit. First
three-layer fixtures demonstrate learned image selection differs from their
original text hash rows. This proves calling the image policy for all five kinds;
it does not prove integrated hash bypass.

Raw visibility matches all 433,562 query/key booleans exactly: 72,361 for the
one-image fixture and 361,201 for two images. This includes leading padding,
text before/between/after images, future-image isolation, and an image span longer
than the sliding window. The native probe writes full masks only for qualification;
the production predicate remains scalar.

Evidence: `~/lucebox-ds4v-policy/artifacts/image-policy/{red.log,green.log,fixtures.log,fixtures/verdict.json}`.
Verdict: **PASS for this unintegrated CPU policy unit.** GPU graph execution,
bias_vl loading, text routing preservation in the integrated backend, HTTP image
behavior, and full-model output remain outside this unit.
