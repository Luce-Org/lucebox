# Pure DS4V image prompt preparation

`prepare_image_prompt` accepts final rendered/tokenized text and ordered
`ImagePatchInput` values containing an existing `ResizePlan` and raw BF16 patches.
It returns one owning value with expanded token IDs and `PromptImage` records,
each binding copied plan/patches to the existing core `ImageLayout`/`TokenSpan`.
There is no RGB retention, caller-supplied layout, second span representation,
model payload abstraction, decoder, tower, HTTP, backend, or cache integration.

The operation enforces the fixed vocabulary 129280 and marker 129264 contract.
It checks final marker cardinality even with zero images, rejects negative or
external-vocabulary input tokens, and accepts at most four images. No-image,
no-marker requests keep their token vector unchanged and have no image records.
The caller remains responsible for obtaining these final tokens from the verified
native tokenizer and for transport schema checks before this operation.

Plans must have consistent positive patch/aligner grids and resized dimensions.
The accepted core's layout builder at position zero verifies the complete source
image budget with all three possible leading pads reserved. Patch word count must
match the grid, and every raw BF16 value must be finite and in [-1,1]. This checks
the normalized tensor contract, not image provenance or whether each value came
from a particular byte-valued source pixel. The unit does not rerun resizing.

Each final layout is then generated at the current expanded position, so prior
images affect later offsets and leading padding correctly. All five image kinds
become vocabulary+kind IDs. The core's whole-block bounds include leading pads;
its narrower visibility bounds start at IMAGE_START and include IMAGE_END.

Limits accept context capacities through INT_MAX, a nonnegative output reserve
not exceeding that capacity, and a positive expanded-token limit capped at
1,048,576 (default 131,072). The hard ceiling bounds memory independently of a
misconfigured context limit. Size checks use uint64 and subtraction before token
allocation or narrowing. Exact context fit passes; compression grants no
exception. All inputs/layouts/counts are validated before the final result is
published. Errors carry bounded category/index text and no partial tokens or
image records. Default value copies own independent patch storage; move ownership
survives destruction of the original transport/input object.

## soulf CPU verification

The tests and a compiling stub were committed first as c24e823. The isolated
remote Release target built and CTest failed with `FAIL: text path`. Implementation
a2489a4 passed the same tests and original source fixtures. The final revision adds
explicit equal-layout separate-request isolation and additional bound cases.

Run from the isolated `~/lucebox-ds4v-prompt` checkout:

```sh
cmake -S server/tools/ds4v_image_prompt -B /tmp/ds4v-image-prompt-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ds4v-image-prompt-build -j2
ctest --test-dir /tmp/ds4v-image-prompt-build --output-on-failure
python3 server/tools/ds4v_image_prompt/verify_fixtures.py /tmp/ds4v-image-prompt-build/ds4v_image_prompt_fixtures ~/ds4v-work/ds4v-preprocess-fixtures-final artifacts/image-prompt/fixture-verdict.json
```

This source-only CMake target compiles the new helper and the accepted
`deepseek4_vision_preprocess.cpp` core. It has no FetchContent, codec, GGML, GPU,
or Python package dependency. The fixture wrapper uses Python's standard library
and checks all 28 saved carrots/corn fixture hashes before the native probe.
The source fixtures are reused read-only; no codec or unrelated RGB tests rerun.

Tests cover every start residue; text before/between/after images; consecutive
images; all cardinality errors including zero-image marker injection; all five
forged external image IDs; malformed plans/patches; the source full-layout budget;
finite normalized BF16 bounds; image counts; exact-fit/context-overflow and
near-INT_MAX limits; transactional second-image failure; independent copies,
moves, and separate same-layout image storage.

Ten original-image cases compare exact types, permutation, absolute spans,
expanded image IDs, unchanged text, and original BF16 patch bytes: carrots and
corn at starts 0, 1, 2, 3, 127. Both consecutive-image orders also pass, using
source fixture lengths to independently establish second starts 109 and 313.
No expected layout is calculated with the implementation under test.

Evidence lives in `~/lucebox-ds4v-prompt/artifacts/image-prompt` with RED/GREEN
logs, fixture logs/verdict, exact final source commit and binary hashes. This is
**unintegrated CPU preparation proof**. The final integration owner must preserve
the result through requests/retries, prevent later token rewrites, add backend
materialization and execution guards, and implement image-safe usage/cache/status
handling. Native tower numerical qualification remains open.
