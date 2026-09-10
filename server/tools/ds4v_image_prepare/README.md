# CPU image preparation composition probe

This qualification-only probe composes accepted data-URL extraction, JPEG/PNG
decode, source RGB preprocessing, the existing built-in DeepSeek4 renderer and
native tokenizer, and owned prompt expansion. It also executes the existing
Jinja renderer for marker-preserving, dropped-marker, and repeated-marker
controls. The tokenizer loads metadata from the converter smoke GGUF; no model
weights, GGML backend, HTTP server, or vision tower are initialized.

`compose.cpp` contains an explicit tiny adapter from normalized text parts to
ChatMessage. It supports only the simple roles/string/text-part fixtures here.
It is **not HttpServer::normalize_chat_messages**, and does not implement its
ToolMemory replay, Responses or Anthropic conversion, request copies, queueing,
compression, cache, snapshot, usage serialization, or generation behavior.
Production component implementations and APIs are unchanged.

The adapter counts markers after real rendering/tokenization, including zero-image
requests, before decoding. It constructs its result transactionally. Transport,
decode, preprocess, and expansion failures discard accumulated records and return
bounded category/index messages. Decoded RGB is retained only for this probe's
byte comparisons; the production prompt payload still contains plan/patches/layout.

## Tests and limits of proof

- Real source carrots/corn JPEG data URLs in both orders, with text before,
  between, and after them. Decoded RGB, BF16 patches, shapes, layout kinds,
  permutation, generated IDs, and absolute spans match immutable source fixtures.
  All ordinary rendered tokens remain unchanged around expanded blocks.
- Both orders expand to 436 tokens in the fixture's built-in DS4 prompt. The first
  block starts at 9; the second starts at 119 for corn then carrots, and 323 for
  carrots then corn. These positions arise from actual rendering/tokenization.
- Exact expanded context fit succeeds and one-token overflow fails, using the
  image-specific preparation helper. Existing text compression admission is not
  executed or qualified here.
- Jinja preserves an image marker successfully, while dropped/repeated markers
  fail. A Jinja-injected marker and a tool-schema object key containing the marker
  fail with zero extracted images. This tests the final token boundary that an
  earlier string-value scan can miss.
- A valid first JPEG followed by a truncated second JPEG fails transactionally.
  Redaction occurs before JSON serialization and removes all data URLs.
- Two separately encoded 19x11 solid PNG requests have identical expanded IDs
  but different patch bytes and independent mutable storage. They are synthetic
  test inputs generated in memory with the pinned LodePNG encoder, not saved
  user artifacts.
- Ordinary text follows the real built-in renderer/tokenizer unchanged.

The fixture wrapper verifies all 28 original carrots/corn fixture hashes before
execution. Logs contain counts, shapes, bounds and hashes, never encoded images
or full prompt data. No source fixture or Python environment is modified.

The initial test/stub commit was 1096fab. Linking the existing Jinja engine also
requires its existing common/unicode.cpp; the harness wiring correction a5aa8a3
then built and recorded the intended RED `text composition failed`. The minimal
adapter at 15a10c5 passed the unchanged interaction cases. Final checks also
assert/report source dimensions and explicitly disable additional GGML providers.

## Reproduce on soulf only

From isolated `~/lucebox-ds4v-cpu`:

```sh
cmake -S server/tools/ds4v_image_prepare -B /tmp/ds4v-image-prepare-build -DCMAKE_BUILD_TYPE=Release -DDS4V_TOKENIZER_GGUF=$HOME/lucebox-ds4v-mix-fix/artifacts/fitter-fix/smoke.gguf -DDS4V_SOURCE_FIXTURES=$HOME/ds4v-work/ds4v-preprocess-fixtures-final
cmake --build /tmp/ds4v-image-prepare-build --target ds4v_image_prepare -j2
python3 server/tools/ds4v_image_prepare/verify.py /tmp/ds4v-image-prepare-build/ds4v_image_prepare ~/lucebox-ds4v-mix-fix/artifacts/fitter-fix/smoke.gguf ~/ds4v-work/ds4v-preprocess-fixtures-final artifacts/cpu-composition/verdict.json
```

The wrapper is the authoritative hash-checking invocation. CTest is also
registered as `ds4v_image_prepare_composition` when both fixture paths are set.
Builds use two jobs; the native composition execution is single-threaded. Only
ggml-base/gguf is linked, with compute backends disabled. The accepted preprocessing
CMake target supplies the pinned libjpeg-turbo and LodePNG dependencies and its
two-job external build. JSON fallback uses the root server's pinned 9cca280a
archive with its SHA256. No server root build is configured.

Retain [the existing preprocessing third-party notices](../ds4v_preprocess_probe/THIRD_PARTY_NOTICES.md)
for Pillow/libjpeg-turbo/LodePNG, and the vendored llama.cpp license. This target
reuses those implementations and pins rather than copying codec code.

Evidence: `~/lucebox-ds4v-cpu/artifacts/cpu-composition` contains RED/GREEN logs,
final source commit, source/binary hashes, dependency/backend configuration, and
fixture verdict. PASS means **CPU preparation composition with the explicit probe
adapter**, not integrated HTTP image input, image routing/attention execution,
vision semantics, tower parity, or paired-GPU performance.
