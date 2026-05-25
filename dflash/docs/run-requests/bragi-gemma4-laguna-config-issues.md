# Run request: bragi gemma4 + laguna config issues block local benchmarking

**Date opened**: 2026-05-25
**Status**: gemma4 chat-template fixed in `f1d30f2` (PR #269 candidate).
The gemma4 `ggml_can_mul_mat` crash root-caused: the entrypoint's
draft auto-resolver was picking the wrong draft GGUF — the qwen3.6
draft (`dflash-draft-3.6-q4_k_m.gguf`, fc.ne[0]=25600=5*5120) instead
of the proper gemma4 drafts (`dflash-gemma-4-{26b,31b}-*-q8_0.gguf`,
fc.ne[0] matches gemma4 hidden exactly). Code in `gemma4_backend.cpp`
is correct; the wrong draft just couldn't satisfy the mul_mat shape
contract. Fix landed as PR #277 (`fix(entrypoint): pick draft GGUF
that matches target architecture`). Laguna OOM asks remain unowned
(config / sidecar metadata, not engine).

## 1. Gemma4 — chat template / thought-token leakage in visible content

Gemma4-26b-A4B-it Q4_K_M loads cleanly on bragi (RTX 5090 MaxQ 24 GB)
and responds to most prompts coherently. But specific prompts trigger
internal `<thought>` / `<channel|>` tokens to leak into the visible
content stream rather than being routed to `reasoning_content`.

Reproduced with simple curl probes (`thinking: {type: "disabled"}` set
explicitly):

| Prompt | Visible content (truncated) |
|---|---|
| `"What is 2+2?"` | `'\n4//thought\n\n4\nthought\n\n4'` |
| `"Hello, world!"` | `'\nHello! How can I help you today?\nthought\n\n Hello! How can I help you today?node_modules/lodash/package.json:\n{\n  "name": "lodash"…'` |
| `"Continue: 1, 2, 3,"` | `'\n4deviserthought\n\n4, 5, 6, 7, 8, 9, 10...\n\nthought\n\n4\nthought\n\n4'` |
| `"What color is the sky?"` | clean ✓ |
| `"Translate to French: cat"` | clean ✓ |

The "thought" string appears verbatim and the model loops, sometimes
then drifting into unrelated content (node_modules JSON). Looks like
the `<channel|>` channel-marker token (id 101 per server log) is
making it into the rendered content when the chat-template renderer
doesn't suppress it, OR the SSE emitter's per-arch token mapping
doesn't catch the gemma4 channel tokens.

Server confirms gemma4 detected:
```
[server] gguf meta: general.name='Gemma 4 26B A4B It' general.architecture='gemma4'
[server] level-2 force-close: <channel|> = 101 (1 token), hard_limit_reply_budget = 512
```

Also: `usage.timings` reports `decode_ms: 0.0`, `decode_tokens_per_sec: 0.0`,
`prefill_ms: 0.0` for gemma4 — timing instrumentation appears to be
qwen35-only, not wired through Gemma4Backend yet.

### Asks for gemma4

1. **Chat template fix**: confirm `chat_template.cpp` renders gemma4
   prompts correctly when `enable_thinking: false` is set (or
   `thinking: {type: disabled}` per Anthropic shape). The garbage
   suggests the template is sending channel markers in the prompt
   that the model then echoes back.
2. **Per-arch token mapping in SSE emitter**: the `<channel|>` open
   tag (and `<|channel>` close, or whatever the pair is) needs the
   same "consume + transition mode" treatment that qwen3.6's
   `<think>` / `</think>` got. Right now it appears to fall through
   to raw text.
3. **Gemma4Backend timings instrumentation**: mirror sindri's
   `3b80fa8` (qwen35 `usage.timings` emit) into `Gemma4Backend::generate`
   so cross-arch perf comparison is possible.

### Validation
A clean smoke after the fix: `curl -d '{"messages":[{"role":"user","content":"What is 2+2?"}],...}'`
should return content `'4'` (or short coherent answer) with NO
`thought` substring. `usage.timings.decode_tokens_per_sec` should be
nonzero.

### Update 2026-05-25 — additional repro detail

After landing the unified bench harness (`--area code|longctx|agent`)
I re-probed gemma4 and confirmed three distinct failure modes:

1. **`//thought\n\n` token leakage** (the original report). Probing
   `"What is 2+2?"` returns
   `'\n2 + 2 = 4//thought\n\n2 + 2 = 4//thought\n\n2 + 2 = 4//thought\n\n2 + 2 ='`
   regardless of thinking mode. The literal substring `//thought\n\n`
   is being rendered to visible content. This isn't `<|channel>` or
   `<channel|>` (those have explicit SSE mappings in http_server.cpp
   lines 1410-1421); it's some other special token whose text content
   is `//thought\n\n`. Suggests the gemma4 vocab has additional
   thought-channel markers we haven't enumerated. Need to dump the
   gemma4 GGUF `added_tokens` list and add SSE mappings for any that
   collide with thought-block control sequences.

2. **`thinking: {type: enabled}` doesn't open a reasoning block** for
   gemma4. With thinking enabled, reasoning_content stays empty and
   the thought-leakage still happens in content. The
   `ChatFormat::GEMMA4` render path in `chat_template.cpp:185-215`
   doesn't reference thinking at all — no `<|channel>` opener emitted
   in the prompt to put the model in thinking mode.

3. **Hard crash on prompts ≥~160 tokens** (NOT just smoke-mc — any
   long-ish prompt). Reproduces deterministically with a single curl
   against image SHA `c344ad7c` (post-chat-template-fix `f1d30f2`,
   so this is independent of the chat-template work):

   ```bash
   curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model":"dflash",
     "messages":[{"role":"user","content":"Continue the following Python code. Output ONLY the function body — no markdown, no explanation, no extra prose:\n\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for"}],
     "max_tokens":256, "thinking":{"type":"disabled"}}'
   ```

   prompt_tokens=169, container exits with status 139 immediately
   after printing the stack:

   ```
   /src/dflash/deps/llama.cpp/ggml/src/ggml.c:3243: GGML_ASSERT(ggml_can_mul_mat(a, b)) failed
   ggml_mul_mat+0x59  [libggml-base.so.0]
   dflash_server+0x154860       ← gemma4 forward path
   dflash_server+0x1130cc
   dflash_server+0x1132d7
   dflash_server+0x10b33e
   dflash_server+0x10d097       ← worker_loop / chat handler
   dflash_server+0x5345a        ← thread entry
   ```

   Trivial prompts ("What is 2+2?", "Hello, world!") at <20 tokens
   succeed fine. Smoke-mc multi-choice at 91 tokens succeeds (per
   newer probe — see Update 2026-05-25 above). The crash threshold
   appears to be somewhere between 91 and 169 tokens, OR is shape-
   specific (Python code in the prompt may interact differently than
   plain English).

   `ggml_can_mul_mat(a, b)` failing on `mul_mat` means two tensors
   have incompatible shapes for matmul.

   **Root cause** (resolved 2026-05-25): addr2line on the unstripped
   binary inside the image resolves `dflash_server+0x154860` to
   `dflash::common::build_draft_graph` — the speculative-decode
   *draft* graph, NOT the gemma4 forward path. The entrypoint's
   draft auto-resolver was picking `dflash-draft-3.6-q4_k_m.gguf`
   (qwen3.6's draft, `fc.ne[0]=25600=5*5120`, target_hidden=5120)
   for every target, including gemma4 (target_hidden=2816 — doesn't
   divide 25600). The proper gemma4 drafts
   (`dflash-gemma-4-{26b,31b}-*-q8_0.gguf`, `fc.ne[0]=16896=6*2816`)
   exist in the same draft dir but never matched the entrypoint's
   `dflash-draft-*.gguf` glob.

   **Fix**: PR #277 — entrypoint derives target family from filename
   and prefers `dflash-{family}-*.gguf` first, falling back to the
   legacy glob. Verified end-to-end on bragi:
   - Before: HE prompt at 169 prompt_tokens → SIGABRT every time.
   - After: spec-decode loads correct draft, HE prompt emits clean
     `i in range(len(numbers)):...` Python continuation.

   No engine-side code changes needed — the gemma4 backend was
   correct all along; it was just being fed a wrong-arch draft.

These three issues block the bragi gemma4 column in our local
benchmark matrix. Until fixed, the only gemma4 data we have is
OpenRouter's hosted version (73-80% on ds4-eval — works fine
upstream).

## 2. Laguna XS.2 — OOM at default max_ctx on 24 GB VRAM

Laguna-xs.2-Q4_K_M loads 18.77 GiB of weights on RTX 5090 MaxQ
(24 GB), then immediately OOMs allocating the KV cache:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8160.04 MiB on device 0:
   cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 8556421120
cache failed: laguna cache: ggml_backend_alloc_ctx_tensors failed
[backend_factory] LagunaBackend init failed
[server] backend creation failed
```

18.77 (model) + 8.16 (KV at max_ctx=98304, tq3_0/tq3_0) ≈ 26.93 GiB,
exceeding 24 GB.

### Asks for laguna

1. **Document max_ctx limits in the laguna sidecar**: add a
   `"recommended_max_ctx_by_vram"` table (or similar) so the
   entrypoint can derive a safe default. e.g.

   ```json
   "recommended_max_ctx_by_vram_gib": {
     "24": 32768,
     "32": 65536,
     "48": 98304
   }
   ```
2. **OR**: tune the laguna container default to use `max_ctx=32768`
   (which fits in ~2.7 GiB KV + 18.77 GiB model = 21.5 GiB, fits 24 GB)
   and bump only when the host card has more headroom.
3. **Documentation**: lucebox profile / model card README should
   mention "Laguna XS.2 (Q4_K_M) needs ≥32 GB VRAM at default 98k
   context; bragi-class 24 GB cards must lower to 32k".

## Impact

Blocking the master sweep's gemma4-26b / gemma4-31b / laguna-xs.2
columns. Until these land, lucebox's bench coverage matrix is:

| Backend | luce-dflash bragi |
|---|---|
| qwen3.6-27b ✓ | full coverage (nothink, think4k, think65k, forge) |
| gemma-4-26b   | blocked — chat template / token leakage |
| gemma-4-31b   | blocked — likely same as 26b |
| laguna-xs.2   | blocked — OOM at default max_ctx on 24 GB |

OR + vidar reference baselines exist for all four models, but the
local luce-dflash measurements that would show quant impact /
per-platform perf can't run until these are fixed.
