# Model load balancing

Model load balancing lets one `dflash_server` process serve requests across
models on different GPUs through one HTTP listener. It prefers the configured
primary and sends new requests to a secondary when the primary cannot accept
them. If all eligible models are busy, requests enter a bounded waiting queue.

Model placement, primary GPU selection and enabling load balancing are separate
settings. For example, Qwen can stay on R9700 and DeepSeek4 on Strix Halo while
either GPU is selected as primary. Every generation request follows this
capacity-based policy, regardless of the request's `model` field. The response
identifies the model that actually generated the answer.

Each model keeps its own tokenizer, defaults, backend and scheduler. Clients
use the existing Chat Completions, Messages and Responses endpoints.

## Configure placement and load balancing

Model placement and load-balancing priority are independent. Each `--model <path>`
begins a model option block; its `--target-device <backend:gpu>` chooses where
that model loads. The first path may also be positional for single-model CLI
compatibility. The first block owns listener options and `--routing-queue-limit`.

`--load-balancing-primary-gpu <backend:gpu>` selects the primary by its
configured target device, independently of block order. It must match exactly one block; if
omitted, the first block is primary. Use explicit device IDs after checking
which GPU each ID names; the example assumes R9700 is `hip:0` and Strix Halo
is `hip:1`.

`--load-balancing` enables primary-first fallback. Without it, only the selected
primary loads and serves requests, even if other model blocks are configured.
With it, all configured models load and the remaining blocks are tried in their
original order after the primary. The old `--load-balancing-next-model` and
`--next-model` flags are removed.

```bash
dflash_server --load-balancing --load-balancing-primary-gpu hip:0 \
  --model /models/qwen38-27b.gguf \
  --model-name qwen --target-device hip:0 \
  --draft /models/qwen38-dflash2.gguf --draft-device hip:0 \
  --max-ctx 4096 --max-concurrency 4 --kv-pool-tokens 16384 \
  --cache-type-k q8_0 --cache-type-v q8_0 --fa-window 0 \
  --default-max-tokens 2048 --hard-limit-reply-budget 1024 \
  --host 127.0.0.1 --port 8080 --routing-queue-limit 32 \
  --model /models/deepseek-v4-flash.gguf \
  --model-name ds4 --target-device hip:1 \
  --max-ctx 8192 --max-concurrency 1 \
  --default-max-tokens 2048 --hard-limit-reply-budget 1024 \
  --prefix-cache-slots 0 --prefill-cache-slots 0 --disk-prefix-cache off \
  --ds4-fused-decode --ds4-expert-top-k 6 --ds4-prefill exact
```

Change only `--load-balancing-primary-gpu hip:1` to prefer DS4 on Strix Halo
while keeping both model placements intact. Remove `--load-balancing` to serve only that
selected primary.

Use the installed model paths and a binary built for both GPU architectures.
Here Qwen uses continuous batching and DS4 uses its existing native worker.
The latter has one active request; raising a supported model's concurrency
selects its paged scheduler. `--max-concurrency` is per model, not a shared
limit or a promise that every sequence will fit in the KV pool.

```json
{"model":"auto","messages":[{"role":"user","content":"Write a Python parser."}],"max_tokens":512,"stream":true}
```

## How requests are balanced

- Every generation request follows operator-configured priority. An omitted,
  empty, `auto`, explicit model name, or other client alias has the same
  load-balancing behavior. There is no model pinning. With balancing enabled, requests try the
  primary and then the remaining models; without it, only the primary serves.
  This is preference with capacity fallback, not round-robin.
- The listener reserves an available model slot. Its scheduler then calls
  `SeqEngine::admit` on the worker thread, which atomically checks actual
  slots, prompt KV reservation and existing decoders' headroom. The listener
  never reads a live KV pool from another thread or estimates free VRAM.
- A busy engine returns the request before any SSE header, output or prefill.
  The listener retries with the next model, re-rendering and re-tokenizing the
  original request and applying that model's defaults. Response `model`
  identifies the model that actually generates.
- If all eligible targets are busy, requests wait in memory, up to
  `--routing-queue-limit` (default 32; 0 rejects immediately). Capacity release
  wakes waiters; a 250 ms retry also observes engine-only changes, disconnects
  and shutdown. Waiting does not consume a GPU sequence slot. Waiters compete
  for the next available reservation; there is no FIFO or starvation guarantee.
- Context/pool limits that can never fit skip that target; when none can fit,
  the request returns 400 instead of waiting.
  Other validation and backend failures terminate the request without fallback.
- A reservation lasts through engine retirement and output draining. A slow
  reader can therefore retain admission capacity after generation finishes.
  Cancellation does not expose a slot while its GPU work is still running.
- Waiting ends when capacity becomes available, the client disconnects, or the
  server stops. There is no server queue deadline or pre-admission SSE heartbeat;
  clients/proxies may impose their own timeout. Shutdown answers waiters with 503
  and joins all worker/client users before releasing model state. Active native
  single-request workers finish generation and send their normal terminal
  response; a real client disconnect still cancels them. Batched workers keep
  their existing shutdown cancellation and error-finalization policy.

`/v1/models` lists loaded model names. With balancing disabled, only the
selected primary is loaded; other configured model blocks are absent from
this list. With balancing enabled, token-counting requests require an
explicit name because tokenizers differ; naming the primary here always counts
with its own tokenizer and never falls back. `/props` and `/status/json` expose
`routing: "primary-first"`, `waiting`, `queue_limit`, and each model's
`capacity`, `in_flight`, execution mode, placement and existing status.
A single-model launch uses the ordinary server path and reports its configured
model identity, regardless of the client alias.

With balancing enabled, model names must be unique, nonempty and cannot be
`auto`. With balancing enabled, startup rejects model-block
options that mutate shared process policy, including KVFlash, Spark, PFlash,
remote drafts and target splitting. Environment variables remain process-wide;
this does not establish arbitrary model/environment isolation. Qwen's explicit
KV types are stored per backend rather than in shared environment state.
With balancing disabled, the selected primary retains single-model CLI
features. Unused blocks are syntax-checked but do not apply process policy.
GPU draft top-K and sampling scratch belong to each worker thread and are
released on their owning device when the worker exits.

## Active requests and load-balancing limits

Load balancing selects a model before generation starts. Once admitted, a
request remains on that model. If decode later exhausts its pool, the existing
scheduler error and retirement behavior still applies. Load balancing does not
move an active stream to another model or recover it from that failure.

Qwen and DS4 have incompatible tokenizers, cache layouts and recurrent state.
Moving an active Qwen request to DS4 would require replaying its conversation
and accepted generated text, then continuing with a different model. Before
implementing that, define stream/model identity, assistant continuation,
thinking/tool-call boundaries, output budgets and cancellation across replay.
Raw KV copying cannot implement this handoff.

Active SSD suspension is not a small extension of the existing prefix cache:
`Qwen35Backend::snapshot_save` rejects paged attention, DeepSeek4's paged path
cannot export/adopt its monolithic snapshots, and `SeqEngine` has no active
sequence suspend/restore API. Those contracts must exist before eviction can
safely preserve a streaming request.

KVFlash currently pages cold attention chunks to host RAM. It is neither a
continuous-batching sequence checkpoint nor an SSD request spool. Integrating
it requires sequence-owned paging state and reclaim/restore synchronization.
Exact same-model suspension also needs KV plus recurrent/compressor state,
accepted/pending tokens, RNG/sampler state, speculative state and response
formatting state. SSD suspension additionally needs atomic durable records,
model/version validation, disk quotas, cancellation cleanup and crash recovery.
None of those mechanisms is enabled by this draft. Requests waiting here have
not begun generation and retain the parsed request in memory only.

## Provenance and checks

The single-listener and per-model loader are adapted from the local runtime
behind [research PR 105](https://github.com/Luce-Org/luce_box/pull/105), commit
`63151ff8`. [Research PR 103](https://github.com/Luce-Org/luce_box/pull/103)
records earlier batched and hybrid serving attempts. This branch adds
model load balancing through primary-first admission, capacity feedback and
bounded waiting;
it does not import the planning, synthesis or benchmark workflows.

`test_model_routing.cpp` exercises real HTTP with deterministic host backends:
independent workers, distinct tokenizers/defaults, primary preference, slot and
KV fallback, permanent rejection, queue overflow/drain, cancellation and
shutdown. `test_seq_slot_manager` verifies the real pool's atomic reservation
and permanent-capacity result. Hardware qualification is reported in the PR;
these host checks alone do not establish numerical GPU correctness, long-context
survival, a latency improvement or production readiness.

### Model load-balancing checks (2026-09-09)

The updated host suite passed 496 checks. Coverage exercises primary-first
load balancing for omitted, automatic, explicit and unknown client aliases; reversed
priority; fallback before response headers; bounded waiting; and cancellation.
Twenty-one CLI checks cover separate model blocks and balancing enablement,
invalid or ambiguous primary selection, GPU index overflow, queue validation,
and rejection of the removed separator flags.

Functional GPU results and launch provenance for the modular interface are
retained under `profile-runs/primary-capacity-routing-20260909/modular/` in the
hub workspace. The R9700-primary run verified fallback, bounded waiting, queue
overflow, cancellation and capacity reuse. The reversed-priority run kept Qwen
on R9700 and DS4 on Strix Halo: a request naming Qwen went to DS4, and a request
naming DS4 fell back to Qwen while DS4 was occupied. With balancing disabled,
both blocks were configured but only the selected Strix Halo model loaded and
served, including requests naming Qwen. All three runs exited cleanly and left
no KFD compute processes.

### Earlier functional qualification (2026-09-09)

The following results predate the modular flags and removal of model pinning;
they describe the earlier implementation, not verification of the current CLI.

Runtime commit `8d550fcf` passed 500 host checks and 11 CLI checks. The HIP
binary includes `gfx1201` and `gfx1151`; the observed devices were R9700 at
`0000:c4:00.0` and Radeon 8060S/Strix Halo at `0000:c5:00.0`.

| Run | Observed result |
| --- | --- |
| Opted-in Qwen C4, 16,384 shared KV tokens; native DS4 C1 | Primary filled to 4/4. A request naming Qwen fell back to DS4. Another primary-named request waited, queue overflow returned 503, and cancellation admitted the waiter to Qwen. |
| Same models, Qwen pool reduced to 1,024 tokens | After the primary emitted a token for a 685-token prompt, a second 685-token prompt fell back with only 1/4 primary slots occupied. DS4 independently tokenized it to 674 tokens. |
| Small deterministic output control | Primary-name and automatic routing returned `routing-ok` on Qwen. Primary-name fallback matched explicitly named DS4 on that prompt. This is not a broad numerical or quality qualification. |
| No opt-in separator | Only Qwen was listed and served the request; the ordinary single-model path had no routing metadata. |
| Cancellation and cleanup | Both balanced runs reused capacity after cancellation. All three runs exited 0 and left no KFD compute processes. The pressure run canceled DS4 during prefill; it does not establish successful long-prompt completion. |

Artifacts, exact launch arguments, binary hashes and raw responses are retained
locally under `profile-runs/primary-capacity-routing-20260909/opt-in/` in the hub
workspace. No DSpark draft was enabled for DS4 in these runs. Active-request
migration, decode-growth recovery, SSD resume and throughput benefits remain
unimplemented or unmeasured as described above.
