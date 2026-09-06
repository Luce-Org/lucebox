# Server engine components

This document describes the server engine structure implemented in the current
code. It is a reference, not a roadmap.

## Ownership overview

```text
server_main
  ├─ builds BackendPlan
  ├─ creates ModelBackend
  ├─ transfers ModelBackend ownership to LuceEngine
  └─ constructs HttpServer, which borrows LuceEngine

LuceEngine
  ├─ owns ModelBackend
  ├─ owns the serving thread
  └─ selects one serving loop
       ├─ serial: HttpServer::worker_loop()
       └─ concurrent: HttpServer::scheduler_loop(SeqEngine &)

HttpServer
  ├─ owns sockets and HTTP protocol state
  ├─ owns the current ServerJob queue
  ├─ prepares model-ready GenerateRequest values
  └─ formats tokens and terminal results for clients
```

The backend outlives `HttpServer` because `server_main` constructs
`LuceEngine` before the server. `HttpServer` stops and joins the serving loop
before its transport and cache state is destroyed.

## Components

### `BackendPlan`

`BackendPlan` is the validated, normalized input to backend construction. Its
groups describe the effective model, placement, cache, speculation, runtime,
and architecture-specific values selected during startup.

The backend factory accepts a plan, projects its values into the selected
architecture config, and returns one owned `ModelBackend`. Architecture
configs own persistent strings; they do not retain pointers into the plan.

Files:

- `server/src/common/backend_plan.cpp`
- `server/src/common/backend_plan_internal.h`
- `server/src/common/backend_factory.{h,cpp}`

### `ModelBackend`

`ModelBackend` is the common model-resource interface implemented by each
architecture adapter. A concrete backend owns weights, caches, snapshots,
model-specific execution state, and any architecture-specific helpers.

Its complete-request generation operations remain the execution mechanism for
serial serving. A backend that supports continuous batching also exposes a
borrowed `SeqEngine`; that object is owned by the backend and remains valid for
the backend lifetime.

File: `server/src/common/model_backend.h`

### `LuceEngine`

`LuceEngine` is the runtime owner. It owns exactly one `ModelBackend` and at
most one serving thread.

Its current public lifecycle is:

```cpp
explicit LuceEngine(std::unique_ptr<ModelBackend> backend);

ModelBackend & backend() noexcept;
bool start_serving(ServingLoops loops, bool allow_concurrent);
void stop_serving();
```

`start_serving()` selects the concurrent loop only when both conditions hold:

- the caller permits concurrent local serving; and
- the backend exposes a `SeqEngine`.

Otherwise it starts the serial loop. The selected callback is owned by the
worker thread. `stop_serving()` requests shutdown and joins that thread before
returning. Destruction also calls `stop_serving()` and then destroys the owned
backend, whose concrete destructor performs backend shutdown.

Files: `server/src/engine/luce_engine.{h,cpp}`

### `HttpServer`

`HttpServer` owns the HTTP-facing state:

- listen and client sockets;
- request parsing and response formatting;
- SSE state and client-disconnect detection;
- tokenizer-dependent request preparation;
- prefix-cache policy and server status;
- the current intrusive `ServerJob` queue.

It borrows `LuceEngine` and, through it, a `ModelBackend`. The references are
valid for the complete `HttpServer` lifetime.

At startup the server supplies its established serial and concurrent loops to
`LuceEngine`. Upstream PFlash forwarding disables concurrent local serving,
so the serial worker remains selected for that configuration.

Files:

- `server/src/server/http_server.{h,cpp}`
- `server/src/server/scheduler.cpp`

### `GenerateRequest` and `GenerateResult`

`GenerateRequest` is the model-ready input shared by backend execution paths.
It owns every token sequence that may be retained during generation:

- prompt tokens;
- speculative hint tokens;
- stall-detection sequences;
- thinking-budget close tokens.

`GenerateResult` owns the completed token vector, timings, typed failure, and
generation metadata. Neither type contains HTTP or JSON state.

File: `server/src/common/generation_types.h`

### Generation channel

The generation channel provides transport-independent request and result
lifetime types:

```text
GenerationQueue::submit(GenerateRequest)
  -> Generation                     consumer handle

GenerationQueue::next()
  -> GenerateRequest + GenerationSource

GenerationSource
  -> TokenBatch
  -> coalesced GenerationProgress
  -> one GenerateCompleted result
```

`Generation` and `GenerationSource` are move-only. Dropping an unfinished
consumer cancels that generation. Losing its producer completes the consumer
with a typed failure. The queue bounds both waiting requests and buffered
tokens; shutdown rejects new requests and completes every live channel.

These channel types are implemented and tested. The live HTTP path still uses
`ServerJob` and the two existing serving loops, so the channel is not yet the
HTTP submission path.

Files:

- `server/src/engine/generation.{h,cpp}`
- `server/test/test_generation.cpp`

### `SeqEngine`

`SeqEngine` is the model-side capability used by concurrent serving. It owns
slot allocation, per-sequence model state, KV capacity, batched prefill and
decode execution, sampling, and retirement.

`HttpServer::scheduler_loop()` currently owns admission order, fair prefill
selection, cancellation, response construction, and non-blocking delivery.
All `SeqEngine` calls come from the single serving thread owned by
`LuceEngine`.

File: `server/src/common/concurrency/seq_engine.h`

## Runtime flow

### Startup

```text
CLI arguments
  -> BackendPlan::build()
  -> create_backend(plan)
  -> LuceEngine(std::move(backend))
  -> HttpServer(engine, tokenizer, config)
  -> LuceEngine::start_serving(...)
```

Backend validation and normalization happen before construction. Reporting,
tokenizer setup, and backend construction read the same effective plan values.

### Serial request

```text
client thread
  -> enqueue ServerJob
serving thread
  -> worker_loop()
  -> prepare prompt, cache state, GenerateRequest, and callbacks
  -> ModelBackend::generate() or restore_and_generate()
  -> stream or format GenerateResult
  -> complete ServerJob
client thread
  -> close request connection
```

Only the serving thread mutates model execution state.

### Concurrent request

```text
client thread
  -> enqueue ServerJob
serving thread
  -> scheduler_loop(SeqEngine &)
  -> admit requests into model slots
  -> select bounded prefill work and all live decode rows
  -> SeqEngine::step()
  -> buffer and flush client output without blocking other slots
  -> retire completed, failed, or cancelled slots
```

Admission order and slot identity remain stable until retirement. A slow or
disconnected client cannot block model progress for other live slots.

### Shutdown

```text
HttpServer::shutdown()
  -> set stopping flag and wake the request queue
  -> LuceEngine::stop_serving()
       -> invoke request_stop callback
       -> join serving thread
  -> close SSE clients
  -> drain remaining ServerJob values
  -> release server-owned cache and transport state
```

`stop_serving()` is idempotent and serializes stop/restart through the full
worker join. `HttpServer` also calls it when `run()` exits. `LuceEngine`
destroys the backend only after the serving thread has joined; the concrete
backend destructor owns its single shutdown call.

## Current boundary

The ownership boundary is active: `LuceEngine` owns the backend and execution
thread, while `HttpServer` owns transport. Request coordination has not fully
crossed that boundary yet: `ServingLoops` and the `ServerJob` queue connect the
runtime owner to the existing HTTP worker and scheduler.

The generation channel is therefore an available engine component, not yet a
live server entry point. This distinction is reflected in the types and tests
rather than hidden behind a second generation path.
