# Server tests

## Layout

Tests are grouped by the component they exercise. Host-only and backend tests
can live together; their existing CMake targets determine dependencies and hardware
requirements. Moving a source does not change its test name or executable.

| Directory | Responsibility |
| --- | --- |
| `server/` | Request parsing, streaming, tools, chat templates, caches, sessions and admission |
| `common/` | IPC, sampling, backend result state, restore deltas, mapping and shared shape/type checks |
| `moe/` | Shared expert placement, routing, swaps, storage, graphs and streaming |
| `qwen3/` | Drafter loading, anchors, query capture, score range and buffer planning |
| `qwen35/` | Positions, recurrent rollback, tensor parallelism, tree guards and tracing |
| `deepseek4/` | DS4 loading, budgets, registry cleanup, tracing and numerical kernels |
| `kvflash/` | Placement reservation, pool sizing and chunk scoring |
| `placement/` | Device placement, residency and layer-split delegation |
| `draft/` | Model-independent draft tree policies |
| `kernels/` | Kernel scheduling policy |
| `support/` | Shared fixtures and their tests |

The historical `qwen35moe` test filenames live in `moe/` because they exercise
shared MoE implementations. `deepseek4/` matches the production directory name;
its existing target names may use either `ds4` or `deepseek4`.

The `test_server_unit` target still aggregates the model-free component sources.
Existing CTest selectors, including the historical `ServerUnitFixture` names,
remain stable. The standalone `test_feature_gate` reports each rule group
individually and builds without the backend stack. Other standalone, benchmark,
smoke and kernel tests remain at the top level with their current targets.

Shared fixtures in `support/` cover assertions, portable environment operations,
tool schemas, streaming constructors and a minimal mock backend. Keep helpers
with only one consumer in that test's `.cpp`. Include production dependencies
explicitly. `fixtures/` holds test data; the common runner and asset helpers remain
at the top level. Source lists in `server/CMakeLists.txt` are explicit, so adding
a file alone does not register a test.

## Run

From the repository root, with an already configured build directory:

```sh
cmake --build build --target test_server_unit -j 4
ctest --test-dir build -R 'test_server_unit\.' --output-on-failure -j 4
./build/test_server_unit --discover_tests
./build/test_server_unit QueryCaptureFixture
```

The build still links the configured backend, but the aggregated component cases
require no downloaded model or GPU execution. CUDA configurations additionally
link backend-specific tests from the top level; those retain their own hardware
requirements. Use `cmake --build build --target check` for the broader configured
suite, including standalone targets and any enabled hardware tests.

## What belongs in a unit test

Call the production helper or component and assert its observable result. Keep
checks of boundaries, error cleanup, ownership, serialization, numerical results,
and supported backend policies. A small test is useful when its failure identifies
a broken contract.

Do not copy a production formula into a lambda, simulate an allocator with a fake
vector, build the expected response JSON and assert against that same JSON, or
assign a field just to read it back. Those tests pass when production regresses.
Keep backend and model integration tests for contracts that cannot be reached
through the existing unit interfaces.

The earlier cleanup removes simulated PFlash threshold/interpolation and continuation
checks and hand-built non-streaming telemetry responses. These did not cover the
HTTP path. End-to-end coverage of those policies and response fields must drive
the real request handler; the remaining unit tests do not establish that coverage.
