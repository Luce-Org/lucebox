# PFlash tests and replay tools

The dedicated `test_pflash_*.cpp` files live here and remain part of the
`test_server_unit` executable. Moving them does not change test names or CTest
discovery. The shared HTTP server harness and its PFlash serving regressions
remain in `test/unit/test_server_unit.cpp`.

`fixtures/` holds the captured agent-client requests and generated request-probe
parity cases. The fixture generator is documented in
[PFlash development tools](../../scripts/pflash/README.md).

The helper executable keeps its target name:

- `pflash_daemon`: drafter-only capture and compression protocol.

```sh
cmake --build server/build --target test_server_unit pflash_daemon -j
server/build/test_server_unit pflash
```

The general GPU-copy and Qwen causal-mask regression tests stay under
`test/unit/`: they check backend behavior also used outside PFlash.
