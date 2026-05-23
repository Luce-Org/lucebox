# Harness followup items

## accept_rate from server log

`harness.metrics_parser.extract_accept_rate_from_log()` now recognizes both the
JSON `[pflash-bandit] {...}` form and the native C++ server's plain-text
`[pflash-bandit] ... accept=...` lines, so live bandit CSV rows can populate
`accept_rate` without changing the server.

## Hermes config bug skip

`HermesAdapter.preflight_check()` intentionally returns
`HERMES_CONFIG_BUG: see .notes/harness-followups.md` until the adapter learns
to write the canonical temp config from `run_hermes.sh`.

## OpenCode provider config skip

`OpenCodeAdapter.preflight_check()` intentionally returns
`PROVIDER_CONFIG_BUG: opencode.json model registration not yet working — see
.notes/harness-followups.md` until the provider registration is fixed.

## codex /v1/responses request shape mismatch

The plan (section 6) flags that codex's `/v1/responses` path has a different
request shape (`input`, `metadata`). The stub server accepts it but the real
C++ server may reject it. File a separate issue if the live codex run fails on
this route.

## pi + codex PATH bootstrap

`run_pi.sh` and `run_codex.sh` are deleted; their path bootstrap fix (from
`project_ee7_multiclient_validated`) needs to be reproduced in the respective
adapter `env_overrides` if the PATH fix was applied to the bash scripts. Check
before running live against real binaries.

## ResourceWarning in test output

The ThreadingHTTPServer proxy leaves dangling socket FDs during test teardown.
The `_start_proxy()` helper returns a `ThreadingHTTPServer` which shuts down
properly but the HTTP connection socket isn't explicitly closed. Low priority —
tests pass, warnings are cosmetic.

## LOC delta vs plan estimate

Plan estimated net negative LOC. Actual: +770 LOC. The test suite (~500 LOC) is
responsible. The bash deletions (375 LOC) don't outweigh the test investment,
which is correct — the tests are the whole point.

## Native live run blocker in this sandbox

The harness side is fixed, but the native `dflash_server` live bandit path
cannot complete here because the sandbox exposes no CUDA-capable device.
The server reaches backend initialization and then exits with
`ggml_backend_cuda_init` failure.
