# Validation v2

This package is the fail-closed preparation and analysis layer for the July 2026
Lucebox validation campaign. It does not contain a benchmark headline and does
not treat a failed GPU process as a completed experiment.

Key contracts:

- A matrix cell is complete only when its newest attempt is `success`, hashes
  match the plan, and every required metric exists.
- Prompt or benchmark case is the independent statistical unit. Repetitions are
  nested technical replicates.
- NIAH uses protocol-native final content and exact normalized equality. The
  expected string appearing inside a longer response is a failure.
- Publication provenance refuses dirty trees and hashes every executable input.
- Raw outputs remain available for rescoring; task-specific answer recovery is
  not part of a primary quality score.

Generate the deterministic 100-case design plus 20 controls:

```bash
python -m harness.validation_v2.cli niah-generate \
  --context-tokens 32768 \
  --tokenizer <immutable-hub-tokenizer-repo> \
  --tokenizer-revision <immutable-hub-commit> \
  --controls \
  --out .validation-v2/niah-32k.jsonl
```

Strictly join and score one hash-bound final response per generated case:

```bash
python -m harness.validation_v2.cli score-niah \
  --cases .validation-v2/niah-32k.jsonl \
  --responses .validation-v2/niah-32k-responses.jsonl \
  --out .validation-v2/niah-32k-scores.json
```

Each response row must contain `prompt_id`, the matrix `prompt_hash`, and
protocol-native `final_content`. Missing, duplicate, unplanned, or hash-mismatched
responses abort scoring; longer strings containing the expected answer still fail.

Validate that every planned matrix cell succeeded:

```bash
python -m harness.validation_v2.cli validate-matrix \
  --plan .validation-v2/matrix.jsonl \
  --results .validation-v2/results.jsonl \
  --require-metric decode_tok_s \
  --require-metric ttft_ms
```

Create the prompt-clustered comparison:

```bash
python -m harness.validation_v2.cli summarize-paired \
  --results .validation-v2/flat-results.jsonl \
  --baseline target-ar --candidate local-approx-ddtree \
  --metric decode_tok_s --out .validation-v2/summary.json
```

Official RULER and LongBench runners remain external pinned inputs. Their commit,
dataset revision, prompts, generation settings, and official scorer hashes must
be recorded in the campaign manifest rather than copied or silently modified.

`protocol.json` freezes the campaign-wide revisions and measurement contract.
`matrix.py` expands a sealed prompt corpus and configuration list into stable,
append-only experiment cells before any GPU output is inspected.

The local Lucebox and upstream llama.cpp DFlash draft GGUFs are separate pinned
inputs. They advertise different architecture identifiers and are not
interchangeable. The MTP comparator likewise uses the pinned MTP-bearing target
GGUF; a target without embedded MTP layers must fail that cell rather than fall
back to ordinary autoregressive decoding. SHA-256 values for all three model
families are part of the protocol.

Build the independent tuning and held-out decode corpora from hash-verified,
commit-pinned public sources:

```bash
python -m harness.validation_v2.corpora
```

This produces the external V100 study's CC0 40-prompt set for tuning and a
balanced ten-family, 100-prompt held-out set from the CC-BY-NC-4.0 Stanford
Alpaca corpus. The source output/labels are deliberately excluded: these files
measure deterministic decode behavior and latency, not task accuracy.
The expected generated hashes are frozen in `protocol.json`; a different hash
is a new campaign, not a resumable continuation of this one.

NIAH generation loads the tokenizer at the supplied immutable Hub revision and
uses its chat template for every count; character-count estimates are rejected
for publication cases. Generated rows record both the requested and actual token
count and the tokenizer-measured needle depth.

`http_runner.py` executes a pre-generated matrix against an OpenAI-compatible
endpoint. It appends every failure/timeout, retries non-success cells on resume,
requires authoritative server token counts, retains rescorable raw responses,
and reads any bearer credential from an environment variable without recording
it in the request artifact. Each configuration must also declare
`expected_configured_contract`, `expected_effective_decode_mode`,
`expected_fallback_reason`, `expected_draft_attached`, and
`expected_draft_loaded`. The runner authenticates to `/props`, verifies those
launch facts against the nested `usage.speculative` response, and refuses to
mark a row complete if any exactness evidence is absent or inconsistent.
Decode throughput comes from `usage.timings.decode_tokens_per_sec`, not total
HTTP wall time. Streaming cells require both a terminal finish reason and the
`[DONE]` sentinel. Raw response artifacts are attempt-qualified, so a retry can
never overwrite a failed attempt. Run concurrency is explicit with
`--concurrency`; every cell must declare the same planned `concurrency` value.
`fixed_work` cells must request `ignore_eos` and still produce exactly
`max_tokens`, while `natural_stop` cells preserve the model's stop behavior.
Parity hashes cover the canonical assistant `reasoning_content` and final
`content` fields together; hashing visible final content alone is forbidden.
Quality scorers still receive final content as a separate field.

Upstream llama.cpp cells set `runtime_family` to `llama.cpp` and must pin the
exact `/props` `build_info`, model alias, and planned speculative type. Because
upstream `/props` does not expose the active speculative route, target-only
cells must report zero draft tokens and DFlash/MTP cells must report positive,
internally consistent `draft_n` and `draft_n_accepted` response timings. These
routes are recorded as `external-unverified`; output parity is evaluated from
the independently retained response bytes and is never inferred from draft
activity alone. Lucebox cells retain the stronger nested launch/response
contract checks described above.
