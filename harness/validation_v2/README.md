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
  --tokenizer-revision <immutable-hub-commit> \
  --controls \
  --out .validation-v2/niah-32k.jsonl
```

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

Build the independent tuning and held-out decode corpora from hash-verified,
commit-pinned public sources:

```bash
python -m harness.validation_v2.corpora
```

This produces the external V100 study's CC0 40-prompt set for tuning and a
balanced ten-family, 100-prompt held-out set from the Apache-2.0 Stanford
Alpaca corpus. The source output/labels are deliberately excluded: these files
measure deterministic decode behavior and latency, not task accuracy.

`http_runner.py` executes a pre-generated matrix against an OpenAI-compatible
endpoint. It appends every failure/timeout, retries non-success cells on resume,
requires authoritative server token counts, retains rescorable raw responses,
and reads any bearer credential from an environment variable without recording
it in the request artifact.
