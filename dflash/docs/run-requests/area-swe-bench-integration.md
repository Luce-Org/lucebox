# Run request: `--area swe` (SWE-bench Verified) integration plan

**Date opened**: 2026-05-25
**Status**: Follow-up to the Phase 1 bench-harness consolidation
(`--area code|longctx|agent` landed in `eb065d9`, `f3116a6`, `3fbeea0`).
This proposes the next gap fill: SWE-bench Verified as `--area swe`.

## Background

`dflash/scripts/fixtures/swe_bench/swe_bench_verified.parquet` (2 MB,
500 verified Python tasks from the original SWE-bench Verified split)
is already in the repo but unused — no bench script currently targets
it.

SWE-bench is the canonical "real coding work" eval: each case ships a
problem statement, a starting repo state, and a ground-truth patch
that, when applied, makes the test suite pass. It complements the
existing `--area code` (HumanEval, decode-only parse check) and
`--area agent` (agent-shape probe on real codex system prompts) by
asking the harder question: **does the model produce a patch that
actually fixes the bug?**

## The integration challenge

Unlike the other --areas, SWE-bench needs *execution* to grade:

1. Apply the model's patch to a snapshot of the target repo
2. Run the project's test suite
3. PASS = tests that previously failed now pass; previously-passing
   tests don't regress (FAIL→PASS, PASS→PASS sets)

That requires either:

- **A**: Containerized sandbox per case (Docker-in-Docker; each SWE-bench
  case ships a Dockerfile pinning the repo + test runner)
- **B**: Reuse the official SWE-bench harness
  (`pip install swebench`) which handles the sandboxing
- **C**: Skip execution; grade by structural heuristics (does the model
  reference the right file? does its patch parse as a unified diff?
  does it mention the failing test by name?)

A and B both need ~20 GB disk + 8+ GB RAM per parallel sandbox; not
suitable for bragi's already-loaded inference workload. C is what
this run-request proposes as the first cut.

## Proposal: two-phase rollout

### Phase 1 — structural grading only (this run-request)

Add `--area swe` that runs N SWE-bench Verified cases through the
unified harness and grades by **structural heuristics only**:

- **target_file_pass**: model output mentions at least one of the
  files in the ground-truth patch's `Files changed` list.
- **diff_format_pass**: response contains a unified diff hunk
  (`@@ -<lineno> +<lineno> @@`) or an `apply_patch` envelope per
  Codex convention.
- **test_name_pass**: response mentions a test function name from the
  case's `FAIL_TO_PASS` list.

PASS = `target_file_pass AND (diff_format_pass OR test_name_pass)`.
Lighter than real execution but catches obvious failure modes:
"model produced narrative without code", "model edited the wrong
file", "model produced markdown explanation but no diff".

Implementation:

1. New module `dflash/scripts/bench_swe_verified.py` (sister of
   `bench_humaneval.py`, `bench_longctx.py`, `bench_agent_cases.py`).
2. At import time, pre-extract N cases from the parquet into an
   in-memory list. Needs `pyarrow` or `pandas` (already a transitive
   dep via existing fixtures). If neither, vendor a JSON export of the
   first N as `fixtures/swe_bench/cases_<n>.json`.
3. Wire into `bench_http_capability.py`:
   - argparse `--area swe`
   - `case_areas` + `select_cases` + `build_prompt` (similar to
     agent-prompt; multi-section)
   - `default_max_tokens(swe) = 4096` (patches can be long)
   - grader dispatch in `run_case`

Suggested per-case structure:

```python
{
  "area": "swe",
  "source": "swe-bench-verified",
  "id": case["instance_id"],              # e.g. "astropy__astropy-12907"
  "kind": "swe-bench",
  "problem_statement": case["problem_statement"],
  "patch_files": [...],                   # from case["patch"] hunks
  "fail_to_pass": case["FAIL_TO_PASS"],   # list of test names
  "system_prompt": "<concise coding-agent prompt asking for unified diff>",
}
```

Estimated effort: half a day. Output rows include the usual unified
schema (provider/server_info/timings/etc.) plus the three structural
booleans.

### Phase 2 — real execution grading

After Phase 1 lands and we have baseline structural-pass numbers,
follow up with execution grading via the official `swebench` harness.
This wants a separate orchestrator (sandbox per case, ~20 GB scratch
per parallel slot) and probably belongs on a dedicated machine, not
bragi. Open a separate run-request when Phase 1 is shipping clean
numbers.

## Phase 1 case-count target

Start with 10 cases (one from each project area in SWE-bench Verified
— astropy, django, flask, matplotlib, pylint, pytest, requests,
scikit-learn, sphinx, sympy). Easy to expand later by bumping the
selector.

## Validation

Smoke against bragi qwen3.6 nothink:

```bash
python dflash/scripts/bench_http_capability.py \
  --url http://127.0.0.1:8080 --area swe --model dflash \
  --no-think --questions 2 --timeout 300 \
  --json-out /tmp/_swe_smoke.json
```

Expected: 2 cases run, both produce non-empty responses, structural
pass rate somewhere between 0% (regressed to noise) and 100%
(structurally correct but possibly semantically wrong). The point is
the new --area runs cleanly through the unified harness, not the
absolute pass number.

## Open questions before starting

1. **Case set size**: 10 (one per project), 50 (sampled), or 500 (full
   verified split)? Lucebox-relevant default likely 10-50.
2. **System prompt**: use a Codex-style one (reuse the
   `fixtures/agent_prompts/codex_*.md` set?) or a SWE-bench-specific
   one ("Output a unified diff...")?
3. **Patch parsing**: should we parse the model's diff and validate
   it against the actual repo (using `git apply --check`) without
   executing tests? That's intermediate between structural pass and
   real execution.

These don't block Phase 1 — happy to pick defaults and iterate.
