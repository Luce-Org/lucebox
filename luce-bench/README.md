# luce-bench

Capability benchmarks for OpenAI-compatible chat-completion endpoints —
6 evaluation areas (`smoke`, `ds4-eval`, `code`, `longctx`, `agent`,
`forge`). Lives inside the [lucebox-hub] monorepo and publishes to
PyPI on tagged releases.

[lucebox-hub]: https://github.com/luce-org/lucebox-hub

## Quick start

```bash
# Smoke test (3 fast cases, ~10s) — runs by default if no --areas given
uvx luce-bench --base-url http://127.0.0.1:8000

# Smoke against OpenRouter
export OPENROUTER_API_KEY=sk-or-...
uvx luce-bench --base-url https://openrouter.ai/api \
  --model qwen/qwen3.6-27b --auth-env OPENROUTER_API_KEY

# Full sweep (all areas) — writes per-area JSON + _summary.md
uvx luce-bench --areas all --name my-machine \
  --base-url http://127.0.0.1:8000

# Single area
uvx luce-bench --areas ds4-eval --base-url http://127.0.0.1:8000
```

Run an unreleased branch directly (e.g. to validate a PR before merge):

```bash
uvx --from "git+https://github.com/luce-org/lucebox-hub@feat/lucebox-docker#subdirectory=luce-bench" \
  luce-bench --base-url http://127.0.0.1:8000
```

Every run prints a version banner as its first line — `[lucebench] vX.Y.Z` —
so stale uvx caches are easy to spot.

## Install

```bash
uvx luce-bench               # one-shot, no venv pollution
uv add luce-bench            # add to a uv-managed project
pip install luce-bench       # plain pip
pip install 'luce-bench[dev]' # + pytest, ruff for contributors
```

`anthropic` is a hard runtime dep (the `forge` area needs it). The
legacy `[forge]` extra still resolves as an empty alias for backward
compatibility with older install commands.

## More examples

```bash
# Single case, json-out for downstream analysis
luce-bench --areas ds4-eval --case-id aime2025-02 \
  --base-url http://localhost:8080 --json-out /tmp/aime02.json

# Limit each area to N questions
luce-bench --areas all --name quick --questions 2 \
  --base-url http://localhost:8080

# Parallel against a stateless gateway (skip on single-GPU local servers)
luce-bench --areas ds4-eval --base-url https://openrouter.ai/api \
  --model openai/gpt-5.4 --auth-env OPENROUTER_API_KEY --parallel 8

# Single-case multi-mode reasoning probe (think / nothink / budget=N / …)
luce-bench-probe --case-id aime2025-02 \
  --url http://localhost:8080 --out-dir ./probes/my-model
```

A sweep writes per-area JSON and a combined `_summary.md` table under
`./snapshots/<name>/`. Each row carries the full request + response
payload + timings (when surfaced by the server).

## What's benchmarked

| Area | Cases | Grader | Source |
|------|-------|--------|--------|
| `smoke` | 3 (arithmetic, capital, sequence) | case-insensitive substring | own — default sanity check |
| `ds4-eval` | 92 (GPQA Diamond, SuperGPQA, AIME2025, COMPSEC) | strict `Answer: X` extract | [antirez/ds4](https://github.com/antirez/ds4) (MIT) |
| `gsm8k` | 100 (test split sample, seed 42) | `#### N` marker, last-number fallback | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (MIT) |
| `truthfulqa-mc1` | 100 (validation split sample, seed 42) | MC letter extract (2–13 choices) | [truthful_qa](https://huggingface.co/datasets/truthful_qa) (Apache-2.0) |
| `hellaswag` | 100 (validation split sample, seed 42) | MC letter extract (A–D endings) | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) (MIT) |
| `code` | 10 (mid-function completion) | `ast.parse(prompt + completion)` | [openai/human-eval](https://github.com/openai/human-eval) (MIT) port |
| `longctx` | 6 frontiers (2k → 64k tokens) | `^Risk:` prefix check | own ports |
| `agent` | N codex-style prompts paired with coding tasks | code-fence / json-tool / apply_patch detect | own ports |
| `agent_recorded` | 25 prompts mined from real local Claude Code + Codex sessions | three-bin tool-schema-coverage (expected tools + files named in reply) | own — mined via `scripts/extract-agentic-fixture.py` |
| `forge` | 7+ tool-calling scenarios | error_type == None | [antoinezambelli/forge](https://github.com/antoinezambelli/forge) 0.7.1 (MIT) |

Each row in the result carries:

- `pass` (bool), `graded` (full grader output)
- `wall_seconds`, `http_status`, `error`
- `prompt_tokens`, `completion_tokens`, `timings` (when surfaced by the server)
- `content`, `reasoning_content`, `finish_reason`, `finish_details`

The default sampling shape is **send-nothing-explicit** — the server
gets to apply its own defaults (model card sampling, provider tuning,
etc). Passing `--temperature 0` would forcibly override that; bench
deliberately omits sampling fields unless the user sets them.

## Programmatic use

```python
from lucebench.areas import ds4_eval
from lucebench.runner import run_case

cases = ds4_eval.load_ds4_eval_cases()
case = next(c for c in cases if c["id"] == "aime2025-02")

row = run_case(
    url="http://localhost:8080",
    case=case,
    model="my-model",
    think=True,
)
graded = ds4_eval.grade_case(case, row)
print(graded["pass"], graded["given"], "/", graded["correct"])
```

## Attribution

This project redistributes evaluation fixtures from upstream MIT-
licensed projects. See `NOTICE` for full attribution; in short:

- ds4-eval cases — `antirez/ds4`, MIT
- GSM8K cases — `openai/gsm8k`, MIT
- TruthfulQA MC1 cases — `truthful_qa`, Apache-2.0
- HellaSwag cases — `Rowan/hellaswag`, MIT
- HumanEval prompts — `openai/human-eval`, MIT
- forge eval scenarios — `antoinezambelli/forge`, MIT

The luce-bench code itself is Apache-2.0.

## Contributing

```bash
git clone https://github.com/luce-org/lucebox-hub
cd lucebox-hub
uv sync --extra dev
uv run pytest luce-bench/tests/
uv run ruff check luce-bench/src luce-bench/tests
```

CI runs the same matrix on Python 3.10–3.13 + a wheel-build check
that verifies fixtures are bundled.
