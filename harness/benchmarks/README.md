# Generation Benchmarks

These checks are separate from the client harness launchers. They compare Lucebox
generation against a llama.cpp baseline on the same target GGUF, using small
deterministic prompts.

Use this when you want to know whether a server change affects output quality or
decode speed. Use `harness/clients/` when you want to know whether Codex,
OpenCode, Open WebUI, Pi, and the other clients still work.

## Bench suites (HumanEval, GSM8K, Math500, Agent)

Run standard LLM and agentic benchmarks against a running Lucebox server:

```bash
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080
```

This sends benchmark prompts through the OpenAI-compatible `/v1/chat/completions`
endpoint and reports tok/s, TTFT, and correctness scores.

### Suites

| Suite   | Description                                        | Scoring          |
|---------|----------------------------------------------------|------------------|
| `he`    | HumanEval code-completion prompts (10)             | tok/s + pass rate |
| `gsm`   | GSM8K arithmetic reasoning prompts (10)            | tok/s + accuracy |
| `math`  | Math500 with `\boxed{}` correctness check (10)     | tok/s + accuracy |
| `agent` | Agentic workloads at 2K/8K/24K context (6)         | TTFT + tok/s     |

### Usage

```bash
# All suites (default)
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080

# Only Math500 correctness
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --suite math

# HumanEval + agent
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --suite he,agent

# Limit to 3 prompts per suite
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --n-sample 3

# Save JSON results
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --json-out /tmp/bench.json
```

### Options

- `--url` (required): Server base URL
- `--suite`: Comma-separated list or `all` (default: `all`)
- `--model`: Model name (default: `luce-dflash`)
- `--n-sample`: Max prompts per suite (default: all in file)
- `--prompts-dir`: Override prompt files directory
- `--json-out`: Write JSON results to this path

### Prompt files

Static JSONL files in `harness/benchmarks/prompts/`:

- `bench_he.jsonl` — HumanEval code-completion
- `bench_gsm.jsonl` — GSM8K arithmetic reasoning
- `bench_math.jsonl` — Math500 with `gold_answer` field
- `bench_agent.jsonl` — Agentic prompts with `bucket` field (2k/8k/24k)

### Correctness

Math500 responses are scored by extracting `\boxed{}` answers and comparing
against gold with normalized math equivalence. Accuracy is reported in the
output but does not gate the exit code.

---

## Lucebox vs llama.cpp

Run from the repo root on the GPU host:

```bash
harness/benchmarks/run_lucebox_vs_llamacpp.sh
```

The runner starts llama.cpp first, runs the prompt set, stops it, then starts
Lucebox and runs the same prompt set. It is sequential on purpose so a 24 GB
card does not need to hold two copies of the target model.

Common overrides:

```bash
MAX_CTX=65536 MAX_TOKENS=512 harness/benchmarks/run_lucebox_vs_llamacpp.sh
LLAMA_SERVER_BIN=/path/to/llama-server harness/benchmarks/run_lucebox_vs_llamacpp.sh
PROMPTS=/tmp/my_prompts.jsonl harness/benchmarks/run_lucebox_vs_llamacpp.sh
```

Each run writes:

- `llamacpp.json`: raw llama.cpp endpoint results
- `lucebox.json`: raw Lucebox endpoint results
- `compare.json`: machine-readable comparison
- `report.md`: speed and expected-output summary

Prompt files are JSONL. Each line needs `id` and either `prompt` or `messages`.
Optional `expect_contains` and `expect_regex` fields define lightweight accuracy
checks.

---

## Compare two DeepSeek4 pull requests

`deepseek4_generation_benchmark_between_prs.py` reproduces an end-to-end comparison of
two upstream Lucebox pull requests on one GPU host. It fetches both PR heads,
prepares SHA-pinned worktrees, builds and unit-tests both with identical settings,
starts their servers sequentially, and delegates endpoint measurement and report
generation to `generation_benchmark.py`.

The command inherits the current process environment. Pass the DeepSeek4 GGUF
with `--model`, or export it as `MODEL`, `DFLASH_TARGET`, or `TARGET`. An
optional shell environment can be loaded explicitly with `--env-file PATH`.

Workload and execution topology are independent options. Select a workload with
`--suite` and its size with `--profile`:

| Suite | Prompt source | Scoring |
|---|---|---|
| `generation` | Five deterministic plumbing prompts | Expected markers |
| `humaneval` | 10 HumanEval functions | Generated-code tests |
| `gsm8k` | 10 GSM8K problems | Numeric answer matching |
| `math` | 10 Math500 problems | Boxed-answer math equivalence |

The `smoke` profile runs one repeat with no warmup and limits scored suites to
three cases. The `full` profile runs every case with one warmup and five measured
repeats. Per-suite defaults size the context and output limit appropriately;
`--case-limit`, `--repeats`, `--warmup-repeats`, `--max-ctx`, and `--max-tokens`
remain available as explicit overrides. HumanEval scoring executes generated
Python with isolated interpreter settings and a timeout, but it is not an OS
sandbox; use it only with a trusted local model.

For compatibility, `--mode smoke` maps to `--suite generation --profile smoke`,
and `--mode full` maps to `--suite gsm8k --profile full`.

`--topology` independently selects one of these build and placement plans:

| Topology | Builds per PR | Default placement |
|---|---|---|
| `hip-monolithic` | HIP server | `hip:0` |
| `hip-hybrid` | HIP server | `hip:0`, hybrid expert placement |
| `hip-layer-split` | HIP server | `hip:0,hip:1`, weights `1,1` |
| `cuda-hip-layer-split` | CUDA server + HIP IPC daemon | `cuda:0,hip:0`, weights `24,19` |

Start with a monolithic smoke run:

```bash
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --suite generation --profile smoke
```

Use the same single-device HIP model with hybrid expert placement as a control:

```bash
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --suite generation --profile smoke \
  --topology hip-hybrid \
  --ds4-expert-top-k 4
```

Test the local HIP layer-split path:

```bash
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --suite generation --profile smoke \
  --topology hip-layer-split \
  --hip-visible-devices 0,1 \
  --target-devices hip:0,hip:1 \
  --layer-split 1,1
```

Test a CUDA parent with a HIP tail shard. Device IDs in `--target-devices`
refer to each backend's visible-device namespace:

```bash
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --suite generation --profile smoke \
  --topology cuda-hip-layer-split \
  --cuda-visible-devices 0 \
  --hip-visible-devices 0 \
  --cuda-architectures 86 \
  --target-devices cuda:0,hip:0 \
  --layer-split 24,19
```

Once the selected topology passes its generation smoke check, select any scored
suite. The build cache is shared across suite and profile changes:

```bash
# Three-case smoke profiles
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --suite humaneval --profile smoke
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --suite gsm8k --profile smoke
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --suite math --profile smoke

# Complete suite, including warmup and five measured repeats
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --suite gsm8k --profile full
```

### Build cache

Source worktrees and build directories persist between runs by default. The
cache lives under `$XDG_CACHE_HOME/lucebox/deepseek4-pr-benchmark/`, or
`~/.cache/lucebox/deepseek4-pr-benchmark/` when `XDG_CACHE_HOME` is unset.
Worktrees are keyed by the immutable PR head SHA; builds are keyed by SHA,
backend, architecture, wave size, and relevant toolchain settings. CMake still
runs its configure and incremental build steps on every invocation, so an
unchanged second run should be a quick no-op build.

The default `current-pair` policy bounds growth around the two PRs being
compared. After both selected PRs build and any enabled unit tests pass, the
runner removes source and build entries for PR SHAs that are no longer in the
pair, along with the fetched `pr-bench` refs that pinned those SHAs in the
repository, so `git gc` can eventually reclaim their objects. For each active
PR it retains the latest configuration for each backend, so HIP and CUDA can
each remain reusable without accumulating old architecture or wave-size
variants. Changing one PR evicts only that PR's previous SHA; swapping baseline
and candidate evicts nothing. Cache operations and benchmark runs are serialized
with a lock scoped to the cache directory (one per repository by default) so
pruning cannot race an active build or server using the same cache.

Only source and compiled artifacts are cached. Unit tests, server startup,
generation requests, comparison, and all result files run fresh every time.
Useful cache controls are:

```bash
# Fetch, compile, and unit-test both PRs without running generation.
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --mode smoke --prepare-only

# Force clean rebuilds for the selected PR SHAs and backend configuration.
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --mode smoke --rebuild

# Run once with isolated disposable worktrees and builds.
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --mode smoke --no-cache

# Put the persistent cache somewhere explicit.
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --mode smoke --cache-dir /mnt/fast/lucebox-pr-benchmark-cache

# Opt out of rolling eviction when rotating among several PR pairs.
python3 harness/benchmarks/deepseek4_generation_benchmark_between_prs.py \
  503 531 --mode smoke --cache-policy keep-all
```

The runner prints `cold`, `reuse`, `resume`, or `rebuild` for each source/build
entry, reports every eviction, and records the same cache decisions in
`metadata.json`. Pruning occurs only after successful builds and any enabled
unit tests, so a broken replacement does not destroy the last usable pair.

Results default to
`benchmark-results/pr<baseline>-vs-pr<candidate>-<topology>-<suite>-<profile>-<timestamp>/`
and include:

- resolved PR commit SHAs and benchmark configuration in `metadata.json`;
- compiler, ROCm, and GPU details in `system-info.txt`;
- build, unit-test, server, and benchmark logs for both PRs;
- raw endpoint reports with per-case correctness for both PRs;
- `compare.json` and a dynamically labelled `report.md` with baseline and
  candidate accuracy, throughput, speedup, and normalized-output equivalence.

The command exits nonzero if unit tests or expected-output checks fail, or if
the normalized baseline and candidate outputs differ. Expected-check failures
do not abort the run early: both PRs are still benchmarked, the failing case
ids are printed, and the full comparison artifacts are written before the
command fails. Use `--allow-output-differences` only when the PR intentionally
changes generation.

The default HIP target is `gfx1151` with `DFLASH_WAVE_SIZE=32`. Override
`--hip-arch` and `--wave-size` together when testing a different architecture.
The `hip-monolithic` topology passes `--ds4-fused-decode` and refuses to
benchmark if startup logs show a hybrid fallback or do not confirm
`fused_decode=on`. The `hip-hybrid` topology omits fused decode and requires the
startup logs to confirm hybrid expert placement. Both single-device HIP modes
default to four routed experts; use `--ds4-expert-top-k 0` for the model
default. The model's shared expert remains active in both modes. Throughput
prefers the server's decode-only timing from
`usage.timings`; total HTTP request time is used only for endpoints that do not
report decode timing.

For mixed builds, `--cuda-architectures` avoids compiling every default CUDA
target. `--target-devices` and `--layer-split` override the topology defaults;
`--peer-access` enables same-backend peer access where supported. Run `--help`
for repeat, prompt, model, port, output, and worktree overrides.
