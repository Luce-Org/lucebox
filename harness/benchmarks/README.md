# Generation Benchmarks

These checks are separate from the client harness launchers. They compare Lucebox
generation against a llama.cpp baseline on the same target GGUF, using small
deterministic prompts.

For the paired ragged C1/C4/C8/C16 serving benchmark and the concurrent
HumanEval/GSM8K/Math500/agent suite runner, see
[`concurrency/`](concurrency/README.md).

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
| `he`    | HumanEval code-completion prompts (10)             | tok/s only       |
| `gsm`   | GSM8K arithmetic reasoning prompts (10)            | tok/s only       |
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

## DeepSeek 4 exact-context benchmark

`deepseek4/ds4_publication_decode_client.py` runs a deterministic streaming
decode workload and records timing, token counts, and response hashes.
`deepseek4/ds4_context_sweep.py` uses the target model's tokenizer to run the
same workload at exact context lengths.

The AMD q=5 hardware launcher is kept separately under
`harness/qualification/deepseek4/` because it changes GPU performance settings
and assumes a specific two-GPU layout.
