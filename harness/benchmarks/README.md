# Generation Benchmarks

These checks are separate from the client harness launchers. They compare Lucebox
generation against a llama.cpp baseline on the same target GGUF, using small
deterministic prompts.

Use this when you want to know whether a server change affects output quality or
decode speed. Use `harness/clients/` when you want to know whether Codex,
OpenCode, Open WebUI, Pi, and the other clients still work.

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
