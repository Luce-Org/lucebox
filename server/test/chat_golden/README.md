# Chat golden corpus

The prompt Lucebox serves for a request must be the prompt the model's official chat template renders.
This directory holds the references and the list of cases that still differ.

| File | What it is |
|---|---|
| `cases.json` | Synthetic OpenAI Chat Completions requests: plain chat, thinking on and off, reasoning efforts, earlier thinking, tool loops (single, parallel, two rounds, a new user message after one), every argument type, whitespace and Unicode edges, text that spells control tokens. |
| `generate.py` | Writes `expected/<model>.json`: each case rendered by the model's official template (Qwen3.8 27B, Qwen3.8-Flash-Next), with the official tokenizer's token IDs. DeepSeek V4.1 has a Python encoder instead of a template; its expected file holds the encoder's own golden pairs from the model repo, read as data. |
| `known_divergences.json` | The cases where Lucebox still differs, each with the reason. |

`test_chat_golden_qwen38` sends every case through the server's whole request path (parsing, normalization, rendering,
tokenization). `test_chat_golden_deepseek41` renders DeepSeek's conversations directly. A case that differs and is not
in `known_divergences.json` fails, and so does a listed case that now matches, so the list is always the current state.
Each run prints how many cases match per model.

```bash
# Regenerate the references (pinned Hub revisions; needs the Hub or its cache).
uv run --project server python server/test/chat_golden/generate.py

# Compare token IDs too, with the real tokenizers.
LUCE_CHAT_GOLDEN_QWEN38_GGUF=/path/to/Qwen3.8-27B.gguf \
LUCE_CHAT_GOLDEN_DS41_GGUF=/path/to/DeepSeek-V4.1-Flash.gguf \
  ./build/test_server_unit test_chat_golden

# Write Lucebox's prompt for each case to a directory, to diff against expected/.
LUCE_CHAT_GOLDEN_DUMP=/tmp/golden ./build/test_server_unit test_chat_golden
```

Without a GGUF, the tests use a byte-level tokenizer with the model's control tokens, which round-trips any prompt
text exactly, so CI compares the text.
