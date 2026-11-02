# Chat golden corpus

The prompt Lucebox serves for a request must be the prompt the model's official chat template renders.
This directory holds the references and the list of cases that still differ.

| File | What it is |
|---|---|
| `cases.json` | Synthetic OpenAI Chat Completions requests: plain chat, thinking on and off, reasoning efforts, earlier thinking, tool loops (single, parallel, two rounds, a new user message after one), every argument type, whitespace and Unicode edges, text that spells control tokens. |
| `generate.py` | Writes `expected/<model>.json`: each case rendered by the model's official template (Qwen3.8 27B, Qwen3.8-Flash-Next, Qwen3.6 27B), with the official tokenizer's token IDs. DeepSeek V4 and V4.1 have DeepSeek's Python encoder (`encoding/encoding_dsv4.py`, `encoding/encoding.py` in the model repos) instead of a template; each case is prepared as vLLM's V4 or V4.1 tokenizer prepares a request and encoded with it, and the encoder's own golden pairs sit beside them as `official_N`, encoded as the encoder's own tests encode them. |
| `known_divergences.json` | The cases where Lucebox still differs, each with the reason. |

`test_chat_golden_qwen38` sends every case through the server's whole request path (parsing, normalization, rendering,
tokenization), once per prompt path: the `qwen3.8` chat codec (`--chat-codec qwen3.8`) for Qwen3.8 27B and Flash-Next, and the
Jinja engine with Flash-Next's official template (the escape hatch, `--chat-template-file`). It covers Qwen3.6 27B
and Qwen3.5 27B with their codecs (`qwen3.6`, `qwen3.5`) and the Jinja engine the same way, and Qwen3.5 once more
with `preserve_thinking: false` (`codec_official`, below). `test_chat_golden_deepseek41` and
`test_chat_golden_deepseek4` do the same for DeepSeek V4.1 and V4 with their codecs (`deepseek4.1`, `deepseek4`), and
render DeepSeek's own golden pairs from their conversations through them. A case that differs and is not in
`known_divergences.json` (keyed `<model>/<path>`) fails, and so does a listed case that now matches, so the list is
always the current state. Each run prints how many cases match per model and path.

| Path | Text | Tokens (real GGUF) |
|---|---|---|
| Qwen3.8 27B, codec | 27 / 27 | 26 / 27 |
| Qwen3.8-Flash-Next, codec | 27 / 27 | 26 / 27 |
| Qwen3.8-Flash-Next, Jinja | 25 / 27 | 25 / 27 |
| Qwen3.6 27B, codec | 26 / 26 | 25 / 26 |
| Qwen3.6 27B, Jinja | 24 / 26 | 24 / 26 |
| Qwen3.5 27B, codec (the server default) | 20 / 26 | 19 / 26 |
| Qwen3.5 27B, codec with `preserve_thinking: false` | 26 / 26 | 25 / 26 |
| Qwen3.5 27B, Jinja | 24 / 26 | 24 / 26 |
| DeepSeek V4.1, codec | 29 / 29 | 28 / 29 |
| DeepSeek V4.1, the encoder's own pairs, codec | 2 / 5 | 2 / 5 |
| DeepSeek V4, codec | 28 / 28 | 27 / 28 |
| DeepSeek V4, the encoder's own pairs, codec | 2 / 4 | 2 / 4 |

Of the encoders' own pairs, two each use the internal `latest_reminder` role, which no API request can send, and one of
V4.1's has images, which the server renders through its image pipeline (not driven here). The Qwen3.8 GGUF checks
Qwen3.6's and Qwen3.5's token IDs too: its vocabulary is theirs with seven audio tokens added (Qwen3.5's and Qwen3.6's
references tokenize alike).

Qwen3.5's six differences by default are deliberate: its template drops earlier turns' thinking (vLLM, SGLang,
llama.cpp and Ollama render it so), which rewrites the turns after the previous question at each new one, so the
prefix cache and PFlash views could not continue. The codec keeps it, as Qwen3.8 does; `preserve_thinking: false`
renders the official prompt. Qwen3.5 also writes scalar tool-call arguments with Python's `str()` (`True`, `None`).

The codecs' one token difference is deliberate: control-token text a user types (`<|im_end|>`, and DeepSeek's turn
markers such as `<｜User｜>`, which its vocabulary types user-defined) stays text, so a message cannot close its turn.
Hugging Face tokenizers turn it into the token.

The model cards name each model's codec (`chat_codec`), and the `deepseek41`, `deepseek4`, `qwen4exp`, `qwen35` and
`qwen35moe` archs default to theirs (`qwen3.5` for `qwen35` and `qwen35moe`), so the server renders with it by default. These archs have no built-in renderer: `--chat-codec none` works only
with `--chat-template-file`.

```bash
# Regenerate the references (pinned Hub revisions; needs the Hub or its cache).
uv run --project server python server/test/chat_golden/generate.py

# Compare token IDs too, with the real tokenizers.
LUCE_CHAT_GOLDEN_QWEN38_GGUF=/path/to/Qwen3.8-27B.gguf \
LUCE_CHAT_GOLDEN_DS41_GGUF=/path/to/DeepSeek-V4.1-Flash.gguf \
LUCE_CHAT_GOLDEN_DS4_GGUF=/path/to/DeepSeek-V4-Flash.gguf \
  ./build/test_server_unit test_chat_golden

# Write Lucebox's prompt for each case to a directory, to diff against expected/.
LUCE_CHAT_GOLDEN_DUMP=/tmp/golden ./build/test_server_unit test_chat_golden
```

Without a GGUF, the tests use a byte-level tokenizer with the model's control tokens, which round-trips any prompt
text exactly, so CI compares the text.
