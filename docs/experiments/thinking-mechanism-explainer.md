# How "thinking on/off" actually works (mechanism explainer)

Companion to `thinking-control-protocol.md`. Spells out **exactly**
what gets sent to the model in each thinking mode for each arch, and
why the same client-side toggle produces such different model
behavior across Qwen3 and Gemma 4.

## What the client sends

The `probe_thinking_control.py` runner constructs three thinking-
related fields per request:

```jsonc
{
  // ... messages, temperature, etc ...
  "chat_template_kwargs": {"enable_thinking": true|false},
  "thinking":             {"type": "enabled"|"disabled",
                            "budget_tokens": 1024 /* optional */}
}
```

* `chat_template_kwargs.enable_thinking` — a flag passed into the
  Jinja/native chat template renderer. **This is the only field that
  affects the prompt the model sees.** Every other thinking-related
  field is either advisory (`thinking`) or post-hoc bookkeeping
  (`reasoning_content` extraction).
* `thinking: {type, budget_tokens}` — the Anthropic-shape opt-in.
  Our server reads it for:
  1. Setting `enable_thinking` if the client didn't send
     `chat_template_kwargs` explicitly.
  2. Wiring the budget envelope's hard force-close at
     `budget_tokens + hard_limit_reply_budget` (default 512).
  3. Tagging the response so downstream tooling knows whether the
     client opted in.

If `chat_template_kwargs.enable_thinking` and `thinking.type`
disagree, the **chat-template flag wins for prompt shape**, and the
`thinking` field wins for the *budget contract* (force-close target).
Probe mode `think-raw-noprompt` deliberately sets them in opposition
to expose that asymmetry.

## What the server's chat template renders

### Qwen3 / 3.5 / 3.6 (ChatFormat::QWEN3)

`dflash/src/server/chat_template.cpp:67-156`. With a system message
+ user message and `add_generation_prompt=true`:

**enable_thinking=true:**
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
<think>
```

The trailing `<think>\n` *pre-opens* the reasoning block. The model
is already inside `<think>...</think>` when it starts decoding, so
all output up to its own `</think>` is reasoning content.

**enable_thinking=false:**
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The trailing block is a *closed and consumed* thinking block —
literal `<think>\n\n</think>\n\n`. The model sees "thinking is
already complete (empty), now produce the answer." The **two blank
lines after `</think>`** are a strong transition cue Qwen was
trained on.

### Gemma 4 (ChatFormat::GEMMA4)

`dflash/src/server/chat_template.cpp:185-275`. Same inputs:

**enable_thinking=true:**
```
<bos><|turn>system
<|think|>
{system}<turn|>
<|turn>user
{user}<turn|>
<|turn>model
```

The `<|think|>` token (id 98) sits at the top of the **system turn**,
not the assistant turn — it's a *global* "this conversation is in
thinking mode" signal. The assistant turn header is bare; the model
itself decides to emit `<|channel>thought\n...<channel|>` for its
reasoning, which our server's parser then routes to
`reasoning_content`.

**enable_thinking=false:**
```
<bos><|turn>system
{system}<turn|>
<|turn>user
{user}<turn|>
<|turn>model
<|channel>thought
<channel|>
```

The system turn omits `<|think|>`. The assistant turn header
includes a *pre-filled empty thought channel* (`<|channel>thought\n<channel|>`).
The intent matches Qwen's `<think></think>`: "thought channel is
done, just answer now." **But unlike Qwen, there's no trailing
transition cue.** The cursor lands immediately after `<channel|>`,
and the model fills in whatever's natural for the context.

## Why this matters — the asymmetry

For Qwen3 the no-think pattern was **trained**: official
`chat_template.jinja` emits `<think>\n\n</think>\n\n` and Qwen was
post-trained on conversations following that pattern. The model
learned "this exact sequence means thinking is done." Result: when
the server pre-fills the block, the model reliably produces a
concise answer.

For Gemma 4 the picture is murkier:

1. **The no-think guard `<|channel>thought\n<channel|>` is a
   community-derived prefill**, not a token sequence the model was
   trained to follow with "now answer briefly" behavior. Google's
   official docs recommend it, but the larger 26B/31B variants
   "sometimes open a thought channel even when `enable_thinking=False`"
   ([source](https://www.opcnew.com/en/gemma-4-thinking-tokens-system-prompt-control)).
2. **No trailing transition cue after `<channel|>`** — Qwen's
   `</think>\n\n` includes two blank lines that train the model "now
   the visible answer comes". Gemma's `<channel|>` is followed
   immediately by the decode cursor; the model picks up whatever the
   training distribution says comes after a closed thought channel
   in context.
3. **Reasoning is emergent in Gemma 4's training, not channel-gated.**
   Google's docs and practitioner reports both note that natural-
   language instructions to "not reason" are inconsistent — the
   model's training to walk through problems step-by-step fires
   regardless of channel markers. So even when the channel-thought
   block is suppressed, the model reasons in `content`.

So the *same client toggle* produces:
- Qwen3 nothink → short answer (~hundreds of tokens, model commits)
- Gemma 4 nothink → long step-by-step in `content`, no channel tags

This is the headline finding from `gemma4-26b-thinking-control-2026-05-25.md`
Addendum 2.

## What we send for each probe mode

For reference (the probe runner is in
`dflash/scripts/probe_thinking_control.py`):

| Mode | `enable_thinking` | `thinking.type` | budget | system | other |
|---|---|---|---|---|---|
| `think-default`             | true  | enabled  | —    | default | — |
| `nothink`                   | false | disabled | —    | default | — |
| `think-low`                 | true  | enabled  | 1024 | default | — |
| `think-medium`              | true  | enabled  | 4096 | default | — |
| `think-raw-noprompt`        | false | enabled  | —    | empty   | exposes template/budget asymmetry |
| `nothink-terse`             | false | disabled | —    | "Answer ONLY the final answer; no reasoning" | tests prompt-side compulsion |
| `nothink-prefill-answer`    | false | disabled | —    | default | appends `{role:"assistant", content:"The answer is "}` to force a commit |
| `nothink-stop-after-answer` | false | disabled | —    | terse   | + stop=["\nReason","\nLet","\nFirst","\nWe ","\nTo ","Reasoning:","Explanation:","Step 1"] |

## What the server does AFTER decoding

Independent of the prompt-side template, the server has two more
levers that affect the response:

1. **`reasoning_content` parser**: scans the decoded stream for the
   per-arch channel markers (`</think>` for qwen, `<channel|>` for
   gemma) and splits the text into `message.reasoning_content` (the
   bit before the close) and `message.content` (the bit after). This
   is purely cosmetic — the model already did all the work.
2. **Budget hard-close**: when the client sent
   `thinking.budget_tokens=N`, the server emits a *force-close*
   token at decode position N (the per-arch close marker). For Qwen
   that's `</think>`; for Gemma 4 it should be `<channel|>` (token
   id 101). Decoding then continues for up to
   `hard_limit_reply_budget=512` more tokens for the visible answer.

   Measured per
   `gemma4-26b-thinking-control-2026-05-25.md` Q3a, the hard-close
   fires at the exact `budget + 512` token count. But the post-close
   content is garbage when the model wasn't ready to wrap up.

## TL;DR

* "Thinking on/off" is a **prompt-template control** — one bool that
  changes the suffix the chat template appends.
* For Qwen3 the template suffix maps to a trained behavior: nothink
  → short answer. For Gemma 4 it doesn't — the model reasons
  regardless of the channel toggle, just in a different field of the
  response.
* The Anthropic-shape `thinking: {type, budget_tokens}` field is
  about the **budget contract** (force-close target), not the prompt
  shape. It's advisory unless `chat_template_kwargs` is missing.
* `reasoning_content` is **post-hoc parsing**, not a compute lever.
  Splitting the response into reasoning + content doesn't change
  what the model did.
