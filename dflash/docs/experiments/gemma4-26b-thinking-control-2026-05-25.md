# Gemma 4 26B-A4B-it — thinking control characterization — 2026-05-25

First-run results of the
[thinking-control protocol](thinking-control-protocol.md) against
Gemma 4 26B-A4B-it (Q4_K_M target, q8_0 dflash draft) on luce-dflash.

* **Server**: luce-dflash on bragi (RTX 5090 Laptop), image
  `lucebox-hub:cuda12` 2026-05-25 13:18 with PR #277 entrypoint
  bind-mounted.
* **Case**: `aime2025-02` (geometry, ground-truth answer = 588).
* **Sampling**: temperature=0, top_p=1.0, max_tokens=8192. **See
  caveat below — Gemma 4's recommended sampling is temp=1.0,
  top_p=0.95, top_k=64.**
* **Raw output**: `dflash/docs/tuning-snapshots/bragi-rtx5090laptop-gemma4-26b-thinking-control-2026-05-25/`

## Summary table

| mode | prompt | comp | content_chars | reasoning_chars | finish | wall |
|---|---|---|---|---|---|---|
| `think-default`       | 664 | 8192 | 1167  | 24229 | length | 43.8s |
| `nothink`             | 666 | 8192 | 12824 | 0     | length | 37.8s |
| `think-low` (b=1024)  | 664 | 1536 | 1022  | 2191  | length | 15.1s |
| `think-medium` (b=4096)| 664 | 4608 | 1231  | 12291 | length | 27.2s |
| `think-raw-noprompt`  | 645 | 8192 | 18733 | 0     | length | 60.3s |

None of the five modes reached the correct answer (588). Geometry
this hard isn't expected to be a slam-dunk for the 26B at temp=0;
the question here is mechanism, not pass rate.

## Q1 — Thinking ON: separation, termination, server detection

**Separation: yes.** `think-default` produced 24229 chars of
`reasoning_content` and 1167 chars of `content`. Reasoning starts
with `*   Triangle $ABC$ with points...` (the model's outline of the
problem). Content starts with a fresh solution attempt — the parser
correctly routed the channel-thought block to `reasoning_content`.

**Termination: no, finish=`length`.** The model never emitted its
`<channel|>` close on its own within 8192 tokens. Reasoning ended
in a runaway repetition loop:

```
        $M$- $N$ is- $P$ is $Q$- $R$- $T$-- ... $S$ $T$ $
```

This is the degenerate-decode pattern seen across all modes (see
"sampling caveat" below).

**Token-level leak: minor.** `reasoning_content` begins with the
literal string `thought\n` — the `<|channel>` open-tag token (id 100)
text is `<|channel>thought` and the parser is stripping `<|channel>`
but keeping `thought`. Not catastrophic, but the SSE emitter mapping
should drop the full pair atomically. Matches vLLM
[issue #38855](https://github.com/vllm-project/vllm/issues/38855)
verbatim.

**`reasoning_tokens` accounting: BROKEN.**
`usage.completion_tokens_details.reasoning_tokens = 0` despite
~8000 reasoning tokens emitted. `finish_details.thinking_tokens = 0`
likewise. The server's per-mode token bookkeeping treats the
thinking phase as content tokens. Visible
`completion_tokens` is correct (it counts every emitted token), but
downstream tooling that wants "decode-only" wall reads `reasoning_tokens=0`
and undercounts the budget.

**Timings: BROKEN.** `usage.timings.decode_ms = 0.0`,
`decode_tokens_per_sec = 0.0` for every gemma4 request. The
Gemma4Backend hasn't had sindri's qwen35 timing instrumentation
(commit `3b80fa8`) ported — open follow-up.

## Q2 — Thinking OFF: drop reasoning or just hide tags?

**Drops reasoning at the boundary, but doesn't save work — content
balloons instead.**

| mode | content_chars | reasoning_chars | total_chars |
|---|---|---|---|
| `think-default` | 1167  | 24229 | 25396 |
| `nothink`       | 12824 | 0     | 12824 |

Total text drops from ~25k to ~13k chars (49% reduction), so the
work *isn't* identical, but the model still produces ~10× more
content text than think-default's content portion. Looks like the
model reasons *inside* `content` when the template-emitted
`<|channel>thought\n<channel|>` guard runs out — exactly the
behavior our `f1d30f2` chat-template fix was supposed to suppress.

**No literal `thought` / `channel` / `<|` / `|>` substring leakage
in nothink content** — the guard is working at the *token* level
(no `<channel|>` text reaches the visible content). What leaks is
the model's *behavior*: it does step-by-step math anyway, just
without the channel markers. Compare to `think-default`'s content
(`1167` chars of compact solution): when the model gets to "speak"
in the reasoning channel, the content portion is naturally short.

**Conclusion**: nothink is half-effective. The server-side parser
+ chat template prevent the channel-tagged thinking from being
emitted, but the model's training to reason still fires; the
reasoning ends up in `content`. Per the
[opcnew analysis](https://www.opcnew.com/en/gemma-4-thinking-tokens-system-prompt-control)
this is reported to be inherent to Gemma 4's training (reasoning is
emergent, not instruction-gated) and natural-language "do not reason"
in the system prompt is unreliable.

## Q3a — Hard-close budget mechanism

**Server-side budget enforcement: WORKS PRECISELY.**

| mode | budget_tokens | comp_tokens | comp - budget |
|---|---|---|---|
| `think-low`     | 1024 | 1536 | 512 |
| `think-medium`  | 4096 | 4608 | 512 |

`comp - budget = 512` for both, matching `/props.budget_envelope.hard_limit_reply_budget = 512` exactly. The
server cleanly closes thinking at `budget_tokens` and reserves 512
tokens for a content reply.

**Post-close coherence: BROKEN.** The content emitted after the
forced thinking-close is garbage on both think-low and think-medium:

`think-low` content (full):
```
To find the area own of the--------- ---- - - - - - - - - - - ...
```

`think-medium` content:
```
To find-the area of the triangle formed by the points $A(0,0)$,
$B(1,1)$, and $C(2,0)$... [solves a different, hallucinated
problem then degenerates]
```

The model gets force-closed mid-reasoning and can't recover to
produce a coherent final answer. Reasoning content also degenerates
*before* the close fires (`$M$-$N$ is-$P$ is $Q$-...`), suggesting
the model was already losing coherence well inside the budget — a
sampling / model issue, not a budget issue (see sampling caveat).

**No thinking-tokens-flow-into-content failure mode.** With the
hard-close active, the post-close output is in `content`, not
`reasoning_content`. The server's transition firmly switches
channels. The badness is the *model's* failure to write a useful
content reply, not the server crossing channels.

## Q3b — Prompt-side control

**`think-raw-noprompt`** (system prompt empty, `enable_thinking=false`
in template, `thinking: {type: enabled}` in body) emitted **zero**
reasoning content and 18733 chars of content. So with the template's
`<|think|>` opener suppressed, the model does *not* self-open a
thought channel even when the server's budget contract says
thinking is enabled.

This confirms what the model card states and what the research
agent found: thinking is gated by the `<|think|>` token in the
**system turn** of the chat template, not by anything in the request
body. The server's `thinking: {type: enabled}` field is only
meaningful when the chat template also primes the model with
`<|think|>`.

**Practical implication**: the chat template
is the source of truth for whether Gemma 4 thinks. The Anthropic-shape
`thinking: {type}` field is just a hint into the chat template
render (via `chat_template_kwargs.enable_thinking`).

## Q3c — Model card guidance

`share/model_cards/gemma-4-26b-a4b-it.json` says verbatim:

> Reasoning-capable via `<|think|>` token at start of system prompt.

Recommended sampling:
```json
"sampling": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0
}
```

`/props.budget_envelope.effort_tiers`: low=1984, medium=7936,
high=15872, max=15872, default `think_max_tokens=15488` (server-side
defaults derived from `max_tokens=16384` minus `hard_limit_reply_budget=512`).

## Q3d — Research agent findings (summary)

Spawned a research agent (see this session's full notes); highlights:

1. Gemma 4 thinking tokens: `<|think|>` (id 98) opens *globally* in
   the system turn; `<|channel>thought` (id 100) opens the per-turn
   reasoning block; `<channel|>` (id 101) closes it. Distinct from
   Qwen3's `<think>` / `</think>` text-level markers.
2. **vLLM issue #38855**: their parser strips `<|channel>`/`<channel|>`
   via `skip_special_tokens=True` before parsing → reasoning leaks
   into `content`. Same failure mode we'd hit if we matched by text
   instead of token-id.
3. **llama.cpp PR #21697**: wired up `--reasoning-budget` for gemma4
   by matching against the channel-thought token-id explicitly. Our
   server's hard-close should target token id **101**, not
   `</think>`.
4. **Google docs recommend pre-seeding** an empty
   `<|channel>thought\n<channel|>` in the assistant turn for the
   26B/31B when you want non-thinking output — because the larger
   variants sometimes open a thought channel even with
   `enable_thinking=false`. Our `f1d30f2` chat-template fix already
   does this pre-seeding; our nothink result above (zero
   `reasoning_content`) confirms it works.
5. **Multi-turn caveat**: Google's docs say strip prior-turn
   thoughts between turns. Our KV-cache currently preserves them —
   open follow-up.
6. F16 GGUFs on CUDA flood `<unused49>` tokens when thinking is
   forced on; Q4_K_M is the workaround (we're already on Q4_K_M).
7. 31B reports "blank line" / continued-thinking-after-close on
   unsloth GGUF discussion #6 — relevant if we hit similar issues
   when we run the same probe against 31B.

## Sampling caveat (read before drawing conclusions)

All five modes were run at temp=0 (greedy decode) for
reproducibility. The model card recommends temp=1.0, top_p=0.95,
top_k=64 for Gemma 4. The pervasive `- - - -` / `$0-0.` /
`$M$-$N$-$P$-` degeneration loops seen across modes are consistent
with greedy-decode collapse on Gemma 4 — *not* with a thinking-control
bug. The mechanism-level findings (parser separation, budget
enforcement, no token leakage, template gating) are robust to
sampling. The "model never closes thinking" and "post-close content
is garbage" findings are likely sampling-driven and should be
re-tested at temp=1.0 before concluding the model fundamentally
can't terminate.

A follow-up run at the model card's recommended sampling is queued.

## Concrete follow-ups

1. **Server: fix `reasoning_tokens` accounting in Gemma4Backend.**
   Currently always 0; need to count tokens emitted between
   `<|channel>thought` (id 100) and `<channel|>` (id 101) and surface
   in both `usage.thinking_tokens` and
   `usage.completion_tokens_details.reasoning_tokens`.
2. **Server: port qwen35 timing instrumentation (`3b80fa8`) to
   Gemma4Backend.** `decode_ms=0, decode_tokens_per_sec=0` is
   pretending nothing happened.
3. **Server: fix the `thought` text prefix leak** at the start of
   `reasoning_content`. The `<|channel>thought` token text needs to
   be dropped atomically by the SSE emitter, not partial-matched.
4. **Server: validate hard-close transition recovery.** When the
   server force-closes thinking, the model often emits garbage in
   the content phase. Two possible mitigations: (a) emit a short
   transitional cue (e.g. `<channel|>\nFinal answer: `) after the
   forced close to nudge the model into answer mode; (b) when the
   model fails to emit useful content within N tokens of the
   forced close, give up and surface a `finish_details.error =
   "post_close_degenerate"` so callers can retry with a larger
   budget.
5. **Re-run probe at the model card's recommended sampling**
   (temp=1.0, top_p=0.95, top_k=64) to disentangle sampling-driven
   degeneration from thinking-control issues. Add a `--sampling`
   knob to `probe_thinking_control.py`.
6. **Multi-turn KV-cache: strip prior-turn thoughts** between turns
   per Google's guidance (research item 5). Likely needs a
   conversation-state pass over `cache_.target_feat` after each
   turn closes.
7. **Repeat against gemma-4-31b** to see whether the 31B's
   reported "continued-thinking-after-close" issue (research item 7)
   manifests in our setup.
