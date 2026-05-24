# Run request: `--area forge` against vidar native ds4-server (DS4f)

**Date opened**: 2026-05-24
**Status**: Fills the missing "DS4f on the native ds4-server" cell of the
forge tool-calling matrix.

## Why

Today we have forge results for 5 OpenRouter-hosted models:

| Model (OR)                  | Pass     |
|-----------------------------|----------|
| deepseek/deepseek-v4-flash  | 30/30    |
| qwen/qwen3.6-27b            | 28/30    |
| google/gemma-4-31B-it       | 26/30    |
| google/gemma-4-26B-A4B-it   | 25/30    |
| poolside/laguna-xs.2:free   |  5/30    |

We do NOT yet have forge against the **native ds4-server** (antirez/ds4
running DS4f directly, vs OR's proxied DS4f). The native run pairs with
our existing `vidar-deepseek-v4-flash-2026-05-23-ds4-eval/` snapshot —
ds4-eval landed 73/92 = 79.3% there — and would tell us whether the
native server's tool-call handling matches OR's (which got a perfect
30/30).

Why this matters:
- Different server stacks can implement the OpenAI/Anthropic tool-call
  contract differently. OR's proxy may be more lenient with arg
  formatting; native may be stricter (or vice versa).
- forge's pass rate is sensitive to protocol conformance, so a non-100%
  result on native (vs OR's 100%) would point at a tool-call wire-format
  divergence.
- Closes the cross-stack forge matrix.

## Asks

Single run from any host that can reach vidar's ds4-server port:

```bash
python3 dflash/scripts/bench_http_capability.py \
  --area forge \
  --url http://<vidar-host>:1236 \
  --model deepseek-v4-flash \
  --timeout 600 \
  --no-think \
  --json-out dflash/docs/tuning-snapshots/vidar-ds4f-2026-05-24-forge/result.json \
  --trace  dflash/docs/tuning-snapshots/vidar-ds4f-2026-05-24-forge/trace.txt
```

Notes:
- Use the **enriched bench script** from `easel/integration/props-uv-squared-clean`
  (post-`cf8eb4b`) so `rows[]` carry per-iteration tool-call detail.
- `--no-think` matches the cross-model default; switch to `--think` only
  if vidar's ds4-server has a separate thinking config.
- `--model` slug is whatever vidar advertises at `GET /v1/models`. If
  the slug differs from the example above, use what's actually served.
- ~10-20 min wall expected (forge scenarios are short).

## Output

Snapshot to `dflash/docs/tuning-snapshots/vidar-ds4f-2026-05-24-forge/`
on integration. The result.json will follow the new schema with
`iterations[]` per scenario. Update `SUMMARY-2026-05-24.md`'s forge
table with the new row.

## Where this runs

- **Preferred: bragi RTX 5090 MaxQ** after the in-flight `--think4k 92`
  finishes. Bragi already has the workflow + credentials wired up.
- **Alternative: sindri RTX 3090 Ti** can reach vidar's port, but the
  sindri queue is long (ds4-eval --no-think still running, then AIME
  budget sweep, then --think4k+65k, then Gemma/Laguna local). Slot in
  on bragi unless that pipeline runs over.
