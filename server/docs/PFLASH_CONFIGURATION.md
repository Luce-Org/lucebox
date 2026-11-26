# PFlash configuration

Prefill compression requires the four PFlash model assets (the trained scoring
head, segment probe, request probe, and segment reranker; see
[Model assets](#model-assets)), an in-process drafter (`--prefill-drafter`),
and a target whose drafter pass returns per-candidate scores (the Qwen3.5/3.6/3.8
family and DeepSeek V4/V4.1).

`--pflash auto` (the default) turns PFlash on whenever `--prefill-drafter` and
all four PFlash assets are available and the launch supports it; otherwise
startup logs why it stays off. `--pflash on` refuses to start if any component
or prerequisite is missing, and `--pflash off` disables it. Invalid environment
values stop startup before model loading and name the offending variable.

The [development tools](../scripts/pflash/README.md) document request-probe
export and parity fixtures; [PFlash tests](../test/pflash/README.md) describe
the fixtures and helper executables.

## Quick launch

Place all four GGUF assets in one directory and set `PFLASH_SCORING_HEAD_GGUF`.
The server auto-discovers `segment-probe.gguf`, `request-probe.gguf`, and
`seg-reranker.gguf` beside it:

```sh
export PFLASH_SCORING_HEAD_GGUF=/path/to/seg-scoring-head.gguf

luce_server /path/to/target.gguf \
  --prefill-drafter /path/to/Qwen3.5-0.8B-BF16.gguf \
  --prefill-skip-park
```

`--prefill-skip-park` stays opt-in. It keeps the target resident while the
drafter scores (used in the PFlash November Update measurements on a 32 GB
R9700); on smaller cards leave it off so the target parks during scoring.

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--pflash` | `auto` | `auto` (on when `--prefill-drafter` and all four PFlash assets are present), `on`, or `off` |
| `--prefill-drafter <path>` | none | In-process PFlash drafter GGUF (Qwen3.5-0.8B BF16) |
| `--prefill-drafter-device <backend:gpu>` | `--draft-device` | GPU the drafter scores on, in process (same backend as an explicit `--target-device`); on a separate GPU, nothing is reserved or parked on the target GPU |
| `--prefill-skip-park` | off | Keep the target and decode draft resident while the drafter scores |
| `--prefill-threshold <N>` | `8192` | Admission threshold in droppable tokens (`0` compresses every prompt) |
| `--prefill-keep-ratio <F>` | release curve + adaptive | Flat keep ratio in `(0, 1]`; replaces both the release curve and adaptive widening |
| `--prefill-curve T:R [T:R ...]` | `8192:0.450 16384:0.350 32768:0.270 65536:0.260 120000:0.240` + adaptive | Piecewise keep-ratio curve; overrides `--prefill-keep-ratio` and disables adaptive widening |
| `--prefill-upstream-base <URL>` | none | Forward compressed prompts to an OpenAI-compatible upstream (compression-proxy mode) |
| `--prefill-upstream-key <KEY>` | none | Bearer token for the upstream |
| `--prefill-upstream-model <NAME>` | none | Model name sent on forwarded requests |

## Environment variables (11 total)

PFlash reads 11 `PFLASH_*` environment variables: 4 model asset paths,
5 operational settings, and 2 diagnostic trace paths.

### Model assets

All four files come from [`Lucebox/pflash`](https://huggingface.co/Lucebox/pflash)
and must open as regular files. Setting any of the three sibling variables to
empty, `0`, `off`, `none`, `false`, or `no` disables that asset and leaves
`--pflash auto` off (`--pflash on` refuses to start).

| Variable | Default | Purpose |
|---|---|---|
| `PFLASH_SCORING_HEAD_GGUF` | unset (required) | Trained block-15 scoring head (`seg-scoring-head.gguf`) |
| `PFLASH_SEGMENT_PROBE_GGUF` | `segment-probe.gguf` beside head | Segment-boundary probe |
| `PFLASH_REQUEST_PROBE_GGUF` | `request-probe.gguf` beside head | Neural request/query probe |
| `PFLASH_SEG_RERANKER_GGUF` | `seg-reranker.gguf` beside head | Segment reranker (validated against loaded drafter and head) |

## Operational environment variables

| Variable | Default | Values / behavior |
|---|---|---|
| `PFLASH_REGION_TOOLS` | `*` | Tools whose outputs the unified view may page (`*` for all, comma-separated names, or empty for none) |
| `PFLASH_RECALL` | `1` | `1` enables recall of dropped lines on follow-up turns; `0` disables recall |
| `PFLASH_RECALL_GATE` | `0.075` | Deficit threshold in `(0, 1)` below which recall is skipped, or `off` to fill each region's cap |
| `PFLASH_FAIL_CLOSED` | `0` | `0` serves the newest content verbatim after a drafter failure (or continues with Stage-1 scores if only the segment reranker fails); `1` returns HTTP 500 on either failure |
| `PFLASH_DRAFTER_SESSIONS` | `2` | Retained drafter scoring sessions (`0`..`16`; `0` scores every prompt from scratch) |

## Diagnostics

| Variable | Default | Purpose |
|---|---|---|
| `PFLASH_TRACE_PATH` | unset | Append per-call selection trace records as JSONL |
| `PFLASH_VIEW_TRACE_PATH` | unset | Append per-request served-view records as JSONL |

## Built-in November Update defaults

Selection, query extraction, and drafter behavior are built in and require no
tuning flags:

- **Drafter execution:** runs blocks `0..15` of Qwen3.5-0.8B with F16 drafter
  KV cache, trims cached pool temporaries after scoring, and offloads retained
  scoring sessions to pageable host memory when released.
- **Segmentation and ranking:** budget-only selection over segment-probe
  boundaries scored by attention-mass density (forced probe mode segments
  whenever the probe is loaded, falling back to 256-token chunks only if probe
  segmentation produces no spans), reranked by the neural segment reranker.
- **Adaptive budget:** widens the keep ratio when the reranker spreads
  relevance across 42%–58% of optional tokens, serves new regions verbatim past
  58% when the raw prompt fits `--max-ctx` (whole prompt on a fresh view, or
  new turns verbatim while preserving already-paged earlier turns on a
  continuing view; otherwise keeps 58%), and maintains a 90% stage-1
  attention-mass floor up to 58%.
- **Query extraction and chat skeleton:** neural request-probe query candidate
  (`candidates=1`, score-weighted blend with window tail/end queries) plus up
  to 512 pinned request tokens; keeps up to 256-token conversational turns
  whole and mixes the last 3 earlier user questions at halving weights.
