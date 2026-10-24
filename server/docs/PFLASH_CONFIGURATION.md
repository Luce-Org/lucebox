# PFlash configuration

With a trained scoring head (`PFLASH_SCORING_HEAD_GGUF`) and prefill
compression on, every setting left unset takes the measured
[release profile](#release-profile); without a head the built-in defaults
apply. Fold, query-grid, stage-2 and oracle experiments remain available.
With prefill compression enabled, selection, query, admission and chat
settings are checked before model loading. Invalid settings stop startup and name the offending
variable. Model schemas and replay files receive their loader checks.

The [development tools](../scripts/pflash/README.md) document request-probe
capture, validation and export; [PFlash tests](../test/pflash/README.md) describe
the fixtures and replay executables.

Serving booleans accept `0` or `1`; unset or empty serving overrides retain their
defaults. Selection and query overrides reject empty values unless explicitly
allowed below. Valid integer overrides retain their existing upper clamps.

## Serving defaults

| Setting | Built-in default | Release profile | Values / behavior |
|---|---|---|---|
| `--pflash` | `off` | | `on`, `off` |
| `--prefill-threshold` | `32000` | `16384` | Admission threshold, in droppable tokens (`0`: every prompt) |
| `--prefill-keep-ratio` | `0.05` | curve | Flat ratio; `--prefill-curve tokens:ratio ...` overrides it. The release curve is `16384:0.315 32768:0.270 65536:0.270 120000:0.225`; either flag replaces it |
| `--prefill-skip-park` | off | | Keep the target resident while the drafter scores; see the release profile |
| `--prefill-drafter-device` | the `--draft-device` | | `backend:gpu` the drafter scores on, in process (the target's backend, one explicit `--target-device`); on another GPU than the target nothing is reserved or parked on the target's |
| `PFLASH_ADMISSION` | `length` | `cache` | `length` or `cache`; cache admission accounts for restorable target KV |
| `PFLASH_REGION_TOOLS` | `*` | | Under the unified view (on whenever the selection gives per-candidate scores; the chat-view settings below apply only to the requests it hands back, see the [server README](../README.md#pflash)): tools whose outputs may be paged; a comma-separated list names tools, empty pages none |
| `PFLASH_RECALL`, `PFLASH_RECALL_GATE` | `1`, `0.075` | | Under the unified view: recall for a new question, and the share of score mass outside the view below which nothing is recalled (`off` recalls up to the regions' caps) |
| `PFLASH_AGENT_PASTE` | `0` | | With cache admission, `1` permits a large newest user paste in a tool session; earlier history stays raw |
| `PFLASH_CHAT_VIEW`, `PFLASH_CHAT_RECALL` | `1`, `1` | | Continue the served view; recall omitted material for a new question |
| `PFLASH_CHAT_COMPRESS_NEW_TOKENS` | `16384` | | Integer >= 1; smaller additions stay verbatim; capped at 2^30 |
| `PFLASH_CHAT_SKELETON_TOKENS` | `256` | | Integer >= 0; capped at 2^20 |
| `PFLASH_CHAT_HISTORY_QUERIES` | `3` | | Integer >= 0; capped at 8 earlier user turns |
| `PFLASH_CHAT_REBUILD_RATIO` | `2` | | Finite number >= 0; `0` rebuilds only at the context limit |
| `PFLASH_CHAT_REBUILD_MIN_TOKENS` | `0` | `65536` | Integer >= 0; smallest view eligible for drift rebuild; capped at 2^30 |
| `PFLASH_ARRIVE` | `0` | `1` | Score only a large newly arriving region |
| `PFLASH_SELECT_PARAGRAPH_JOIN` | `1` | `0` | Insert paragraph breaks between nonadjacent kept spans |
| `PFLASH_FALLBACK_SPREAD` | off | `0.11` | Positive finite threshold; unset, empty, `0` or `off` disables it. Applies to whole-prompt selections, not region plans |
| `PFLASH_FAIL_CLOSED` | `0` | | `0` serves raw after compression failure when it fits; `1` returns the error. Keep-history admission still serves raw |
| `PFLASH_DRAFTER_SESSIONS` | `2` | | Integer >= 0; capped at 16; `0` scores from scratch. Stage 2 uses its own rows and ignores this setting |
| `PFLASH_DRAFTER_TRIM_POOL` | `1` | | Release cached temporary allocations from a resident drafter's pool |
| `PFLASH_SESSION_OFFLOAD_PINNED` | `0` | | Plain host memory; `1` opts into pinned host allocations |

## Selection and model assets

| Variable | Built-in default | Release profile | Values / requirements |
|---|---|---|---|
| `PFLASH_SCORING_HEAD_GGUF` | unset | | Trained head file; otherwise built-in attention-mass scorer. Setting it selects the release profile |
| `PFLASH_SEGMENT_PROBE_GGUF` | unset | | Segment probe file; otherwise fixed chunks |
| `PFLASH_REQUEST_PROBE_GGUF` | unset | | Request probe file; required for query-probe selection outside the test stub |
| `PFLASH_SELECT_MODE` | unset: legacy | `budget_only` | `budget_only`, `top_p`, `top_k`; setting a mode enables strict selection |
| `PFLASH_SELECT_CHUNK_SIZE` | scheduled in strict mode | `256` | Positive integer; 128 below 500 input tokens, 512 below 3000, otherwise 1024 |
| `PFLASH_SELECT_QUERY_TOKENS` | `8` | `128` | Integer in [1, 512] |
| `PFLASH_SELECT_QUERY_PARSER` | `arbitrary_tail` | | `arbitrary_tail` or `latest_user`; default uses the latest user turn in chat |
| `PFLASH_SELECT_QUERY_HEAD` | `0` | `1` | `1` adds prose around pasted material; needs strict selection |
| `PFLASH_SELECT_QUERY_HEAD_TOKENS`, `PFLASH_SELECT_QUERY_TAIL_TOKENS` | `256` each | | Integers in [2, 512] |
| `PFLASH_SELECT_QUERY_PROBE` | `0` | `1` with the request probe | `1` replaces scoring windows with request-probe query tokens; needs strict head selection |
| `PFLASH_SELECT_SEGMENTS` | `auto` | `probe` with the segment probe | `auto`, `fixed`, `probe`; auto uses probe segments when loaded and the probe finds at least 4 boundaries; probe always does |
| `PFLASH_SELECT_SCORE` | `auto` | `density` | `auto`, `sum`, `density`; auto uses density for probe segments, sum for fixed chunks |
| `PFLASH_SELECT_SCORER`, `PFLASH_SELECT_SPLIT` | `head`, `0.5` | | `head`, `legacy`, `split`; split fraction finite and in (0, 1) |
| `PFLASH_SELECT_TOP_P`, `PFLASH_SELECT_TOPK` | `0.95`, unset | Top-p finite in (0, 1]; top-k positive integer, required in `top_k` mode |
| `PFLASH_SELECT_STRATEGY` | `segments` | `segments` or `fold`; fold needs `budget_only` and the head scorer |
| `PFLASH_SELECT_FOLD_K`, `PFLASH_SELECT_FOLD_CAP`, `PFLASH_SELECT_FOLD_HEAD` | `20`, `1500`, `150` | K and cap >= 1, head >= 0; with fold, head cannot exceed cap |
| `PFLASH_SELECT_FOLD_PARAGRAPHS` | `1` | Paragraph completion as fold's last fallback |

Query-probe, identifier and spread switches modify selection; they do not
enable strict selection on their own.

## Relative queries and retained experiments

| Variable | Built-in default | Values / requirements |
|---|---|---|
| `PFLASH_QUERY_CANDIDATES`, `PFLASH_QUERY_PIN` | `0`, `0` | Candidates in [0, 8], pin tokens in [0, 4096]; positive enables the feature. The release profile sets `1` and `512` with both probes, unless request stage 2 is set or the query probe is off |
| `PFLASH_QUERY_UNIT_SCORE`, `PFLASH_QUERY_AGG` | `max`, `softmax` | `max` or `mean`; `softmax` or `wmax` |
| `PFLASH_QUERY_TEMP`, `PFLASH_QUERY_NULL_MARGIN` | `1`, `0` | Finite temperature > 0; finite null margin >= 0 |
| `PFLASH_QUERY_EXTENT`, `PFLASH_QUERY_PIN_MARGIN` | `gap`, `gap` | `gap` or finite number >= 0; pin margin also accepts `inf` |
| `PFLASH_QUERY_TW`, `PFLASH_QUERY_END` | `0.5`, `0.25` | Explicit shares in (0, 1), sum < 1; TW also accepts `score` |
| `PFLASH_REQUEST_PROBE_POLICY`, `PFLASH_SELECT_QUERY_PROBE_RECENCY_PIN` | GGUF policy | JSON policy file; recency pin in [0, 4096]; checked by the loader |
| `PFLASH_REQUEST_STAGE2_GGUF` | unset | Experimental look-back combiner; needs compatible request probe and trained head |
| `PFLASH_EVAL_REF_IDENT_QUERY` | `off` | `off` or `heur`; heuristic evaluation arm needs strict head selection and excludes query-probe selection |
| `PFLASH_QUERY_GRID`, `PFLASH_QUERY_GRID_OUT` | unset | Replay input and append-only result file; grid needs query-probe selection and an output path |
| `PFLASH_QUERY_ORACLE_DIR` | unset | Oracle evaluation output during grid replay |
| `PFLASH_QWEN35_LEGACY_SCORER` | `0` | Force the original Qwen3.5 scorer |
| `PFLASH_SELECT_QUERY_PROBE_STUB` | `0` | Test-only request-probe stub |
| `PFLASH_COMPRESS_ANCHOR_RADIUS`, `PFLASH_COMPRESS_MAX_ANCHOR_HITS` | tier default | Legacy anchor controls |
| `PFLASH_FREEZE_HOT_WINDOW` | `2` | FlowKV's recent uncompressed message window |

Query tuning overrides require positive candidates or pinning. TW/END blend
overrides additionally require candidates. Candidates, pinning and grid replay
need the segment probe and cannot combine with request stage 2.

`PFLASH_TRACE_PATH` and `PFLASH_VIEW_TRACE_PATH` append selection and served-view
JSONL. `PFLASH_TRACE_REQUEST_FEATURES`, `PFLASH_TRACE_REQUEST_LOGITS` and
`PFLASH_TRACE_STAGE2_FEATURES` retain the feature/logit diagnostics.

## Release profile

The PFlash-v2 measurements (Qwen3.8-27B target, Qwen3.5-0.8B drafter, R9700)
used the values in the release-profile columns above. They are the defaults
whenever prefill compression is on and `PFLASH_SCORING_HEAD_GGUF` is set, so
the measured configuration needs only the model files:

```sh
export PFLASH_SCORING_HEAD_GGUF=/path/to/head.gguf
export PFLASH_SEGMENT_PROBE_GGUF=/path/to/segment-probe.gguf
export PFLASH_REQUEST_PROBE_GGUF=/path/to/request-probe.gguf

luce_server /path/to/target.gguf \
  --prefill-drafter /path/to/Qwen3.5-0.8B.gguf \
  --pflash on --prefill-skip-park
```

The server installs the profile's environment variables at startup and logs
how many it applied and which ones it kept from the environment. A variable
that is already set keeps its value, and `--prefill-threshold`,
`--prefill-keep-ratio` or `--prefill-curve` on the command line replace the
profile's threshold or curve. `PFLASH_SELECT_QUERY_PROBE=1` needs the request
probe, and the query candidates and pin need both probes. They also yield to
the experiment arms they cannot combine with: another scorer, the identifier
query, request stage 2, or `PFLASH_SELECT_QUERY_PROBE=0`. Without a head the
built-in defaults apply, as before.

`--prefill-skip-park` stays opt-in. It keeps the target resident while the
drafter scores, which the measurements used on a 32 GB card at
`--max-ctx 139264`; on smaller cards the two may not fit together.

The obsolete `PFLASH_COLD_HISTORY` and `PFLASH_AGENT_HISTORY` names have been
removed: cache admission keeps tool history raw, with paste compression behind
`PFLASH_AGENT_PASTE`. The active experimental families above remain intact.
