# DS4 production exact-differential trace

The production trace is disabled unless `DFLASH_DS4_EXACT_TRACE_PATH` names an
output file. Enabling it adds synchronous device readbacks and is diagnostic
only. It does not alter model arithmetic, routing, cache mutation, capture
selection, or ROCTX ranges.

## Schema and comparison contract

Every JSONL record uses schema `lucebox.ds4.exact-diff/v1`. The runner writes a
`manifest` record with the revision, binary SHA-256, target and drafter SHA-256,
prompt-byte SHA-256, fixed request configuration, exact width, and tolerances.
The production backend appends request, layer, cache, capture, logits, snapshot,
reset, continuation-token, and completion records. Each `request_start` carries
the bounded production prompt token-ID vector as well as its little-endian
signed-int32 hash, so trace consumers can authenticate the real producer
schema rather than reconstructing tokenization.
The backend also appends one `model_config` record containing the exact SWA
window and every distinct compressor boundary. Boundary relations include the
numeric boundary, so matrix validation proves before/on/after coverage for each
configured value rather than accepting one representative boundary.

The comparator always uses q=1 as the oracle and stops at the first mismatch.
Generated text is never an oracle. It aligns records by profile, request,
committed position, layer, token position, and field. Routing IDs, token IDs,
cache positions, counts, and state hashes are exact. Floating values use:

| Field | absolute tolerance | relative tolerance |
|---|---:|---:|
| routing weights | `1e-6` | `1e-6` |
| DSpark capture rows | `1e-5` | `1e-5` |
| final logits | `1e-4` | `1e-4` |

The inclusive rule is `abs(a-b) <= atol + rtol * max(abs(a), abs(b))`. Any NaN
or infinity is a hard failure, including two matching infinities. Manifests
must use these exact built-in tolerances; trace inputs cannot weaken them. HC, raw KV,
compressed KV, attention-compressor state, indexer-compressor state, and
indexer KV use exact deterministic byte hashes. Their trace records also carry
byte counts. DSpark capture records contain every value in every captured row;
the comparator applies the declared tolerance to the complete row and rejects
missing capture positions.

## Correctness matrix

The `reset` profile runs the same request twice with both prefix caches disabled.
The `snapshot` profile runs it twice with the full-prompt snapshot cache enabled,
so the second request must restore the first request's state. For every profile,
q=1, q=2, q=3, and q=4 use the same prompt bytes and request configuration.
To cover all three tail widths with one pinned prompt, its production token count
must satisfy `N % 12 == 11`; q=2, q=3, and q=4 then end with widths 1, 2, and 3.
The comparator rejects a trace set that does not actually contain those rows.
Every request must end successfully and emit exactly one completion and token
record. The repeated request must use the same prompt and sampling configuration
and produce the same continuation tokens; reset repetitions also compare final
logits. A full-prompt restored request performs no prefill, so its DSpark state
is authenticated through the snapshot's cache, logits, and feature hashes
rather than through impossible duplicate capture rows.
Across all eight traces, the comparator requires one binary, revision, model,
prompt, tolerance contract, and fixed request configuration. Only the profile,
width, exact-band setting, and per-process port may vary. The observed prompt
tokens, generation count, sampling settings, and width must also agree with the
manifest and with every other trace.

The production position filter records:

- ordinary q=2, q=3, and q=4 steps;
- observed tail widths 1, 2, and 3;
- steps before, on, and after every model compressor ratio;
- steps before, on, and after the model SWA boundary;
- every hash-routed and learned-router layer at each selected step;
- the first and repeated/reset request;
- snapshot save and restore events;
- the beginning and end of the DSpark final capture window;
- final cache position, declared-tolerance logits, greedy token, and all
  continuation token IDs.

The comparator fails if a required matrix relation has no trace evidence. At
every selected step it also requires the complete oracle layer-key set, checks
the committed cache position and exact-band readout policy, and requires both
ends of each non-restored request's four-token DSpark final-capture window.

## Commands

Generate all traces (two profiles, four exact widths):

The q=2 through q=4 runs require a server revision that implements
`DFLASH_DS4_EXACT_PREFILL_BANDS`. On a revision without exact bands, comparison
fails closed because the required step widths and tail coverage are absent.

```bash
python3 harness/ds4_exact_diff.py run \
  --binary server/build-hip/dflash_server \
  --target /models/target.gguf \
  --draft /models/dspark.gguf \
  --prompt /data/prompt.txt \
  --target-device hip:0 \
  --output-dir /tmp/ds4-exact-diff
```

Compare q=2, q=3, and q=4 against q=1:

```bash
python3 harness/ds4_exact_diff.py compare \
  --trace-dir /tmp/ds4-exact-diff
```

The run command hashes its inputs before launching the server, forces greedy
sampling, keeps exact attention, disables approximate/fused verification, and
uses `--chunk q` with the exact-band flag disabled for q=1 and enabled for
q=2..4. `--server-arg` rejects options that could override those fixed
invariants, and a run without `--draft` clears inherited DSpark activation.
Raw traces and server logs belong outside Git.
