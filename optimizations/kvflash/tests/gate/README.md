# KVFlash Gate A scripts

GPU tests for KVFlash session restore (see `server/docs/PREFIX_CACHE.md`,
section "KVFlash session restore", and `server/docs/ENVIRONMENT.md`). They start
`luce_server` themselves, one at a time, and need a GPU, a model file and
Python 3 (standard library only).

## What Gate A checks

`run_gate.sh` runs the same 10-turn chat twice against a fresh server each time.

- Arm A: `LUCE_KVFLASH_SESSION=0`. Session restore is off, so every turn
  re-prefills the whole prompt from token 0 (cold). Every turn must log
  `prefix_len=0`.
- Arm B: `LUCE_KVFLASH_SESSION=1`. Live restore is on and replays arm A's prompts
  verbatim. From turn 2 the log must show a nonzero `prefix_len` (the restored
  prefix) and a `live restore` line.

`cmp_gate.py` then compares the two arms turn by turn and prints server-side
medians (prefill seconds, decode tokens per second, wall seconds) taken from the
`chat DONE` log lines, dropping the warm-up request.

`run_scenarios.sh` runs regression scenarios in `scripts/scenarios.py`. Each one
sends setup requests to a live server, then probe requests, and compares each
probe answer with a cold reference (a fresh server that sees only that probe).

1. Pooled restore past the pool: session A over two turns, an unrelated session
   B, then A extended by one turn and A with an edit near token 9000.
2. The 512-token boundary: a long prompt, a short prompt whose snapshot lands on
   the 512 boundary, and probes that share its first ~640 tokens and diverge.
3. Image: a text session, an unrelated image session, and a text probe that
   shares the image prompt's leading text. Needs `KV_MMPROJ`; skipped otherwise.

## Parity settings

Both arms and all references use the same settings: the same model file,
default KV cache type, `--max-ctx 131072`, `--kvflash 8192` (pool size in
tokens), temperature 0, seed 1, a first turn of about 12500 filler tokens, 10
turns, 80 output tokens (64 in scenarios), and a warm-up request before the
measured turns. Only `LUCE_KVFLASH_SESSION` differs between the arms. Scenarios
also set `LUCE_KVFLASH_RESTORE_CONSUME=1`.

## Usage

Export the GPU selection yourself (`CUDA_VISIBLE_DEVICES` or
`HIP_VISIBLE_DEVICES`); the scripts set no device variable.

```
export KV_GATE_DIR=/path/with/space      # output, lock file and TMPDIR live here (required)
export KV_MODEL=/path/to/model.gguf      # target model (required)
export KV_MMPROJ=/path/to/mmproj.gguf    # optional: enables scenario 3
export KV_DRAFT=/path/to/draft.gguf      # optional: enables mode dflash2
export KV_PORT=18290                     # optional

run_gate.sh /path/to/luce_server plain 10      # BINARY MODE [TURNS] [TAG]
run_scenarios.sh /path/to/luce_server all      # BINARY [1|2|3|all] [TAG]
```

Output goes to `$KV_GATE_DIR/out/<mode>-<tag>/` (`provenance.txt` with the
binary path and md5, `gate4_A.out`, `gate4_B.out`, `cmp.out`) and
`$KV_GATE_DIR/out/scenarios-<tag>/` (`scenarios.out`). Mode `dflash2` adds
`--draft $KV_DRAFT` and exits with a SKIP message when `KV_DRAFT` is unset.
The scripts refuse to start while another gate holds the lock or any
`luce_server` process is running, so check the GPU is idle first.

## Pass criteria

- Gate A: every turn's answer in arm B is byte-identical to arm A
  (`IDENTICAL 10/10`, exit code 0). For the result to mean anything, arm A must
  show `prefix_len=0` on every turn and arm B a nonzero `prefix_len` from turn 2.
- Scenarios: every probe answer equals its cold reference, giving
  `SCENARIO n: PASS`. A skipped scenario 3 is not a failure.
