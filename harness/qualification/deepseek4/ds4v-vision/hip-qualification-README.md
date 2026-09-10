# Prepared native HIP qualification

**Executed: numerical ISSUES (exit3).** The first standalone component run is recorded in `native-hip-first/`. All probe processes completed; corn repeat was byte-identical, but feature and embedding comparisons failed unchanged thresholds. The harness never launches/stops converters, a text server, or the operator service.

Current deployed harness SHA256 is `152af330bb37f77f8e6f9c795ae46c12f787e8587c42fc210aeb671b205296d5`. It accepts a completed text-proof directory or `--component-only` as argument1, a fresh absolute evidence directory as argument2, and `--gpu-window-released` as argument3. Component-only execution is independently authorized while CPU conversion runs; it makes no text/chat acceptance claim. Both modes require a released window and double checks for inactive operator/PID0, free8016/8217, empty KFD process inventory, and at least8GiB host and discrete VRAM available. Component mode passed independent review in `component-window-review.md`.

The selected runtime is `4bf727077cf007352997798edc52f32c1f887023` in `soulf:~/lucebox-ds4v-runtime`. The existing `/tmp/ds4v-runtime-hip-build/ds4v_vision_probe` and its three GGML shared libraries are pinned by SHA256 in the harness. Probe SHA256 is `4dc9430cd56ca5afe6e649adfa22ad5065b2c97cede86d0ac0b75b2f9296377e`. The executed device check confirmed hip:0 as Radeon RX7900XT/gfx1100; device1 enumerates as gfx1151.

For the completed-text-proof mode, run manually with **both exact directories**, never a `latest` pointer:

```sh
bash /tmp/ds4v-hip-qualification.sh \
  /home/marcelorm/lucebox-ds4v-mix-parallel/artifacts/fitter-fix/load-proof-EXACT_COMPLETED_RUN \
  /home/marcelorm/lucebox-ds4v-runtime/artifacts/hip-qualification-UNIQUE_RUN \
  --gpu-window-released
```

These example directory suffixes are placeholders. The text evidence must be an existing `load-proof-*` directory under the serial or parallel converter's `artifacts/fitter-fix`. It must contain `harness.exit=0`, a cleanup record, the passing text/math/speculation verdict and a recorded server PID that is no longer present. The new absolute HIP evidence directory must not exist, and its parent must exist. Gate failures create no HIP evidence and make no GPU call. The explicit release flag records the caller's already-granted GPU window; it is not automatic authorization.

The harness then:

1. Uses the existing immutable CPU-reference venv interpreter with `-I` under `env -i`. Only HOME, fixed PATH/locale and two-thread CPU math limits remain. All inherited device masks/overrides, GGML/DS4/DFLASH controls, Python settings and dynamic-loader overrides are removed.
2. Verifies the exact source commit, clean tracked source, pinned binary/libraries/compare script, accepted mmproj SHA256 `58eb6b63243df2db21261ced5568b385b04991f38d45d39b781309497abd4b1c`, original reference manifest and every used patch/feature/embedding file. Records resolved transitive shared-library hashes, Python, environment, harness and text-proof file hashes in `provenance.json`.
3. Runs `--load-only 4096 129280 hip:0` and requires the actual log to identify `backend=ROCm0 requested=hip:0`, device0 as 7900 XT/gfx1100 and device1 as gfx1151. Only hip:0 receives a runtime/weight allocation. Device enumeration does not execute the graph on device1. An unavailable HIP backend fails; there is no CPU fallback command.
4. Runs carrots 42×61, corn 23×34, then corn again, each as a separate sequential hip:0 process. Stage dumping is disabled to keep scratch bounded. The existing probe reports weights/scratch/encode duration and exercises its release/error checks. This measures complete probe behavior including its host transfers; it is not a warmed persistent-runtime throughput benchmark.
5. Runs the **unchanged** selected `compare.py` twice. The second native directory contains the new corn result and explicit symlinks to the first carrots result because the comparator requires both names. It does not run carrots twice or replace any reference. Shape, finite, max-absolute, RMSE and cosine measurements remain in `comparison.json` and `repeat-comparison.json`. Corn repeat byte identity is reported separately.

The fixed gates remain features max-absolute≤0.25, RMSE≤0.03, cosine≥0.9995; embeddings max-absolute≤0.75, RMSE≤0.08, cosine≥0.9990. Either fixed comparison's exit **3** is preserved as the final harness exit even if a later step also fails; it never becomes a success. Features and embeddings have separate summary statuses, so passing embeddings cannot erase feature ISSUES. A repeat-byte mismatch also yields overall ISSUES/exit3; it is separately identified rather than changing the numerical thresholds. Execution/shape/provenance failures remain unqualified, with their error and individual process exits recorded. Interrupted/timed-out runs exit 130/143 or 124 unless an already-recorded numerical exit3 takes precedence.

Each process has an exact command/PID/log/exit and `*.time.json` containing `wait4` user/system time, wall time and peak RSS KiB. `*.memory.jsonl` samples process status, host memory and DRM VRAM/GTT usage every half-second. DRM observations are device-wide and may include unrelated allocations; they are not an isolated per-process GPU peak. `memory-before.json`, `memory-after.json`, `outputs.sha256.json`, `summary.json` and `harness.exit` complete the evidence. Each process has a 900-second ceiling.

On interruption or timeout, cleanup terminates only the harness's unreaped direct child whose PID it recorded; after five seconds it may kill that same child. It uses no name matching, port cleanup, GPU reset or other-process signal. Inputs, references, venv, source and libraries are read-only. Output files remain in the fresh evidence directory even on failure.

Preparation checks: soulf `bash -n` and embedded Python `ast.parse` both passed without executing qualification code or the probe. Parent subsequently reviewed the actual probe/comparator contract, text-verdict schema, pinned inputs, clean environment, PID ownership and preserved numerical failure status, then copied the script to soulf and verified its hash and shell syntax. No qualification PASS, HIP feature parity, maximum-grid behavior or decoder/HTTP acceptance is claimed by preparation.

## First component result

`native-hip-first/summary.json` records all commands, PIDs and exits. All native probes exit0; both comparisons exit3. Corn feature cosine0.98593858015, carrots0.99388401645; embedding cosine0.99263256728 and0.99645496479 respectively. All four fail their unchanged cosine gate. Corn repeat is byte-identical. No HTTP/decoder image support is qualified by this result.
