# DS4V vision qualification kit

Reproduction material for the DS4V vision numerical gate, shipped so the work can be
continued without rebuilding the measurement setup. Context and the full narrative are in
`docs/ds4v-continuation-status.md` and `docs/ds4v-image-serving.md`.

The scripts are reproduced verbatim from the operator host (Strix Halo + 7900 XT, Linux
ROCm). They carry hard-coded host paths and their own idle/resource guards; read
`hip-qualification-README.md` before running anything.

## Where the gate stands

Against the frozen same-GPU original-source reference, thresholds unchanged:

| Check | Result | Gate |
|---|---|---|
| Embeddings corn / carrots | 0.999525047 / 0.999721786 | pass |
| Corn features (cosine) | 0.999063593 | fail (0.9995) |
| Corn features (maxabs) | 0.947265625 | fail |
| Carrots features | cosine 0.999538399 pass, maxabs 0.26416015625 | fail (maxabs) |

Earlier and retained results:

- first native HIP run, runtime `4bf7270`: corn 0.985939, carrots 0.993884, embeddings also
  below gate, harness exit 3 (`native-hip-first/`).
- scoped biased-linear-rounding fix `be8b0f1` (biases promoted to F32 operands on GPU
  backends only) produced the passing embeddings and the corn cosine above
  (`native-hip-scoped/`).
- CPU portability is a separate retained failure: original CPU corn feature cosine
  0.99822935.
- three-way comparison (`source-rocm-reference/hip-supervision-confirmed/three-way.json`):
  the original-source corn HIP forward also fails the unchanged CPU gate (0.997729789,
  maxabs 2.73828125), and native HIP fails against source HIP (features 0.991196939,
  embeddings 0.994697537). CPU/GPU portability therefore does not explain the native
  discrepancy.

## Contents

- `hip-qualification.sh`, `runtime-qualification.sh` - component-only and runtime windows
  with the idle, port, KFD and memory guard.
- `hip-linear-qualification.sh`, `hip-attention-qualification.py`,
  `hip-norm-qualification.py`, `hip-unbiased-qualification.py`, `hip-lt-qualification.sh`,
  `hip-lt-concurrent-qualification.py`, `hip-lt-retry-qualification.py` - per-component
  qualification, each with its review note where one exists.
- `target-hip-qualification-policy.md` (+ review), `component-window-review.md` - the
  adopted same-GPU reference policy and the window review.
- `how-source.md`, `how-backend.md`, `native-tower-brief.md`, `native-tower-rubric.md` -
  the source/backend contracts and the tower acceptance rubric.
- `source-rocm-reference/` - reference environment receipts (`constraints.txt`,
  `cpu-runtime-info.py`, `libtorch-hip-*ldd.txt`, MIOpen receipts, script hashes),
  the freeze and comparison tooling (`freeze-source-reference.py`,
  `compare-corn-three-way.py`) and the control reports.
- `native-hip-first/`, `native-hip-scoped/` - the run summaries and comparisons.
- `mmproj-byte-proof.py`, `reference-fixtures.py`, `capture-comparator-runtime.py`,
  `patch-bias-diagnostic.py` - supporting checks.

## Reproducing

1. Pin the reference environment from `source-rocm-reference/constraints.txt` and rebuild
   the original-source Torch/ROCm control (see `source-rocm-reference/evidence/` for the
   exact library and package receipts).
2. Freeze original-source corn and carrots outputs on the same device
   (`freeze-source-reference.py`) and confirm repeat stability.
3. Run the native component window (`hip-qualification.sh --component-only`) under the
   guard, then the corrected full-tower run.
4. Compare with `compare-corn-three-way.py`; no threshold is adjusted between runs.

## Open decision: what counts as done

The adopted policy keeps every numeric threshold unchanged, but the original-source corn
HIP forward itself fails the CPU feature gate. "Match the frozen same-GPU source" and
"pass 0.9995" are therefore different targets, and the residual corn deviation sits
between them. This needs an owner decision before the tower can be declared qualified.

## Next experiments

- Capture the two preselected real corn projections (patch projection, block-0 QKV) and
  compare them against the frozen source outputs. The synthetic biased projection already
  matches the original GPU source exactly at 1024/1024 values with the source's
  first-heuristic and 76 MiB workspace configuration
  (`artifacts/ds4v-step2/hipblaslt-real-projections/`, not in this kit).
- The residual is sparse rather than structural: corn cosine is close to the gate while
  maxabs is 0.947265625, which points at per-element tie and product-rounding behaviour.
  The direct fused source biased output still differs in 88 tiny tie cases, so a fused
  versus unfused rounding contract is the leading candidate.
- The sensitive-block diagnostic found no semantic or BF16-boundary discrepancy for corn
  blocks 12 and 31, and a single-thread original-source control reproduced the two-thread
  reference bitwise.

## Known gaps

- The two source images (`corn.jpeg`, `carrots.jpeg`) are not in this repository and the
  frozen reference outputs are tied to them. They were deleted from the reference host;
  they can be recovered from the base64 payloads in the trial captures or provided on
  request.
- Guard journals, per-run `guard.json` snapshots and HTTP captures are not included here.
- No production image-chat acceptance has been run against this tower state.
