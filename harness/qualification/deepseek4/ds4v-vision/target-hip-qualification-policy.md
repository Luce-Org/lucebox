# Prospective 7900 XT vision qualification

Status: adopted prospectively after two independent PASS reviews. No corrected full-tower candidate output has been produced under this policy. The original CPU fixtures and their thresholds remain unchanged and all comparisons to them stay visible.

## Evidence motivating the target reference

The unchanged original source, fixed AMD Torch2.10/ROCm7.2.4 environment, original weights and identical corn patches produced a source-HIP feature cosine of0.997729789 and maximum absolute difference2.73828125 against original CPU. These fail the existing feature gate. Source-HIP embeddings pass the original embedding gate. This demonstrates that the existing CPU feature threshold is not portable even to the original implementation on this target GPU; it does not establish native correctness. The existing native HIP implementation also fails against source HIP, with feature cosine0.991196939 and embedding cosine0.994697537.

Evidence: `source-rocm-reference/hip-supervision-confirmed/three-way.json`. The narrowly corrected original-source runner only logs actual device identity and matches the observed marketing name. Its source SHA256 is `cd34fd3df07963bea9014ccc6f405ec95400a7b8b70e4233fe5d3ead7dc9468c`. Model code, weights, precision boundaries, default SDPA, image patches and library versions are unchanged.

## Reference freeze before candidate execution

Use the same original-source runner on exactly Radeon RX7900XT/gfx1100, selected by UUID `GPU-93a97448a27aeff3`, with its pinned environment and weights. Retain the first completed corn output. Run one corn repeat and two carrots runs in fresh processes/directories. Require each image's features and embeddings to repeat byte-identically; otherwise stop qualification and investigate reference stability without choosing a favorable sample.

Freeze the first source outputs for both images, their repeat hashes, the existing CPU manifest and patch hashes, runtime/library provenance, hardware identity, exact runner and comparison scripts in one manifest before executing the corrected native full tower. Never replace references based on candidate results. The small exact-dyadic biased-linear regression is independent of this full-tower policy and may execute first.

## Unchanged numeric gates, explicit scope

For both images separately, require exact expected dimensions, finite values and every existing gate against frozen same-GPU source output:

- Final normalized features: maximum absolute difference<=0.25, RMSE<=0.03, cosine>=0.9995.
- Aligner embeddings: maximum absolute difference<=0.75, RMSE<=0.08, cosine>=0.9990.
- Native corn repeat must be byte-identical. Failed metrics remain machine-visible failures with nonzero qualification exit.

Publish native-vs-original-CPU and source-HIP-vs-original-CPU results beside the target comparison using the unchanged thresholds. A target PASS cannot relabel those CPU portability failures as PASS. CPU functional/regression suites remain required, while numerical CPU tower qualification remains separately unresolved unless its original gate actually passes.

A successful result qualifies only this native tower on this 7900XT/software configuration. It does not qualify Strix vision execution, other GPU architectures, CPU tower numerics, full decoder parity, other vision model architectures, or image chat. The supported placement keeps the tower on7900XT; Strix continues to own tail language experts. Memory limits, unsupported-path errors, transactional loading and exact projector/preprocessing contracts still apply.

## Integration and behavior remain separate

Only a passing target tower may clear the tower dependency for the selected heterogeneous runtime integration. Actual HTTP image input, correct image-dependent answers for both fixtures and equal-layout/different-image isolation, malformed-input behavior, text regression, GPU ownership, memory/latency and cleanup remain required. Sparse decoder prefill remains explicitly approximate.

Do not sweep tolerances, source versions, precision modes or reference backends to obtain a pass. Any further change to qualification policy must be prospective, separately motivated and independently reviewed, with old failures retained.
