# Prospective 7900 XT qualification policy review

**Verdict: PASS as a prospective, target-scoped numerical policy.** Reviewed
`target-hip-qualification-policy.md` at SHA256
`62b3daef05bcaaac175f6907b8748d64952a4bf96baab316bab9b2140451929f`
before any corrected full-tower output was produced. The post-review status-line
adoption did not change the reviewed procedure. This review does not qualify the
current or corrected native tower.

The target reference is scientifically motivated by observed source behavior,
not selected from candidate results: the unchanged original source on the
7900 XT fails the existing CPU feature threshold, while its embedding remains
within the existing embedding threshold. A same-device source reference controls
the backend-dependent reduction schedule that the original CPU fixture cannot.
The policy keeps the already published CPU comparisons and their failures
visible, so a target result cannot be presented as CPU equivalence.

The freeze is suitably prospective and resists result shopping. It requires the
same original model, weights, patches, BF16 boundaries, default source operations,
pinned software and exact 7900 XT identity; byte-identical source repeats for
corn and carrots; a manifest containing all source outputs and provenance before
the corrected native full tower runs; and no later reference replacement based
on candidate output. The corrected source runner's only semantic change is the
observed marketing-name check and identity logging; its SHA256 is frozen as
`cd34fd3df07963bea9014ccc6f405ec95400a7b8b70e4233fe5d3ead7dc9468c`.

Acceptance remains demanding: both images must independently pass every existing
feature and embedding threshold with exact shapes and finite values, failures
must remain machine-visible, and the native corn repeat must be byte-identical.
The dyadic biased-linear regression can run before the reference freeze because
it neither executes the full tower nor supplies a reference output.

Any PASS is limited to the native tower on the recorded 7900 XT/software tuple.
It does not qualify CPU numerics, gfx1151, other accelerators, decoder behavior,
HTTP image input, image-dependent answers, isolation, resource limits or cleanup.
Those separate gates remain required. The main residual scientific limitation is
fixture breadth: two images establish the selected deployment gate, not general
cross-image or cross-backend equivalence. No tolerance, source implementation or
reference backend may be changed after seeing corrected candidate results.

This was a read-only policy and evidence review. No build, model execution, GPU
operation, server action or runtime source edit was performed.
