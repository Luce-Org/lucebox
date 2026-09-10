# Scoped HIP linear qualification harness review

**Verdict: PASS for execution only after the scoped tiny regression and CPU
identity gates pass and the parent explicitly releases the GPU lane.** Reviewed
`hip-linear-qualification.sh` at SHA256
`8d3e34a5df237da0e4bb3098ff03e1d6694e91d547700879e927ddb2ff5c6806`.
This is a read-only harness review, not a tower qualification result.

The harness pins candidate source `be8b0f1b07f1a3a034ce1d7333fd0d3402754c60`,
probe `79f4928a5172001eef205b100d7578f77560a566d4179fe6755951a15438b483`,
the three reused GGML libraries, projector, comparator, original CPU manifest,
canonical 7900 XT source manifest and its freeze. Source policy SHA256
`62b3daef05bcaaac175f6907b8748d64952a4bf96baab316bab9b2140451929f`
matches the adopted prospective policy. The frozen target manifest and freeze
are `677b5ef033d009a4c44f9fcf7207276e55c4ec8968044ed237f709a108cb3f86`
and `8ab35a8a7adc8c66678af7f50b6de610777f0cf03077e108d1d381957cbe8ce0`.
They record first-output selection, byte-identical source repeats for both images,
the exact runner and device, and unchanged thresholds before candidate execution.

The revised integrity boundary is complete. It hashes all target and original CPU
patch, feature and embedding bytes before the first GPU operation, includes both
sets in provenance, and rehashes both after all comparisons. It also rechecks all
pinned files. Candidate execution uses only the frozen target patches. The two
original-CPU comparisons are reported separately and their numerical exit 3 does
not control target acceptance.

Target acceptance requires both the first and repeat comparisons to complete
normally and apply every unchanged feature and embedding gate to both images.
Corn output must also repeat byte for byte. Either target comparator exit 3 or a
repeat mismatch produces final exit 3. Execution, shape, hash, load, device or
timeout failures remain unqualified. Once a target comparator records numerical
exit 3, the finalizer cannot turn it into success or hide it behind a later error.

The idle-window, exact device, no-fallback, owned-child cleanup, timeout and fresh
evidence-directory controls are retained from the previously reviewed harness.
The run remains scoped to the 7900 XT tower and does not qualify CPU portability,
gfx1151, HTTP image behavior, the decoder, text regression or full serving.

No build, model execution, GPU operation, server action or source edit was made
during this review.
