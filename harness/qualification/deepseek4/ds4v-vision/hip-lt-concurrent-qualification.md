# Native HIP image qualification

Prepared, not executed. `hip-lt-concurrent-qualification.py` supersedes the
idle-only draft for this prospective numerical window. It does not start a
server or authorize a paired GPU HTTP test.

The candidate is `6137f4305400247fed98d2144634c184e2bc6b13`. Its clean HIP
probe is `635a49b45d116a62d405898230389f41e77aa51d47437c02ffa647919fd458d7`.
CPU execution already preserves all four prior full-image outputs exactly;
the older CPU-versus-source feature failures remain visible.

Release requires the reviewed concurrent guard, concrete immutable workload
policies, and the accepted six-lane linear receipt for this same candidate.
The three full-image lanes are carrots, corn, and a second corn execution.
Each is a separate direct child of the live operator guard, sees one Radeon,
and must execute all 67 fused biased projections with the fixed 76 MiB
workspace. The guard checks the existing Strix operator and its resources
before, during, and after every lane. Unrelated opaque non-KFD processes are
a recorded visibility limitation; this is not exclusive device ownership or
a performance benchmark.

The original target policy is unchanged, SHA256
`62b3daef05bcaaac175f6907b8748d64952a4bf96baab316bab9b2140451929f`:

| Output | Maximum error | RMSE | Minimum cosine |
| --- | ---: | ---: | ---: |
| Features | 0.25 | 0.03 | 0.9995 |
| Embeddings | 0.75 | 0.08 | 0.9990 |

All stages and both images must pass against the frozen first original-source
Radeon outputs. Both corn outputs must repeat byte-for-byte. Separate CPU
portability reports cannot replace the target comparison or hide its failure.
The comparator, references, complete Python/numpy runtime inventory, candidate
binary and actual linked HIP libraries are pinned. The umbrella `libggml.so`
is a pinned build artifact but is not linked into this standalone probe.

After comparisons, input/source checks repeat and a final locked read-only
operator preflight must pass before the report can indicate success. A
successful numerical report would permit the planned production integration;
it would not establish that HTTP image input or image-based answers work.
