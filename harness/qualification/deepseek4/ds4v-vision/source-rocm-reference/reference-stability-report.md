# First-source HIP reference freeze

**SOURCE_REFERENCE_STABILITY_PASS.** Corn and carrots each produced byte-identical features and embeddings in two independent fresh processes on the same7900XT. The FIRST completed output for each image is now the frozen reference. No corrected native full-tower output was inspected or used to select it. This establishes reference stability for the adopted target-GPU policy; it does not accept any native candidate or resolve the original CPU feature failure.

The exact parent policy was independently reviewed at SHA256 `7edde20ee70b804cc903b827dbea1dbc9b8d43d9d352e22f82672758430f9682`. The parent changed only the status sentence after both independent PASS reviews; adopted policy SHA256 is `62b3daef05bcaaac175f6907b8748d64952a4bf96baab316bab9b2140451929f`. Both text versions and hashes are pinned in the provenance manifest, and their bodies were checked identical apart from the Status line. The initial copy raced the parent's status update; the first freeze correctly rejected the reviewed-hash mismatch before writing a reference directory. The exact reviewed text was restored to a separate file and verified against7edde20e; both versions remain preserved.

## Canonical reference and provenance

Remote canonical directory:
`/home/marcelorm/ds4v-work/source-rocm210-reference/source-hip-reference`

Its `manifest.json` uses the existing comparison schema: each image has vit_grid, aligner_grid and patches/features/embeddings entries with file, shape and SHA256. Patch files are exact copies of the immutable original CPU patches. Feature/embedding files are exact copies of FIRST source-HIP outputs, never repeat-selected samples. All canonical files are read-only.

- Canonical `manifest.json` SHA256:`677b5ef033d009a4c44f9fcf7207276e55c4ec8968044ed237f709a108cb3f86`.
- Separate remote `~/ds4v-work/source-rocm210-reference/source-hip-reference-freeze.json` SHA256:`8ab35a8a7adc8c66678af7f50b6de610777f0cf03077e108d1d381957cbe8ce0`.
- Frozen UTC:`2026-09-05T01:40:51.838089+00:00`.
- Source runner SHA256:`cd34fd3df07963bea9014ccc6f405ec95400a7b8b70e4233fe5d3ead7dc9468c`.
- Original CPU manifest SHA256:`38d12f30b9a10ed2f0e99bf4d40a07357ab5e8c4880151c58cb90414e4a53f4f`.

The freeze manifest pins first/repeat raw-byte hashes and report hashes; exact source argv/environment/PIDs/resources; first actual hardware identities; all14 wheel hashes/versions, install and private MIOpen provenance;77 actual loaded shared-library paths and SHA256 values; original source/index/weight inventory/patch identities; source supervision and comparison scripts; both policy hashes and all unchanged numerical gates. Repeats were checked with both SHA256 and whole-file byte comparisons. No original CPU reference was changed.

## Source execution and resources

All source forwards use the same unchanged original modules, weights, BF16/F32 boundaries, default SDPA, original hashed patches, AMD Torch/runtime and private MIOpen. Actual device identity is `Radeon RX 7900 XT`, gfx1100, UUID selection`GPU-93a97448a27aeff3`,20464 MiB. Source processes use two CPU threads. The earlier exact-name failure and identity-only query remain separate; they ran no model.

| Lane | Actual Python PID | Exit | Process seconds | Forward seconds | Peak RSS KiB | GPU allocated / reserved bytes |
|---|---:|---:|---:|---:|---:|---|
| FIRST corn, retained | 3362755 | 0 | 5.758920 | 0.798916 | 2504072 | 1176237056 / 1186988032 |
| Corn repeat | 3366985 | 0 | 5.234832 | 0.389909 | 2503140 | 1176237056 / 1186988032 |
| FIRST carrots, retained | 3367102 | 0 | 5.763091 | 0.855355 | 2517764 | 2089592320 / 2134900736 |
| Carrots repeat | 3367383 | 0 | 5.750177 | 0.855419 | 2517508 | 2089592320 / 2134900736 |

Every lane passed operator inactive/MainPID0, no ports8016/8217, empty KFD and≥8 GiB host/discrete-memory preflight. Each had a300-second deadline; no timeout or signal occurred. Each source child exited before the next launch. After each, KFD was empty and discrete free VRAM returned to21430087680 bytes. GPU ownership was released to the parent immediately after the three new source stability children exited; subsequent manifest work was CPU-only. The timing difference between first/repeat corn was not used to select a reference or make a speed claim.

## Frozen FIRST output hashes

| Image | Stage | SHA256, also matched by its repeat |
|---|---|---|
| Corn | Features | `5790c492de2618560f81bac3ab8a70282a272be0e1885b95246621aab0bf4bb8` |
| Corn | Embeddings | `80a30a096a9dd84e91f8d47d63b1cf00b3eded88f6e472b1018bbdabb697ab86` |
| Carrots | Features | `270b09f7b62d47162137613df78c5735284dee5b2d31a42892e12ec9631d57b1` |
| Carrots | Embeddings | `4eaf4a6de24d0b9c6cab3d42ec13a3a74e7a06627c21262106e6b059ac4bbb4f` |

## Original CPU portability remains separate

| Image/stage | Max absolute | RMSE | Cosine | Original gate |
|---|---:|---:|---:|---|
| Corn features | 2.73828125 | 0.007125659433 | 0.997729789065 | FAIL |
| Corn embeddings | 0.09130859375 | 0.003126610779 | 0.999115331698 | PASS |
| Carrots features | 0.1484375 | 0.002524693483 | 0.999693162950 | PASS |
| Carrots embeddings | 0.0302734375 | 0.001367435885 | 0.999825389498 | PASS |

All outputs are finite and shapes match the original manifest. Feature gates remain maxabs≤0.25/RMSE≤0.03/cosine≥0.9995; embedding gates remain maxabs≤0.75/RMSE≤0.08/cosine≥0.9990. The failed corn CPU feature gate remains a failure. The previously measured native-HIP/source-HIP comparison still fails both corn stages; nothing in this freeze relabels it.

## Review and limits

The independent recommendation is `target-hip-acceptance-review.md`. Target-GPU fidelity is a justified, explicitly different question from cross-device CPU portability. Matching the original source on this GPU cannot rule out an underlying source-HIP backend defect shared by another implementation; it is not absolute numerical ground truth. That limitation, the unchanged CPU failures, default-SDPA dispatch dependence and the experimental AMD fork/host tuple must remain visible. Two stable fixtures are not universal reproducibility or accuracy coverage. The corrected native GEMM regression, target-tower gates and separate end-to-end image behavior remain required before claiming working vision.

Local copies: `source-hip-reference-freeze.json`, `evidence/source-hip-reference-manifest.json`, `source-corn-repeat/`, `source-carrots-first/`, `source-carrots-repeat/`, and corresponding `*-supervision/` directories under this report's directory. The local manifest is a review copy; complete canonical tensor files remain on soulf. Copied manifest hashes were rechecked locally. No further GPU execution occurred after release.
