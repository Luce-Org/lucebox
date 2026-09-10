# Original-source corn HIP control — completed diagnostic

**The source HIP forward completed, but source-HIP/CPU feature portability fails the unchanged gate. Native HIP also fails against source HIP for both features and embeddings. Native vision remains NOT QUALIFIED.** These results separate a source backend portability effect from an additional native discrepancy; they do not establish its exact cause or justify widening thresholds.

## Execution and identity

The first released attempt used the original frozen runner17ba9d66 unchanged. It exited1 at the exact marketing-name guard before weight loading/forward. That evidence is preserved in `hip-supervision/` and `hip-control-initial-failure.md`.

The parent then authorized one identity-only query and a narrow name correction if the actual device agreed. Identity-only Python PID3362155 exited0, reporting exactly one visible device: `Radeon RX 7900 XT`, gfx1100,21458059264 bytes (20464 MiB), selected by ROCr UUID `GPU-93a97448a27aeff3`, PCI bus198/device0. The mismatch was the expected string's `AMD ` prefix.

The reviewed correction changes only that exact expected name and prints actual properties before asserting them. The original source runner and original `vision.py` remain unchanged. `radeon-name-correction.diff` records the two-line correction. New runner `source-forward-radeon-name.py` SHA256:
`cd34fd3df07963bea9014ccc6f405ec95400a7b8b70e4233fe5d3ead7dc9468c`.

The parent received this hash before execution and reviewed the diff. Supervision reused the operator/idle/memory/hash guards, created fresh `hip-supervision-confirmed/` and `hip-corn-confirmed/`, and imposed a300-second deadline. The exact Python command/environment are in `hip-supervision-confirmed/run.json`; direct `os.wait4` supervision recorded the actual Python PID/resource usage instead of GNU time. It signals only that unreaped direct child if necessary.

- Before:operator inactive/MainPID0; no ports8016/8217; KFD empty;36154867712 bytes host available;21430087680 bytes discrete VRAM free.
- Actual corrected Python PID3362755, exit0, elapsed5.758919639 seconds; user4.532378/system0.985126 seconds; peak RSS2504072 KiB.
- Original source forward0.798916269 seconds; GPU peak allocated1176237056 bytes, reserved1186988032 bytes.
- After:operator inactive/PID0, ports free, KFD empty, discrete VRAM back to21430087680 free bytes. GPU ownership was released to the parent immediately after completion.

There was exactly one actual source corn HIP forward. The initial guard failure and identity-only query did not run a model. No additional image, Torch version, SDPA variant, source math, native code, operator, converter or fixture was changed.

## Fixed three-way comparison

All compared tensors are finite and have exact expected shapes:features782×1024, embeddings96×4096. The CPU-only comparison verified the original manifest and every input file hash. Existing fixed gates remain features maxabs≤0.25/RMSE≤0.03/cosine≥0.9995; embeddings maxabs≤0.75/RMSE≤0.08/cosine≥0.9990. All three comparisons retain their own verdicts.

| Stage | Pair | Max absolute | RMSE | Cosine | Gate |
|---|---|---:|---:|---:|---|
| Features | Source HIP vs original CPU | 2.73828125 | 0.007125659433 | 0.997729789065 | FAIL |
| Features | Native HIP vs source HIP | 1.2039794921875 | 0.014023936578 | 0.991196938632 | FAIL |
| Features | Native HIP vs original CPU | 2.9617919921875 | 0.017743029365 | 0.985938580153 | FAIL |
| Embeddings | Source HIP vs original CPU | 0.09130859375 | 0.003126610779 | 0.999115331698 | PASS |
| Embeddings | Native HIP vs source HIP | 0.16259765625 | 0.007664476777 | 0.994697536836 | FAIL |
| Embeddings | Native HIP vs original CPU | 0.176513671875 | 0.009035238955 | 0.992632567277 | FAIL |

The new AMD Torch environment's earlier CPU outputs were byte-identical to the immutable original CPU fixtures. The source-HIP/CPU gap therefore appears when executing the original graph on HIP in this controlled environment. This observation is bounded to this image/runtime/default dispatch; it is not a universal backend-error estimate. Native-HIP/source-HIP still fails both stages, so source portability does not explain away the native discrepancy. The old native-HIP outputs are the parent's completed frozen `hip-component-first/native` files, not a new native run.

## Exact output identities

| Stage | Producer | SHA256 |
|---|---|---|
| Features | Original CPU | `aa7c43be7182759f83881cf823661bf52c14b73222645d1ec37b14c6502bc982` |
| Features | Source HIP | `5790c492de2618560f81bac3ab8a70282a272be0e1885b95246621aab0bf4bb8` |
| Features | Native HIP | `59bd19a13750d07f7f1018c32c5a43c4ae2cd7a7ff132da3f6201b08c400cc4e` |
| Embeddings | Original CPU | `c96d59ae722ad8ac31299aabb4e758b788a1ee4be30ea94b833c753721229040` |
| Embeddings | Source HIP | `80a30a096a9dd84e91f8d47d63b1cf00b3eded88f6e472b1018bbdabb697ab86` |
| Embeddings | Native HIP | `a398c9c10a7b2bbeb63f4b910bf5f71bb9fefd0aa388b276a3bf06ca3e280a06` |

Comparison script SHA256:`3741e93cab886c9050e2aa484236a6b15d7aa8b6f3ed55ae1e8acfe37cf4d555`; comparison process exit0 means metrics completed, not numerical acceptance.

Local evidence is under this report's directory: `hip-identity-supervision/`, `hip-supervision-confirmed/{run.json,hip-corn.log,memory.jsonl,three-way.json,three-way.log}`, and `hip-corn-confirmed/report.json`. Complete output tensor files remain in the corresponding soulf root `~/ds4v-work/source-rocm210-reference/`. Original runner17ba9d66, private library hash, original source/weight/reference hashes and exact pins remain preserved. This report does not qualify another image, source repeat stability, end-to-end image answers or later server integration.
