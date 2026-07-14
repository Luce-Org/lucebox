# Vendored llama.cpp/ggml snapshot

This directory contains the ggml-only subset used by Lucebox Hub.

- Source repository: https://github.com/Luce-Org/lucebox-ggml
- Source base branch: `luce-dflash`
- Source base commit: `6fbe72d67069136bbd370be703e1d4f441b5e942`
- Included merged PR: `#35` (`0fe65d9354b7c5da52a7741d2e37ba85f0d0c925`)
- Included test PR: `#37` (`0699be81480428f01b9b7ac49a09a2d51c77f8df`)
- Included upstream backport: `llama.cpp #22298` (`9725a313be0528214c4a02fed906ddaf7b3f712e`)
- Reconstruction: `luce-dflash@6fbe72d67069136bbd370be703e1d4f441b5e942` plus cherry-picked PRs `#35`, `#37`, and upstream `llama.cpp #22298`
- Vendored paths: `LICENSE`, `common/jinja`, `common/log.h`, `common/unicode.*`, `ggml`, `gguf-py`

Open ggml feature PRs are intentionally not included until they are merged, except for explicitly listed hub test PRs.

## Hub-local deltas

- `ggml/src/ggml-cuda/gated_delta_net*`: Lucebox selects the existing grouped-column
  Gated DeltaNet kernel for Ampere prefill launches of at least 512 tokens while
  retaining the classic kernel for decode and partial chunks. The force/disable
  environment controls remain available for differential testing. This policy is
  maintained locally until it is suitable for upstreaming to `lucebox-ggml`.
