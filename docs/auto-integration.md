# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-27T22:56:45-04:00 through 2026-05-27T22:56:45-04:00
Current stack tip before this refresh: `a448139` (already aligned with `origin/main` `4f4d82e`; this run found no upstream-base advance and revalidated all open non-draft contributor PR heads with fresh fetches, ancestry checks, and direct merge probes).

## Included in the current stack

| PR | Head branch | State | Notes |
|---:|---|---|---|
| #288 | `fix/laguna-chat-template` | included | Current head `5e8136a` is an ancestor of the refreshed stack. |
| #287 | `feat/gemma4-timings` | included | Current head `b3163f4` is an ancestor of the refreshed stack. |
| #284 | `fix/draft-safetensors-rope-theta` | included | Current head `697198a` is an ancestor of the refreshed stack. |
| #276 | `fix/qwen36-claude-code-tool-calling` | included | Current head `0e3c79a` is an ancestor of the refreshed stack. |
| #274 | `feat/pflash-drafter-ee7` | included | Current head `5037b28` is an ancestor of the refreshed stack. |
| #266 | `feat/harness-typed-adapters` | included | Current head `17525ea` is an ancestor of the refreshed stack. |
| #152 | `main` | included | Current head `cf735be` is an ancestor of the refreshed stack. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | included | Current head `f2fbf62` is an ancestor of the refreshed stack. |
| #174 | `split/gemma4-14-small-vram-docs` | selectively included | The docs-only VMM guidance remains ported to current `server/README.md` after the upstream README refresh. The older inherited Gemma4 commits on this branch are not cherry-picked. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / selectively included | Not an ancestor after the `dflash/` → `server/` migration, but its safetensors SWA config parsing and causal-mask support remain present under current draft/common code. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / selectively included | Not an ancestor after layout migration, but the daemon reset regression test and follow-up daemon reset fixes remain carried. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / selectively included | The old `dflash/CMakeLists.txt` change is obsolete; current `server/CMakeLists.txt` explicitly resolves CUDA architectures and carries Blackwell/GB10 handling. |
| #39 | `feat/moe-35b-a3b` | selectively included | This run mined the still-useful draft safetensors `config.json`/YaRN handling into current `server/src/draft/*`: `block_size`, `mask_token_id`, `rope_scaling` beta/original-context fields, and graph-side YaRN RoPE parameters are now wired while keeping current qwen35moe/DDTree code. Older MoE loader/FFN/target-graph pieces remain superseded by current `server/src/qwen35moe` and `server/src/qwen35` architecture. |
| #285 | `feat/lucebox-docker` | partially included / draft dependency | Draft PR, not part of the non-draft target, but an earlier head is carried as the Docker/CLI/bench integration dependency. Latest draft head `32961a1` is not required by any non-draft PR this run. |

## Attempted this run

| PR | Outcome | Notes |
|---:|---|---|
| all non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated reconciliation worktree `/tmp/luce-auto-cron-20260527-224700` confirmed `origin/main` was already merged and then attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39. Every direct probe conflicted; consolidated conflict output is retained in `/tmp/luce-merge-probes-20260527-224700.txt`. No PR head newly became mechanically mergeable since the previous refresh. |
| #39 | partially integrated | Fresh conflicted worktree `/tmp/luce-attempt-pr39-20260527-222159` plus tmux-driven Codex session `luce-pr39-codex-20260527-222159` produced `/tmp/pr39-survivorship-20260527-222159.txt`. Codex found the MoE loader/FFN/target-graph pieces already superseded by current qwen35moe/DDTree code, but identified draft safetensors config parsing and YaRN RoPE as portable. This run ported that subset into `server/src/draft/draft_safetensors_loader.cpp` and `server/src/draft/draft_graph.cpp`; old `dflash/` files and obsolete tests were not resurrected. |
| #221 | not integrated | Fresh conflicted worktree `/tmp/luce-attempt-pr221-20260527-2149` plus tmux-driven Codex session `luce-pr221-codex-2149` produced `/tmp/pr221-feasibility-20260527-2149.txt`. Codex concluded direct merge is unsafe: PR #221 layers prefix-cache warm/ghost behavior and speculator dispatch on top of the older MTP foundation, mixes legacy `dflash/` paths with current `server/`, adds `qwen36` MTP pathing likely superseded by PR #237's current naming, and should wait for a current-layout PR #237/equivalent MTP foundation before selectively porting prefix-cache/range-warm pieces. |
| #237 | not integrated | Fresh worktree merge probe in `/tmp/luce-auto-cron-20260527-214507` reproduced the same broad conflicts across current `server/CMakeLists.txt`, common backend/MTP interfaces, Qwen35 loader/graph/backend files, tests, and deleted legacy `dflash/scripts/server.py` / `dflash/src/server/server_main.cpp`; prior tmux-driven Codex session `luce-pr237-codex-2124` wrote `/tmp/pr237-feasibility-20260527-2124.txt`: selective port is feasible, but a direct conflict-marker resolution is unsafe because the current native server CLI/config, Qwen35 MoE/remote-draft/thinking-budget behavior, and restore/prefix-cache semantics must be preserved with MTP off by default first. |
| #177 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr177-20260527-2020` produced conflicts in deleted legacy `dflash/CMakeLists.txt`, moved Gemma4 header/source/test paths, `server/src/errors.cpp`, and `server/src/internal.h`. A tmux-driven Claude assessment session was launched (`luce-pr177-claude-2020`) but exited without producing a report; the raw conflict shape still confirms this older Gemma4 KV-correctness split is not mechanically mergeable into the current `server/src/gemma4` architecture. |
| #153 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr153-20260527-2020` produced conflicts in legacy `dflash/CMakeLists.txt`, current `server/src/internal.h`, `server/src/qwen35/gguf_target_loader.cpp`, `server/src/qwen35/qwen35_target_graph.cpp`, and path-mapped MTP docs/tests. Codex tmux session `luce-pr153-codex-2020` wrote `/tmp/pr153-feasibility-20260527-2020.txt`: selective port is feasible, but not as conflict-marker resolution; current-layout loader/graph/token embedding/MoE/cache semantics need deliberate design. |
| #39 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr39-20260527-2020` produced broad conflicts across deleted legacy `dflash/*` files plus current draft graph/safetensors loader and MoE smoke tests. A tmux-driven Claude assessment session was launched (`luce-pr39-claude-2020`) but exited without producing a report; the branch remains an older MoE/DDTree stack that needs a survivorship review against current qwen35moe support. |
| #183 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr183-20260527-1959` produced unmerged paths across deleted legacy `dflash/CMakeLists.txt`, moved Gemma4 source/header files, `server/src/errors.cpp`, and `server/src/internal.h`. Codex tmux session `luce-pr183-codex-2003` wrote `/tmp/pr183-feasibility-20260527-1959.txt`: safe normal resolution is not feasible; PR #183 is an older Gemma4 MTP split-chain head whose direct semantic change needs a future current-layout MTP foundation port, likely after #237 or a replacement common MTP design. |
| #182 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr182-20260527-2039` conflicted on deleted legacy `dflash/CMakeLists.txt`, moved Gemma4 target/draft/MTP source and tests, `server/src/errors.cpp`, and `server/src/internal.h`. Claude exited without a report; Codex session `luce-pr182-codex2-20260527-2039` wrote `/tmp/pr182-feasibility-20260527-2039.txt`: the MTP assistant loader/step graph are portable only as a current-layout `server/src/gemma4` rewrite, not as direct conflict resolution. |
| #181 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr181-20260527-2039` hit the same old Gemma4 split-chain/current-layout boundary as #180/#182, including moved Gemma4 target/draft files, `server/src/errors.cpp`, and `server/src/internal.h`. It overlaps current Gemma4 DFlash backend and should be mined only after deciding the current Gemma4 MTP port shape. |
| #180 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr180-20260527-2039` conflicted on legacy `dflash/CMakeLists.txt`, moved Gemma4 target/draft files and tests, `server/src/errors.cpp`, and `server/src/internal.h`. Its draft-loader/quant pieces overlap current `server/src/gemma4` loader/backend work and are not mechanically mergeable. |
| #154 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr154-20260527-2039` conflicted in current Qwen35 loader/graph/test files plus old-layout MTP docs/scripts/tests. Codex session `luce-pr154-codex-20260527-2039` wrote `/tmp/pr154-feasibility-20260527-2039.txt`: the linear native MTP decode semantics are portable with moderate risk, but require a selective current-layout Qwen35 MTP design preserving current MoE/NVFP4/server abstractions. |
| #131 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr131-20260527-2039` produced broad conflicts across `.gitmodules`, deleted `dflash/*`, current Gemma4 source/test files, `server/src/internal.h`, `server/src/errors.cpp`, and pFlash adapter/test code. Claude exited without a report; Codex session `luce-pr131-codex2-20260527-2039` wrote `/tmp/pr131-feasibility-20260527-2039.txt`: broad Gemma4 support can be mined, but direct merge would reintroduce the old `dflash27b` API and obsolete build layout. |
| #137 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr137-20260527-2105` conflicted only on deleted legacy `dflash/CMakeLists.txt`. Manual diff inspection shows the PR's single commit retargets the obsolete old CMake file; current `server/CMakeLists.txt` already owns CUDA architecture/BSA handling. Suggested close or author retarget to current `server/` layout. |
| #135 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr135-20260527-220258` again conflicted in `server/src/internal.h`, `server/src/qwen35/qwen35_target_graph.cpp`, and `server/test/test_dflash.cpp`. tmux-driven Codex session `luce-pr135-codex-2203` produced `/tmp/pr135-feasibility-20260527-2203.txt`: low-level `n_seqs`/batched target graph mechanics are selectively portable with moderate-to-high risk, but the old multi-request scheduler loop is not a small conflict-resolution port because current `ModelBackend`/`run_daemon()` lack the old request-slot queue/runtime model. Safe next subset is isolated aligned-bucket helper/selftest, then a disabled qwen35 batch-probe after a deliberate current-layout `n_seqs` graph port. |
| upstream sync | checked | `origin/main` remains `4f4d82e`; `easel/auto-integration` was already based on that tip, so no upstream merge was needed this run. |

## Previously attempted / still not integrated

| PR | Head branch | State | Why held |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | attempted / blocked-needs-human / selective-port | Revalidated this run in `/tmp/luce-attempt-pr237-20260527-2124`; Codex report says selective current-layout MTP foundation port is feasible, but not by resolving the merge markers directly. First port current server MTP CLI/config with MTP off by default, preserve Qwen35 MoE/remote-draft/thinking-budget behavior, and defer or design restore/prefix-cache MTP warm semantics. |
| #221 | `feat/mtp-prefix-warm-ghost` | attempted / blocked-needs-human / dependency | Revalidated this run in `/tmp/luce-attempt-pr221-20260527-2149`; Codex report `/tmp/pr221-feasibility-20260527-2149.txt` says direct merge is unsafe and PR #221 should be a follow-on selective port after a current-layout #237/equivalent MTP foundation, salvaging prefix-cache MTP warm/range-warm behavior and dispatcher/benchmark lessons while leaving old `dflash/`/`qwen36` artifacts behind. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / superseded-by-current-architecture | Revalidated in the previous run. Its direct `h_prev` assertion has no current target location until Gemma4 MTP runtime is reintroduced in the current `server/src/gemma4` architecture. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | attempted / selective-port | Revalidated this run in `/tmp/luce-attempt-pr182-20260527-2039`; Codex report says the assistant loader/MTP graph are portable only as current-layout `server/src/gemma4` work. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | attempted / likely superseded | Revalidated this run in `/tmp/luce-attempt-pr181-20260527-2039`; conflicts overlap current Gemma4 DFlash backend and the #182 MTP port decision. |
| #180 | `split/gemma4-08-draft-loader-quant` | attempted / likely superseded | Revalidated this run in `/tmp/luce-attempt-pr180-20260527-2039`; conflicts overlap current Gemma4 loader/backend work and are not mechanically mergeable. |
| #177 | `split/gemma4-06-kv-correctness` | attempted / not mechanically mergeable | Revalidated this run in `/tmp/luce-attempt-pr177-20260527-2020`; conflicts are on the old Gemma4 split-chain/current-layout boundary. Needs selective correctness review, not a branch merge. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | attempted / selective-port | Revalidated this run in `/tmp/luce-attempt-pr154-20260527-2039`; Codex report says the linear native MTP decode semantics are portable with moderate risk, but require current-layout Qwen35 MTP design work. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Revalidated this run in `/tmp/luce-attempt-pr153-20260527-2020`; Codex report says a current-layout Qwen35 MTP port is feasible but needs loader/graph/cache/MoE design rather than a mechanical merge. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | attempted / stale | Revalidated this run in `/tmp/luce-attempt-pr137-20260527-2105`; conflicts only on deleted legacy `dflash/CMakeLists.txt`. Current `server/CMakeLists.txt` supersedes it. Suggested close unless re-targeted to current server layout. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | attempted / selective-port | Revalidated this run in `/tmp/luce-attempt-pr135-20260527-220258`; Codex report `/tmp/pr135-feasibility-20260527-2203.txt` says the low-level aligned-bucket and `n_seqs` batched-target graph pieces are portable only after deliberate current-layout graph/API work. The old daemon scheduler loop should not be reintroduced into `server/test/test_dflash.cpp`; design a current `ModelBackend`/Qwen35Backend multi-request scheduler first. |
| #131 | `feature/gemma4-support` | attempted / selective-port | Revalidated this run in `/tmp/luce-attempt-pr131-20260527-2039`; Codex report says broad Gemma4 support can be mined, but direct merge would resurrect old `dflash27b`/`dflash/` layout. |
| #39 | `feat/moe-35b-a3b` | partially integrated / mostly superseded | Revalidated this run in `/tmp/luce-attempt-pr39-20260527-222159`; Codex report `/tmp/pr39-survivorship-20260527-222159.txt` found current qwen35moe/DDTree code supersedes the old loader/FFN/target-graph pieces. The safe remaining draft safetensors config/YaRN subset was ported this run; remaining test ideas can be reauthored against current APIs if desired. |

## Draft / excluded

Draft PRs remain outside the primary contributor integration target: #289, #286, #285, #278, #275, #273, #265, #249, #193, #75. #285 remains partially carried only as an integration dependency.

## Suggested close / author action

- #137: close or ask author to re-target current `server/CMakeLists.txt`.
- #48: close as superseded by current CUDA architecture/Blackwell handling unless the author has a specific missing Blackwell case.
- #237: keep open for a deliberate current-layout MTP foundation port; direct merge is unsafe, but the Codex report identifies a viable compile-first path with MTP disabled by default.
- #39: remaining old branch content is mostly superseded by current qwen35moe/DDTree code after this run's draft config/YaRN selective port; ask author/maintainers whether any of the old MoE smoke coverage should be reauthored against current APIs before closing.
- #183/#182/#181/#180/#177/#154/#153/#131: do not close solely as conflicts yet; ask authors or maintainers to identify which current-layout Gemma4/MTP pieces are still intended after the newer `server/src/gemma4` architecture and #237 direction.

## Validation run

This run performed:

- 2026-05-27 22:46 preflight: `date -Is`, primary `git status --short` (clean), branch/remotes, `gh auth status`, `claude auth status --text`, and `codex --version` with the real user HOME/config.
- `git fetch --prune origin` and `git fetch --prune easel`.
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...`.
- Targeted fetches for all open non-draft PR refs (`origin/pr/<n>`).
- `git merge-base --is-ancestor origin/pr/<n> easel/auto-integration` classification checks: #288, #287, #284, #276, #274, #266, #152, and #142 are current ancestors; #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39 remain non-ancestor/selective-port candidates.
- Isolated reconciliation/probe worktree `/tmp/luce-auto-cron-20260527-224700`; `git merge --no-edit origin/main` reported already up to date.
- Fresh direct merge probes for all currently non-ancestor non-draft PR refs: #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39; all direct probes conflicted and were aborted in the isolated worktree; consolidated output retained at `/tmp/luce-merge-probes-20260527-224700.txt`.
- `git diff --check -- docs/auto-integration.md` (clean).
- No source/build validation was rerun because this refresh changed only integration metadata; the previous source validation result remains unchanged: local CMake/CUDA configure fails before project compilation on unsupported `sm_52` compiler identification in this WSL CUDA setup.

Previous retained validation/audit details:

- 2026-05-27 22:20 preflight: `date -Is`, primary `git status --short` (clean), branch/remotes, `gh auth status`, `claude auth status --text`, and `codex --help` with the real user HOME/config.
- `git fetch --prune origin` and `git fetch --prune easel`.
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...`.
- Targeted fetches for all open non-draft PR refs (`origin/pr/<n>`).
- `git merge-base --is-ancestor origin/pr/<n> easel/auto-integration` classification checks: #288, #287, #284, #276, #274, #266, #152, and #142 are current ancestors; #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39 remain non-ancestor/selective-port candidates.
- Isolated reconciliation/probe worktree `/tmp/luce-auto-cron-20260527-222142`; `git merge --no-edit origin/main` reported already up to date.
- Fresh direct merge probes for all currently non-ancestor non-draft PR refs: #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39; all direct probes conflicted and were aborted in the isolated worktree; consolidated output retained at `/tmp/luce-merge-probes-20260527-222142.txt`.
- Fresh isolated PR #39 conflicted worktree `/tmp/luce-attempt-pr39-20260527-222159`.
- tmux-driven Codex assessment session `luce-pr39-codex-20260527-222159`, producing `/tmp/pr39-survivorship-20260527-222159.txt`.
- Selective PR #39 port: current draft safetensors loader now reads `block_size`, `mask_token_id`, `dflash_config.target_layer_ids` sanity, and YaRN `rope_scaling` from `config.json`; current draft graph now passes loaded YaRN RoPE fields to `ggml_rope_ext`.
- `git diff --check -- server/src/draft/draft_graph.cpp server/src/draft/draft_safetensors_loader.cpp docs/auto-integration.md` (clean).
- CMake configure/build attempt for `dflash_common` failed before compiling project code because the local CUDA toolkit defaults compiler identification to unsupported `sm_52` (`ptxas fatal : Value 'sm_52' is not defined for option 'gpu-name'`); retrying with `-DCMAKE_CUDA_ARCHITECTURES=89` hit the same pre-project compiler-identification failure. No project compile result was available in this WSL CUDA setup.

- `git merge --no-edit origin/main` check in the metadata worktree (no merge needed; `origin/main` was already an ancestor of `d465a7e`)
- `date -Is`
- `git status --short`, `git remote -v`, branch and revision checks in the primary checkout
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status`
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text`
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --help`
- `git fetch --prune origin`
- `git fetch --prune easel`
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...`
- Targeted fetches for all open non-draft PR refs (`origin/pr/<n>`)
- `git cherry easel/auto-integration origin/pr/<n>` and `git merge-base --is-ancestor origin/pr/<n> easel/auto-integration` classification checks
- Isolated metadata/probe worktree `/tmp/luce-auto-cron-20260527-214507`
- `git merge --no-edit origin/main` in that worktree (already up to date)
- Fresh direct merge probes for all currently non-ancestor non-draft PR refs: #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39; all conflicted and were aborted in the isolated worktree; consolidated output retained at `/tmp/luce-merge-probes-20260527-214507.txt`
- Fresh isolated PR #221 conflicted worktree `/tmp/luce-attempt-pr221-20260527-2149`
- tmux-driven Codex assessment session `luce-pr221-codex-2149`, producing `/tmp/pr221-feasibility-20260527-2149.txt`
- `git diff --check -- docs/auto-integration.md` before this metadata commit (clean)
- Isolated metadata worktree `/tmp/luce-auto-cron-20260527-212401`
- Fresh isolated PR #237 merge probe in `/tmp/luce-attempt-pr237-20260527-2124`
- tmux-driven Codex assessment session `luce-pr237-codex-2124`, producing `/tmp/pr237-feasibility-20260527-2124.txt`
- `git diff --check -- docs/auto-integration.md` before this metadata commit (clean)
- Isolated metadata worktree `/tmp/luce-auto-cron-20260527-2020`
- Fresh isolated PR #177 merge probe in `/tmp/luce-attempt-pr177-20260527-2020`
- Fresh isolated PR #153 merge probe in `/tmp/luce-attempt-pr153-20260527-2020`
- Fresh isolated PR #39 merge probe in `/tmp/luce-attempt-pr39-20260527-2020`
- tmux-driven Claude assessment sessions `luce-pr177-claude-2020` and `luce-pr39-claude-2020` (both exited before producing report files)
- tmux-driven Codex assessment session `luce-pr153-codex-2020`, producing `/tmp/pr153-feasibility-20260527-2020.txt`
- `git diff --check -- docs/auto-integration.md` before metadata commit (clean)
- Fresh isolated PR #182 merge probe in `/tmp/luce-attempt-pr182-20260527-2039`
- Fresh isolated PR #181 merge probe in `/tmp/luce-attempt-pr181-20260527-2039`
- Fresh isolated PR #180 merge probe in `/tmp/luce-attempt-pr180-20260527-2039`
- Fresh isolated PR #154 merge probe in `/tmp/luce-attempt-pr154-20260527-2039`
- Fresh isolated PR #131 merge probe in `/tmp/luce-attempt-pr131-20260527-2039`
- tmux-driven Claude assessment sessions `luce-pr182-claude-20260527-2039` and `luce-pr131-claude-20260527-2039` (both exited before producing report files)
- tmux-driven Codex assessment sessions `luce-pr182-codex2-20260527-2039`, `luce-pr154-codex-20260527-2039`, and `luce-pr131-codex2-20260527-2039`, producing `/tmp/pr182-feasibility-20260527-2039.txt`, `/tmp/pr154-feasibility-20260527-2039.txt`, and `/tmp/pr131-feasibility-20260527-2039.txt`
- `git diff --check HEAD^1..HEAD` after the upstream merge and metadata refresh (clean)
- Fresh isolated PR #137 merge probe in `/tmp/luce-attempt-pr137-20260527-2105`
- Manual PR #137 diff inspection against old `dflash/CMakeLists.txt` and current `server/CMakeLists.txt`
- Fresh isolated PR #135 merge probe in `/tmp/luce-attempt-pr135-20260527-2105`
- tmux-driven Codex assessment session `luce-pr135-codex-2105` (exited without producing `/tmp/pr135-feasibility-20260527-2105.txt`)
- tmux-driven Claude assessment session `luce-pr135-claude-2109` (hit max-turn limit before producing `/tmp/pr135-feasibility-20260527-2109.txt`)
- `git diff --check -- docs/auto-integration.md` before this metadata commit (clean)
- Fresh isolated PR #135 merge probe in `/tmp/luce-attempt-pr135-20260527-220258`
- tmux-driven Codex assessment session `luce-pr135-codex-2203`, producing `/tmp/pr135-feasibility-20260527-2203.txt`
- `git diff --check -- docs/auto-integration.md` before this metadata commit (clean)
- `uv run --directory luce-bench --with pytest pytest -q tests/test_runner.py tests/test_smoke_area.py` (passed in previous run; skipped this run because only metadata changed)

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at start and was only edited for the final metadata refresh after probe verification.
- Retained worktree `/tmp/luce-auto-cron-20260527-224700` for this metadata refresh and probe audit; consolidated direct-merge probe output is `/tmp/luce-merge-probes-20260527-224700.txt`.
- Retained worktree `/tmp/luce-auto-cron-20260527-222142` for the prior selective-port commit and probe audit.
- Retained conflicted worktree `/tmp/luce-attempt-pr39-20260527-222159` for audit; it contains unmerged PR #39 probe state and Codex report `/tmp/pr39-survivorship-20260527-222159.txt`.
- Retained worktree `/tmp/luce-auto-cron-20260527-214507` for this metadata refresh until pushed/verified.
- Retained conflicted worktree `/tmp/luce-attempt-pr221-20260527-2149` for audit; it contains unmerged PR #221 probe state and Codex report `/tmp/pr221-feasibility-20260527-2149.txt`.
- Retained worktree `/tmp/luce-auto-cron-20260527-212401` for this metadata refresh until pushed/verified.
- Retained conflicted worktree `/tmp/luce-attempt-pr237-20260527-2124` for audit; it contains unmerged PR #237 probe state.
- Retained prior metadata/conflicted worktrees from earlier runs as previously reported; cleanup would be separate maintenance.
- Retained worktree `/tmp/luce-auto-cron-20260527-2105` for this metadata refresh until pushed/verified.
- Retained conflicted worktree `/tmp/luce-attempt-pr137-20260527-2105` for audit; it contains unmerged PR #137 probe state.
- Retained conflicted worktree `/tmp/luce-attempt-pr135-20260527-2105` for audit; it contains unmerged PR #135 probe state.
- Retained conflicted worktree `/tmp/luce-attempt-pr135-20260527-220258` for audit; it contains unmerged PR #135 probe state and Codex report `/tmp/pr135-feasibility-20260527-2203.txt`.
- Retained prior conflicted worktrees from earlier runs as previously reported; cleanup would be separate maintenance.
