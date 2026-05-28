# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T02:25:47-04:00
Current base: `origin/main` `4f4d82e`
Current integration tip before this refresh: `easel/auto-integration` `13ae15c`

This branch is maintained as a reproducible patch stack over `origin/main`. The
primary checkout was clean at the start of this unattended run. This refresh
found a new non-draft update to PR #284 (`63bba30`) that was no longer an
ancestor of `easel/auto-integration`; it merged cleanly on top of the current
stack and was integrated. Upstream `origin/main` did not advance. The remaining
non-ancestor non-draft PRs were re-probed in the isolated integration worktree;
every direct merge still conflicts and remains a selective-port/design task
rather than a mechanical merge.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #284 | `fix/draft-safetensors-rope-theta` | `63bba30` | included | Newly integrated this run; rejects non-finite `rope_theta` values from draft safetensors `config.json` by adding an `std::isfinite` guard. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Current head is an ancestor of the refreshed stack. |
| #274 | `feat/pflash-drafter-ee7` | `5037b28` | included | Current head is an ancestor of the refreshed stack. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Current head is an ancestor of the refreshed stack. |
| #152 | `main` | `cf735be` | included | Current head is an ancestor of the refreshed stack. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | Current head is an ancestor of the refreshed stack. |
| #174 | `split/gemma4-14-small-vram-docs` | `8b1caba` | selectively included | The useful small-VRAM/VMM documentation is already ported into the current `server/README.md`; the remaining old Gemma4 split-chain commits are not ancestors. |
| #94 | `feat/dflash-qwen36-swa-draft` | `d2f9c9d` | absorbed / selectively included | Current draft/common code already carries the safetensors SWA config parsing and causal-mask behavior; the old branch conflicts on moved/rewritten draft files. |
| #62 | `fix/issue-55-stable-kv-pad` | `0ce6832` | absorbed / selectively included | Daemon reset regression coverage and reset fixes remain carried in the current server layout; the old branch still conflicts on legacy test paths. |
| #48 | `fix/consumer-blackwell-auto-detect` | `858b84b` | superseded / selectively included | Current `server/CMakeLists.txt` owns CUDA architecture resolution and Blackwell/GB10 handling; the PR edits deleted legacy `dflash/CMakeLists.txt`. |
| #39 | `feat/moe-35b-a3b` | `c86ec86` | partially integrated / mostly superseded | Prior run ported draft safetensors `config.json`/YaRN survivorship into current `server/src/draft/*`; the remaining loader/FFN/target-graph pieces are superseded by current qwen35moe/DDTree code. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the non-draft target, but an earlier Docker/CLI/bench integration dependency remains carried. |

Recently closed contributor PRs #287 and #288 are no longer open non-draft
integration targets; their heads remain in the historical stack where already
carried.

## Attempted this run

| PR | Outcome | Notes |
|---:|---|---|
| upstream sync | checked | In isolated worktree `/tmp/luce-auto-cron-20260528-022439`, `git merge --no-edit origin/main` reported `Already up to date.` |
| #284 | integrated | `git merge --no-edit origin/pr/284` succeeded, adding `#include <cmath>` and an `std::isfinite` guard to `server/src/draft/draft_safetensors_loader.cpp`. |
| all remaining non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated direct probes attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39. Every direct probe conflicted and was aborted in the isolated worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-022439.txt`. |
| #237 | prior delegated/manual feasibility retained | Conflicted probe worktree `/tmp/luce-pr237-feas-20260528-014920` remains for inspection. Claude exited with only `Error: Reached max turns (12)` (`/tmp/pr237-claude-feasibility-20260528-014920.txt`), so no Claude conclusion is claimed. Codex completed (`/tmp/pr237-codex-feasibility-20260528-014920.txt`) and confirmed #237 is a manual current-layout port rather than a conflict-marker resolution. |
| #221 | prior delegated/manual feasibility retained | Prior conflicted probe worktree `/tmp/luce-pr221-feas-20260528-002537` remains the current evidence. Claude exited with only `Error: Reached max turns (18)` (`/tmp/pr221-claude-feasibility-20260528-002537.txt`); Codex report `/tmp/pr221-codex-feasibility-20260528-002537.txt` says #221 should wait for #237/equivalent MTP foundation. |
| #183 | prior delegated/manual feasibility retained | Prior conflicted probe worktree `/tmp/luce-pr183-feas-20260528-004611` remains the current evidence. Claude exited with only `Error: Reached max turns (18)`, and Codex produced read-only inspection output without a final recommendation. Manual inspection confirms a current-layout Gemma4 MTP design is required. |
| #177 | prior delegated/manual feasibility retained | Prior conflicted probe worktree `/tmp/luce-pr177-feas-20260527-234303` and two Claude max-turn exits remain the current evidence. Manual inspection shows old-layout Gemma4 KV/target-graph work that needs deliberate mapping into current `server/` APIs. |
| #135 | prior delegated/manual feasibility retained | Conflicted probe worktree `/tmp/luce-pr135-feas-20260528-020651` remains for inspection. Codex completed with a usable report (`/tmp/pr135-codex-feasibility-20260528-020651.txt`) describing the current-layout scheduler/graph port order and required preservation points. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge probe still conflicts. Safest current-layout starting subset remains disabled-by-default MTP metadata/interfaces (`server/src/common/gguf_metadata.h`, `MtpSource`, `server/src/common/mtp_*`) before enabling runtime MTP, while preserving qwen35moe, remote-draft, PFlash, budget hooks, and token-callback behavior. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Should wait for #237/equivalent MTP foundation; then selectively port unique dual-speculator routing, MTP head-KV WARM snapshot/restore, partial `warm_head_kv_range`, native HTTP/daemon `speculator` request plumbing, and focused prefix-cache MTP tests. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / current-layout design required | Unique value is Gemma4 assistant/MTP graph and target hidden-state capture/asymmetric KV tests, but it is old-layout and overlaps current `server/src/gemma4/*`; port only after deciding how Gemma4 MTP composes with the existing feature-complete Gemma4 backend/DFlash target. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | selective-port | Portable only as current-layout `server/src/gemma4` assistant loader/MTP graph work. Prior Codex report: `/tmp/pr182-feasibility-20260527-2039.txt`. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | likely superseded / selective-port | Overlaps current Gemma4 DFlash backend and should be mined only after the current Gemma4 MTP shape is chosen. |
| #180 | `split/gemma4-08-draft-loader-quant` | likely superseded / selective-port | Overlaps current Gemma4 loader/backend code; not mechanically mergeable. |
| #177 | `split/gemma4-06-kv-correctness` | selective-port / blocked-needs-human | Old-layout additions `dflash/include/gemma4.h`, `dflash/src/gemma4_target_graph.cpp`, `dflash/src/gemma4_target_loader.cpp`, and Gemma4 tests need deliberate mapping into current `server/` APIs. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | selective-port | Linear native MTP decode semantics are portable with moderate risk after a current-layout Qwen35 MTP design. Prior Codex report: `/tmp/pr154-feasibility-20260527-2039.txt`. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Current-layout Qwen35 MTP port is feasible but requires loader/graph/cache/MoE design, not conflict-marker resolution. Prior Codex report: `/tmp/pr153-feasibility-20260527-2020.txt`. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | stale / suggested close | Edits only deleted legacy `dflash/CMakeLists.txt`; close or ask author to retarget current `server/CMakeLists.txt`. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Prior Codex report: `/tmp/pr135-codex-feasibility-20260528-020651.txt`. Port order: add `n_seqs` primitives while preserving current HEAD behavior, port Qwen35 batched graph/probe path as comparison-only, move aligned-bucket helper/selftest into current server tests, then design multi-slot `Qwen35Backend` state and scheduler API before production exposure. |
| #131 | `feature/gemma4-support` | selective-port | Broad Gemma4 support can be mined, but direct merge would resurrect old `dflash27b`/`dflash/` layout. Prior Codex report: `/tmp/pr131-feasibility-20260527-2039.txt`. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / verify-only | Remaining conflicts are old-layout draft/common paths; current code carries the useful SWA pieces. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / verify-only | Remaining conflicts are old-layout tests; current code carries reset behavior. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / suggested close | Current CMake supersedes the old `dflash/CMakeLists.txt` change. |
| #39 | `feat/moe-35b-a3b` | partially integrated / mostly superseded | Ask maintainers whether any remaining old MoE smoke coverage should be reauthored against current qwen35moe APIs before closing. Prior Codex report: `/tmp/pr39-survivorship-20260527-222159.txt`. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #289, #286, #285, #278, #275, #273, #265, #249, #193,
and #75. #289 (`feat(qwen35moe): pipelined hybrid MoE decode with GPU/CPU
overlap`) remains draft-only and was not attempted. #285 remains partially
carried only as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T02:23:50-04:00 during preflight, 2026-05-28T02:25:47-04:00 while refreshing this manifest.
- Primary checkout `git status --short` was clean before work began and stayed clean while probing in worktrees.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel`.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --help` succeeded.
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft PR refs.
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...` enumerated all open PRs.
- `git merge-base --is-ancestor origin/pr/<n> <stack>` classification checks: #284, #276, #274, #266, #152, and #142 are current ancestors after this refresh; #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39 remain non-ancestor/selective-port candidates.
- Isolated reconciliation/probe worktree `/tmp/luce-auto-cron-20260528-022439`; `git merge --no-edit origin/main` reported already up to date.
- `git merge --no-edit origin/pr/284` succeeded and integrated PR #284 head `63bba30`.
- Fresh direct merge probes for all remaining non-ancestor non-draft PR refs: #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39; all direct probes conflicted and were aborted in the isolated worktree; consolidated output retained at `/tmp/luce-merge-probes-20260528-022439.txt`.
- `git diff --check origin/main...HEAD` reported only pre-existing whitespace warnings in `luce-bench/src/lucebench/fixtures/forge_eval/scenarios/_model_quality.py` and `_stateful_model_quality.py`.
- `git diff --check easel/auto-integration...HEAD -- server/src/draft/draft_safetensors_loader.cpp` and `git diff --check HEAD~1..HEAD` both passed for this run's changed source.
- `cmake -S server -B /tmp/luce-build-20260528-022439 -DLUCE_BUILD_TESTS=ON` still fails before project compilation in this WSL CUDA setup because compiler identification selects unsupported `sm_52` (`ptxas fatal: Value 'sm_52' is not defined for option 'gpu-name'`).

## Notes

- Primary checkout `/home/erik/Projects/luce2` stayed clean during probing and was modified only after the verified integration result was ready.
- Retained worktree `/tmp/luce-auto-cron-20260528-022439` for this source integration and direct-merge probe audit.
- Retained direct-probe log `/tmp/luce-merge-probes-20260528-022439.txt`.
- Retained CMake configure directory `/tmp/luce-build-20260528-022439` for inspection of the local CUDA compiler-identification failure.
- Prior retained conflicted worktrees and agent reports remain as listed in earlier manifest revisions; cleanup is separate maintenance.
