# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T01:22:15-04:00
Current base: `origin/main` `4f4d82e`
Current integration tip before this refresh: `easel/auto-integration` `db9e839`

This branch is maintained as a reproducible patch stack over `origin/main`. The
primary checkout was clean at the start of this unattended run. This refresh
rechecked upstream/main and the open PR set, found no upstream-base advance and
no newly mergeable non-draft PR head, and therefore made no source stack
changes. It refreshed direct-merge probe evidence for every currently
non-ancestor non-draft PR and preserves the prior manual/delegated feasibility
reports for the large selective-port candidates.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #284 | `fix/draft-safetensors-rope-theta` | `697198a` | included | Current head is an ancestor of `easel/auto-integration`. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Current head is an ancestor of `easel/auto-integration`. |
| #274 | `feat/pflash-drafter-ee7` | `5037b28` | included | Current head is an ancestor of `easel/auto-integration`. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Current head is an ancestor of `easel/auto-integration`. |
| #152 | `main` | `cf735be` | included | Current head is an ancestor of `easel/auto-integration`. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | Current head is an ancestor of `easel/auto-integration`. |
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
| upstream sync | checked | In isolated worktree `/tmp/luce-auto-cron-20260528-012215`, `git merge --no-edit origin/main` reported `Already up to date.` |
| all non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated direct probes attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39. Every direct probe conflicted and was aborted in the isolated worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-012215.txt`. |
| #237 | prior delegated/manual feasibility retained | Prior conflicted probe worktree `/tmp/luce-pr237-feas-20260528-000006` remains the current evidence. Claude exited with only `Error: Reached max turns (12)` (`/tmp/pr237-claude-feasibility-20260528-000006.txt`), so no Claude conclusion is claimed. Codex completed (`/tmp/pr237-codex-feasibility-20260528-000006.txt`) and confirmed #237 is not mechanically mergeable: stale deleted `dflash/` server paths plus semantic conflicts in MTP source selection, qwen35/qwen35moe GGUF metadata, graph-builder hidden-state capture vs MoE router capture, remote-draft/PFlash coexistence, and CMake/test wiring. |
| #221 | prior delegated/manual feasibility retained | Prior conflicted probe worktree `/tmp/luce-pr221-feas-20260528-002537` remains the current evidence. Claude exited with only `Error: Reached max turns (18)` (`/tmp/pr221-claude-feasibility-20260528-002537.txt`), so no Claude conclusion is claimed. Codex produced a large usable report (`/tmp/pr221-codex-feasibility-20260528-002537.txt`) confirming #221 should wait for #237/equivalent MTP foundation before selectively porting dual-dispatch and prefix-WARM behavior. |
| #183 | prior delegated/manual feasibility retained | Prior conflicted probe worktree `/tmp/luce-pr183-feas-20260528-004611` remains the current evidence. Claude exited with only `Error: Reached max turns (18)` (`/tmp/pr183-claude-feasibility-20260528-004611.txt`), and Codex produced read-only inspection output without a final recommendation (`/tmp/pr183-codex-feasibility-20260528-004611.txt`). Manual inspection confirms #183's unique value is Gemma4 target-graph MTP integration, but current `server/src/gemma4/*` already has a different feature-complete backend/DFlash target shape. |
| #177 | prior delegated/manual feasibility retained | Prior conflicted probe worktree `/tmp/luce-pr177-feas-20260527-234303` and two Claude max-turn exits remain the current evidence. Manual inspection shows #177 is old-layout Gemma4 KV/target-graph work that maps mostly from deleted `dflash/*` into current `server/*` and needs a deliberate current-layout correctness port. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge probe still conflicts; previous delegated worktree/Codex report remains the current guidance. Minimal viable current-layout port starts with `server/src/common/mtp_*`, `server/src/common/gguf_metadata.h`, `server/src/qwen35/qwen35_mtp*`, and `server/test/test_common_mtp_orchestrator.cpp`, then deliberate merges into backend factory, qwen35 graph builders/loader/backend/daemon, native server CLI, and CMake/CTest while preserving qwen35moe, remote-draft, PFlash, and token-callback behavior. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Should wait for #237/equivalent MTP foundation; then selectively port unique dual-speculator routing, MTP head-KV WARM snapshot/restore, partial `warm_head_kv_range`, native HTTP/daemon `speculator` request plumbing, and focused prefix-cache MTP tests. Do not restore deleted `dflash/scripts/*` or `dflash/src/qwen36/*` paths. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / current-layout design required | Unique value is Gemma4 assistant/MTP graph and target hidden-state capture/asymmetric KV tests, but it is old-layout and overlaps current `server/src/gemma4/*`; port only after deciding how Gemma4 MTP composes with the existing feature-complete Gemma4 backend/DFlash target. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | selective-port | Portable only as current-layout `server/src/gemma4` assistant loader/MTP graph work. Prior Codex report: `/tmp/pr182-feasibility-20260527-2039.txt`. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | likely superseded / selective-port | Overlaps current Gemma4 DFlash backend and should be mined only after the current Gemma4 MTP shape is chosen. |
| #180 | `split/gemma4-08-draft-loader-quant` | likely superseded / selective-port | Overlaps current Gemma4 loader/backend code; not mechanically mergeable. |
| #177 | `split/gemma4-06-kv-correctness` | selective-port / blocked-needs-human | Old-layout additions `dflash/include/gemma4.h`, `dflash/src/gemma4_target_graph.cpp`, `dflash/src/gemma4_target_loader.cpp`, and Gemma4 tests need deliberate mapping into current `server/` APIs. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | selective-port | Linear native MTP decode semantics are portable with moderate risk after a current-layout Qwen35 MTP design. Prior Codex report: `/tmp/pr154-feasibility-20260527-2039.txt`. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Current-layout Qwen35 MTP port is feasible but requires loader/graph/cache/MoE design, not conflict-marker resolution. Prior Codex report: `/tmp/pr153-feasibility-20260527-2020.txt`. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | stale / suggested close | Edits only deleted legacy `dflash/CMakeLists.txt`; close or ask author to retarget current `server/CMakeLists.txt`. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | selective-port | Low-level aligned-bucket and `n_seqs` mechanics need deliberate current `ModelBackend`/Qwen35Backend graph/API work. Prior Codex report: `/tmp/pr135-feasibility-20260527-2203.txt`. |
| #131 | `feature/gemma4-support` | selective-port | Broad Gemma4 support can be mined, but direct merge would resurrect old `dflash27b`/`dflash/` layout. Prior Codex report: `/tmp/pr131-feasibility-20260527-2039.txt`. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / verify-only | Remaining conflicts are old-layout draft/common paths; current code carries the useful SWA pieces. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / verify-only | Remaining conflicts are old-layout tests; current code carries reset behavior. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / suggested close | Current CMake supersedes the old `dflash/CMakeLists.txt` change. |
| #39 | `feat/moe-35b-a3b` | partially integrated / mostly superseded | Ask maintainers whether any remaining old MoE smoke coverage should be reauthored against current qwen35moe APIs before closing. Prior Codex report: `/tmp/pr39-survivorship-20260527-222159.txt`. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #289, #286, #285, #278, #275, #273, #265, #249, #193,
and #75. #285 remains partially carried only as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T01:21:22-04:00 during preflight, 2026-05-28T01:22:15-04:00 for the reconciliation worktree name.
- Primary checkout `git status --short` was clean before work began and remained clean while probing in worktrees.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel`.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` succeeded (`codex-cli 0.130.0`).
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft PR refs.
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...` enumerated all open PRs.
- Targeted fetches for all open non-draft PR refs (`origin/pr/<n>`).
- `git merge-base --is-ancestor origin/pr/<n> easel/auto-integration` classification checks: #284, #276, #274, #266, #152, and #142 are current ancestors; #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39 remain non-ancestor/selective-port candidates.
- Isolated reconciliation/probe worktree `/tmp/luce-auto-cron-20260528-012215`; `git merge --no-edit origin/main` reported already up to date.
- Fresh direct merge probes for all currently non-ancestor non-draft PR refs: #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39; all direct probes conflicted and were aborted in the isolated worktree; consolidated output retained at `/tmp/luce-merge-probes-20260528-012215.txt`.
- `git diff --check` on the metadata-only change passed.

No source/build validation was rerun because this refresh changes only
integration metadata. Previous source validation remains unchanged: local
CMake/CUDA configure fails before project compilation in this WSL CUDA setup
because compiler identification selects unsupported `sm_52` (`ptxas fatal:
Value 'sm_52' is not defined for option 'gpu-name'`).

## Notes

- Primary checkout `/home/erik/Projects/luce2` stayed clean during probing and was modified only after the verified metadata commit was ready.
- Retained worktree `/tmp/luce-auto-cron-20260528-012215` for this metadata refresh and direct-merge probe audit.
- Retained direct-probe log `/tmp/luce-merge-probes-20260528-012215.txt`.
- Prior retained conflicted worktrees and agent reports remain as listed in earlier manifest revisions; cleanup is separate maintenance.
