# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T06:55:39-04:00
Current base: `origin/main` `4f4d82e`
Current integration tip before this refresh: `easel/auto-integration` `d735948`

This branch is maintained as a reproducible patch stack over `origin/main`. The
primary checkout was clean at the start of this unattended run. Upstream
`origin/main` was already merged into the stack, and no new ready contributor PR
head needed a code merge beyond the current `easel/auto-integration` tip. This
refresh revalidated the ready/non-draft PR set, reran direct probes for the
remaining old-layout/non-ancestor PRs, and added a fresh Codex feasibility pass
for #181. Codex confirmed #181 is feasible only as a current-layout Gemma4 draft
runtime selective port: keep the useful Gemma4 quantizer metadata, error-handling
intent, draft runtime concepts, and smoke-test coverage, but discard the stale
old `dflash/`/top-level `server/src/gemma4_*.cpp` layout.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #289 | `pipeline_moe` | `593266a` | included | Adds pipelined hybrid Qwen35 MoE decode with persistent decode state, optimized FFN routing, telemetry, and tests. Integrated with a manual signature conflict resolution that preserves the current `BudgetHook`/close-kind API; the inaccessible submodule pointer from the PR was not adopted. |
| #284 | `fix/draft-safetensors-rope-theta` | `63bba30` | included | Current head is an ancestor of the refreshed stack; rejects non-finite `rope_theta` values from draft safetensors `config.json`. |
| #278 | `fix-pflash-drafter-backend-precision-submit` | `fdfcbda` | included | Adds legacy CUDA drafter precision fallback via shared backend precision policy and BF16/F16 tensor conversion; conflict with existing Q8_0 GGUF allocation support was resolved in the prior refresh by preserving Q8_0 when present and otherwise using the policy-selected allocation type. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Current head is an ancestor of the refreshed stack. |
| #274 | `feat/pflash-drafter-ee7` | `5037b28` | included | Current head is an ancestor of the refreshed stack. |
| #273 | `feat-cpp-server-gemma4-layer-split-adapter` | `79abba9` | included | Current head is an ancestor of the refreshed stack; adds the Gemma4 target-layer-split adapter and loader/graph support. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Current head is an ancestor of the refreshed stack. |
| #265 | `feat-cpp-server-target-layer-split-prep` | `054af28` | included | Current head is an ancestor of the refreshed stack; tightens layer-split validation/cleanup across backend factory, layer-split backend, Qwen35 adapter, server main, and unit coverage. |
| #152 | `main` | `cf735be` | included | Current head is an ancestor of the refreshed stack. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | Current head is an ancestor of the refreshed stack. |
| #177 | `split/gemma4-06-kv-correctness` | `0a95d4b` | selectively included | This refresh ported the low-risk current-layout survivor: Gemma4 SWA cache writes now split on ring-buffer wrap in `server/src/gemma4/gemma4_graph.cpp`; the old-layout loader/graph/test files still need deliberate mapping before the PR can be fully closed. |
| #174 | `split/gemma4-14-small-vram-docs` | `8b1caba` | selectively included | The useful small-VRAM/VMM documentation is already ported into the current `server/README.md`; the remaining old Gemma4 split-chain commits are not ancestors. |
| #94 | `feat/dflash-qwen36-swa-draft` | `d2f9c9d` | absorbed / selectively included | Current draft/common code already carries the safetensors SWA config parsing and causal-mask behavior; the old branch conflicts on moved/rewritten draft files. |
| #62 | `fix/issue-55-stable-kv-pad` | `0ce6832` | absorbed / selectively included | Daemon reset regression coverage and reset fixes remain carried in the current server layout; the old branch still conflicts on legacy test paths. |
| #48 | `fix/consumer-blackwell-auto-detect` | `858b84b` | superseded / selectively included | Current `server/CMakeLists.txt` owns CUDA architecture resolution and Blackwell/GB10 handling; the PR edits deleted legacy `dflash/CMakeLists.txt`. |
| #39 | `feat/moe-35b-a3b` | `c86ec86` | partially integrated / mostly superseded | Prior run ported draft safetensors `config.json`/YaRN survivorship into current `server/src/draft/*`; the remaining loader/FFN/target-graph pieces are superseded by current qwen35moe/DDTree code. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the non-draft target, but an earlier Docker/CLI/bench integration dependency remains carried. |

Recently closed contributor PRs #287 and #288 are no longer open non-draft
integration targets; their heads remain in the historical stack where already
carried.

## Fresh probe results from this run

| PR | Outcome | Notes |
|---:|---|---|
| upstream sync | checked | In isolated worktree `/tmp/luce-auto-cron-20260528-065539`, `git merge --no-edit origin/main` reported `Already up to date.` |
| current integrated PRs | checked | Current heads for #289, #284, #278, #276, #274, #273, #266, #265, #152, and #142 are ancestors of `easel/auto-integration` `d735948`; no new ready contributor head required a code merge this run. |
| remaining non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated probes attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39. Every direct probe conflicted and was aborted in the isolated worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-065539.txt`. |
| #181 | delegated feasibility refreshed | Claude in tmux exited without producing report content in `/tmp/pr181-claude-feasibility-20260528-065539.txt`. Codex in tmux completed at `/tmp/pr181-codex-feasibility-20260528-065539.txt`: direct conflict resolution is unsafe; portable pieces are Gemma4 support in `server/scripts/quantize_draft_q8.py`, thread-local/optional Gemma4 error reporting intent, draft runtime concepts already partly covered by current `server/src/gemma4` and generic draft code, and stale smoke-test intent. Recommended port order is current-layout quantizer metadata alignment, optional current-layout error API, replacement of hard-coded Gemma4 draft overrides with reliable GGUF metadata, capture-layer validation, current-style tests, then evaluate prefix-direct cached draft performance separately. |

## Prior probe results (retained)

| PR | Outcome | Notes |
|---:|---|---|
| #237 | prior delegated/manual feasibility retained | Conflicted probe worktree `/tmp/luce-pr237-feas-20260528-023928` remains for inspection. Codex completed in tmux and produced `/tmp/pr237-codex-feasibility-20260528-023928.txt`, confirming the portable value is the native Qwen35/Qwen3.6 MTP stack (`mtp_interface`, chain runner/orchestrator, MTP GGUF metadata, hidden capture, backend path, native server flags, and tests). It should be ported into `server/` in layers rather than by accepting the conflicted merge. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge probe still conflicts. Prior Codex report `/tmp/pr237-codex-feasibility-20260528-023928.txt` recommended a layered full MTP port. This run's narrower Claude report `/tmp/pr237-claude2-feasibility-20260528-061548.txt` tightened the guidance: do not cherry-pick a server-only subset, because the apparent additive flags are MTP-coupled and adjacent server/parser/SSE edits would regress current PFlash IPC drafter plus Anthropic/Responses/tool-call handling. Next useful action is a deliberate full MTP architecture/port. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Fresh Claude feasibility report `/tmp/pr221-claude2-feasibility-20260528-043924.txt` says this should not be direct-merged: path and namespace migrations plus a highly diverged `qwen35_backend` require a feature port. After #237/equivalent MTP foundation, port pure-add MTP files and tests into `server/`, add `ModelBackend`/`DFlashTarget`/`StepGraph` hidden-state and speculator hooks additively, then hand-port Qwen35 backend dispatcher, daemon wiring, bench scripts, and prefix-cache WARM tests. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / selective-port | Fresh Claude feasibility report `/tmp/pr183-claude2-feasibility-20260528-051446.txt` says direct merge is unsafe because the PR targets the retired `dflash/` + `dflash27b` layout and duplicates current `server/src/gemma4` loader/graph work. Portable value is Gemma4 MTP graph/loader, long-context KV correctness, asymmetric KV/`h_prev` capture, dead-cast hardening, and MTP/KV tests. Port order: KV fix, `h_prev`/asymmetric KV hook, extracted `gemma4_mtp_loader.cpp`, `gemma4_mtp_graph.cpp`, hardening, then tests/CMake. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | blocked-needs-human / selective-port | Fresh Codex feasibility report `/tmp/pr182-codex-feasibility-20260528-063427.txt` says no direct cherry-pick is safe because the branch adds old `dflash/` Gemma4 MTP graph/loader code while current HEAD lacks the corresponding `server/src/gemma4` MTP structs and cache fields. Port order: add MTP declarations/cache fields to `gemma4_internal.h`, port assistant loader pieces into `gemma4_loader.cpp`, add new `gemma4_mtp_graph.cpp` with the PR's cleanup commits folded in, then wire `server/CMakeLists.txt` and current-layout MTP loader/shape tests. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | blocked-needs-human / selective-port | Fresh Codex feasibility report `/tmp/pr181-codex-feasibility-20260528-065539.txt` says direct conflict resolution is unsafe because the PR is mostly old-layout Gemma4 draft runtime code (`dflash27b`, top-level `server/src/gemma4_*.cpp`, and deleted `dflash/CMakeLists.txt`) while current HEAD owns Gemma4 under `server/src/gemma4` with generic draft plumbing. Portable pieces: Gemma4 quantizer metadata support in `server/scripts/quantize_draft_q8.py`, current-layout error reporting hardening, draft runtime metadata/capture concepts, and reauthored smoke tests. Port order: align quantizer metadata with `draft_gguf_loader.cpp`, add optional current-layout Gemma4 error API only if needed, replace hard-coded Gemma4 draft overrides with reliable GGUF metadata, validate capture-layer IDs, then port tests. |
| #180 | `split/gemma4-08-draft-loader-quant` | likely superseded / selective-port | Overlaps current Gemma4 loader/backend code; not mechanically mergeable. |
| #177 | `split/gemma4-06-kv-correctness` | partially integrated / selective-port | This run ported PR #177's SWA ring-buffer wrap-safe K/V cache write into current `server/src/gemma4/gemma4_graph.cpp`. Direct merge still conflicts and the remaining old-layout `dflash/include/gemma4.h`, `dflash/src/gemma4_target_graph.cpp`, `dflash/src/gemma4_target_loader.cpp`, and Gemma4 tests need deliberate mapping into current `server/` APIs before #177 is fully integrated. |
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
dependency awareness: #291, #290, #286, #285, #275, #249, #193, and #75. #285
remains partially carried only as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T06:55:39-04:00 at preflight and manifest refresh.
- Primary checkout `git status --short` was clean before work began.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel`.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` succeeded (`codex-cli 0.130.0`).
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft PR refs.
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...` enumerated all open PRs and showed #291, #290, #286, #285, #275, #249, #193, and #75 as drafts/excluded.
- `git merge-base --is-ancestor origin/pr/<n> easel/auto-integration` showed #289, #284, #278, #276, #274, #273, #266, #265, #152, and #142 were current integrated non-draft heads.
- Isolated reconciliation/probe worktree `/tmp/luce-auto-cron-20260528-065539`; `git merge --no-edit origin/main` reported already up to date.
- Fresh direct merge probes for remaining non-ancestor non-draft PR refs: #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39; all direct probes conflicted and were aborted in the isolated worktree; consolidated output retained at `/tmp/luce-merge-probes-20260528-065539.txt`.
- Claude delegation for #181: tmux run exited without report content in `/tmp/pr181-claude-feasibility-20260528-065539.txt`.
- Codex delegation for #181: tmux run completed and produced `/tmp/pr181-codex-feasibility-20260528-065539.txt`, recommending a current-layout Gemma4 draft-runtime selective port rather than accepting the conflicted PR files.
- `git diff --check` passed for this refresh's manifest change.
- Search for exact merge conflict markers (`^(<<<<<<<|=======|>>>>>>>)`) under the reconciliation worktree returned no results.
- No CMake configure/build was rerun because no code changed in this refresh; validation focused on repo state, direct probes, delegated feasibility, diff sanity, and conflict-marker checks.

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at preflight and matched fetched `easel/auto-integration` (`d735948`).
- Retained worktree `/tmp/luce-auto-cron-20260528-065539` for direct-merge probe audit and final commit preparation until the pushed branch is reviewed.
- Retained direct-probe log `/tmp/luce-merge-probes-20260528-065539.txt`.
- Retained #181 probe worktree `/tmp/luce-pr181-feas-20260528-065539` plus reports `/tmp/pr181-claude-feasibility-20260528-065539.txt` (empty report) and `/tmp/pr181-codex-feasibility-20260528-065539.txt` (completed feasibility).
- Retained previous #182 probe worktree `/tmp/luce-pr182-feas-20260528-063427` plus reports `/tmp/pr182-claude-feasibility-20260528-063427.txt` (turn-limit only) and `/tmp/pr182-codex-feasibility-20260528-063427.txt` (completed feasibility).
- Retained previous #237 Claude probe worktree `/tmp/luce-pr237-claude-20260528-061548` plus reports `/tmp/pr237-claude-feasibility-20260528-061548.txt` (turn-limit only) and `/tmp/pr237-claude2-feasibility-20260528-061548.txt` (completed feasibility).
- Retained conflicted #177 probe worktree `/tmp/luce-pr177-feas-20260528-053512` plus Claude turn-limit reports `/tmp/pr177-claude-feasibility-20260528-053512.txt` and `/tmp/pr177-claude2-feasibility-20260528-053512.txt`.
- Retained previous conflicted #183 probe worktree `/tmp/luce-pr183-feas-20260528-051446` plus Claude reports `/tmp/pr183-claude-feasibility-20260528-051446.txt` and `/tmp/pr183-claude2-feasibility-20260528-051446.txt`.
- Retained previous conflicted #221 probe worktree `/tmp/luce-pr221-feas-20260528-043924` plus Claude reports `/tmp/pr221-claude-feasibility-20260528-043924.txt` and `/tmp/pr221-claude2-feasibility-20260528-043924.txt`.
- Retained earlier worktrees `/tmp/luce-auto-cron-20260528-051446`, `/tmp/luce-auto-cron-20260528-042532`, `/tmp/luce-auto-cron-20260528-035325`, `/tmp/luce-auto-cron-20260528-033958`, `/tmp/luce-auto-cron-20260528-032612`, and `/tmp/luce-auto-cron-20260528-031117`, direct-probe logs `/tmp/luce-merge-probes-20260528-051446.txt`, `/tmp/luce-merge-probes-20260528-042532.txt`, `/tmp/luce-merge-probes-20260528-035325.txt`, `/tmp/luce-merge-probes-20260528-033958.txt`, `/tmp/luce-merge-probes-20260528-032612.txt`, and `/tmp/luce-merge-probes-20260528-031117.txt`, and prior CMake configure directories for inspection.
- Prior retained conflicted worktrees and agent reports remain as listed in earlier manifest revisions; cleanup is separate maintenance.
