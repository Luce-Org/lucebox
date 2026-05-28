# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T15:36:00-04:00
Current base: `origin/main` `3ba525e0`
Current integration tip before this refresh: `easel/auto-integration` `1ae2f2cb`
Refreshed stack tip prepared in this run: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was not used directly because it still carries unrelated
local modifications; this refresh used isolated worktree
`/tmp/luce-auto-integration-20260528` on top of fetched `easel/auto-integration`.
`origin/main` remains at `3ba525e0`, and the currently maintained stack already
includes the sampled layer-split fix from #295. Since the prior refresh, a new
draft follow-up (#297) appeared stacked on #295, so this run is a manifest update
and inventory refresh rather than a new integration merge.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #296 | `fix/pascal-multiarch-q4km-draft` | upstream | included through upstream | `origin/main` now contains this PR via merge commit `3ba525e0`; the integration stack was reconciled over it. |
| #295 | `fix-layer-split-sampling` | `a9aedf7` | included | Supports sampled requests in target layer split by retaining prefill logits through Qwen35/Gemma4 prefix snapshots. Manual conflict resolution kept #295's always-capture semantics over the prior conditional optimization. |
| #294 | `feat/server-passthrough-proxy` | `0883c2e` | included | Adds server passthrough proxy wiring, piecewise keep-ratio curve, query survival checks, and unit coverage. |
| #292 | `feat-backend-ipc-payload-pipe-open` | `90bc52f` | included | Adds backend IPC payload-pipe support for remote draft feature/noise payloads, with integration-only pipe-drain hardening and shared feature-slice storage helper. |
| #289 | `pipeline_moe` | `0ffab8a` | included | Pipelined hybrid Qwen35 MoE decode update is carried. The inaccessible submodule pointer from earlier PR history remains excluded. |
| #284 | `fix/draft-safetensors-rope-theta` | upstream | included through upstream | Reads and validates `rope_theta` from draft safetensors `config.json`. |
| #278 | `fix-pflash-drafter-backend-precision-submit` | upstream | included through upstream | `origin/main` contains this PR. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path. |
| #274 | `feat/pflash-drafter-ee7` | `e64a2b8` | included | Adaptive pFlash compression docs/config, transitive anchor default, EE7/drafter updates, and Qwen3 closed-`<think>` Jinja prefill docs/tests are carried. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Typed harness adapters and format-aware session-inject proxy. |
| #265 | `feat-cpp-server-target-layer-split-prep` | upstream | included through upstream | Upstream main contains this PR. |
| #177 | `split/gemma4-06-kv-correctness` | `0a95d4b` | selectively included | Previously carried split-on-wrap SWA KV writes and larger SWA ring allocation for chunked long-context prefill in `server/src/gemma4/gemma4_loader.cpp`. Remaining TQ3/large-head KV alignment and tests still require manual design. |
| #174 | `split/gemma4-14-small-vram-docs` | `8b1caba` | selectively included | Useful small-VRAM/VMM documentation is already ported into current `server/README.md`. |
| #152 | `main` | `cf735be` | included | Gemma 4 RTX 4090 backend helpers already carried. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | FP16 safetensors drafter support already carried. |
| #94 | `feat/dflash-qwen36-swa-draft` | `d2f9c9d` | absorbed / selectively included | Current draft/common code carries safetensors SWA config parsing and causal-mask behavior; remaining branch edits target old paths. |
| #62 | `fix/issue-55-stable-kv-pad` | `0ce6832` | absorbed / selectively included | Current server layout carries daemon reset behavior and regression coverage; remaining branch conflicts are legacy tests. |
| #48 | `fix/consumer-blackwell-auto-detect` | `858b84b` | superseded / selectively included | Current `server/CMakeLists.txt` owns CUDA architecture resolution and Blackwell/GB10 handling. |
| #39 | `feat/moe-35b-a3b` | `c86ec86` | partially integrated / mostly superseded | Draft safetensors `config.json`/YaRN survivorship was previously ported; remaining old loader/FFN/target-graph pieces are superseded by current qwen35moe/DDTree code. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the non-draft target, but earlier Docker/CLI/bench integration dependency remains carried. |

## Fresh probe results from this run

| PR | Outcome | Notes |
|---:|---|---|
| upstream sync | checked / no-op | `origin/main` remains `3ba525e0`; fetched `easel/auto-integration` is `1ae2f2cb`, and no upstream merge was needed for this refresh. |
| open inventory | checked | Open non-draft PRs still include #295, #294, #292, #289, #276, #274, #266, #237, #221, #154, #153, #152, #142, #137, #135, #94, and #48. Draft #297 is stacked on #295 and stays outside the non-draft stack. |
| stack ancestry | checked | `git merge-base --is-ancestor origin/pr/295 HEAD` returned yes in the worktree; `origin/pr/297`, `origin/pr/291`, and `origin/pr/290` were not ancestors of the current stack tip. The included upstream PRs #296, #284, #278, and #265 remain present through `origin/main`. |
| historical conflict status | carried forward | The earlier direct-probe findings for #237, #221, #154, #153, #137, #135, #94, and #48 remain the basis for their blocked/stale classifications. |
| review note | recorded | #291 and #290 stay draft/conflicting; they remain on the hold list for later manual porting work if their dependency chain becomes worthwhile. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge still conflicts across current server/common/qwen35 files and old `dflash/` paths. Port common MTP interfaces/runners and tests under `server/`, add MTP fields to `BackendArgs`/`ModelBackend`, reauthor Qwen35 hidden/logit/rollback capture against current `StepGraph`/MoE changes, then wire `qwen35_mtp*`, CLI flags, CMake, and focused tests. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Requires #237/equivalent MTP foundation, then a current-layout feature port for hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | selective-port | Linear native MTP decode semantics are portable after a current-layout Qwen35 MTP design. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Feasible but requires loader/graph/cache/MoE design work, not conflict-marker resolution. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | stale / suggested close | Edits only deleted legacy `dflash/CMakeLists.txt`; close or ask author to retarget current `server/CMakeLists.txt`. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Port `n_seqs` primitives/test helpers first, then design multi-slot `Qwen35Backend` state and scheduler API. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / verify-only | Remaining conflicts are old-layout draft/common paths; current code carries the useful SWA pieces. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / suggested close | Current CMake supersedes the old `dflash/CMakeLists.txt` change. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #297, #291, #290, #286, #285, #275, #249, #193, and #75.
#297 is a draft Laguna follow-up stacked on #295. #286 is the current draft
auto-integration snapshot. #285 remains partially carried only as an integration
dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T15:36:00-04:00 at preflight.
- Primary checkout `/home/erik/Projects/luce2` still contains unrelated local modifications, so the refresh work happened in clean worktree `/tmp/luce-auto-integration-20260528`.
- `git status --short --branch` in the worktree reported `## auto-integration-maint-20260528...easel/auto-integration`.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 40` refreshed the open PR inventory, including new draft follow-up #297.
- `gh pr view` was used for #295, #297, #291, and #290 to confirm bodies, draft state, and stack relationships.
- `git merge-base --is-ancestor origin/pr/295 HEAD` returned yes in the worktree; `origin/pr/297`, `origin/pr/291`, and `origin/pr/290` were not ancestors of the current stack tip.
- `origin/main` and fetched `easel/auto-integration` remained at `3ba525e0` and `1ae2f2cb` respectively, so no upstream merge was needed.
- `git diff --check` passed before this manifest refresh.

## Notes

- Primary checkout `/home/erik/Projects/luce2` retains unrelated uncommitted changes and is intentionally left untouched.
- Retained integration worktree `/tmp/luce-auto-integration-20260528`; no new probe worktree or log was created in this refresh.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
