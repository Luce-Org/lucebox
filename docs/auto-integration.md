# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T15:47:19-04:00
Current base: `origin/main` `3ba525e0`
Current integration tip before this refresh: `easel/auto-integration` `c244ecd4`
Refreshed stack tip prepared in this run: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was clean at preflight, but its local `auto-integration`
branch was behind fetched `easel/auto-integration`, so this refresh used isolated
worktree `/tmp/luce-auto-cron-20260528-1547` on top of fetched
`easel/auto-integration`. `origin/main` remains at `3ba525e0`. This run updated
the stack ancestry with the latest non-draft #276 head. The direct merge was
clean, but its only tree change duplicated Qwen3.6 Claude-code tool-calling
server unit coverage that the integration stack already carried; this refresh
therefore records #276 as a merge parent and drops the duplicate test block in a
separate integration-only fix. The rest of the non-draft open inventory was
rechecked; the remaining non-ancestors still require selective current-layout
ports rather than direct merges.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #296 | `fix/pascal-multiarch-q4km-draft` | upstream | included through upstream | `origin/main` contains this PR via merge commit `3ba525e0`; the integration stack is reconciled over it. |
| #295 | `fix-layer-split-sampling` | `a9aedf7` | included | Supports sampled requests in target layer split by retaining prefill logits through Qwen35/Gemma4 prefix snapshots. |
| #294 | `feat/server-passthrough-proxy` | `0883c2e` | included | Adds server passthrough proxy wiring, piecewise keep-ratio curve, query survival checks, and unit coverage. |
| #292 | `feat-backend-ipc-payload-pipe-open` | `90bc52f` | included | Adds backend IPC payload-pipe support for remote draft feature/noise payloads, with integration-only pipe-drain hardening and shared feature-slice storage helper. |
| #289 | `pipeline_moe` | `0ffab8a` | included | Pipelined hybrid Qwen35 MoE decode update is carried. The inaccessible submodule pointer from earlier PR history remains excluded. |
| #284 | `fix/draft-safetensors-rope-theta` | upstream | included through upstream | Reads and validates `rope_theta` from draft safetensors `config.json`. |
| #278 | `fix-pflash-drafter-backend-precision-submit` | upstream | included through upstream | `origin/main` contains this PR. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `5e861b4` | included | Latest head is a merge parent. Its Qwen3.6-27B Anthropic/Claude-code tool-call tests were already present in the stack, so the duplicate block from the merge was removed as an integration-only cleanup. |
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
| upstream sync | checked / no-op | `origin/main` remains `3ba525e0`; fetched `easel/auto-integration` was `c244ecd4`, already based on current upstream. |
| #276 | integrated with integration-only cleanup | Fetched `origin/pr/276` at `5e861b4`; `git merge --no-ff --no-edit origin/pr/276` completed cleanly in `/tmp/luce-auto-cron-20260528-1547`. Its only tree change duplicated existing `server/test/test_server_unit.cpp` tool-calling coverage and left a bad duplicate block, so this run restored the file to the pre-merge stack content while preserving #276 as a merge parent. |
| open inventory | checked | Open non-draft PRs are #295, #294, #292, #289, #276, #274, #266, #237, #221, #154, #153, #152, #142, #137, #135, #94, and #48. Draft/excluded PRs are listed below. |
| stack ancestry | checked | After the #276 merge, ancestor checks pass for #295, #294, #292, #289, #276, #274, #266, #152, and #142. Upstream-included PRs #296, #284, #278, and #265 remain present through `origin/main`. |
| remaining non-ancestor non-draft PRs | direct merge probes conflicted | Fresh isolated probes in `/tmp/luce-probe-20260528-1547` attempted #237, #221, #154, #153, #137, #135, #94, and #48 on top of the refreshed stack. All direct probes conflicted and were aborted in the probe worktree. Consolidated log: `/tmp/luce-merge-probes-20260528-1547.txt`. |
| #237 delegated recheck | still blocked-needs-human / selective-port | Claude print-mode under tmux exited without a usable report (`/tmp/claude-pr237-1547-report.txt` stayed empty). Codex under tmux produced `/tmp/codex-pr237-1547-report.txt`: direct merge is unsafe; conflicts span old `dflash/` rename/delete paths, `ModelBackend`/`BackendArgs`/`DFlashTarget`/`StepGraph` API drift, Qwen35 backend/target-graph drift, CMake/tests, and obsolete old server files. Recommended path is a selective current-layout MTP port, starting with common MTP interfaces/runner/orchestrator and disabled-by-default MTP parsing, then backend/target/CLI wiring. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge and delegated review confirm this requires a selective port: move common MTP interfaces/runners/orchestrator under `server/src/common`, add MTP source/config stubs while preserving current budget/telemetry/remote-draft fields, reauthor Qwen35 hidden/logit/rollback capture against current `StepGraph`/MoE/layer-split code, then wire `qwen35_mtp*`, CLI flags, CMake, and focused tests. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Requires #237/equivalent MTP foundation first, then a current-layout feature port for hidden-state/speculator hooks, backend dispatcher, daemon wiring, prefix WARM tests, and cache/snapshot behavior. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | selective-port | Linear native MTP decode semantics are portable only after a current-layout Qwen35 MTP foundation exists. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Feasible but requires loader/graph/cache/MoE design work, not conflict-marker resolution. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | stale / suggested close | Edits only deleted legacy `dflash/CMakeLists.txt`; close or ask author to retarget current `server/CMakeLists.txt`. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Port `n_seqs` primitives/test helpers first, then design multi-slot `Qwen35Backend` state and scheduler API. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / verify-only | Remaining conflicts are old-layout draft/common paths; current code carries the useful SWA pieces. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / suggested close | Current CMake supersedes the old `dflash/CMakeLists.txt` change. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #297, #291, #290, #286, #285, #275, #249, #193, and #75.
#297 is a draft Laguna follow-up stacked on #295. #291/#290 are draft/conflicting
Gemma4 residency work. #286 is the current draft auto-integration snapshot. #285
remains partially carried only as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T15:47:19-04:00 at preflight.
- Primary checkout `/home/erik/Projects/luce2` was clean at preflight (`git status --short` returned no paths), on local branch `auto-integration`, with remotes `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`. A later #237 delegated setup command was accidentally launched from the primary checkout; its conflicted merge state was immediately aborted with `git merge --abort`, and a follow-up `git status --short` returned clean.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel` with repo/workflow scopes.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --help` succeeded.
- `git fetch --prune origin` and `git fetch --prune easel` completed separately; open non-draft PR refs were fetched explicitly.
- `git merge --no-ff --no-edit origin/pr/276` succeeded in `/tmp/luce-auto-cron-20260528-1547`; the duplicate test block introduced by that merge was then removed by restoring `server/test/test_server_unit.cpp` to the pre-merge stack content.
- Direct conflict probes for #237, #221, #154, #153, #137, #135, #94, and #48 ran in `/tmp/luce-probe-20260528-1547`; all conflicted and were aborted there.
- Codex delegated #237 feasibility review ran through tmux session `codex-pr237-1547` and produced `/tmp/codex-pr237-1547-report.txt`. Claude delegated review through `claude-pr237-1547` did not produce a usable report.
- `git diff --check` passed before commit.
- Targeted conflict-marker scan over `docs/auto-integration.md` and `server/test/test_server_unit.cpp` found no merge markers.
- Local CMake/test execution was not rerun in this WSL environment because prior runs hit the known CUDA compiler-identification blocker before project configure; CI remains the authoritative CMake verifier for CUDA/server build health.

## Notes

- Primary checkout `/home/erik/Projects/luce2` is clean again and remains behind fetched `easel/auto-integration`; the pushed branch is the source of truth.
- Retained worktrees/logs from this run: `/tmp/luce-auto-cron-20260528-1547`, `/tmp/luce-probe-20260528-1547`, `/tmp/luce-pr237-delegated-20260528-1547`, `/tmp/luce-merge-probes-20260528-1547.txt`, `/tmp/claude-pr237-1547-report.txt`, and `/tmp/codex-pr237-1547-report.txt`.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
