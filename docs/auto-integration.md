# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T21:38:23-04:00
Current base: `origin/main` `8782d07a`
Current integration tip before this refresh: `easel/auto-integration` `9dd1160f`
Refreshed stack merge commit prepared in this run: none; stack already current
Final manifest commit prepared after stack merge: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The upstream base, writable remote tip, and carried contributor PR heads were
already current at the start of this unattended run. This refresh revalidated
open non-draft PR classification and repeated direct merge probes for the
still-unintegrated PRs in fresh isolated worktrees. No new non-draft PR head
appeared, so no additional external-agent feasibility run was needed beyond the
retained prior Codex/Claude triage for the same unchanged conflict classes.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #303 | `fix/harness-portable-run-dirs` | upstream `05b008a0` | included through upstream and stack | Harness portable run/cache directories and automatic client-install fallback are in `origin/main`; the stack preserved local compatibility docs and helpers. |
| #302 | `fix/harness-model-paths` | upstream | included through upstream | Harness launcher model-path override behavior and documentation are in `origin/main`. |
| #301 | `fix/ddtree-test-harness` | upstream | included through upstream | DDTree test harness fixes are in `origin/main`. |
| #300 | `fix/sigterm-gpu-unload` | upstream | included through upstream | SIGTERM GPU-unload fix remains in upstream. |
| #299 | `feat/draft-swa-flag` | upstream | included through upstream | Draft SWA env/flag support remains in upstream. |
| #298 | `fix/gemma4-destructor-link` | upstream | included through upstream | Gemma4 destructor-link fix remains in upstream. |
| #292 | `feat-backend-ipc-payload-pipe-open` | upstream / `90bc52f` | included through upstream and stack | `origin/main` carries backend IPC payload-pipe support; the stack already contained the same PR head before it landed upstream. |
| #297 | `feat-server-laguna-layer-split-adapter-v2` | `53dd1686` | included / now draft | Laguna target-layer-split adapter remains carried in the easel stack. The GitHub PR is currently draft, so it is excluded from the non-draft target but retained as an already-carried dependency. |
| #295 | `fix-layer-split-sampling` | `a9aedf7d` | included | Target layer-split sampling support remains an ancestor of the stack. |
| #294 | `feat/server-passthrough-proxy` | `0883c2ef` | included | Server passthrough proxy wiring, piecewise keep-ratio curve, query survival checks, and unit coverage are carried. |
| #289 | `pipeline_moe` | `0ffab8a1` | included | Pipelined hybrid Qwen35 MoE decode update remains an ancestor of the stack. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `5e861b4d` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path is carried. |
| #274 | `feat/pflash-drafter-ee7` | `e64a2b80` | included | Adaptive pFlash composition, EE7/drafter updates, docs, tests, and follow-up fixes are carried. |
| #266 | `feat/harness-typed-adapters` | `17525eae` | included | Typed harness adapters and format-aware session-inject proxy are carried. |
| #152 | `main` | `cf735bee` | included | Gemma 4 RTX 4090 backend helpers are carried. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62f` | included | FP16 safetensors drafter support is carried. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the primary non-draft target, but earlier Docker/CLI/bench integration dependency remains partially carried. The current draft head `82ebf982` is not fully an ancestor. |

## Fresh probe results from this run

| PR | Outcome | Notes |
|---:|---|---|
| current upstream/base | checked | `git merge --no-edit origin/main` in isolated worktree `/tmp/luce-cron-20260528-213918/reconcile` reported `Already up to date.`; no upstream reconciliation commit was needed. |
| current integrated PRs | checked | Ancestor checks still pass for carried open non-draft PR refs: #295, #294, #289, #276, #274, #266, #152, and #142. Draft #297 also remains an ancestor. |
| upstream-adjacent PRs | checked | #292/#298/#299/#300/#301/#302/#303 are included through `origin/main`. |
| #237 | blocked-needs-human / selective-port | Fresh direct merge probe still conflicts across deleted legacy `dflash/` server files, `server/CMakeLists.txt`, backend factory/model backend, MTP common interfaces/orchestrator, Qwen35 loader/graph/backend/target files, and tests. Prior tmux-driven Claude/Codex review found this is not a safe mechanical merge because the old MTP API/loader/graph/decode path collides with current remote-draft, layer-split, pFlash, thinking-budget, telemetry, and MoE hooks. |
| #221 | blocked-needs-human / dependency | Fresh direct merge probe still conflicts across old prefix-cache/MTP/common/Qwen35 files, server tests, legacy script paths, and benchmark artifacts. It still depends on a current-layout #237-equivalent MTP foundation before a useful port can be made. |
| #154 | blocked-needs-human / dependency | Fresh direct merge probe conflicts in legacy `dflash/CMakeLists.txt`, MTP docs, `server/src/internal.h`, Qwen35 loader/target graph, MTP smoke/contract tests, and `test_dflash.cpp`. Portable only after current-layout Qwen35 MTP exists. |
| #153 | blocked-needs-human / dependency | Fresh direct merge probe shows the same old-layout MTP integrated runtime conflict class as #154; requires current-layout loader/graph/cache design work. |
| #137 | suggested-close/superseded | Fresh direct merge probe conflicts only in legacy `dflash/CMakeLists.txt`; current `server/CMakeLists.txt` owns CUDA arch/BSA handling. |
| #135 | blocked-needs-human / selective-port | Fresh direct merge probe conflicts in `server/src/internal.h`, `server/src/qwen35/qwen35_target_graph.cpp`, and `server/test/test_dflash.cpp`. A prior tmux-driven Codex inspection concluded it is not mechanically resolvable: the PR's batched `n_seqs` target stepping and inline scheduler are based on the old monolithic harness and collide with current `TargetLoadPlan`, partial cache/layer ownership, FWHT K rotation, MoE/router capture, `last_token_logits_only`, daemon dispatch, layer-split daemon, draft IPC, peer access, sampler, prefix snapshot, and extracted module structure. |
| #94 | suggested-close/superseded | Fresh direct merge probe conflicts in current draft graph/safetensors loader/internal files. Current code already carries the useful SWA draft parsing and causal-mask behavior; remaining branch edits are old-layout/obsolete. |
| #48 | suggested-close/superseded | Fresh direct merge probe conflicts only in legacy `dflash/CMakeLists.txt`; current server CMake supersedes the old Blackwell arch detection patch. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | First port only the common MTP API surface to `server/src/common`: no-op `DFlashTarget`/`ModelBackend` hooks, `mtp_interface`, chain runner/orchestrator, CMake entries, and `test_common_mtp_orchestrator`; after that compiles, port Qwen35 MTP graph/loader/runtime while preserving current remote-draft, pFlash, layer-split, thinking-budget, telemetry, and MoE hooks. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Revisit after #237-equivalent MTP foundation; then port hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests to current `server/` layout. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | blocked-needs-human / dependency | Mine linear MTP decode semantics after current-layout Qwen35 MTP exists. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / dependency | Mine loader/graph/cache/test ideas after current-layout Qwen35 MTP exists. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Design a current-layout multi-request scheduler in the qwen35 daemon/graph-builder layer, including multi-request state ownership, batched `n_seqs` target stepping, copy-in/copy-back validation, and tagged streaming. Do not resurrect the old monolithic `test_dflash.cpp` scheduler mechanically. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | suggested-close/superseded | Ask author to close or retarget to current `server/CMakeLists.txt` if anything remains. |
| #94 | `feat/dflash-qwen36-swa-draft` | suggested-close/superseded | Useful behavior appears absorbed; ask author/maintainers whether any remaining old-layout tests should be reauthored before close. |
| #48 | `fix/consumer-blackwell-auto-detect` | suggested-close/superseded | Close or retarget to current `server/CMakeLists.txt` if still needed. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
ongoing dependency awareness: #304, #297, #291, #290, #286, #285, #275, #249,
and #193. #286 is the draft PR for the current auto-integration snapshot. #304
is a draft LLM auto context compaction PR observed again in this run; it was not
integrated because it is draft.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T21:38:23-04:00 for the manifest refresh timestamp.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status`, `HOME=/home/erik /home/erik/.local/bin/claude auth status --text`, and `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` all succeeded.
- `git fetch --prune origin` and `git fetch --prune easel` completed separately; targeted pull-ref fetch recreated open PR refs for #304, #297, #295, #294, #291, #290, #289, #286, #285, #276, #275, #274, #266, #249, #237, #221, #193, #154, #153, #152, #142, #137, #135, #94, and #48.
- Isolated reconciliation worktree `/tmp/luce-cron-20260528-213918/reconcile` was created from `easel/auto-integration`; `git merge --no-edit origin/main` reported `Already up to date.`
- Ancestor checks still pass for carried open non-draft PR refs: #295, #294, #289, #276, #274, #266, #152, and #142. Draft #297 remains carried at `53dd1686`; draft #285 remains only partially carried.
- Fresh direct merge probes were run in isolated worktrees for the still-unintegrated non-draft PRs #237, #221, #154, #153, #137, #135, #94, and #48; each still conflicts as summarized above. Probe log: `/tmp/luce-cron-20260528-213918/probe-results.txt`.
- No new non-draft PR head required another external-agent run this cycle. The retained prior tmux-driven Codex/Claude feasibility inspections for the unchanged conflict classes remain applicable.
- Manifest-only validation on the reconciliation worktree: `git diff --check HEAD -- docs/auto-integration.md` passed.
- Full stack whitespace validation `git diff --check origin/main...HEAD` still reports only known pre-existing whitespace warnings outside this run's change (`luce-bench/src/lucebench/fixtures/forge_eval/scenarios/_model_quality.py`, `_stateful_model_quality.py`, and `scripts/docker_build_env.sh`).
- Anchored conflict-marker scan on the reconciliation worktree `git grep -n -E '^(<<<<<<<|>>>>>>>|=======)( |$)' -- . ':!server/eval/humaneval_plus/humanevalplus.jsonl'` found no markers.

## Notes

- This run produced a manifest-only refresh on top of `9dd1160f`; no source stack rewrite was needed because `origin/main`, `easel/auto-integration`, and all carried non-draft PR heads were unchanged.
- Retained reconciliation worktree `/tmp/luce-cron-20260528-213918/reconcile` contains this manifest refresh.
- Retained fresh conflict probe worktrees `/tmp/luce-cron-20260528-213918/probe-pr237`, `/tmp/luce-cron-20260528-213918/probe-pr221`, `/tmp/luce-cron-20260528-213918/probe-pr154`, `/tmp/luce-cron-20260528-213918/probe-pr153`, `/tmp/luce-cron-20260528-213918/probe-pr137`, `/tmp/luce-cron-20260528-213918/probe-pr135`, `/tmp/luce-cron-20260528-213918/probe-pr94`, and `/tmp/luce-cron-20260528-213918/probe-pr48` are intentionally left for review because they are conflicted.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
