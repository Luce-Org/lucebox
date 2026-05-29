# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-29T00:22:51-04:00
Current base: `origin/main` `8782d07a`
Current integration tip before this refresh: `easel/auto-integration` `7735f660`
Refreshed stack merge commit prepared in this run: none; stack already current
Final manifest commit prepared after stack/probe refresh: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
At this run's start `easel/auto-integration` was already based on current
`origin/main` (`0` behind / `377` ahead), and every currently mergeable
non-draft contributor PR head was already an ancestor of the stack. This refresh
revalidated auth/tooling, fetched current pull refs, repeated direct worktree
merge probes for the remaining conflicted non-draft PRs, and ran a fresh Codex
feasibility review for PR #237 plus a Claude Code attempt for PR #135. No source
stack rewrite was made.

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
| #297 | `feat-server-laguna-layer-split-adapter-v2` | `53dd1686` | included / draft at final check | Laguna target-layer-split adapter remains carried in the easel stack. It is currently draft and excluded from the non-draft target, but retained as an already-carried dependency. |
| #295 | `fix-layer-split-sampling` | `a9aedf7d` | included | Target layer-split sampling support remains an ancestor of the stack. |
| #294 | `feat/server-passthrough-proxy` | `0883c2ef` | included | Server passthrough proxy wiring, piecewise keep-ratio curve, query survival checks, and unit coverage are carried. |
| #289 | `pipeline_moe` | `0ffab8a1` | included | Pipelined hybrid Qwen35 MoE decode update remains an ancestor of the stack. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `5e861b4d` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path is carried. |
| #274 | `feat/pflash-drafter-ee7` | `e64a2b80` | included | Adaptive pFlash composition, EE7/drafter updates, docs, tests, and follow-up fixes are carried. |
| #266 | `feat/harness-typed-adapters` | `17525eae` | included | Typed harness adapters and format-aware session-inject proxy are carried. |
| #152 | `main` | `cf735bee` | included | Gemma 4 RTX 4090 backend helpers are carried. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62f` | included | FP16 safetensors drafter support is carried. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the primary non-draft target, but earlier Docker/CLI/bench integration dependency remains partially carried. The draft head has moved and is not fully an ancestor. |

## Validation run

This run performed:

- `date -Is` -> 2026-05-29T00:18:22-04:00 for preflight and 2026-05-29T00:22:51-04:00 for the manifest refresh timestamp.
- Primary checkout preflight: `git status --short` was clean; branch was `auto-integration`; remotes were `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- Auth/tooling checks with real user credentials succeeded: `gh auth status`, `claude auth status --text`, and a harmless `codex --help` smoke check.
- `git fetch --prune origin` and `git fetch --prune easel` completed successfully.
- Open PR enumeration used `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json number,title,isDraft,author,headRefName,headRepositoryOwner,updatedAt,mergeable,baseRefName --jq ...`.
- Fetched open non-draft PR refs explicitly: #295, #294, #289, #276, #274, #266, #237, #221, #154, #153, #152, #142, #137, #135, #94, and #48.
- `git rev-list --left-right --count origin/main...easel/auto-integration` reported `0` behind and `377` ahead before this manifest commit.
- `git merge-base --is-ancestor` checks pass for carried open non-draft PR refs: #295, #294, #289, #276, #274, #266, #152, and #142.
- Reconciliation worktree `/tmp/luce-auto-cron-20260529-001903/reconcile` was created from `easel/auto-integration`; `origin/main` is already an ancestor and no base merge was required.
- Repeated direct merge probes from `easel/auto-integration` in `/tmp/luce-auto-cron-20260529-001903/` for #237, #221, #154, #153, #137, #135, #94, and #48. All still conflict in the same conflict classes recorded below.
- Fresh delegation: Codex ran in tmux session `luce237-luce-auto-cron-20260529-001903` and produced `/tmp/luce237-luce-auto-cron-20260529-001903-codex-report.txt`, confirming #237 is feasible only as a semantic selective port into the current `server/` layout. Claude Code ran in tmux session `luce135-luce-auto-cron-20260529-001903` and ended at max turns with no usable report (`/tmp/luce135-luce-auto-cron-20260529-001903-claude-report.txt`).
- `git diff --check -- docs/auto-integration.md` passed after this manifest update.

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Head | Current status | Probe result / next useful action |
|---:|---|---:|---|---|
| #237 | `feat/dflash-mtp-foundation` | `02c6a6c4` | blocked-needs-human / salvage-port | Probe worktree `/tmp/luce-auto-cron-20260529-001903/pr-237-probe` has delete/update conflicts in old `dflash/scripts/server.py`, `dflash/src/common/backend_factory.cpp`, `dflash/src/server/server_main.cpp`; file-location conflicts for new `server/src/common/mtp_*`, `server/src/qwen35/qwen35_mtp*`, and `server/test/test_common_mtp_orchestrator.cpp`; and semantic conflicts in `server/CMakeLists.txt`, `backend_factory.h`, `step_graph.h`, qwen35 loader/graph/backend/dflash target, and `server/test/test_dflash.cpp`. Fresh Codex report says selective port is feasible but not a clean merge: ignore obsolete old-layout files, add MTP source resolution/CLI/factory pieces, preserve current qwen35moe, pFlash, layer-split, remote-draft, prefix/disk-cache, budget-hook, and sentinel behavior, and compile-test focused targets only after manual conflict resolution. |
| #221 | `feat/mtp-prefix-warm-ghost` | `05502974` | blocked-needs-human / dependent salvage-port | Probe worktree `/tmp/luce-auto-cron-20260529-001903/pr-221-probe` conflicts on old `dflash/` scripts/backend files; `server/src/common/dflash_target.h`, `model_backend.h`, `step_graph.h`, `gguf_mmap.h`; MTP file-location paths; Qwen35 loader/graph/backend/dflash target areas; and `server/test/test_dflash.cpp` / MTP tests. Prior Codex report classifies it as not direct-merge and not superseded: port #237-equivalent MTP foundation first, then mine #221 for prefix-cache MTP WARM behavior, snapshot/restore head KV, partial/range warm, warm-completion callback, per-request dispatcher/PFlash protocol, and `server/test/test_prefix_cache_mtp.cpp` coverage while excluding stale old-layout benchmark artifacts. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | `2f4ede79` | blocked-needs-human / dependency | Probe worktree `/tmp/luce-auto-cron-20260529-001903/pr-154-probe` conflicts on old `dflash/CMakeLists.txt`, file-location moves for MTP docs/tests and `f16_convert.cu`, plus semantic conflicts in `internal.h`, `gguf_target_loader.cpp`, `qwen35_target_graph.cpp`, and `test_dflash.cpp`. Mine linear MTP decode semantics after current-layout Qwen35 MTP exists. |
| #153 | `xabicasa/dflash-mtp-integrated` | `e9b17cb1` | blocked-needs-human / dependency | Probe worktree `/tmp/luce-auto-cron-20260529-001903/pr-153-probe` conflicts on old CMake, moved MTP docs/tests and `f16_convert.cu`, and the same core Qwen35/internal target files. Mine loader/graph/cache/test ideas after current-layout Qwen35 MTP exists. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | `561b0ac1` | blocked-needs-human / selective-port | Fresh probe worktree `/tmp/luce-auto-cron-20260529-001903/pr-135-probe` has three semantic conflicts: `server/src/internal.h`, `server/src/qwen35/qwen35_target_graph.cpp`, and `server/test/test_dflash.cpp`. Prior Codex classified it as selective current-layout port, not direct merge: salvage opt-in `--target-cache-slots` / `SLOT <id>`, tagged stream demux, request commands, aligned-bucket scheduler and batch validation. Fresh Claude Code attempt reached max turns with no usable report, so prior Codex guidance remains the usable delegation evidence. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | `297fc74e` | suggested-close/superseded | Probe worktree `/tmp/luce-auto-cron-20260529-001903/pr-137-probe` only conflicts on deleted old `dflash/CMakeLists.txt`. Ask author to close or retarget to current `server/CMakeLists.txt` if anything remains. |
| #94 | `feat/dflash-qwen36-swa-draft` | `d2f9c9dd` | suggested-close/superseded | Probe worktree `/tmp/luce-auto-cron-20260529-001903/pr-94-probe` conflicts in `server/src/draft/draft_graph.cpp`, `draft_safetensors_loader.cpp`, and `server/src/internal.h`; useful behavior appears absorbed. Ask author/maintainers whether any remaining old-layout tests should be reauthored before close. |
| #48 | `fix/consumer-blackwell-auto-detect` | `858b84b6` | suggested-close/superseded | Probe worktree `/tmp/luce-auto-cron-20260529-001903/pr-48-probe` only conflicts on deleted old `dflash/CMakeLists.txt`. Close or retarget to current `server/CMakeLists.txt` if still needed. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
ongoing dependency awareness: #304, #297, #291, #290, #286, #285, #275, #249,
and #193. #286 is the draft PR for the current auto-integration snapshot. #304
is a draft LLM auto context compaction PR and excluded because it is draft. #297
is draft at this run's enumeration and remains carried as an already-integrated
draft dependency.

## Retained worktrees / logs

The conflicted probe worktrees were intentionally retained for manual follow-up
because safe cleanup would require resolving or discarding conflicted indexes:

- `/tmp/luce-auto-cron-20260529-001903/pr-237-probe`
- `/tmp/luce-auto-cron-20260529-001903/pr-221-probe`
- `/tmp/luce-auto-cron-20260529-001903/pr-154-probe`
- `/tmp/luce-auto-cron-20260529-001903/pr-153-probe`
- `/tmp/luce-auto-cron-20260529-001903/pr-137-probe`
- `/tmp/luce-auto-cron-20260529-001903/pr-135-probe`
- `/tmp/luce-auto-cron-20260529-001903/pr-94-probe`
- `/tmp/luce-auto-cron-20260529-001903/pr-48-probe`

The clean reconciliation worktree `/tmp/luce-auto-cron-20260529-001903/reconcile`
was also left in place to avoid worktree force-deletion in an unattended run.

Agent reports/logs retained:

- `/tmp/luce237claude224833-report.txt` (prior Claude Code max-turns without usable report)
- `/tmp/luce237codex224833-report.txt` (prior Codex salvage-port report for #237)
- `/tmp/luce221codex-20260528-230909-report.txt` (prior Codex dependent salvage-port report for #221)
- `/tmp/luce135claude-20260528-232939-report.txt` (prior Claude Code max-turns without usable report)
- `/tmp/luce135codex-20260528-232939-report.txt` (prior Codex selective-port report for #135)
- `/tmp/luce237-luce-auto-cron-20260529-001903-codex-report.txt` (fresh Codex selective-port report for #237)
- `/tmp/luce135-luce-auto-cron-20260529-001903-claude-report.txt` (fresh Claude Code max-turns without usable report for #135)

## Notes

This run produced a manifest-only refresh on top of `7735f660`; no source stack
rewrite was needed because `origin/main`, `easel/auto-integration`, and all
carried mergeable non-draft PR heads were already current. The next useful work
remains a human-reviewed selective port of #237's MTP foundation into the current
`server/` layout. After that, mine #221's WARM-cache/dispatcher behavior and
#153/#154's native/integrated MTP semantics. #135 remains confirmed by Codex as a
selective current-layout port focused on the qwen35 daemon scheduler and batched
target-step API. #137 and #48 look like old `dflash/CMakeLists.txt` changes that
should be closed or retargeted, and #94 appears largely superseded by current
draft/SWA support.
