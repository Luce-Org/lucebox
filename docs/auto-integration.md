# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T22:07:55-04:00
Current base: `origin/main` `8782d07a`
Current integration tip before this refresh: `easel/auto-integration` `a98ffb79`
Refreshed stack merge commit prepared in this run: none; stack already current
Final manifest commit prepared after stack/probe refresh: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The upstream base, writable remote tip, and carried contributor PR heads were
already current at the start of this unattended run. This refresh revalidated
the open PR set, fetched current pull refs, confirmed that all mergeable
non-draft contributor heads are ancestors of the integration tip, and repeated
worktree merge probes for the remaining conflicted PRs. No source stack rewrite
was made.

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
| #297 | `feat-server-laguna-layer-split-adapter-v2` | `53dd1686` | included | Laguna target-layer-split adapter remains carried in the easel stack. The GitHub PR is non-draft again and its current head is an ancestor of the integration tip. |
| #295 | `fix-layer-split-sampling` | `a9aedf7d` | included | Target layer-split sampling support remains an ancestor of the stack. |
| #294 | `feat/server-passthrough-proxy` | `0883c2ef` | included | Server passthrough proxy wiring, piecewise keep-ratio curve, query survival checks, and unit coverage are carried. |
| #289 | `pipeline_moe` | `0ffab8a1` | included | Pipelined hybrid Qwen35 MoE decode update remains an ancestor of the stack. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `5e861b4d` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path is carried. |
| #274 | `feat/pflash-drafter-ee7` | `e64a2b80` | included | Adaptive pFlash composition, EE7/drafter updates, docs, tests, and follow-up fixes are carried. |
| #266 | `feat/harness-typed-adapters` | `17525eae` | included | Typed harness adapters and format-aware session-inject proxy are carried. |
| #152 | `main` | `cf735bee` | included | Gemma 4 RTX 4090 backend helpers are carried. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62f` | included | FP16 safetensors drafter support is carried. |
| #285 | `feat/lucebox-docker` | draft | partial draft dependency | Draft PR, outside the primary non-draft target, but earlier Docker/CLI/bench integration dependency remains partially carried. The draft head has moved (`d8549958` observed on `easel/feat/lucebox-docker`) and is not fully an ancestor. |

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T22:07:55-04:00 for the run timestamp.
- Primary checkout preflight: `git status --short` was clean; branch was `auto-integration`; remotes were `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- Auth/tooling checks with real user credentials succeeded: `gh auth status`, `claude auth status --text`, and a harmless Codex CLI help check.
- `git fetch --prune origin` and `git fetch --prune easel` completed successfully.
- Open PR enumeration used `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json number,title,author,isDraft,headRefName,headRepositoryOwner,headRepository,baseRefName,updatedAt,mergeable,url --jq ...`.
- Fetched open non-draft PR refs explicitly: #295, #294, #289, #276, #274, #266, #237, #221, #154, #153, #152, #142, #137, #135, #94, #48, and #297.
- `git rev-list --left-right --count origin/main...HEAD` reports `0` behind and `368` ahead before the manifest commit.
- `git merge-base --is-ancestor` checks pass for carried open non-draft PR refs: #297, #295, #294, #289, #276, #274, #266, #152, and #142.
- Repeated direct merge probes from `easel/auto-integration` in `/tmp/luce-auto-cron-20260528-220855/` for #237, #221, #154, #153, #137, #135, #94, and #48. All still conflict.
- Delegation for #237: Claude Code was run through tmux but reached `--max-turns` without a usable report (`/tmp/luce237claude220908-report.txt`). Codex was then run through tmux on the same conflicted worktree and produced a read-only feasibility report (`/tmp/luce237codex221133-report.txt`) recommending staged selective port rather than direct merge resolution.

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Head | Current status | Probe result / next useful action |
|---:|---|---:|---|---|
| #237 | `feat/dflash-mtp-foundation` | `02c6a6c4` | blocked-needs-human / selective-port | Probe worktree `/tmp/luce-auto-cron-20260528-220855/pr-237-probe` has delete/update conflicts in old `dflash/scripts/server.py`, `dflash/src/common/backend_factory.cpp`, `dflash/src/server/server_main.cpp`; file-location conflicts for new `server/src/common/mtp_*`, `server/src/qwen35/qwen35_mtp*`, and `server/test/test_common_mtp_orchestrator.cpp`; and semantic conflicts in `server/CMakeLists.txt`, `backend_factory.h`, `step_graph.h`, qwen35 loader/graph/backend/dflash target, and `server/test/test_dflash.cpp`. Codex recommends a staged selective port: first `backend_factory.{h,cpp}` MTP source/config, then `server_main.cpp` CLI flags, CMake MTP sources/tests, common target API hidden-capture hooks, Qwen35 graph/loader/backend hooks, and finally new MTP files/tests. |
| #221 | `feat/mtp-prefix-warm-ghost` | `05502974` | blocked-needs-human / dependency | Probe worktree `/tmp/luce-auto-cron-20260528-220855/pr-221-probe` conflicts on old `dflash/` scripts/backend files, adds many benchmark artifacts, and conflicts in common target APIs plus Qwen35 graph/loader/backend/dflash target. Revisit after #237-equivalent MTP foundation; then port hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests to current `server/` layout. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | `2f4ede79` | blocked-needs-human / dependency | Probe worktree `/tmp/luce-auto-cron-20260528-220855/pr-154-probe` conflicts on old `dflash/CMakeLists.txt`, file-location moves for MTP docs/tests and `f16_convert.cu`, plus semantic conflicts in `internal.h`, `gguf_target_loader.cpp`, `qwen35_target_graph.cpp`, and `test_dflash.cpp`. Mine linear MTP decode semantics after current-layout Qwen35 MTP exists. |
| #153 | `xabicasa/dflash-mtp-integrated` | `e9b17cb1` | blocked-needs-human / dependency | Probe worktree `/tmp/luce-auto-cron-20260528-220855/pr-153-probe` conflicts on old CMake, moved MTP docs/tests and `f16_convert.cu`, and the same core Qwen35/internal target files. Mine loader/graph/cache/test ideas after current-layout Qwen35 MTP exists. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | `561b0ac1` | blocked-needs-human / selective-port | Probe worktree `/tmp/luce-auto-cron-20260528-220855/pr-135-probe` has semantic conflicts in `server/src/internal.h`, `server/src/qwen35/qwen35_target_graph.cpp`, and `server/test/test_dflash.cpp`. Design a current-layout multi-request scheduler in the qwen35 daemon/graph-builder layer instead of resurrecting the old monolithic scheduler mechanically. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | `297fc74e` | suggested-close/superseded | Probe worktree `/tmp/luce-auto-cron-20260528-220855/pr-137-probe` only conflicts on deleted old `dflash/CMakeLists.txt`. Ask author to close or retarget to current `server/CMakeLists.txt` if anything remains. |
| #94 | `feat/dflash-qwen36-swa-draft` | `d2f9c9dd` | suggested-close/superseded | Probe worktree `/tmp/luce-auto-cron-20260528-220855/pr-94-probe` conflicts in `server/src/draft/draft_graph.cpp`, `draft_safetensors_loader.cpp`, and `server/src/internal.h`; useful behavior appears absorbed. Ask author/maintainers whether any remaining old-layout tests should be reauthored before close. |
| #48 | `fix/consumer-blackwell-auto-detect` | `858b84b6` | suggested-close/superseded | Probe worktree `/tmp/luce-auto-cron-20260528-220855/pr-48-probe` only conflicts on deleted old `dflash/CMakeLists.txt`. Close or retarget to current `server/CMakeLists.txt` if still needed. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
ongoing dependency awareness: #304, #291, #290, #286, #285, #275, #249, and
#193. #286 is the draft PR for the current auto-integration snapshot. #304 is a
draft LLM auto context compaction PR observed in this run and excluded because
it is draft. #297 is no longer draft and is included above.

## Retained worktrees / logs

The conflicted probe worktrees were intentionally retained for manual follow-up
because safe cleanup would require resolving or discarding conflicted indexes:

- `/tmp/luce-auto-cron-20260528-220855/pr-237-probe`
- `/tmp/luce-auto-cron-20260528-220855/pr-221-probe`
- `/tmp/luce-auto-cron-20260528-220855/pr-154-probe`
- `/tmp/luce-auto-cron-20260528-220855/pr-153-probe`
- `/tmp/luce-auto-cron-20260528-220855/pr-137-probe`
- `/tmp/luce-auto-cron-20260528-220855/pr-135-probe`
- `/tmp/luce-auto-cron-20260528-220855/pr-94-probe`
- `/tmp/luce-auto-cron-20260528-220855/pr-48-probe`

Agent reports/logs retained:

- `/tmp/luce237claude220908-report.txt` (Claude Code max-turns without usable report)
- `/tmp/luce237codex221133-report.txt` (Codex read-only feasibility report)

## Notes

This run produced a manifest-only refresh on top of `a98ffb79`; no source stack
rewrite was needed because `origin/main`, `easel/auto-integration`, and all
carried mergeable non-draft PR heads were already current. The only non-draft
state change relevant to inclusion is #297 returning from draft to open; its
head `53dd1686` was already carried.
