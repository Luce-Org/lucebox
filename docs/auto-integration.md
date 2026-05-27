# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-27T19:19:38-04:00
Current branch tip before this metadata refresh: `0e2d839` (`auto-integration`, equal to `easel/auto-integration` before this metadata-only refresh).

## Included in the current stack

| PR | Head branch | State | Notes |
|---:|---|---|---|
| #288 | `fix/laguna-chat-template` | included | Merged as `9a392e7`; current head `5e8136a` remains an ancestor of `easel/auto-integration`. |
| #287 | `feat/gemma4-timings` | included | Merged as `bf7306e`; current head `b3163f4` remains an ancestor of `easel/auto-integration`. |
| #284 | `fix/draft-safetensors-rope-theta` | included | Current head `697198a` is an ancestor of `easel/auto-integration`. |
| #276 | `fix/qwen36-claude-code-tool-calling` | included | Current head `0e3c79a` is an ancestor of `easel/auto-integration`. |
| #274 | `feat/pflash-drafter-ee7` | included | Current head `5037b28` is an ancestor of `easel/auto-integration`. |
| #266 | `feat/harness-typed-adapters` | included | Current head `17525ea` is an ancestor of `easel/auto-integration`. |
| #152 | `main` | included | Current head `cf735be` is an ancestor of `easel/auto-integration`. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | included | Current head `f2fbf62` is an ancestor of `easel/auto-integration`. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / selectively included | Not an ancestor after the `dflash/` → `server/` migration, but its safetensors SWA config parsing and causal-mask support are present under `server/src/draft/` and `server/src/common/dflash_draft_graph.cpp`. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / selectively included | Not an ancestor after layout migration, but the daemon reset regression test exists as `server/test/test_daemon_reset_merge_resolution.py` and the stack carries follow-up daemon reset fixes. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / selectively included | The old `dflash/CMakeLists.txt` change is obsolete; `server/CMakeLists.txt` now resolves CUDA architectures explicitly, including Blackwell/GB10 handling. |
| #174 | `split/gemma4-14-small-vram-docs` | selectively included | Ported the documentation-only head commit `8b1caba` to current `server/README.md` as `6a54c27`, updating the CMake command for the current layout and omitting stale follow-up wording. Earlier commits on the branch are inherited Gemma4 code and remain outside this docs-only port. |
| #285 | `feat/lucebox-docker` | partially included / draft dependency | Integrated through `dd69a25`; draft branch has since advanced to `32961a1`, outside the primary non-draft contributor target unless it becomes a required dependency. |

## Attempted this run

| PR | Outcome | Notes |
|---:|---|---|
| #237 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr237-20260527-190917` still produced 23 unmerged paths across `server/CMakeLists.txt`, backend factory, MTP/StepGraph interfaces, Qwen35 loader/backend/graph files, tests, and deleted legacy `dflash/` server entry points. Claude was unavailable due session limit; Codex tmux session `luce-pr237-codex-1914` completed a read-only feasibility note at `/tmp/pr237-feasibility.txt`: a safe minimal merge is not feasible; lowest-risk path is a fresh manual port with MTP disabled by default and existing draft/pflash/remote-draft/thinking-budget behavior preserved. |
| #221 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr221-20260527-190917` produced 36 unmerged paths and many legacy `dflash/bench/results` artifacts. Codex tmux session `luce-pr221-1909` attempted a current-layout port (moved Qwen36 MTP pieces and edited namespace/backend paths) but left conflict markers/unmerged paths, so no changes were accepted into the stack. This remains dependent on a coherent #237-style MTP foundation port first. |
| #174 | previously partially integrated | A prior read-only delegated assessment confirmed the final docs-only commit was safe to port to `server/README.md`; committed as `6a54c27`. The older inherited Gemma4 commits on the branch were not cherry-picked. |
| #137 | previously not integrated | Manual cherry-pick probe in `/tmp/luce-auto-run-20260527-1815` produced a modify/delete conflict on deleted legacy `dflash/CMakeLists.txt`; the current `server/CMakeLists.txt` architecture handling makes the old standalone BSA/sm_89 patch stale. |
| #135 | previously not integrated | Manual cherry-pick probe conflicts in `server/src/internal.h`, `server/src/qwen35/qwen35_target_graph.cpp`, and `server/test/test_dflash.cpp`. Codex tmux review (`codex-pr135-1819`) concluded it is a selective architecture port, not a safe cherry-pick: the old daemon scheduler must be redesigned around current `HttpServer`/`ModelBackend`/Qwen35 cache APIs while preserving newer MoE, KV rotation, snapshot, `last_token_logits_only`, remote-draft, and tool-hint behavior. |

## Held / not yet included

| PR | Head branch | State | Why held |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | DIRTY | Valuable, but direct and delegated read-only reviews confirm it needs a selective architecture port to current `server/` rather than a normal merge. |
| #221 | `feat/mtp-prefix-warm-ghost` | DIRTY | Has direct + external-agent attempt evidence; needs architecture-aware selective port, not a normal merge. |
| #183 | `split/gemma4-11a-target-mtp-integration` | DIRTY | Older Gemma4 split chain; held pending dependency/supersession review against #237 and current Gemma4 backend. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | DIRTY | Older Gemma4 split chain; likely superseded by #183/#237 direction. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | DIRTY | Older Gemma4 split chain; likely superseded. |
| #180 | `split/gemma4-08-draft-loader-quant` | DIRTY | Older Gemma4 split chain; likely superseded. |
| #177 | `split/gemma4-06-kv-correctness` | DIRTY | Older Gemma4 split chain; may need selective correctness-port review. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | DIRTY | Older dFlash MTP line stacked on #153; suggested superseded by #237 if that remains survivor. |
| #153 | `xabicasa/dflash-mtp-integrated` | DIRTY | Older dFlash MTP line; suggested superseded by #237. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | blocked-needs-human / stale | Revalidated this run: legacy `dflash/CMakeLists.txt` only; current `server/CMakeLists.txt` supersedes it. Suggested close unless the author re-targets current server layout. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Revalidated this run with manual and Codex tmux review; concept is potentially useful but requires a current-tree scheduler redesign, not conflict resolution. |
| #131 | `feature/gemma4-support` | DIRTY | Broad older Gemma4 support stack; overlaps later Gemma4 work and draft #193. |
| #39 | `feat/moe-35b-a3b` | DIRTY | Older MoE/draft work conflicts with newer stacks. |

## Draft / excluded

Draft PRs remain outside the primary contributor integration target: #286, #285, #278, #275, #273, #265, #249, #193, #75. #285 is carried only as an integration dependency and needs a deliberate refresh if its latest draft head becomes required.

## Validation run

This run refreshed repository/auth/fetch status, re-enumerated open PRs, confirmed the currently included heads, and performed fresh worktree-based attempts for PR #237 and PR #221. Only this manifest changed in the final branch; no conflicted agent edits were accepted. Validation covered:

- `date -Is`
- `git status --porcelain=v1; git branch --show-current; git remote -v; git worktree list`
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status`
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text`
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version`
- `git fetch --prune origin`
- `git fetch --prune easel`
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...`
- `git merge-base --is-ancestor <PR head> HEAD` for open non-draft heads
- `git fetch origin pull/237/head:refs/remotes/origin/pr/237` and `git fetch origin pull/221/head:refs/remotes/origin/pr/221`
- Fresh isolated merge probes in `/tmp/luce-attempt-pr237-20260527-190917` and `/tmp/luce-attempt-pr221-20260527-190917`
- tmux-driven Codex feasibility/reconciliation attempts: `luce-pr237-codex-1914` and `luce-pr221-1909`
- `git status --short` in the primary checkout before and after the metadata refresh
- `git diff --check -- docs/auto-integration.md`

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at start; only this manifest was changed and committed.
- PR #237 fresh attempt retained conflicted worktree `/tmp/luce-attempt-pr237-20260527-190917`; Codex wrote `/tmp/pr237-feasibility.txt`.
- PR #221 fresh attempt retained conflicted worktree `/tmp/luce-attempt-pr221-20260527-190917`; Codex attempted edits but still left 36 unmerged paths and no accepted stack changes.
- Retained conflicted PR #221 worktree from prior run: `/tmp/luce-pr221-delegated-20260527-1727`.
- Retained prior metadata-update worktree from prior run: `/tmp/luce-auto-report-update-20260527-1727`.
- Retained prior metadata refresh worktree from prior run: `/tmp/luce-auto-report-20260527-175019`.
