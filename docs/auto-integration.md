# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-27T17:50:19-04:00
Current branch tip before this metadata refresh: `223e835` (`easel/auto-integration`).

## Included in the current stack

| PR | Head branch | State | Notes |
|---:|---|---|---|
| #288 | `fix/laguna-chat-template` | included | Current head `5e8136a` is an ancestor of `easel/auto-integration`. |
| #287 | `feat/gemma4-timings` | included | Current head `b3163f4` is an ancestor of `easel/auto-integration`. |
| #284 | `fix/draft-safetensors-rope-theta` | included | Current head `697198a` is an ancestor of `easel/auto-integration`. |
| #276 | `fix/qwen36-claude-code-tool-calling` | included | Current head `0e3c79a` is an ancestor of `easel/auto-integration`. |
| #274 | `feat/pflash-drafter-ee7` | included | Current head `5037b28` is an ancestor of `easel/auto-integration`. |
| #266 | `feat/harness-typed-adapters` | included | Current head `17525ea` is an ancestor of `easel/auto-integration`. |
| #152 | `main` | included | Current head `cf735be` is an ancestor of `easel/auto-integration`. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | included | Current head `f2fbf62` is an ancestor of `easel/auto-integration`. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / selectively included | Not an ancestor after the `dflash/` → `server/` migration, but its safetensors SWA config parsing and causal-mask support are present under `server/src/draft/` and `server/src/common/dflash_draft_graph.cpp`. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / selectively included | Not an ancestor after layout migration, but the daemon reset regression test exists as `server/test/test_daemon_reset_merge_resolution.py` and the stack carries follow-up daemon reset fixes. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / selectively included | The old `dflash/CMakeLists.txt` change is obsolete; `server/CMakeLists.txt` now resolves CUDA architectures explicitly, including Blackwell/GB10 handling. |
| #285 | `feat/lucebox-docker` | partially included / draft dependency | Integrated through `dd69a25`; draft branch has since advanced to `852d6fe`, outside the primary non-draft contributor target unless it becomes a required dependency. |

## Attempted this run

| PR | Outcome | Notes |
|---:|---|---|
| #221 | not integrated | Prior retained worktree `/tmp/luce-pr221-delegated-20260527-1727` has direct + Claude + Codex attempt evidence. This run revalidated that current head `0550297` still has 12 unmatched patch-id commits; it remains an architecture-aware selective-port task rather than a normal merge. |
| #237 | not integrated | Revalidated current head `02c6a6c` still has 13 unmatched patch-id commits. Prior direct/Claude/Codex attempts did not produce a coherent port to current `server/`. |

## Held / not yet included

| PR | Head branch | State | Why held |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | DIRTY | Valuable, but prior direct/Claude/Codex attempts did not produce a coherent buildable port to current `server/`. Needs selective architecture port. |
| #221 | `feat/mtp-prefix-warm-ghost` | DIRTY | Has direct + external-agent attempt evidence; needs architecture-aware selective port, not a normal merge. |
| #183 | `split/gemma4-11a-target-mtp-integration` | DIRTY | Older Gemma4 split chain; held pending dependency/supersession review against #237 and current Gemma4 backend. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | DIRTY | Older Gemma4 split chain; likely superseded by #183/#237 direction. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | DIRTY | Older Gemma4 split chain; likely superseded. |
| #180 | `split/gemma4-08-draft-loader-quant` | DIRTY | Older Gemma4 split chain; likely superseded. |
| #177 | `split/gemma4-06-kv-correctness` | DIRTY | Older Gemma4 split chain; may need selective correctness-port review. |
| #174 | `split/gemma4-14-small-vram-docs` | DIRTY | Title is docs-only, but current diff also contains inherited Gemma4 code because the branch stacks on older split work; not a safe standalone cherry-pick. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | DIRTY | Older dFlash MTP line stacked on #153; suggested superseded by #237 if that remains survivor. |
| #153 | `xabicasa/dflash-mtp-integrated` | DIRTY | Older dFlash MTP line; suggested superseded by #237. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | DIRTY | Older build/config line targeting deleted `dflash/CMakeLists.txt`; stale after layout migration. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | DIRTY | Older scheduler work; held for selective-port review. |
| #131 | `feature/gemma4-support` | DIRTY | Broad older Gemma4 support stack; overlaps later Gemma4 work and draft #193. |
| #39 | `feat/moe-35b-a3b` | DIRTY | Older MoE/draft work conflicts with newer stacks. |

## Draft / excluded

Draft PRs remain outside the primary contributor integration target: #286, #285, #278, #275, #273, #265, #249, #193, #75. #285 is carried only as an integration dependency and needs a deliberate refresh if its latest draft head becomes required.

## Validation run

No code integration was committed this run. Validation was limited to repository/auth/fetch status, PR classification, and feature-presence checks:

- `date -Is`
- `git status --porcelain=v1; git branch --show-current; git remote -v`
- `gh auth status`
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text`
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version`
- `git ls-remote --heads origin main; git ls-remote --heads easel auto-integration`
- `git fetch --prune origin`
- `git fetch --prune easel`
- `git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'` (reported the known submodule warning for `dflash/deps/Block-Sparse-Attention` at `bcddec6`)
- `git cherry HEAD origin/pr/<N>` for open non-draft heads
- `git grep` checks for SWA, daemon reset, and CUDA-architecture migration evidence

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at start and kept clean except for a final fast-forward to the metadata commit.
- Retained conflicted PR #221 worktree: `/tmp/luce-pr221-delegated-20260527-1727`.
- Retained prior metadata-update worktree: `/tmp/luce-auto-report-update-20260527-1727`.
- This metadata refresh worktree: `/tmp/luce-auto-report-20260527-175019`.
