# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-27T20:20:41-04:00 through 2026-05-27T20:27:00-04:00
Current stack tip before this metadata refresh: `7f2973c` (already aligned with `origin/main` `3474edf` and `easel/auto-integration` `7f2973c`; this run is a metadata/triage refresh after additional worktree probes).

## Included in the current stack

| PR | Head branch | State | Notes |
|---:|---|---|---|
| #284 | `fix/draft-safetensors-rope-theta` | included | Current head `697198a` is an ancestor of the refreshed stack. |
| #276 | `fix/qwen36-claude-code-tool-calling` | included | Current head `0e3c79a` is an ancestor of the refreshed stack. |
| #274 | `feat/pflash-drafter-ee7` | included | Current head `5037b28` is an ancestor of the refreshed stack. |
| #266 | `feat/harness-typed-adapters` | included | Current head `17525ea` is an ancestor of the refreshed stack. |
| #152 | `main` | included | Current head `cf735be` is an ancestor of the refreshed stack. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | included | Current head `f2fbf62` is an ancestor of the refreshed stack. |
| #174 | `split/gemma4-14-small-vram-docs` | selectively included | The docs-only VMM guidance remains ported to current `server/README.md` after the upstream README refresh. The older inherited Gemma4 commits on this branch are not cherry-picked. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / selectively included | Not an ancestor after the `dflash/` → `server/` migration, but its safetensors SWA config parsing and causal-mask support remain present under current draft/common code. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / selectively included | Not an ancestor after layout migration, but the daemon reset regression test and follow-up daemon reset fixes remain carried. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / selectively included | The old `dflash/CMakeLists.txt` change is obsolete; current `server/CMakeLists.txt` explicitly resolves CUDA architectures and carries Blackwell/GB10 handling. |
| #285 | `feat/lucebox-docker` | partially included / draft dependency | Draft PR, not part of the non-draft target, but an earlier head is carried as the Docker/CLI/bench integration dependency. Latest draft head `32961a1` is not required by any non-draft PR this run. |

## Attempted this run

| PR | Outcome | Notes |
|---:|---|---|
| #177 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr177-20260527-2020` produced conflicts in deleted legacy `dflash/CMakeLists.txt`, moved Gemma4 header/source/test paths, `server/src/errors.cpp`, and `server/src/internal.h`. A tmux-driven Claude assessment session was launched (`luce-pr177-claude-2020`) but exited without producing a report; the raw conflict shape still confirms this older Gemma4 KV-correctness split is not mechanically mergeable into the current `server/src/gemma4` architecture. |
| #153 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr153-20260527-2020` produced conflicts in legacy `dflash/CMakeLists.txt`, current `server/src/internal.h`, `server/src/qwen35/gguf_target_loader.cpp`, `server/src/qwen35/qwen35_target_graph.cpp`, and path-mapped MTP docs/tests. Codex tmux session `luce-pr153-codex-2020` wrote `/tmp/pr153-feasibility-20260527-2020.txt`: selective port is feasible, but not as conflict-marker resolution; current-layout loader/graph/token embedding/MoE/cache semantics need deliberate design. |
| #39 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr39-20260527-2020` produced broad conflicts across deleted legacy `dflash/*` files plus current draft graph/safetensors loader and MoE smoke tests. A tmux-driven Claude assessment session was launched (`luce-pr39-claude-2020`) but exited without producing a report; the branch remains an older MoE/DDTree stack that needs a survivorship review against current qwen35moe support. |
| #183 | not integrated | Fresh worktree merge probe in `/tmp/luce-attempt-pr183-20260527-1959` produced unmerged paths across deleted legacy `dflash/CMakeLists.txt`, moved Gemma4 source/header files, `server/src/errors.cpp`, and `server/src/internal.h`. Codex tmux session `luce-pr183-codex-2003` wrote `/tmp/pr183-feasibility-20260527-1959.txt`: safe normal resolution is not feasible; PR #183 is an older Gemma4 MTP split-chain head whose direct semantic change needs a future current-layout MTP foundation port, likely after #237 or a replacement common MTP design. |
| upstream sync | unchanged | `origin/main` remained at `3474edf`; `easel/auto-integration` was already based on that tip at the start of this run. |

## Previously attempted / still not integrated

| PR | Head branch | State | Why held |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Prior fresh worktree and Codex attempts found broad conflicts in server CMake/backend/MTP/Qwen35 graph/test code. Valuable, but needs a deliberate current-layout MTP foundation port rather than conflict-marker resolution. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Prior direct + Codex attempts left many unmerged paths and legacy artifacts; depends on a coherent #237-style MTP foundation first. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / superseded-by-current-architecture | Revalidated in the previous run. Its direct `h_prev` assertion has no current target location until Gemma4 MTP runtime is reintroduced in the current `server/src/gemma4` architecture. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | pending / likely superseded | Older Gemma4 split chain; should be assessed only after the MTP foundation decision. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | pending / likely superseded | Older Gemma4 split chain; overlaps current Gemma4 DFlash backend. |
| #180 | `split/gemma4-08-draft-loader-quant` | pending / likely superseded | Older Gemma4 split chain; overlaps current loader/backend work. |
| #177 | `split/gemma4-06-kv-correctness` | attempted / not mechanically mergeable | Revalidated this run in `/tmp/luce-attempt-pr177-20260527-2020`; conflicts are on the old Gemma4 split-chain/current-layout boundary. Needs selective correctness review, not a branch merge. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | pending / likely superseded | Older dFlash MTP line stacked on #153; likely superseded by #237 or replacement MTP foundation. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Revalidated this run in `/tmp/luce-attempt-pr153-20260527-2020`; Codex report says a current-layout Qwen35 MTP port is feasible but needs loader/graph/cache/MoE design rather than a mechanical merge. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | blocked-needs-human / stale | Legacy `dflash/CMakeLists.txt` only; current `server/CMakeLists.txt` supersedes it. Suggested close unless re-targeted to current server layout. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Prior manual and Codex review concluded the concept requires a current-tree scheduler redesign around `HttpServer`/`ModelBackend`/Qwen35 cache APIs. |
| #131 | `feature/gemma4-support` | pending / likely superseded | Broad older Gemma4 support stack; overlaps later current Gemma4 work and draft #193. |
| #39 | `feat/moe-35b-a3b` | attempted / likely superseded | Revalidated this run in `/tmp/luce-attempt-pr39-20260527-2020`; direct merge conflicts with current qwen35moe/draft graph work. Needs a MoE/DDTree survivorship review rather than a branch merge. |

## Draft / excluded

Draft PRs remain outside the primary contributor integration target: #286, #285, #278, #275, #273, #265, #249, #193, #75. #285 remains partially carried only as an integration dependency.

## Suggested close / author action

- #137: close or ask author to re-target current `server/CMakeLists.txt`.
- #48: close as superseded by current CUDA architecture/Blackwell handling unless the author has a specific missing Blackwell case.
- #183/#182/#181/#180/#177/#154/#153/#131/#39: do not close solely as conflicts yet; ask authors or maintainers to identify which current-layout Gemma4/MTP pieces are still intended after the newer `server/src/gemma4` architecture and #237 direction.

## Validation run

This run performed:

- `date -Is`
- `git status --short`, `git remote -v`, branch and revision checks in the primary checkout
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status`
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text`
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version`
- `git fetch --prune origin`
- `git fetch --prune easel`
- `gh pr list --repo Luce-Org/lucebox-hub --state open --limit 200 --json ... --jq ...`
- Targeted fetches for all open non-draft PR refs (`origin/pr/<n>`)
- `git cherry easel/auto-integration origin/pr/<n>` and `git merge-base --is-ancestor origin/pr/<n> easel/auto-integration` classification checks
- Isolated metadata worktree `/tmp/luce-auto-cron-20260527-2020`
- Fresh isolated PR #177 merge probe in `/tmp/luce-attempt-pr177-20260527-2020`
- Fresh isolated PR #153 merge probe in `/tmp/luce-attempt-pr153-20260527-2020`
- Fresh isolated PR #39 merge probe in `/tmp/luce-attempt-pr39-20260527-2020`
- tmux-driven Claude assessment sessions `luce-pr177-claude-2020` and `luce-pr39-claude-2020` (both exited before producing report files)
- tmux-driven Codex assessment session `luce-pr153-codex-2020`, producing `/tmp/pr153-feasibility-20260527-2020.txt`
- `git diff --check -- docs/auto-integration.md` before metadata commit (clean)
- `git diff --check origin/main..HEAD` after metadata commit (reported two pre-existing blank-line-at-EOF warnings in `luce-bench/src/lucebench/fixtures/forge_eval/scenarios/_model_quality.py` and `_stateful_model_quality.py`; not introduced by this run)
- `uv run --directory luce-bench --with pytest pytest -q tests/test_runner.py tests/test_smoke_area.py` (passed in the previous refresh; not rerun because this run changed only `docs/auto-integration.md`)

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at start and was not edited directly during reconciliation.
- Retained worktree `/tmp/luce-auto-cron-20260527-2020` for this metadata refresh until pushed/verified.
- Retained conflicted worktree `/tmp/luce-attempt-pr177-20260527-2020` for audit; it contains unmerged PR #177 probe state.
- Retained conflicted worktree `/tmp/luce-attempt-pr153-20260527-2020` for audit; it contains unmerged PR #153 probe state and corresponds to `/tmp/pr153-feasibility-20260527-2020.txt`.
- Retained conflicted worktree `/tmp/luce-attempt-pr39-20260527-2020` for audit; it contains unmerged PR #39 probe state.
- Retained prior conflicted worktrees from earlier runs as previously reported; cleanup would be separate maintenance.
