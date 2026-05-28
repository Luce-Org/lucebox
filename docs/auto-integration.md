# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T14:49:06-04:00
Current base: `origin/main` `3ba525e0`
Current integration tip before this refresh: `easel/auto-integration` `72355a11`
Refreshed stack tip prepared in this run: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was clean at the start of this unattended run, but the local
primary branch was still behind/diverged from the fetched `easel/auto-integration`
tip; all reconciliation and validation work therefore happened in isolated
worktree `/tmp/luce-auto-cron-20260528-144958`. `origin/main` did not advance
beyond `3ba525e0`, and every currently open non-draft contributor PR that can be
directly integrated remains present in the stack. This refresh is manifest-only:
it records the latest PR inventory, rerun direct-conflict probes for the remaining
non-ancestor PRs, and the improved CI state after the prior integration-only CMake
fix. The GitHub `Build (cmake + uv sync --extra megakernel)` and `uv workspace`
checks now pass for PR #286 at tip `72355a11`; `cuda12` was still pending after a
six-minute watch window.

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
| upstream sync | checked / no-op | `origin/main` remains `3ba525e0`; it is already an ancestor of `easel/auto-integration` `72355a11`, so no upstream merge was required this run. |
| open non-draft inventory | checked | Open non-draft PRs were #295, #294, #292, #289, #276, #274, #266, #237, #221, #154, #153, #152, #142, #137, #135, #94, and #48. Ancestor checks passed for #295, #294, #292, #289, #276, #274, #266, #152, and #142; #296, #284, #278, and #265 are included through `origin/main`. |
| integration-only CI fix | improved / awaiting one check | For PR #286 at `72355a11`, `Build (cmake + uv sync --extra megakernel)` now passes and `uv workspace` passes. `cuda12` was still pending after `gh pr checks 286 --watch --interval 30` ran for six minutes; no new failure was observed. |
| remaining non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated probes attempted `--no-commit --no-ff` merges for #237, #221, #154, #153, #137, #135, #94, and #48 against stack tip `72355a11` in probe branch `auto-integration-probe-20260528-144958`. Every direct probe conflicted and was aborted in the isolated probe worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-144958.txt`. |
| #237 manual/delegated status | still blocked-needs-human | No new #237 head since the prior deep manual/delegated recheck (`02c6a6c4`). This run's direct probe again showed broad conflicts across old `dflash/` paths, common MTP interfaces, Qwen35 graph/backend files, CMake, daemon/server wiring, and tests. Prior Claude/Codex tmux attempts did not produce a trusted ready-to-apply resolution, and the required next step remains a deliberate current-layout MTP port rather than conflict-marker resolution. |

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
dependency awareness: #291, #290, #286, #285, #275, #249, #193, and #75. #286 is
the current draft auto-integration snapshot. #285 remains partially carried only
as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T14:49:06-04:00 at preflight.
- Primary checkout `git status --short` was clean before work began; `git branch --show-current` reported `auto-integration`.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel` with repo/workflow scopes.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --help` succeeded (`codex help ok`).
- `git fetch --prune origin` and `git fetch --prune easel` completed separately; targeted fetches recreated current open non-draft contributor PR refs.
- `origin/main`, fetched `easel/auto-integration`, local primary `HEAD`, and prepared worktree `HEAD` were checked as `3ba525e0`, `72355a11`, `f13a7459`, and `72355a11` respectively before this manifest commit. The primary local branch is still behind/diverged from fetched `easel/auto-integration`, so it was not used for reconciliation.
- Worktree `/tmp/luce-auto-cron-20260528-144958` confirmed `origin/main` is already an ancestor of `HEAD`; no upstream merge was needed.
- Isolated probe worktree `/tmp/luce-probe-20260528-144958` reran direct conflict probes for all remaining non-ancestor non-draft PRs.
- Ancestor checks passed for included contributor PR refs #295, #294, #292, #289, #276, #274, #266, #152, and #142; #296, #284, #278, and #265 are included through upstream main.
- `gh pr checks 286 --repo Luce-Org/lucebox-hub` reported `Build (cmake + uv sync --extra megakernel)` pass and `uv workspace` pass for `72355a11`; `cuda12` remained pending. A six-minute `gh pr checks 286 --watch --interval 30` timed out with `cuda12` still pending and the other two checks still passing.
- `git diff --check` passed before this manifest refresh.
- Targeted conflict-marker scan over `docs/auto-integration.md` found no merge markers before commit.

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at preflight but local `HEAD` (`f13a7459`) still lags/diverges from fetched `easel/auto-integration` (`72355a11`). The pushed branch from this run should be treated as source of truth; primary fast-forward is deferred because the local branch divergence needs supervised cleanup.
- Retained integration worktree `/tmp/luce-auto-cron-20260528-144958`, probe worktree `/tmp/luce-probe-20260528-144958`, and probe log `/tmp/luce-merge-probes-20260528-144958.txt`.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
