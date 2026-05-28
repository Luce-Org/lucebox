# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T16:09:55-04:00
Current base: `origin/main` `3ba525e0`
Current integration tip before this refresh: `easel/auto-integration` `f7d4493a`
Refreshed stack tip prepared in this run: this commit

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was clean at preflight, though its local branch is behind
and divergent from the pushed `easel/auto-integration` history; this run used an
isolated worktree from the fetched easel tip. `origin/main` was already contained
in the integration branch. One new non-draft contributor PR, #298, appeared since
the previous run and was merged cleanly on top of the current stack.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #298 | `fix/gemma4-destructor-link` | `32ba96e` | included this run | Cleanly merged. Marks `Gemma4LayerSplitAdapter` destructor `noexcept` and catches shutdown exceptions to avoid newer libstdc++ termination-helper link dependency. |
| #295 | `fix-layer-split-sampling` | `a9aedf7` | included | Support sampled requests in target layer split. |
| #294 | `feat/server-passthrough-proxy` | `0883c2e` | included | Server passthrough proxy wiring, piecewise keep-ratio curve, query survival checks, and unit coverage. |
| #292 | `feat-backend-ipc-payload-pipe-open` | `90bc52f` | included | Backend IPC payload-pipe support for remote draft feature/noise payloads, with integration-only pipe-drain hardening and shared feature-slice storage helper. |
| #289 | `pipeline_moe` | `0ffab8a` | included | Pipelined hybrid Qwen35 MoE decode update is carried. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `5e861b4` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path. |
| #274 | `feat/pflash-drafter-ee7` | `e64a2b8` | included | Adaptive pFlash compression docs/config, transitive anchor default, backup-suffix ignores, EE7/drafter updates, and Qwen3 closed-`<think>` Jinja prefill docs/tests are carried. |
| #266 | `feat/harness-typed-adapters` | `17525ea` | included | Typed harness adapters and format-aware session-inject proxy. |
| #152 | `main` | `cf735be` | included | Gemma 4 RTX 4090 backend helpers already carried. |
| #142 | `xabicasa/dflash-safetensors-draft-fp16` | `f2fbf62` | included | FP16 safetensors drafter support already carried. |
| #94 | `feat/dflash-qwen36-swa-draft` | `d2f9c9d` | absorbed / selectively included | Current draft/common code carries safetensors SWA config parsing and causal-mask behavior; remaining branch edits target old paths. |
| #48 | `fix/consumer-blackwell-auto-detect` | `858b84b` | superseded / selectively included | Current `server/CMakeLists.txt` owns CUDA architecture resolution and Blackwell/GB10 handling. |

## Attempted this run

| PR | Outcome | Notes |
|---:|---|---|
| upstream sync | checked | `origin/main` `3ba525e0` was already included in `easel/auto-integration` `f7d4493a`; no upstream merge commit was needed. |
| #298 | integrated | Clean merge in `/tmp/luce-auto-cron-20260528-161028`; reviewed the two-file destructor diff and kept PR changes unchanged. |
| #237 | still blocked-needs-human / selective-port | Fresh direct probe conflicts across deleted old `dflash/` scripts/server paths, `server/CMakeLists.txt`, common backend factory/model/step graph files, Qwen35 graph/backend/target files, new MTP common/Qwen35 files, and tests. Needs a current-layout MTP foundation port, not conflict-marker resolution. |
| #221 | still blocked-needs-human / dependency | Fresh direct probe again conflicts broadly and also tries to add old benchmark result artifacts plus old `dflash/` hook/server/backend paths. Depends on #237-equivalent MTP foundation. |
| #154 | selective-port | Fresh direct probe conflicts on retired `dflash/CMakeLists.txt`, MTP docs/file-location moves, `internal.h`, Qwen35 target loader/graph, and MTP tests. Portable only after current-layout Qwen35 MTP design. |
| #153 | blocked-needs-human / selective-port | Fresh direct probe has similar old-layout MTP loader/graph/test conflicts. |
| #137 | stale / suggested close | Fresh probe only conflicts on deleted legacy `dflash/CMakeLists.txt`; retarget current `server/CMakeLists.txt` or close. |
| #135 | blocked-needs-human / selective-port | Fresh probe conflicts in `internal.h`, Qwen35 target graph, and old dFlash tests; still requires scheduler/API redesign against current server layout. |
| #94 | absorbed / verify-only | Fresh probe conflicts in current draft graph/safetensors loader and old internal layout; useful SWA behavior is already represented in current code. |
| #48 | superseded / suggested close | Fresh probe conflicts only on deleted legacy `dflash/CMakeLists.txt`; current CMake supersedes it. |

Consolidated probe log: `/tmp/luce-merge-probes-20260528-161028.txt`.
Probe worktree retained: `/tmp/luce-probe-20260528-161028`.

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Port common MTP interfaces/runners and tests under `server/`, add MTP fields to `BackendArgs`/`ModelBackend`, reauthor Qwen35 hidden/logit/rollback capture against current `StepGraph`/MoE changes, then wire `qwen35_mtp*`, CLI flags, CMake, and focused tests. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Requires #237/equivalent MTP foundation, then a current-layout feature port for hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | selective-port | Linear native MTP decode semantics are portable after a current-layout Qwen35 MTP design. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Feasible but requires loader/graph/cache/MoE design work, not direct merge. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | stale / suggested close | Edits only deleted legacy `dflash/CMakeLists.txt`; close or ask author to retarget current `server/CMakeLists.txt`. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Port `n_seqs` primitives/test helpers first, then design multi-slot `Qwen35Backend` state and scheduler API. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / verify-only | Remaining conflicts are old-layout draft/common paths; current code carries useful SWA pieces. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / suggested close | Current CMake supersedes the old `dflash/CMakeLists.txt` change. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #297, #291, #290, #286, #285, #275, #249, #193, and #75.
#286 is the current draft auto-integration snapshot. #285 remains partially
carried only as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T16:09:55-04:00 at preflight.
- Primary checkout `git status --short` was clean before work began.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel` with repo/workflow scopes.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` succeeded with `codex-cli 0.130.0`.
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft contributor PR refs.
- Isolated integration worktree `/tmp/luce-auto-cron-20260528-161028` was created from fetched `easel/auto-integration`.
- `git merge --no-ff --no-edit origin/main` reported already up to date.
- `git merge --no-ff --no-edit origin/pr/298` completed cleanly.
- `git diff --check easel/auto-integration..HEAD` passed before this manifest refresh.
- Local `cmake -S server -B /tmp/luce-cmake-20260528-161028 -DLUCE_BUILD_TESTS=ON` failed before project configuration due to the known WSL CUDA compiler-id/toolchain blocker: `ptxas fatal : Value 'sm_52' is not defined for option 'gpu-name'`.
- GitHub checks for PR #286 were inspected before this push: `uv workspace (lock + sync + import smoke)` passed; `cuda12` and `Build (cmake + uv sync --extra megakernel)` were pending on the prior draft auto-integration tip.

## Notes

- Primary checkout `/home/erik/Projects/luce2` was left untouched and clean. It is still locally divergent from fetched `easel/auto-integration` (`git rev-list --left-right --count HEAD...easel/auto-integration` was `2 35` at preflight) and should not be treated as the source of truth until supervised reconciliation.
- Retained worktrees: `/tmp/luce-auto-cron-20260528-161028` and `/tmp/luce-probe-20260528-161028`.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
