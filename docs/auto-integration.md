# Auto-integration manifest

Repository: `Luce-Org/lucebox-hub`
Integration branch: `auto-integration`
Writable remote: `easel`
Upstream remote: `origin` / `Luce-Org`
Last refresh: 2026-05-28T13:17:57-04:00
Current base: `origin/main` `315a9bdb`
Current integration tip before this refresh: `easel/auto-integration` `161014ae`
Refreshed stack tip prepared in this run: `11f7424b`

This branch is maintained as a reproducible patch stack over `origin/main`.
The primary checkout was clean at the start of this unattended run. Since the
prior refresh, `origin/main` advanced by the merge commit for PR #284, so this
run first synced that upstream change into the integration branch. All previously
integrated non-draft contributor PR refs remained ancestors of the stack, and no
new non-draft contributor PR head required a fresh integration merge. After the
first manifest-only push, CI exposed integration-only build breakage; this
refresh also removes duplicate CMake test target registrations and installs
libcurl development headers for the new server target in CI/Docker builds.

## Included in the current stack

| PR | Head branch | Head | State | Notes |
|---:|---|---:|---|---|
| #294 | `feat/server-passthrough-proxy` | `0883c2e` | included | Adds server passthrough proxy wiring, piecewise keep-ratio curve, query survival checks, and unit coverage. Manual conflict resolution from the previous refresh preserved the stack's canonical Qwen3 drafter loader while adopting #294's qwen3.5/qwen35 path-based drafter arch inference for compatibility overloads. |
| #292 | `feat-backend-ipc-payload-pipe-open` | `90bc52f` | included | Adds backend IPC payload-pipe support for remote draft feature/noise payloads, with integration-only pipe-drain hardening and shared feature-slice storage helper. |
| #289 | `pipeline_moe` | `0ffab8a` | included | Pipelined hybrid Qwen35 MoE decode update is carried. The inaccessible submodule pointer from earlier PR history remains excluded. |
| #284 | `fix/draft-safetensors-rope-theta` | `63bba30` | included | Reads and validates `rope_theta` from draft safetensors `config.json`. |
| #278 | `fix-pflash-drafter-backend-precision-submit` | upstream | included through upstream | `origin/main` now contains this PR via merge commit `6a6b0081`; the integration stack was rebased over it. |
| #276 | `fix/qwen36-claude-code-tool-calling` | `0e3c79a` | included | Qwen3.6-27B tool-calling fix for Claude-code Anthropic path. |
| #274 | `feat/pflash-drafter-ee7` | `2c19f66` | included | Adaptive pFlash compression docs/config, transitive anchor default, backup-suffix ignores, EE7/drafter updates, and Qwen3 closed-`<think>` Jinja prefill docs/tests are carried. |
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
| upstream sync | checked | `origin/main` remains `6a6b0081`; `easel/auto-integration` and the primary checkout both started at `575034db`, with no upstream-base merge needed in this run. |
| current integrated PRs | checked | `git merge-base --is-ancestor origin/pr/<n> HEAD` shows #294, #292, #289, #284, #276, #274, #266, #152, and #142 are ancestors of the refreshed stack. #278 and #265 are included through `origin/main`. |
| remaining non-ancestor non-draft PRs | direct merge probes still conflicted | Fresh isolated probes attempted `--no-commit --no-ff` merges for #237, #221, #183, #182, #181, #180, #177, #174, #154, #153, #137, #135, #131, #94, #62, #48, and #39 against current stack tip `575034db`. Every direct probe conflicted and was aborted in the isolated probe worktree. Consolidated output: `/tmp/luce-merge-probes-20260528-123547.txt`. |
| #237 manual/delegated status | still blocked-needs-human | No new #237 head since the prior deep manual/delegated recheck. This run's direct probe again showed broad conflicts across old `dflash/` paths, common MTP interfaces, Qwen35 graph/backend files, CMake, daemon/server wiring, and tests. Prior Claude/Codex tmux attempts did not produce a trusted ready-to-apply resolution, and the required next step remains a deliberate current-layout MTP port rather than conflict-marker resolution. |
| integration-only CI fix | fixed | GitHub checks for the manifest-only push `9b7972a1` failed during CMake configure because several tests were registered twice in `server/CMakeLists.txt` and the new `dflash_server` target required libcurl development headers on the hosted runner. This refresh removes the duplicate target block and adds an explicit CI `apt-get install libcurl4-openssl-dev` step plus Docker builder headers. A follow-up attempt showed `Jimver/cuda-toolkit` suffixes `non-cuda-sub-packages`, so libcurl was moved out of that action's package list into the explicit apt step. |

## Pending / blocked-needs-human / selective-port candidates

| PR | Head branch | Current status | Next useful action |
|---:|---|---|---|
| #237 | `feat/dflash-mtp-foundation` | blocked-needs-human / selective-port | Direct merge still conflicts across current server/common/qwen35 files and old `dflash/` paths. Port common MTP interfaces/runners and tests under `server/`, add MTP fields to `BackendArgs`/`ModelBackend`, reauthor Qwen35 hidden/logit/rollback capture against current `StepGraph`/MoE changes, then wire `qwen35_mtp*`, CLI flags, CMake, and focused tests. |
| #221 | `feat/mtp-prefix-warm-ghost` | blocked-needs-human / dependency | Requires #237/equivalent MTP foundation, then a current-layout feature port for hidden-state/speculator hooks, backend dispatcher, daemon wiring, and WARM tests. |
| #183 | `split/gemma4-11a-target-mtp-integration` | blocked-needs-human / selective-port | Direct merge is unsafe because it targets retired `dflash/` plus flat Gemma4 paths. Port KV fix, `h_prev`/asymmetric KV hook, MTP loader/graph, hardening, and tests into current `server/src/gemma4/*`. |
| #182 | `split/gemma4-10-mtp-loader-step-graph` | blocked-needs-human / selective-port | Port MTP declarations/cache fields, assistant loader pieces, a current `gemma4_mtp_graph.cpp`, CMake wiring, and tests into current layout. |
| #181 | `split/gemma4-09-dflash-draft-runtime` | blocked-needs-human / selective-port | Mine quantizer metadata/error-reporting/runtime metadata concepts and reauthor tests against current Gemma4/draft plumbing. |
| #180 | `split/gemma4-08-draft-loader-quant` | blocked-needs-human / selective-port | Reconcile quantizer metadata with current `draft_gguf_loader.cpp`, validate capture/target-layer IDs, and reauthor smoke tests. |
| #177 | `split/gemma4-06-kv-correctness` | partially integrated / selective-port | SWA split-on-wrap and larger SWA ring allocation are ported. Remaining work: TQ3/large-head KV alignment/type resolution, FA rotation path, and adapted `test_gemma4_kv_tq3.cpp`. |
| #154 | `xabicasa/dflash-mtp-speculative-loop` | selective-port | Linear native MTP decode semantics are portable after a current-layout Qwen35 MTP design. |
| #153 | `xabicasa/dflash-mtp-integrated` | blocked-needs-human / selective-port | Feasible but requires loader/graph/cache/MoE design work, not conflict-marker resolution. |
| #137 | `xabicasa/dflash-build-cmake-sm89-bsa` | stale / suggested close | Edits only deleted legacy `dflash/CMakeLists.txt`; close or ask author to retarget current `server/CMakeLists.txt`. |
| #135 | `xabicasa/dflash-multi-request-scheduler-batched-target-step` | blocked-needs-human / selective-port | Port `n_seqs` primitives/test helpers first, then design multi-slot `Qwen35Backend` state and scheduler API. |
| #131 | `feature/gemma4-support` | selective-port | Broad Gemma4 support can be mined, but direct merge would resurrect old `dflash27b`/`dflash/` layout. |
| #94 | `feat/dflash-qwen36-swa-draft` | absorbed / verify-only | Remaining conflicts are old-layout draft/common paths; current code carries the useful SWA pieces. |
| #62 | `fix/issue-55-stable-kv-pad` | absorbed / verify-only | Remaining conflicts are old-layout tests; current code carries reset behavior. |
| #48 | `fix/consumer-blackwell-auto-detect` | superseded / suggested close | Current CMake supersedes the old `dflash/CMakeLists.txt` change. |
| #39 | `feat/moe-35b-a3b` | partially integrated / mostly superseded | Ask maintainers whether any remaining old MoE smoke coverage should be reauthored against current qwen35moe APIs before closing. |

## Draft / excluded

Draft PRs remain outside the primary non-draft integration target except for
dependency awareness: #291, #290, #286, #285, #275, #249, #193, and #75. #286 is
the current draft auto-integration snapshot. #285 remains partially carried only
as an integration dependency.

## Validation run

This run performed:

- `date -Is` -> 2026-05-28T12:34:53-04:00 at preflight.
- Primary checkout `git status --short --branch` was clean before work began and reported `## auto-integration...easel/auto-integration`.
- `git branch --show-current` reported `auto-integration` in the primary checkout.
- `git remote -v` verified `origin=https://github.com/Luce-Org/lucebox-hub` and `easel=https://github.com/easel/lucebox-hub`.
- `GH_CONFIG_DIR=/home/erik/.config/gh XDG_CONFIG_HOME=/home/erik/.config HOME=/home/erik gh auth status` succeeded for account `easel` with repo/workflow scopes.
- `HOME=/home/erik /home/erik/.local/bin/claude auth status --text` succeeded for the Claude Team account.
- `HOME=/home/erik /home/linuxbrew/.linuxbrew/bin/codex --version` succeeded with `codex-cli 0.130.0`.
- `git fetch --prune origin` and `git fetch --prune easel` completed; targeted fetches recreated current open non-draft contributor PR refs.
- `origin/main`, `easel/auto-integration`, and local `HEAD` were checked as `6a6b0081`, `575034db`, and `575034db` respectively; `git rev-list --left-right --count HEAD...easel/auto-integration` returned `0 0`.
- Isolated probe worktree `/tmp/luce-probe-20260528-123547` reran direct conflict probes for all remaining non-ancestor non-draft PRs.
- Ancestor checks passed for included contributor PR refs #294, #292, #289, #284, #276, #274, #266, #152, and #142; #278 and #265 are included through upstream main.
- `git diff --check` passed for this manifest refresh and the integration-only CI fix.
- Targeted conflict-marker scan over `docs/auto-integration.md` found no merge markers.
- GitHub checks on intermediate commit `9b7972a1` were inspected with `gh pr checks 286 --watch` and `gh run view 26588281995 --log-failed`; `uv workspace` passed, while CMake configure failed on duplicate test targets and missing CURL dev headers. Follow-up commit `c967d8e9` was inspected with `gh run view 26588595195 --log-failed`; it reached the CUDA setup action and failed because `Jimver/cuda-toolkit` transformed `libcurl4-openssl-dev` into a non-existent CUDA-suffixed package, so the final fix installs libcurl via a separate apt step.
- Local `cmake -S server -B /tmp/luce-cmake-fix-124202 ...` could not reach project configure due to the known local WSL CUDA compiler-id blocker (`ptxas fatal: Value 'sm_52' is not defined`), so CI is the authoritative CMake verifier for this fix.

## Notes

- Primary checkout `/home/erik/Projects/luce2` was clean at preflight and matched fetched `easel/auto-integration` (`575034db`).
- Retained probe worktree `/tmp/luce-probe-20260528-123547` plus direct-probe log `/tmp/luce-merge-probes-20260528-123547.txt`.
- Prior retained worktrees, probe logs, agent reports, and configure directories remain as listed in earlier manifest revisions; cleanup is separate maintenance.
