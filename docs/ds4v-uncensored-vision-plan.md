# DS4V uncensored vision GGUF for Strix Halo plus 7900XT plan

Build an uncensored DeepSeek V4 Flash Vision GGUF that runs on the operator pair.
Start from the OrcaRouter abliterated parent with vision intact.
Quant with the prometheusAIR imatrix recipe for a 128 GiB plus 24 GiB budget.
Serve with asymmetric expert parallelism per the Lucebox report.
PR ids in order. DS4V-1 then DS4V-2 then DS4V-3.
Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.
Style rules. i dont want any abstract metaphors. Write like hemingway.

## How to read this

One box is one unit of work. Every box names the evidence that checks it. A nested box is a substep of the box above it. Check a box only when its evidence exists. A file. A log line. A test run. Or a SHA.
The program runs `playbooks/autopilot-stack.md`. The root builds the chain. The operator lands it with her own clicks.
Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.
Style rules. i dont want any abstract metaphors. Write like hemingway.

## Program checklist

### Arm the program

- [ ] State the protocol and this plan to the operator, then stop. Start execution only on her explicit go.
- [ ] On her go, adopt the run objective with this exact text. "`docs/ds4v-uncensored-vision-plan.md`, DS4V-1 then DS4V-2 then DS4V-3, Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked., the root builds the chain and the operator lands it, done when every box is checked with its evidence."
- [ ] Read these from trunk at program start. Re-read them at every tick.
  - [ ] `git show origin/main:skills/poteto-mode/playbooks/autopilot-stack.md`
  - [ ] `git show origin/main:skills/swarm/SKILL.md`
  - [ ] `git show origin/main:.pi/skills/verify-lucebox/SKILL.md`
  - [ ] `git show origin/main:skills/poteto-mode/playbooks/opening-a-pr.md`
  - [ ] `git show origin/main:skills/how/SKILL.md`
- [ ] Arm the 30-minute audit tick as a bash polling loop with an explicit iteration cap. Never leave the cadence to memory.
- [ ] Use this tick prompt, verbatim. "Re-read the execution playbook from trunk and the run objective. Audit the operation against both and fix drift in this tick. Probe every active lane and judge progress by side effects only. Stand down a stuck lane and dispatch its replacement now. Then send the operator a status message, whether or not anything changed, with the queue table of PR, owner, state, and head SHA, the verdicts since the last tick, what merged, open operator gates, and blockers."
- [ ] On the operator hold or stand down order, send every owner a zero writes order at once.

### Run owner passes

- [ ] Run one owner pass per PR with the full lifecycle the execution playbook names.
- [ ] Follow this dependency graph. Start dependent work only after its parent merges.
  - [ ] DS4V-1 and DS4V-2 are independent and first. Both branch from `main`.
  - [ ] DS4V-3 after DS4V-1 and DS4V-2.
- [ ] Hold the file boundaries. DS4V-1 touches only `docs/ds4v-baseline.md` and `scripts/ds4v-baseline.sh`. DS4V-2 touches only `docs/ds4v-source.md` and `scripts/ds4v-fetch-source.sh`. DS4V-3 touches only `scripts/ds4v-quant.sh` and `docs/ds4v-quant.md` and `share/model_cards/ds4v-vision.json`.
- [ ] Hold the review gate. No PR changes an interaction. All three stop at merge ready without an operator media review.

### PR mechanics, for every PR

- [ ] Resolve the forge once. Default to `gh`. If `command -v origin` succeeds and Origin can resolve the repository, use `origin pr` for every PR operation. Record any fallback to `gh`. Never require `gt`.
- [ ] Open the PR ready, never draft, with `origin pr create --status open --base main` or `gh pr create --base main` according to the resolved forge. A stack child targets its parent branch.
- [ ] Run the repo lint and typecheck once before the PR facing push. Push with hooks on.
- [ ] Run `/unslop` over the diff before each commit and `/no-comments` before review.
- [ ] Triage every Bugbot and security reviewer comment per `../references/bugbot-triage.md`.
- [ ] Rebase onto current trunk before babysit and again before the merge ready report.

### Verdict and merge, for every PR

- [ ] At the merge ready head SHA, run the swarm per `skills/swarm/SKILL.md`. One gates lane. The ten live lanes from the PR Verify live block. The perf lane from its Verify perf block. One audit lane that reads the diff and the receipts and distrusts the PR body.
- [ ] Clean only when every lane is `PASS`. Findings go back to the owner. A new head gets a fresh swarm and a fresh verdict.
- [ ] The root appends each clean PR to the one linear base branch stack and the operator lands it bottom up. A rebase that changes a patch id sends that PR back through verification.

### Boot recipe, for every live lane

- [ ] Fetch the PR head with `git fetch origin` and check out the exact head SHA.
- [ ] Start the backend on the Strix Halo plus 7900XT pair and wait for `/props.build` to answer.
- [ ] Deliver input only through the bash driven harness. Name the read only diagnostics.
- [ ] Save every proof file under `/tmp/swarm-DS4V/worker-1` and return the paths with the report.

## Reproduce the asymmetric baseline on the operator pair (DS4V-1)

**Depends on.** None.

**Files.**

- [ ] Create `docs/ds4v-baseline.md` with the measured setup and commands.
- [ ] Create `scripts/ds4v-baseline.sh` with the launch and curl proof steps.

**Build.**

- [ ] Record the server SHA and both model SHAs in `docs/ds4v-baseline.md`.

**You see.**

- [ ] A `curl` call to `/props.build` answers with the expected image tag.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Run `ctest --output-on-failure -R deepseek4_unit` in `server/build` and keep the log.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `fast mechanical model (setup default)` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Run the same text prompt at trunk and head. Save `artifacts/ds4v-1/lane-1-compare.json`. Pass when both sides return HTTP 200 with non empty text.
- [ ] Lane 2. Chat smoke over the heterogeneous path. Send the LRU prompt from the verify skill. Save `artifacts/ds4v-1/lane-2-chat.json`. Pass when the response holds generated text.
- [ ] Lane 3. Build identity. Read `/props.build` from the running server. Save `artifacts/ds4v-1/lane-3-props.json`. Pass when the file names the expected image tag.
- [ ] Lane 4. Prefill probe. Send the 2k prompt used in the Lucebox report. Save `artifacts/ds4v-1/lane-4-prefill.json`. Pass when prompt processing exceeds 300 tok/s.
- [ ] Lane 5. Decode probe. Generate 128 tokens from the same prompt. Save `artifacts/ds4v-1/lane-5-decode.json`. Pass when decode exceeds 40 tok/s.
- [ ] Lane 6. DSpark acceptance. Read the served response header for the spec flag. Save `artifacts/ds4v-1/lane-6-spec.json`. Pass when the flag reports true.
- [ ] Lane 7. Placement proof. Read the server log for the owner lines. Save `artifacts/ds4v-1/lane-7-placement.log`. Pass when both devices appear as owners.
- [ ] Lane 8. Determinism. Send the same prompt twice. Save `artifacts/ds4v-1/lane-8-repeat.json`. Pass when both answers match byte for byte.
- [ ] Lane 9. Model list. Read `/v1/models` from the same server. Save `artifacts/ds4v-1/lane-9-models.txt`. Pass when the call returns 200 or 404 with a body.
- [ ] Lane 10. Cleanup. Run the verify skill cleanup. Save `artifacts/ds4v-1/lane-10-cleanup.log`. Pass when the instance is gone and the proof files remain.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Decode tok/s on the 2k prompt with 128 output tokens.
- [ ] Probe. Run `scripts/ds4v-baseline.sh` at trunk and at the head, interleaved.
- [ ] Baseline. Record the trunk value first.
- [ ] Rule. Head ties or beats trunk. Fail when head trails by more than 5 percent.

**Review gate.** None. DS4V-1 is not review-gated.

**Merge.**

- [ ] Root records a clean verdict at the exact head SHA.
- [ ] The owner rebases onto current trunk after the verdict with patch id unchanged.

## Lock the abliterated vision source with provenance (DS4V-2)

**Depends on.** None.

**Files.**

- [ ] Create `docs/ds4v-source.md` with the parent repo and the tensor manifest.
- [ ] Create `scripts/ds4v-fetch-source.sh` with the exact download commands.

**Build.**

- [ ] Verify all 48 shards and 72633 tensors match the manifest by name and shape.

**You see.**

- [ ] The manifest lists the vision tower and the aligner as present.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Run `python3 scripts/ds4v-fetch-source.sh --check-only` and keep the checksum log.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `fast mechanical model (setup default)` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Run the same text prompt against the stock parent and the abliterated source. Save `artifacts/ds4v-2/lane-1-text.json`. Pass when both answers match in shape and the ablated one refuses less.
- [ ] Lane 2. Vision presence. List `vision.*` tensors in the manifest. Save `artifacts/ds4v-2/lane-2-vision.txt`. Pass when the count equals 259.
- [ ] Lane 3. Aligner presence. List `aligner.*` tensors in the manifest. Save `artifacts/ds4v-2/lane-3-aligner.txt`. Pass when the count equals 4.
- [ ] Lane 4. Router bias. List `bias_vl` tensors in the manifest. Save `artifacts/ds4v-2/lane-4-bias.txt`. Pass when the count equals 43.
- [ ] Lane 5. Image smoke. Describe the Earth image with the reference implementation. Save `artifacts/ds4v-2/lane-5-earth.json`. Pass when the answer names Earth.
- [ ] Lane 6. Text capability. Score the MMLU sample with both checkpoints. Save `artifacts/ds4v-2/lane-6-mmlu.json`. Pass when the delta stays within 1 point.
- [ ] Lane 7. Tokenizer. Encode the OpenAI style image message with the reference encoder. Save `artifacts/ds4v-2/lane-7-encode.json`. Pass when token ids match the reference.
- [ ] Lane 8. Draft head. List `mtp.*` blocks in the manifest. Save `artifacts/ds4v-2/lane-8-mtp.txt`. Pass when the count equals 3.
- [ ] Lane 9. License. Read the model `LICENSE` from the source repo. Save `artifacts/ds4v-2/lane-9-license.txt`. Pass when the text names MIT.
- [ ] Lane 10. Cleanup. Remove the scratch download cache. Save `artifacts/ds4v-2/lane-10-cleanup.log`. Pass when the manifest and proof files remain.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Minutes to verify all shard checksums on the build machine.
- [ ] Probe. Run `scripts/ds4v-fetch-source.sh --check-only` twice on the same host.
- [ ] Baseline. Record the first run value first.
- [ ] Rule. Second run ties or beats the first. Fail when it trails by more than 20 percent.

**Review gate.** None. DS4V-2 is not review-gated.

**Merge.**

- [ ] Root records a clean verdict at the exact head SHA.
- [ ] The owner rebases onto current trunk after the verdict with patch id unchanged.

## Ship a vision GGUF tuned for the operator pair (DS4V-3)

**Depends on.** DS4V-1 and DS4V-2.

**Files.**

- [ ] Create `scripts/ds4v-quant.sh` with the imatrix quant recipe.
- [ ] Create `docs/ds4v-quant.md` with the rung table and the budget math.
- [ ] Create `share/model_cards/ds4v-vision.json` with the placement and budget.

**Build.**

- [ ] Run the quant recipe and record the GGUF SHAs in `docs/ds4v-quant.md`.

**You see.**

- [ ] A `curl` image prompt returns a correct scene description.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Assert the GGUF keeps `bias_vl` on all 43 layers and the mmproj loads.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `fast mechanical model (setup default)` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Run the same text prompt at the DS4V-1 baseline and at this head. Save `artifacts/ds4v-3/lane-1-compare.json`. Pass when both return HTTP 200 with non empty text.
- [ ] Lane 2. Image description. Describe the carrots image through `/v1/chat/completions`. Save `artifacts/ds4v-3/lane-2-carrots.json`. Pass when the answer names carrots.
- [ ] Lane 3. Second image. Describe the corn image through the same endpoint. Save `artifacts/ds4v-3/lane-3-corn.json`. Pass when the answer names corn.
- [ ] Lane 4. Missing projector. Send an image prompt without mmproj loaded. Save `artifacts/ds4v-3/lane-4-nommproj.json`. Pass when the server answers 400 cleanly.
- [ ] Lane 5. Text still works. Send the LRU prompt with mmproj loaded. Save `artifacts/ds4v-3/lane-5-text.json`. Pass when the response holds generated text.
- [ ] Lane 6. Placement proof. Read the server log for the owner lines. Save `artifacts/ds4v-3/lane-6-placement.log`. Pass when both devices appear as owners.
- [ ] Lane 7. Decode probe. Generate 128 tokens from the 2k prompt. Save `artifacts/ds4v-3/lane-7-decode.json`. Pass when decode exceeds 35 tok/s.
- [ ] Lane 8. Determinism. Send the same image prompt twice. Save `artifacts/ds4v-3/lane-8-repeat.json`. Pass when both answers match byte for byte.
- [ ] Lane 9. Build identity. Read `/props.build` from the running server. Save `artifacts/ds4v-3/lane-9-props.json`. Pass when the file names the expected image tag.
- [ ] Lane 10. Cleanup. Run the verify skill cleanup. Save `artifacts/ds4v-3/lane-10-cleanup.log`. Pass when the instance is gone and the proof files remain.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Decode tok/s on the 2k prompt with 128 output tokens, plus prefill tok/s on the same prompt.
- [ ] Probe. Run the DS4V-1 baseline script and the DS4V-3 vision script interleaved on the same pair.
- [ ] Baseline. Record the DS4V-1 value first.
- [ ] Rule. Vision head stays within budget. Fail when decode trails the text baseline by more than 20 percent.

**Review gate.** None. DS4V-3 is not review-gated.

**Merge.**

- [ ] Root records a clean verdict at the exact head SHA.
- [ ] The owner rebases onto current trunk after the verdict with patch id unchanged.

## Close the program

- [ ] Every box above is checked with its evidence.
- [ ] Reply to the operator with the report the execution playbook names.

## Appendix A. Prototype evidence

The Lucebox report at `https://www.lucebox.com/blog/deepseek-v4-asymmetric-parallelism` measures 51 tok/s median decode with asymmetric expert parallelism.
PR 604 is merged. It adds the RX 7900 XT plus Strix Halo dual GPU profile with 45 to 47 tok/s decode.
HF API lists 4 prometheusAIR rungs from 66 GiB to 108 GiB plus a sub GiB mmproj file.
OrcaRouter parent keeps vision. Its GGUF is text only.
Unproven. No GPU run happened on this Mac. All throughput numbers above are cited, not measured here.

## Appendix B. Alternatives rejected

Re-abliterate from scratch. Rejected. The OrcaRouter parent already bakes the edit with measured evals.
OrcaRouter GGUF directly. Rejected. It drops the vision tower.
Unsloth quants directly. Rejected for now. First shards read empty at check time.
Qwen mmproj path in Lucebox. Rejected for DS4. Lucebox vision gates on Qwen35 only.

## Appendix C. Risks

VRAM budget. The 95 GiB rung plus KV at long context nears the pair budget. Watch the 1M context setting.
Upstream llama dot cpp drift. Vision support merged days ago. Pin the commit in the quant script.
Safety. Abliterated weights comply with harmful requests. Keep them local and never serve them publicly.
This checkout lacks `server/` sources. All GPU work runs on the Strix Halo machine with submodules present.

## Appendix D. Links and reading list

Read `skills/how/SKILL.md` before the placement review in DS4V-1.
Read `skills/interrogate/SKILL.md` before the quant recipe review in DS4V-3.
Keep the trail per `skills/show-me-your-work/SKILL.md`.
The verify surface is `.pi/skills/verify-lucebox/SKILL.md`.
