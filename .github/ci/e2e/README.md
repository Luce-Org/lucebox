# Model e2e on lucebox3

`model-e2e.yml` runs on lucebox3's self-hosted runner: Qwen3.8-27B on the R9700
(gfx1201, HIP index 0) and DeepSeek V4 Flash on the Strix Halo (gfx1151, HIP
index 1). Each model's files, flags and GPU are in `select_models.py`.

## What lucebox3 needs

- **The models** in `/opt/models` (or the repository variable
  `LUCEBOX_MODELS_DIR`):
  - `Qwen3.8-27B-UD-IQ4_XS.gguf`
  - `qwen38-dflash2-q8_0.gguf` (the draft)
  - `DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf`

  A missing file skips a PR's job and fails a baseline run.
- **A readable kernel log**, so GPU faults during the run are caught: passwordless
  `sudo dmesg` for the runner user (as `gpu-tests-amd` already uses), or
  `kernel.dmesg_restrict=0`. Without it every run warns that GPU errors were not
  checked.
- **ccache** (optional): cold builds take ~100 s without it. The build directory
  and remembered passes live in the runner user's `~/.cache/lucebox-e2e`.

The job sees every user's GPU processes through `/sys/class/kfd/kfd/proc`, which
any user can read.

## Baselines

Every merge to main runs the models whose code changed since their baseline
and uploads each result as the artifact `model-e2e-baseline-<model>-<device>`
(kept 90 days) unless it fails. Every job compares with the newest one, so a
merged change that alters the output becomes the reference for the next PRs. To
refresh a baseline by hand, e.g. after a ROCm upgrade, run the workflow on main
with `update_baseline` ticked. Until the first baseline exists, jobs still fail on
crashes, hangs and failed checks but cannot detect changed output.

## Benchmarking by hand on lucebox3

A job waits up to 4 minutes for other GPU users, then skips with a warning. To
keep jobs off the machine for longer, stop its runner service
(`sudo ./svc.sh stop` in the runner directory) and start it again when you're
done. This also pauses `gpu-tests-amd`.

## Running the tests

```bash
cd .github/ci/e2e && uv run --with pytest --no-project python -m pytest -q
```
