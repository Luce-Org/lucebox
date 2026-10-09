# Gemma 4 31B on RTX 4090

This Lucebox path serves `gemma-4-31B-it-abliterated-Q4_K_M.gguf` through a
Gemma 4 MTP + TurboQuant llama.cpp backend. The DFlash runtime in this
repository is still the Qwen/Laguna research path; Gemma 4 uses libllama because
it needs the Gemma 4 graph, tokenizer template, TurboQuant KV cache, and MTP
assistant support.

Default workstation paths:

```bash
LUCEBOX_GEMMA4_MODEL=/mnt/c/Users/adyba/Downloads/gemma-4-31B-it-abliterated-Q4_K_M.gguf
LUCEBOX_GEMMA4_MTP_MODEL=/home/tdamre/models/AtomicChat-gemma-4-31B-it-assistant-GGUF/gemma-4-31B-it-assistant.Q4_K_S.gguf
LUCEBOX_LLAMA_SERVER=/home/tdamre/src/atomic-llama-cpp-turboquant/build-cuda124/bin/llama-server
```

The current validated Atomic TurboQuant checkout is
`514e600c84f50a4ba31ca0e3ce6d5560f24c2524` on
`feature/turboquant-kv-cache`.

The assistant GGUF above was reconverted from the cached
`google/gemma-4-31B-it-assistant` snapshot with Atomic's converter so its
metadata uses the `gemma4_assistant` architecture expected by `--mtp-head`,
then quantized for MTP serving. The default now uses AtomicChat's prebuilt
Q4_K_S GGUF because it is the best measured 70k throughput variant on this
RTX 4090. The launcher also pins the assistant `token_embd.weight` tensor to
CUDA0; leaving that small tensor CPU-mapped was the main remaining MTP draft
latency bottleneck. The locally converted F16 intermediate is kept at
`/home/tdamre/models/gemma-4-31B-it-assistant-atomic-f16.gguf`.

Start from Windows PowerShell:

```powershell
.\scripts\Start-LuceboxGemma4090.ps1 -Command Start
```

The Windows launcher applies a best-effort `nvidia-smi -lgc 2100,2700` graphics
clock lock before start/restart and resets it on stop. Use `-SkipGpuClockLock`
to leave clocks unchanged. For controlled A/B tests, `-Model` can point the
same recipe at a different local GGUF without editing the default launcher.

Or from WSL:

```bash
./scripts/lucebox-gemma4-4090.sh start
```

The server listens on `http://127.0.0.1:18191` by default and exposes
OpenAI-compatible `/v1/chat/completions` plus llama.cpp `/completion`.
The launcher sets `--reasoning off` so OpenAI chat replies populate
`message.content` by default. The default launcher profile uses Atomic's
`--mtp-head` path with TurboQuant `turbo4` K/V and block-size-4 MTP:

```bash
LUCEBOX_GEMMA4_MTP_STYLE=atomic
LUCEBOX_GEMMA4_CTX_SIZE=70080
LUCEBOX_GEMMA4_DRAFT_CTX_SIZE=2048
LUCEBOX_GEMMA4_DRAFT_BLOCK_SIZE=4
LUCEBOX_GEMMA4_GPU_LAYERS_DRAFT=all
LUCEBOX_GEMMA4_DRAFT_OVERRIDE_TENSOR=token_embd.weight=CUDA0
LUCEBOX_GEMMA4_CACHE_TYPE_K=turbo4
LUCEBOX_GEMMA4_CACHE_TYPE_V=turbo4
LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_K=turbo4
LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_V=turbo4
LUCEBOX_GEMMA4_CACHE_RAM=0
LUCEBOX_GEMMA4_NO_KV_OFFLOAD=0
LUCEBOX_GEMMA4_POLL=100
LUCEBOX_GEMMA4_POLL_BATCH=1
LUCEBOX_GEMMA4_PRIORITY=2
LUCEBOX_GEMMA4_PRIORITY_BATCH=2
LUCEBOX_GEMMA4_THREADS_HTTP=1
```

Verify the reply path and single-stream decode floor:

```bash
python3 scripts/verify_gemma4_4090.py --base-url http://127.0.0.1:18191 --threshold 70
```

Probe long-context chat stability:

```bash
python3 scripts/probe_gemma4_context.py --base-url http://127.0.0.1:18191 --ctx 70080 --targets 70000 --cache-type-k turbo4 --cache-type-v turbo4 --max-tokens 64
```

Current RTX 4090 measurements:

- `40960` context, q8_0 K/V, MTP draft 4, prompt cache disabled: 3-run verifier passed with minimum `63.78 tok/s` and average `75.10 tok/s`.
- `40960` context was restored after the higher-context attempts and passed fresh verifier runs at `63.97 tok/s` and `65.21 tok/s`.
- `40960` context, q8_0 K/V long-context chat probe through about `38955` prompt tokens completed successfully, with decode speed dropping to `16.12 tok/s` at the top end.
- `49152` context, q8_0 K/V, MTP draft 4, default `-b 2048 -ub 512`: loaded and answered, but failed the speed gate with minimum `2.50 tok/s` and average `2.91 tok/s`.
- `49152` context, q8_0 K/V, MTP draft 1, `-b 512 -ub 128`: loaded and answered at `56.82 tok/s` then `51.70 tok/s`, then hit an HTTP 500 parser failure from malformed generated text, so it is not stable or above the speed gate.
- `65536` context, q8_0 K/V, MTP draft 4, `-b 512 -ub 128`: loaded and answered, but only reached `5.58 tok/s`.
- `65536` context, q8_0 K/V, MTP draft 1, `-b 512 -ub 128`: loaded and answered, but only reached `33.35 tok/s`.
- Earlier `65536` attempts with the normal `-ub 512` path loaded q8_0 K/V but failed first generation with CUDA OOM in the MTP flash-attention path.
- A May 12 Atomic MTP recheck at `65536` context with q8_0 K/V, `-b 2048 -ub 512`, and `--draft-block-size 4` loaded fully on CUDA with only about `346 MiB` VRAM free. It answered the first chat verifier request at `39.30 tok/s`, then crashed on the second request with `GGML_ASSERT(i01 >= 0 && i01 < ne01)` after draft truncation, so this q8_0 profile is not stable.
- The same Atomic q8_0 K/V profile with `--draft-block-size 2` loaded but crashed on the first chat request with `std::runtime_error: Invalid token`. `--draft-block-size 1` is rejected by Atomic because valid values are `2` through `32`.
- After a Docker/WSL/GPU runtime recovery on May 11, the previous `speed-faall` profile degraded to about `3.55 tok/s` at `40960` context even with clocks boosted. The `speed-mmq` build is the best current fallback at about `31.44 tok/s` with q8_0 K/V on GPU; `69632` with `--no-kv-offload` loaded and answered but only reached `19.73 tok/s`.
- `65536` context, Atomic TurboQuant `turbo4` K/V, Gemma 4 assistant via `--mtp-head`, `--draft-block-size 3`: loaded and answered at `38.29 tok/s`; MTP accepted `82/88` draft tokens.
- `65536` context, Atomic TurboQuant `turbo4` K/V, Gemma 4 assistant via `--mtp-head`, `--draft-block-size 4`: loaded and answered at `48.52 tok/s`; MTP accepted `93/102` draft tokens.
- The earlier Windows launcher default path for the same `65536`/`turbo4`/MTP block-size-4 recipe started successfully on `http://127.0.0.1:18191` and verified at `44.57 tok/s` with `93/102` MTP draft tokens accepted.
- `71680` context, Atomic TurboQuant `turbo4` K/V, Gemma 4 assistant via `--mtp-head`, `--draft-block-size 4`, launched successfully from the Windows launcher with about `22.33 GiB` VRAM used and `1.81 GiB` free. A short verifier run reached `43.90 tok/s`, so it is still below the `60 tok/s` gate.
- `71680` context, Atomic TurboQuant `turbo4` K/V, Gemma 4 assistant via `--mtp-head`, `--draft-block-size 3`, reached `58.10 tok/s` on a 128-token verifier and `59.01 tok/s` on a 512-token verifier, with `335/350` MTP draft tokens accepted on the longer run. This is the best measured 70k launcher recipe so far, but it remains below the requested `70 tok/s` hard gate.
- Raising the graphics clock floor to `2520 MHz` and testing `--poll 100 --poll-batch 1 --prio 2 --prio-batch 3` did not improve the `71680`/`turbo4`/block-size-3 profile; the 128-token poll/priority run reached `57.67 tok/s`.
- Quantizing the assistant head from F16 (`911 MiB`) to Q4_K_M (`338 MiB`) reduced loaded VRAM by about `250 MiB` at `71680` context and let the first 128-token verifier prompt pass the `70 tok/s` gate at `73.74 tok/s`. A 3-run verifier was still not stable above the gate, with `54.52 tok/s` minimum and `62.68 tok/s` average across the three fixed prompts.
- A Q8_0 assistant (`491 MiB`) was slower than Q4_K_M on the same 3-run verifier: `51.12 tok/s` minimum and `58.23 tok/s` average.
- With the Q4_K_M assistant, a corrected `71680` context probe using a `70034`-token chat prompt completed successfully: prompt processing was `1457.52 tok/s`, decode was `37.17 tok/s`, and MTP accepted `37/51` draft tokens. This confirmed the turbo4 path answers at 70k tokens, but fully populated 70k-context decode remains below the requested `70 tok/s` gate.
- Tightening the context from `71680` to `70080` while preserving a 70k+ usable window improved the Q4_K_M assistant 128-token verifier minimum to `56.13 tok/s` and average to `64.15 tok/s` at block size 3.
- AtomicChat's Q4_K_S assistant at `70080` context with `--draft-block-size 4` became the current validated 70k recipe after pinning only the assistant `token_embd.weight` to CUDA0 with `--override-tensor-draft token_embd.weight=CUDA0`: a 3-run 128-token verifier passed the `70 tok/s` floor with `70.98 tok/s` minimum and `87.31 tok/s` average, and a 3-run 512-token verifier passed with `77.28 tok/s` minimum and `90.62 tok/s` average.
- The same default recipe was restarted through the Windows PowerShell launcher used by the desktop shortcut and re-verified: the 3-run 128-token verifier passed with `72.14 tok/s` minimum and `88.28 tok/s` average, and the 3-run 512-token verifier passed with `78.13 tok/s` minimum and `91.69 tok/s` average.
- The same Q4_K_S/block-size-4 profile answered a `70034`-token chat prompt at `70080` context: prompt processing was `1373.00 tok/s`, decode was `37.92 tok/s`, and MTP accepted `41/64` draft tokens.
- AtomicChat Q4_K_M was slightly worse than Q4_K_S at `70080` context (`55.04 tok/s` minimum, `63.04 tok/s` average on the 128-token verifier). AtomicChat Q5_K_M was also worse on the 512-token verifier (`59.05 tok/s` minimum, `69.35 tok/s` average).
- Rebuilding Atomic with `GGML_CUDA_FORCE_MMQ=ON` did not improve the Q4_K_S/block-size-4 profile; the 3-run 128-token verifier reached `55.14 tok/s` minimum and `67.71 tok/s` average.
- `LLAMA_MTP_SKIP_STREAK_THRESHOLD=1` made the Q4_K_S/block-size-4 profile worse, dropping the 3-run 128-token verifier to `47.99 tok/s` minimum and `50.35 tok/s` average.
- Increasing `-ub` to `1024` did not materially improve the same profile (`56.58 tok/s` minimum and `68.07 tok/s` average). Reducing logical batch to `-b 1024 -ub 1024` was worse (`54.63 tok/s` minimum and `65.97 tok/s` average).
- Disabling continuous batching through `LLAMA_ARG_CONT_BATCHING=false` was only a small improvement on the 128-token verifier (`56.70 tok/s` minimum and `68.35 tok/s` average); the 1024-token verifier still missed the floor (`57.77 tok/s` minimum and `68.80 tok/s` average).
- Request-level `backend_sampling=true` also stayed below the floor (`56.92 tok/s` minimum and `68.29 tok/s` average). Forcing cuBLAS 16F compute and raising the GPU clock floor to `2520 MHz` were both worse than the default run.
- A fresh default May 12 re-run of the Q4_K_S/block-size-4 `70080` profile still failed the hard `70 tok/s` every-run gate: `51.52 tok/s` minimum and `63.79 tok/s` average across the three fixed 128-token verifier prompts. MTP acceptance was prompt-dependent (`93/102`, `74/158`, and `77/150`).
- Disabling the MTP depth-2 pipeline with `LLAMA_PIPELINE_DEPTH2=0` did not fix the low-acceptance prompts: the same 128-token verifier reached only `53.37 tok/s` minimum and `65.08 tok/s` average. Lowering MTP draft block size to 2 was worse (`49.11 tok/s` minimum and `50.45 tok/s` average), and raising it to 5 was clearly worse (`43.16 tok/s` minimum and `49.90 tok/s` average).
- Keeping K at `turbo4` but changing V to `turbo2` also failed the floor (`46.14 tok/s` minimum and `54.87 tok/s` average). The CUDA turbo2-V path was slower and had worse MTP acceptance (`61/196`, `81/136`, and `70/166`) than the default turbo4-V profile.
- Requantizing the local Q4_K_M GGUF to Atomic `TQ4_1S` produced `C:\Users\adyba\Downloads\gemma-4-31B-it-abliterated-TQ4_1S.gguf` (`18,563.83 MiB`, file type `TQ4_1S`), but it is not usable for the 70k profile on the RTX 4090: startup tried to allocate `30,783.28 MiB` of CUDA model buffer before KV cache and failed model loading.
- A detached May 13 priority/polling check at `70080` context with Q4_K_S MTP, `--poll 100 --poll-batch 1 --prio 2 --prio-batch 2 --threads-http 1`, improved the three fixed chat-format prompts only to `61.41-63.12 tok/s`. These flags are now the launcher defaults because they reduce latency a little, but they still do not satisfy the strict `70 tok/s` every-run gate.
- Fast-forwarding Atomic from `2e81dc5f6` to `514e600c8` and rebuilding improved the default `70080`/`turbo4`/Q4_K_S/block-size-4 profile, but not enough: the 3-run 128-token verifier reached `55.90 tok/s` minimum and `66.90 tok/s` average. The same rebuilt tree with AtomicChat's documented `turbo3`/block-size-3 profile was worse on chat-format prompts, with a `53.92 tok/s` low prompt.
- Forcing the target `token_embd.weight` onto CUDA with `--override-tensor token_embd.weight=CUDA0` removed the target's `CPU_Mapped model buffer` and loaded with about `525 MiB` VRAM free, but it was much slower: the same verifier fell to `10.90 tok/s` minimum and `13.14 tok/s` average. Full target tensor residency at 70k is therefore not compatible with the requested speed gate on this RTX 4090 profile.
- Requantizing only the target token embedding down from `q6_K` to `q4_K` produced `C:\Users\adyba\Downloads\gemma-4-31B-it-abliterated-Q4_K_M-tokenemb-q4k.gguf` and let the target load fully on CUDA with about `1.2 GiB` VRAM free, but the 3-run 128-token verifier still failed badly at `14.56 tok/s` minimum and `15.57 tok/s` average. The extra VRAM headroom did not offset the cost of full target CUDA residency.
- Starting the same `70080`/Q4_K_S/block-size-4 profile with `--no-host` did not change the major model/KV/compute buffer placement and still failed the 3-run verifier at `54.61 tok/s` minimum and `65.67 tok/s` average.
- Passing `-ngld all` makes the assistant offload intent explicit and matches Atomic's helper script, but it did not change the effective buffers from auto mode; the same 3-run verifier reached only `55.01 tok/s` minimum and `66.08 tok/s` average.
- A separate build of `test1111111111111112/llama-cpp-turboquant-gemma4` at `e93b7c5` was compiled with CUDA 12.4 to test its Gemma 4 D=256/512 turbo4 kernel path against the same local Q4_K_M target. That fork does not implement the Gemma `--mtp-head` assistant path, and its no-MTP `70080`/turbo4 verifier was not viable: output degraded to repeated non-English tokens and decode stayed at only `36.63 tok/s` minimum and `36.69 tok/s` average. This rules it out as a direct replacement and makes a risky kernel transplant insufficient on its own.
- `71680` context cold-prefill stability probe with a `70035`-token chat prompt completed successfully: prompt processing was `1292.04 tok/s`, decode was `23.51 tok/s`, and MTP accepted `19/31` draft tokens. This confirms the 70k prompt can answer on the Atomic TurboQuant path, but not at the requested decode floor.
- `71680` context probe with a `65590`-token chat prompt also completed successfully: prompt processing was `1298.25 tok/s`, decode was `29.05 tok/s`, and MTP accepted `21/30` draft tokens.
- `65536` context, Atomic TurboQuant `turbo4` K/V, `--draft-block-size 6`: loaded and answered at `22.31 tok/s`; acceptance dropped to `78/234`.
- `65536` context, Atomic TurboQuant `turbo3` K/V, `--draft-block-size 4`: loaded and answered at `26.57 tok/s`; MTP accepted `71/166` draft tokens.
- Copying the exact Q4_K_M target from `/mnt/c/Users/adyba/Downloads` to WSL ext4 at `/home/tdamre/models/gemma-4-31B-it-abliterated-Q4_K_M.gguf` matched SHA-256 `d015e259562bfaa50a9fcca388bbda6a443fbf1c492272197d15282b3b8afaac`, but did not improve the 70k floor: the 3-run 128-token verifier reached `55.33 tok/s` minimum and `66.28 tok/s` average.
- Re-testing Atomic's documented `turbo3` K/V with `--draft-block-size 3` against the ext4 Q4_K_M target was worse than the launcher default, reaching only `54.00 tok/s` minimum and `56.25 tok/s` average.
- A local `SuperGemma4-31b-abliterated.Q4_K_M.gguf` target loaded but returned empty `/completion` content during the verifier, so it is not a compatible replacement for the requested target path.
- Requantizing the verified Q4_K_M target to `Q3_K_M` produced `/home/tdamre/models/gemma-4-31B-it-abliterated-Q3_K_M-from-Q4_K_M.gguf` (`14,563.82 MiB`, `3.98 BPW`), but it still missed the floor (`54.91 tok/s` minimum, `61.57 tok/s` average) and showed visible repeated text on one fixed prompt.
- Requantizing the verified Q4_K_M target to `Q2_K` produced a numerically fast profile (`76.37 tok/s` minimum, `79.72 tok/s` average), but output quality collapsed into repeated fragments such as punctuation runs and repeated `la`, so it is not a valid user-facing solution despite crossing the raw timing threshold.
- A May 13 check confirmed the local cached Google assistant weights were already current for inference: Hugging Face `main` moved from `cffbbd2` to `4735700`, but only the README changed and the LFS object IDs for `model.safetensors` and `tokenizer.json` stayed unchanged.
- `--no-op-offload` did not improve the Q4_K_S/block-size-4 recipe (`54.81 tok/s` minimum, `66.06 tok/s` average). The F16 assistant was slower (`40.96 tok/s` minimum, `50.47 tok/s` average). Forcing target CPU threads to 16 or 4 was also worse (`52.97 tok/s` and `53.07 tok/s` minimum respectively).
- Built-in `q4_0` K/V cache loaded but missed the floor (`47.49 tok/s` minimum) and produced visibly degraded text on one fixed prompt. Mixed `q4_0` K with `turbo4` V aborted in CUDA flash attention during warmup, so the default remains TurboQuant `turbo4` for both K and V.
- `--no-mmap` loaded and answered but did not improve the default (`56.09 tok/s` minimum, `67.51 tok/s` average). Q4_K_S with `--draft-block-size 3` also missed (`57.00 tok/s` minimum, `64.45 tok/s` average).
- With the draft embedding override enabled, the `70080` context profile answered a `70034`-token chat prompt successfully after a Windows-wrapper restart: prompt processing was `1374.83 tok/s`, decode was `46.96 tok/s`, and MTP accepted `41/63` draft tokens. This validates the full 70k prompt path, while the strict `70 tok/s` floor is enforced by the fixed short-prompt single-stream verifier above.
