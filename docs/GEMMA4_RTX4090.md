# Gemma 4 31B on RTX 4090

This Lucebox path serves `gemma-4-31B-it-abliterated-Q4_K_M.gguf` through a
Gemma 4 MTP + TurboQuant llama.cpp backend. The DFlash runtime in this
repository is still the Qwen/Laguna research path; Gemma 4 uses libllama because
it needs the Gemma 4 graph, tokenizer template, TurboQuant KV cache, and MTP
assistant support.

Default workstation paths:

```bash
LUCEBOX_GEMMA4_MODEL=/mnt/c/Users/adyba/Downloads/gemma-4-31B-it-abliterated-Q4_K_M.gguf
LUCEBOX_GEMMA4_MTP_MODEL=/home/tdamre/models/gemma-4-31B-it-assistant-atomic-f16.gguf
LUCEBOX_LLAMA_SERVER=/home/tdamre/src/atomic-llama-cpp-turboquant/build-cuda124/bin/llama-server
```

The assistant GGUF above was reconverted from the cached
`google/gemma-4-31B-it-assistant` snapshot with Atomic's converter so its
metadata uses the `gemma4_assistant` architecture expected by `--mtp-head`.

Start from Windows PowerShell:

```powershell
.\scripts\Start-LuceboxGemma4090.ps1 -Command Start
```

The Windows launcher applies a best-effort `nvidia-smi -lgc 2100,2700` graphics
clock lock before start/restart and resets it on stop. Use `-SkipGpuClockLock`
to leave clocks unchanged.

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
LUCEBOX_GEMMA4_CTX_SIZE=65536
LUCEBOX_GEMMA4_DRAFT_CTX_SIZE=2048
LUCEBOX_GEMMA4_DRAFT_BLOCK_SIZE=4
LUCEBOX_GEMMA4_CACHE_TYPE_K=turbo4
LUCEBOX_GEMMA4_CACHE_TYPE_V=turbo4
LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_K=turbo4
LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_V=turbo4
LUCEBOX_GEMMA4_CACHE_RAM=0
LUCEBOX_GEMMA4_NO_KV_OFFLOAD=0
```

Verify the reply path and single-stream decode floor:

```bash
python3 scripts/verify_gemma4_4090.py --base-url http://127.0.0.1:18191 --threshold 60
```

Probe long-context chat stability:

```bash
python3 scripts/probe_gemma4_context.py --base-url http://127.0.0.1:18191 --ctx 40960 --targets 8192,16384,32768,38912
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
- After a Docker/WSL/GPU runtime recovery on May 11, the previous `speed-faall` profile degraded to about `3.55 tok/s` at `40960` context even with clocks boosted. The `speed-mmq` build is the best current fallback at about `31.44 tok/s` with q8_0 K/V on GPU; `69632` with `--no-kv-offload` loaded and answered but only reached `19.73 tok/s`.
- `65536` context, Atomic TurboQuant `turbo4` K/V, Gemma 4 assistant via `--mtp-head`, `--draft-block-size 3`: loaded and answered at `38.29 tok/s`; MTP accepted `82/88` draft tokens.
- `65536` context, Atomic TurboQuant `turbo4` K/V, Gemma 4 assistant via `--mtp-head`, `--draft-block-size 4`: loaded and answered at `48.52 tok/s`; MTP accepted `93/102` draft tokens. This is the current best high-context TurboQuant + MTP result but still below the `60 tok/s` gate.
- The Windows launcher default path for the same `65536`/`turbo4`/MTP block-size-4 recipe started successfully on `http://127.0.0.1:18191` and verified at `44.57 tok/s` with `93/102` MTP draft tokens accepted.
- `71680` context, Atomic TurboQuant `turbo4` K/V, Gemma 4 assistant via `--mtp-head`, `--draft-block-size 4`, launched successfully from the Windows launcher with about `22.33 GiB` VRAM used and `1.81 GiB` free. A short verifier run reached `43.90 tok/s`, so it is still below the `60 tok/s` gate.
- `71680` context cold-prefill stability probe with a `70035`-token chat prompt completed successfully: prompt processing was `1292.04 tok/s`, decode was `23.51 tok/s`, and MTP accepted `19/31` draft tokens. This confirms the 70k prompt can answer on the Atomic TurboQuant path, but not at the requested decode floor.
- `71680` context probe with a `65590`-token chat prompt also completed successfully: prompt processing was `1298.25 tok/s`, decode was `29.05 tok/s`, and MTP accepted `21/30` draft tokens.
- `65536` context, Atomic TurboQuant `turbo4` K/V, `--draft-block-size 6`: loaded and answered at `22.31 tok/s`; acceptance dropped to `78/234`.
- `65536` context, Atomic TurboQuant `turbo3` K/V, `--draft-block-size 4`: loaded and answered at `26.57 tok/s`; MTP accepted `71/166` draft tokens.
