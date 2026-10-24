# DFlash Server — User Manual

> **Fast AI inference on your Windows PC with an NVIDIA GPU.**

DFlash is a high-performance local AI server. It runs large language models
(like Qwen 27B) on your GPU using speculative decoding — a technique that
generates text 2–5× faster than standard inference. Once running, it exposes an
OpenAI-compatible API that works with any chat client.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 (64-bit) | Windows 11 |
| GPU | NVIDIA GTX 1060 (6 GB) | RTX 3090 / 4090 (24 GB) |
| VRAM | 8 GB (smaller models) | 24 GB (Qwen 27B Q4) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB free | 40 GB free |
| NVIDIA Driver | 535+ | Latest Game Ready or Studio |
| Python | 3.10+ | 3.12 |

> **Note:** You do NOT need to install the CUDA Toolkit. The server binary
> includes everything it needs. Just make sure your NVIDIA driver is up to date.

---

## Quick Start

### Step 1: Download Models

You need two model files — a **target** (the big model) and a **draft** (a
small helper that speeds things up).

Install the Hugging Face CLI if you don't have it:

```powershell
pip install huggingface-hub[cli]
```

Download the models (about 18 GB total):

```powershell
# Create a folder for models
mkdir models\draft

# Download the main model (~16 GB)
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models

# Download the draft model (~1.8 GB)
huggingface-cli download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf --local-dir models\draft
```

### Step 2: Start the Server

The simplest way is to run the server directly:

```powershell
.\dflash_server.exe models\Qwen3.6-27B-Q4_K_M.gguf ^
  --draft models\draft\dflash-draft-3.6-q4_k_m.gguf ^
  --ddtree --ddtree-budget 22 --fa-window 2048 --port 8080
```

You should see output indicating the model is loaded and the server is
listening on `http://localhost:8080`.

### Step 3: Talk to the Server

Open another terminal and send a request:

```powershell
curl http://localhost:8080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"qwen\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello! What can you do?\"}]}"
```

Or use any OpenAI-compatible client by pointing it at `http://localhost:8080`.

---

## Using the Python Launcher

For a more convenient experience, use the included `run.py` script. It handles
tokenization and chat templates automatically.

First, install the required Python package:

```powershell
pip install transformers
```

Then run:

```powershell
python scripts\run.py --prompt "Write a Python function to sort a list"
```

You can also pipe input:

```powershell
echo "Explain quantum computing in simple terms" | python scripts\run.py
```

### Useful Options

| Option | Description |
|--------|-------------|
| `--prompt "..."` | The text prompt to send |
| `--n-gen 512` | Maximum tokens to generate (default: 256) |
| `--target path` | Path to target model (default: `models/Qwen3.6-27B-Q4_K_M.gguf`) |
| `--draft path` | Path to draft model (default: `models/draft/`) |
| `--budget 22` | DDTree budget — higher = more speculative (default: 22) |
| `--system "..."` | System prompt |
| `--kv-tq3` | Enable TQ3 KV cache for longer context (up to 256K) |

---

## Server API Endpoints

Once running, the server provides these endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check (returns 200 when ready) |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | OpenAI Chat Completions API |
| `POST /v1/responses` | OpenAI Responses API (for Codex) |
| `POST /v1/messages` | Anthropic Messages API (for Claude Code) |

---

## Connecting Clients

DFlash works as a drop-in backend for popular AI tools:

- **Open WebUI**: Set the OpenAI API base URL to `http://localhost:8080/v1`
- **Continue (VS Code)**: Add a custom model with base URL `http://localhost:8080`
- **Codex CLI**: Set `OPENAI_BASE_URL=http://localhost:8080/v1`
- **Any OpenAI SDK**: Point `base_url` to `http://localhost:8080/v1`

---

## Troubleshooting

### "CUDA error" or "no CUDA device"

- Make sure your NVIDIA driver is version 535 or newer
- Run `nvidia-smi` in a terminal to verify your GPU is detected
- Restart your PC after a driver update

### Out of memory (OOM)

- Close other GPU-heavy applications (games, other AI tools)
- Use a smaller model or quantization
- Add `--kv-tq3` to reduce KV cache memory usage

### Port already in use

- Change the port: `--port 8081`
- Or find what's using port 8080: `netstat -ano | findstr 8080`

### Server starts but generation is slow

- Make sure you're using both `--ddtree` and `--draft` flags
- Check that your GPU is not thermal throttling (`nvidia-smi` shows temperature)
- Close background GPU workloads

### Python launcher can't find the binary

- Run from the folder where `dflash_server.exe` is located
- Or set the path: `python scripts\run.py --bin path\to\dflash_server.exe`

---

## Performance Tips

- **DDTree budget**: The default of 22 works well for most tasks. Higher values
  (e.g., 32) may help for code generation but use more VRAM.
- **FA window**: `--fa-window 2048` is optimal for most use cases. Only increase
  if you need the model to attend to very long prior context.
- **TQ3 KV cache**: Use `--kv-tq3` if you need very long context (32K+). It
  uses ~3× less memory than the default F16 cache with minimal quality loss.
- **Power limit**: For sustained workloads, setting a power limit
  (`nvidia-smi -pl 220` on RTX 3090) can improve efficiency without
  significant speed loss.
