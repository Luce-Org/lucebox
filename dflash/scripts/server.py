"""
OpenAI-compatible HTTP server on top of test_dflash.

    pip install fastapi uvicorn transformers
    python3 scripts/server.py                 # serves on :8000

    curl http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"luce-dflash","messages":[{"role":"user","content":"hi"}],"stream":true}'

Drop-in for Open WebUI / LM Studio / Cline by setting
  OPENAI_API_BASE=http://localhost:8000/v1  OPENAI_API_KEY=sk-any

Streams tokens as Server-Sent Events using the OpenAI delta format.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware          # FIX 1: add CORS
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool
from transformers import AutoTokenizer

from _prefill_hook import (
    PrefillConfig, add_cli_flags, config_from_args,
    compress_text_via_daemon,
)
from prefix_cache import DaemonStdoutBus, PrefixCache


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = Path(os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
))
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / ("test_dflash" + (".exe" if sys.platform == "win32" else ""))
DEFAULT_BUDGET = 22
MODEL_NAME = "luce-dflash"

# Architecture strings stored in `general.architecture` of every GGUF this
# server can drive. test_dflash dispatches by GGUF arch internally:
#   qwen35 / qwen36  -> existing DFlash + DDTree pipeline
#   laguna           -> dflash27b::run_laguna_daemon() (no spec-decode)
# server.py just needs to omit --draft + the DFlash/DDTree flags when the
# arch doesn't support speculative decoding yet.
_QWEN35_ARCHES = {"qwen35", "qwen36"}
_LAGUNA_ARCHES  = {"laguna"}

_ALLOWED_TEMPLATE_KWARGS = frozenset({"enable_thinking", "tools", "add_generation_prompt"})


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


def _read_gguf_architecture(gguf_path: Path) -> str:
    """Return the 'general.architecture' string from a GGUF file, or '' on error.

    Logs a warning to stderr if the GGUF read fails, since that means the
    server will silently pick the non-Gemma4 daemon path and use the wrong
    argv shape. Caller should treat empty string as 'detection failed' and
    decide accordingly.
    """
    try:
        from gguf import GGUFReader  # type: ignore
        import numpy as np
        r = GGUFReader(str(gguf_path))
        f = r.fields.get("general.architecture")
        if f is None or not f.data:
            return ""
        p = f.parts[f.data[0]]
        if not isinstance(p, np.ndarray):
            return ""
        return bytes(p).decode("utf-8", errors="replace").strip()
    except Exception as e:
        import sys
        print(f"[server] WARNING: failed to read general.architecture from {gguf_path}: {e}",
              file=sys.stderr)
        return ""


_QWEN35_FAMILY_TOKENIZERS = {
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
}
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"

_LAGUNA_FAMILY_TOKENIZERS = {
    "Laguna-XS.2": "poolside/Laguna-XS.2",
    "Laguna-XS":   "poolside/Laguna-XS.2",
    "laguna-xs2":  "poolside/Laguna-XS.2",
}


def _read_gguf_str(reader, key: str) -> str | None:
    f = reader.fields.get(key)
    if f is None or not f.data:
        return None
    import numpy as np
    p = f.parts[f.data[0]]
    if not isinstance(p, np.ndarray):
        return None
    try:
        return bytes(p).decode("utf-8", errors="replace")
    except Exception:
        return None


def _arch_from_gguf(gguf_path: Path) -> str:
    """Return the value of ``general.architecture`` from the GGUF, or 'unknown'.

    server.py uses this to dispatch between the qwen35 stack (test_dflash +
    DFlash + DDTree) and the laguna stack (test_laguna_daemon, autoregressive
    only). 'unknown' falls back to the qwen35 path so existing setups keep
    working when the field is missing.
    """
    try:
        from gguf import GGUFReader  # type: ignore
        r = GGUFReader(str(gguf_path))
        v = _read_gguf_str(r, "general.architecture")
        return v.lower() if v else "unknown"
    except Exception:
        return "unknown"


def _tokenizer_id_from_gguf(gguf_path: Path) -> str:
    default = "Qwen/Qwen3.5-27B"
    try:
        from gguf import GGUFReader  # type: ignore
        r = GGUFReader(str(gguf_path))
        arch = (_read_gguf_str(r, "general.architecture") or "").lower()
        family = _LAGUNA_FAMILY_TOKENIZERS if arch in _LAGUNA_ARCHES else _QWEN35_FAMILY_TOKENIZERS
        if arch in _LAGUNA_ARCHES:
            default = next(iter(_LAGUNA_FAMILY_TOKENIZERS.values()))
        for key in ("general.basename", "general.name"):
            val = _read_gguf_str(r, key)
            if val is None:
                continue
            for known, repo in family.items():
                if known.lower() in val.lower():
                    return repo
    except Exception:
        pass
    return default


def parse_reasoning(
    text: str,
    thinking_enabled: bool = True,
    started_in_thinking: bool = False,
) -> tuple[str, str | None]:
    # Qwen chat templates can prefill `<think>\n` into the prompt, so the
    # generated output contains only the reasoning body plus `</think>`.
    parts = text.partition(THINK_OPEN_TAG)
    saw_open_tag = bool(parts[1])
    rest = parts[2] if saw_open_tag else parts[0]
    if THINK_CLOSE_TAG not in rest:
        if thinking_enabled and (started_in_thinking or saw_open_tag):
            return "", (rest.strip() or None)
        return rest.strip(), None
    reasoning, _, content = rest.partition(THINK_CLOSE_TAG)
    return content.strip(), (reasoning.strip() or None)


def prompt_starts_in_thinking(prompt: str) -> bool:
    return bool(re.search(r"<think>\s*$", prompt))


def consume_stream_piece(window: str, mode: str, piece: str):
    outputs = []
    holdback = max(len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG))
    window += piece
    while True:
        if mode == "reasoning":
            idx = window.find(THINK_CLOSE_TAG)
            if idx != -1:
                pre = window[:idx]
                if pre:
                    outputs.append(("reasoning_content", pre))
                window = window[idx + len(THINK_CLOSE_TAG):]
                mode = "content"
                continue
            if len(window) > holdback:
                safe = window[:-holdback]
                if safe:
                    outputs.append(("reasoning_content", safe))
                window = window[-holdback:]
            break

        idx = window.find(THINK_OPEN_TAG)
        if idx != -1:
            pre = window[:idx]
            if pre:
                outputs.append(("content", pre))
            window = window[idx + len(THINK_OPEN_TAG):]
            mode = "reasoning"
            continue
        if len(window) > holdback:
            safe = window[:-holdback]
            if safe:
                outputs.append(("content", safe))
            window = window[-holdback:]
        break

    return outputs, window, mode


def flush_stream_deltas(window: str, mode: str):
    if not window:
        return []
    kind = "reasoning_content" if mode == "reasoning" else "content"
    return [(kind, window)]


# FIX 2: _content_to_str helper used for BOTH OpenAI and Anthropic message
# content fields (str | list[dict]). Previously OpenAI list[dict] content
# was passed raw to the tokenizer and caused a crash.
def _content_to_str(content: "str | list[dict]") -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class ChatMessage(BaseModel):
    role: str
    # FIX 2 cont: accept list[dict] in the model but always stringify it
    content: str | list[dict]


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    temperature: float | None = None   # 0 = greedy, >0 = sample
    seed: int | None = None             # rng seed for sampling
    top_p: float | None = None         # nucleus, applied when temperature > 0
    top_k: int | None = None           # top-k, applied when temperature > 0
    frequency_penalty: float | None = None  # OAI -> rep_pen = 1 + freq_pen (sampling only)
    stop: list[str] | str | None = None  # FIX 3: accept stop field (Open WebUI sends it)
    chat_template_kwargs: dict | None = None


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict]


class AnthropicMessagesRequest(BaseModel):
    model: str = MODEL_NAME
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    stop_sequences: list[str] | None = None
    chat_template_kwargs: dict | None = None


def _samp_suffix(req) -> str:
    # Render ` samp=temp,top_p,top_k,rep_pen[,seed]` tail when the request asks for
    # non-greedy decoding. Empty string keeps the daemon protocol greedy-compatible.
    t  = float(getattr(req, "temperature", 0.0) or 0.0)
    if t <= 0.0:
        return ""
    tp = float(getattr(req, "top_p", 1.0) or 1.0)
    tk = int(getattr(req, "top_k", 0) or 0)
    rp = float(getattr(req, "frequency_penalty", 0.0) or 0.0) + 1.0
    seed = int(getattr(req, "seed", 0) or 0)
    return f" samp={t:.4f},{tp:.4f},{tk},{rp:.4f},{seed}"


def build_app(target: Path, draft: Path | None, bin_path: Path, budget: int, max_ctx: int,
              tokenizer: AutoTokenizer, stop_ids: set[int],
              prefill_cfg: PrefillConfig | None = None,
              drafter_tokenizer: AutoTokenizer | None = None,
              prefix_cache_slots: int = 4,
              prefill_cache_slots: int = 4,
              arch: str = "qwen35",
              use_pflash: bool = False) -> FastAPI:
    import asyncio
    app = FastAPI(title="Luce DFlash OpenAI server")

    # FIX 1: CORS middleware so Open WebUI / browser frontends on other ports
    # can reach this server without being blocked by the browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    daemon_lock = asyncio.Lock()

    r_pipe, w_pipe = os.pipe()
    if sys.platform == "win32":
        import msvcrt
        os.set_inheritable(w_pipe, True)
        stream_fd_val = int(msvcrt.get_osfhandle(w_pipe))
    else:
        stream_fd_val = w_pipe

    bin_abs = str(Path(bin_path).resolve())
    dll_dir = str(Path(bin_abs).parent / "bin")
    env = {**os.environ}
    if sys.platform == "win32":
        env["PATH"] = dll_dir + os.pathsep + str(Path(bin_abs).parent) + os.pathsep + env.get("PATH", "")

    if arch == "gemma4":
        # Gemma4 binary uses named flags (--model, --draft) instead of positional args.
        # draft is the safetensors directory, not a resolved file.
        cmd = [bin_abs,
               "--model", str(target),
               "--draft", str(draft),
               "--daemon",
               "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={stream_fd_val}"]
        if use_pflash:
            cmd.append("--pflash")
    elif arch in _LAGUNA_ARCHES:
        # test_dflash detects arch=laguna from the GGUF and dispatches
        # internally to dflash27b::run_laguna_daemon(). No --draft, no
        # --fast-rollback, no --ddtree (no Laguna spec-decode draft yet).
        # Tokens stream as int32 LE on stream_fd terminated by -1, byte-
        # identical to the qwen35 path so SSE/stream consumers stay shared.
        cmd = [bin_abs, str(target), "--daemon",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={stream_fd_val}"]
    else:
        if draft is None:
            raise SystemExit("qwen35 arch requires --draft model.safetensors")
        cmd = [bin_abs, str(target), str(draft), "--daemon",
               "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={stream_fd_val}"]
    if sys.platform == "win32":
        daemon_proc = subprocess.Popen(cmd, close_fds=False, env=env,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, bufsize=0)
    else:
        daemon_proc = subprocess.Popen(cmd, pass_fds=(w_pipe,), env=env,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, bufsize=0)
    os.close(w_pipe)

    bus = DaemonStdoutBus(daemon_proc.stdout)

    def _resolve_kv_k_type():
        kv = "q8_0"
        if os.environ.get("DFLASH27B_KV_F16", "0") != "0":
            kv = "f16"
        if os.environ.get("DFLASH27B_KV_Q4", "0") != "0":
            kv = "q4_0"
        if os.environ.get("DFLASH27B_KV_TQ3", "0") != "0":
            kv = "tq3_0"
        if os.environ.get("DFLASH27B_KV_K"):
            kv = os.environ["DFLASH27B_KV_K"].lower()
        return kv

    _fa_window = int(os.environ.get("DFLASH27B_FA_WINDOW", 2048))
    prefix_cache = PrefixCache(
        daemon_stdin=daemon_proc.stdin,
        await_reply=bus.await_reply,
        daemon_lock=daemon_lock,
        tokenizer=tokenizer,
        kv_k_type=_resolve_kv_k_type(),
        fa_window=_fa_window,
        cap=prefix_cache_slots,
    )
    if prefill_cfg is not None and prefill_cache_slots > 0:
        prefix_cache.init_full_cache(prefill_cache_slots)

    @app.on_event("startup")
    async def _startup():
        bus.start(asyncio.get_running_loop())
        await prefix_cache.startup_sync()

    # FIX 4: /health endpoint — Open WebUI and many clients ping this before
    # sending requests. Without it they show a permanent "disconnected" badge.
    @app.get("/health")
    def health():
        alive = daemon_proc.poll() is None
        if not alive:
            return JSONResponse({"status": "error", "detail": "daemon exited"}, status_code=503)
        return {"status": "ok"}

    # FIX 5: richer /v1/models response — Open WebUI uses `context_length` and
    # `created` to populate the model picker and context-bar correctly.
    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "luce",
                "created": 1700000000,
                "context_length": max_ctx,          # shown in Open WebUI header
                "max_context_length": max_ctx,
            }],
        }

    def _ids_to_bin(ids: list[int]) -> Path:
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return Path(path)

    def _render_messages(msgs_list: list[dict],
                         template_kwargs: dict | None = None
                         ) -> tuple[Path, list[int], str]:
        """Apply chat template to msgs_list and return (bin path, ids, raw prompt).

        The raw prompt is returned for spec-prefill: when compression fires we
        re-tokenise it with the drafter vocab.

        ``template_kwargs`` is passed through to ``apply_chat_template`` so callers
        can toggle template knobs like ``enable_thinking`` per-request.

        Thinking is disabled by default (enable_thinking=False) because Qwen3.6's
        think mode wrecks DFlash acceptance rates. Clients can opt in by sending
        ``"chat_template_kwargs": {"enable_thinking": true}`` in the request.
        """
        tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True,
                            "enable_thinking": False}
        tpl_kwargs.update(
            {k: v for k, v in (template_kwargs or {}).items() if k in _ALLOWED_TEMPLATE_KWARGS}
        )
        prompt = tokenizer.apply_chat_template(msgs_list, **tpl_kwargs)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        return _ids_to_bin(ids), ids, prompt

    def _thinking_enabled(kwargs: dict | None) -> bool:
        if kwargs:
            return kwargs.get("enable_thinking", True)
        return True

    # FIX 2 applied: always call _content_to_str on message content
    def _tokenize_prompt(req: ChatRequest) -> tuple[Path, list[int], list[dict], bool]:
        msgs = [{"role": m.role, "content": _content_to_str(m.content)}
                for m in req.messages]
        path, ids, prompt = _render_messages(msgs, req.chat_template_kwargs)
        think = _thinking_enabled(req.chat_template_kwargs) and prompt_starts_in_thinking(prompt)
        return path, ids, msgs, think

    def _maybe_compress(msgs: list[dict], prompt_bin: Path, prompt_ids: list[int],
                        template_kwargs: dict | None = None
                        ) -> tuple[Path, list[int]]:
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_ids
        if not prefill_cfg.should_compress(len(prompt_ids)):
            return prompt_bin, prompt_ids
        if drafter_tokenizer is None:
            return prompt_bin, prompt_ids

        last_user_idx = next((i for i in range(len(msgs) - 1, -1, -1)
                              if msgs[i]["role"] == "user"), None)
        if last_user_idx is None:
            return prompt_bin, prompt_ids
        long_text = msgs[last_user_idx]["content"]

        compressed_text = compress_text_via_daemon(
            daemon_stdin=daemon_proc.stdin,
            r_pipe=r_pipe,
            drafter_tokenizer=drafter_tokenizer,
            cfg=prefill_cfg,
            prompt_text=long_text,
            skip_park=prefill_cfg.skip_park,
        )

        new_msgs = list(msgs)
        new_msgs[last_user_idx] = {"role": "user", "content": compressed_text}
        new_bin, new_ids, _ = _render_messages(new_msgs, template_kwargs)
        try:
            prompt_bin.unlink()
        except Exception:
            pass
        return new_bin, new_ids

    def _token_stream(r, n_gen):
        generated = 0
        hit_stop = False
        while True:
            b = os.read(r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    # FIX 6: _collect_tokens_sync — non-streaming paths previously called
    # list(_token_stream(...)) directly (blocking the event loop) or used
    # an async comprehension over _astream_tokens inside daemon_lock
    # (risking a deadlock if the threadpool stalled). Using run_in_executor
    # offloads the blocking os.read loop to a thread without holding any
    # asyncio primitive across the thread boundary.
    async def _collect_tokens_sync(r, n_gen) -> list[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: list(_token_stream(r, n_gen)))

    async def _astream_tokens(r, n_gen):
        generated = 0
        hit_stop = False
        loop = asyncio.get_running_loop()
        while True:
            b = await loop.run_in_executor(None, os.read, r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    # FIX 7: _write_cmd helper — centralises stdin write+flush and guards
    # against a dead daemon so callers get a clean 503 instead of a hang.
    def _write_cmd(cmd_line: str):
        if daemon_proc.poll() is not None:
            raise RuntimeError("dflash daemon has exited unexpectedly")
        daemon_proc.stdin.write(cmd_line.encode("utf-8"))
        daemon_proc.stdin.flush()

    def _build_cmd_line(req, cur_bin, cur_ids, gen_len, prefix_cache,
                        prompt_ids, full_snap_prep_ref: list,
                        compression_fired: bool):
        """
        FIX 8: extracted cmd_line construction so both streaming and
        non-streaming paths share identical logic and can't diverge.
        Returns (cmd_line, snap_prep).
        full_snap_prep_ref is a 1-element list used as an out-param.
        """
        if compression_fired:
            full_snap_prep = prefix_cache.prepare_full_snap(prompt_ids)
            full_snap_prep_ref[0] = full_snap_prep
            samp = _samp_suffix(req)
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                return f"{cur_bin} {gen_len} snap={len(cur_ids)}:{fslot}" + samp + "\n", None
            else:
                return f"{cur_bin} {gen_len}" + samp + "\n", None
        else:
            full_snap_prep_ref[0] = None
            hit = prefix_cache.lookup(cur_ids)
            snap_prep = prefix_cache.prepare_inline_snap(cur_ids)
            if hit:
                slot, _prefix_len = hit
                cmd_line = f"RESTORE {slot} {cur_bin} {gen_len}"
            else:
                cmd_line = f"{cur_bin} {gen_len}"
            if snap_prep:
                cmd_line += f" snap={snap_prep[1]}:{snap_prep[0]}"
            return cmd_line + _samp_suffix(req) + "\n", snap_prep

    def _gen_len_for(prompt_len: int, max_tokens: int) -> int:
        return min(max_tokens, max_ctx - prompt_len - 20)

    # ── /v1/chat/completions ────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_bin, prompt_ids, raw_msgs, started_in_thinking = _tokenize_prompt(req)
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    full_snap_prep_ref = [None]
                    snap_prep = None

                    full_hit = prefix_cache.lookup_full(prompt_ids)
                    if full_hit is not None:
                        slot, cached_cur_bin, cached_cur_ids_len = full_hit
                        cur_bin = Path(cached_cur_bin)
                        prompt_len = cached_cur_ids_len
                        gen_len = _gen_len_for(prompt_len, req.max_tokens)
                        if gen_len <= 0:
                            try: prompt_bin.unlink()
                            except Exception: pass
                            err = {"id": completion_id, "object": "chat.completion.chunk",
                                   "created": created, "model": MODEL_NAME,
                                   "choices": [{"index": 0, "delta": {},
                                                "finish_reason": "length"}]}
                            yield f"data: {json.dumps(err)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(req) + "\n"
                    else:
                        cur_bin, cur_ids = await asyncio.to_thread(
                            _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                        prompt_len = len(cur_ids)
                        gen_len = _gen_len_for(prompt_len, req.max_tokens)
                        if gen_len <= 0:
                            try: cur_bin.unlink()
                            except Exception: pass
                            err = {"id": completion_id, "object": "chat.completion.chunk",
                                   "created": created, "model": MODEL_NAME,
                                   "choices": [{"index": 0, "delta": {},
                                                "finish_reason": "length"}]}
                            yield f"data: {json.dumps(err)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        compression_fired = (cur_bin != prompt_bin)
                        cmd_line, snap_prep = _build_cmd_line(
                            req, cur_bin, cur_ids, gen_len, prefix_cache,
                            prompt_ids, full_snap_prep_ref, compression_fired)

                    # FIX 7: guard against dead daemon
                    try:
                        _write_cmd(cmd_line)
                    except RuntimeError as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    head = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0,
                                     "delta": {"role": "assistant"},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(head)}\n\n"
                    window, mode = "", ("reasoning" if started_in_thinking else "content")

                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            outputs, window, mode = consume_stream_piece(
                                window, mode, tokenizer.decode([tok_id]))
                            for kind, text in outputs:
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created, "model": MODEL_NAME,
                                    "choices": [{"index": 0,
                                                 "delta": {kind: text},
                                                 "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                        for kind, text in flush_stream_deltas(window, mode):
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created, "model": MODEL_NAME,
                                "choices": [{"index": 0,
                                             "delta": {kind: text},
                                             "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    finally:
                        if full_hit is None:
                            try: cur_bin.unlink()
                            except Exception: pass
                        else:
                            try: prompt_bin.unlink()
                            except Exception: pass

                    full_snap_prep = full_snap_prep_ref[0]
                    if full_snap_prep is not None:
                        fslot, _ = full_snap_prep
                        prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
                    elif snap_prep:
                        prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

                    tail = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {},
                                     "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(tail)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        async with daemon_lock:
            full_snap_prep_ref = [None]
            snap_prep = None

            full_hit = prefix_cache.lookup_full(prompt_ids)
            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
                gen_len = _gen_len_for(prompt_len, req.max_tokens)
                if gen_len <= 0:
                    try: prompt_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                        status_code=400)
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(req) + "\n"
            else:
                cur_bin, cur_ids = await asyncio.to_thread(
                    _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                prompt_len = len(cur_ids)
                gen_len = _gen_len_for(prompt_len, req.max_tokens)
                if gen_len <= 0:
                    try: cur_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                        status_code=400)
                compression_fired = (cur_bin != prompt_bin)
                cmd_line, snap_prep = _build_cmd_line(
                    req, cur_bin, cur_ids, gen_len, prefix_cache,
                    prompt_ids, full_snap_prep_ref, compression_fired)

            try:
                _write_cmd(cmd_line)
            except RuntimeError as e:
                return JSONResponse({"detail": str(e)}, status_code=503)

            # FIX 6: use run_in_executor instead of list() blocking event loop
            tokens = await _collect_tokens_sync(r_pipe, gen_len)

            full_snap_prep = full_snap_prep_ref[0]
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
            elif snap_prep:
                prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            try: prompt_bin.unlink()
            except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        cleaned, reasoning = parse_reasoning(
            text,
            thinking_enabled=_thinking_enabled(req.chat_template_kwargs),
            started_in_thinking=started_in_thinking,
        )
        msg = {"role": "assistant", "content": cleaned}
        if reasoning:
            msg["reasoning_content"] = reasoning
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": msg,
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": prompt_len,
                      "completion_tokens": len(tokens),
                      "total_tokens": prompt_len + len(tokens)},
        })

    # ── Anthropic Messages API ──────────────────────────────────────────────

    def _tokenize_anthropic(req: AnthropicMessagesRequest
                            ) -> tuple[Path, list[int], list[dict], bool]:
        msgs = []
        system_text = _content_to_str(req.system) if req.system else None
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        for m in req.messages:
            msgs.append({"role": m.role, "content": _content_to_str(m.content)})
        path, ids, prompt = _render_messages(msgs, req.chat_template_kwargs)
        think = _thinking_enabled(req.chat_template_kwargs) and prompt_starts_in_thinking(prompt)
        return path, ids, msgs, think

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest):
        prompt_bin, prompt_ids, raw_msgs, started_in_thinking = _tokenize_anthropic(req)
        msg_id = "msg_" + uuid.uuid4().hex[:24]

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    full_snap_prep_ref = [None]
                    snap_prep = None

                    full_hit = prefix_cache.lookup_full(prompt_ids)
                    if full_hit is not None:
                        slot, cached_cur_bin, cached_cur_ids_len = full_hit
                        cur_bin = Path(cached_cur_bin)
                        cur_ids = None
                        prompt_len = cached_cur_ids_len
                        gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                        if gen_len <= 0:
                            try: prompt_bin.unlink()
                            except Exception: pass
                            err = {"type": "error",
                                   "error": {"type": "invalid_request_error",
                                             "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}}
                            yield f"event: error\ndata: {json.dumps(err)}\n\n"
                            return
                        cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(req) + "\n"
                    else:
                        cur_bin, cur_ids = await asyncio.to_thread(
                            _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                        prompt_len = len(cur_ids)
                        gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                        if gen_len <= 0:
                            try: cur_bin.unlink()
                            except Exception: pass
                            err = {"type": "error",
                                   "error": {"type": "invalid_request_error",
                                             "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}}
                            yield f"event: error\ndata: {json.dumps(err)}\n\n"
                            return
                        compression_fired = (cur_bin != prompt_bin)
                        cmd_line, snap_prep = _build_cmd_line(
                            req, cur_bin, cur_ids, gen_len, prefix_cache,
                            prompt_ids, full_snap_prep_ref, compression_fired)

                    message_start = {
                        "type": "message_start",
                        "message": {
                            "id": msg_id, "type": "message", "role": "assistant",
                            "model": req.model or MODEL_NAME,
                            "content": [], "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": prompt_len, "output_tokens": 0},
                        },
                    }
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

                    try:
                        _write_cmd(cmd_line)
                    except RuntimeError as e:
                        yield f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'server_error','message':str(e)}})}\n\n"
                        return

                    out_tokens = 0
                    window, mode = "", ("reasoning" if started_in_thinking else "content")
                    block_index = 0
                    active_kind = "thinking" if mode == "reasoning" else "text"
                    block = {"type": active_kind}
                    if active_kind == "thinking":
                        block["thinking"] = ""
                    else:
                        block["text"] = ""
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': block})}\n\n"
                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            out_tokens += 1
                            outputs, window, mode = consume_stream_piece(
                                window, mode, tokenizer.decode([tok_id]))
                            for kind, text in outputs:
                                target_kind = "thinking" if kind == "reasoning_content" else "text"
                                if target_kind != active_kind:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    block_index += 1
                                    active_kind = target_kind
                                    new_block = {"type": active_kind, active_kind: ""}
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': new_block})}\n\n"
                                delta_type = "thinking_delta" if target_kind == "thinking" else "text_delta"
                                delta_key = "thinking" if target_kind == "thinking" else "text"
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': delta_type, delta_key: text}})}\n\n"
                        for kind, text in flush_stream_deltas(window, mode):
                            target_kind = "thinking" if kind == "reasoning_content" else "text"
                            if target_kind != active_kind:
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                block_index += 1
                                active_kind = target_kind
                                new_block = {"type": active_kind, active_kind: ""}
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': new_block})}\n\n"
                            delta_type = "thinking_delta" if target_kind == "thinking" else "text_delta"
                            delta_key = "thinking" if target_kind == "thinking" else "text"
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': delta_type, delta_key: text}})}\n\n"
                    finally:
                        if full_hit is None:
                            try: cur_bin.unlink()
                            except Exception: pass
                        else:
                            try: prompt_bin.unlink()
                            except Exception: pass

                    full_snap_prep = full_snap_prep_ref[0]
                    if full_snap_prep is not None:
                        fslot, _ = full_snap_prep
                        prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
                    elif snap_prep:
                        prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                    msg_delta = {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": out_tokens},
                    }
                    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming Anthropic
        async with daemon_lock:
            full_snap_prep_ref = [None]
            snap_prep = None

            full_hit = prefix_cache.lookup_full(prompt_ids)
            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
                gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                if gen_len <= 0:
                    try: prompt_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"type": "error", "error": {"type": "invalid_request_error",
                         "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                        status_code=400)
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(req) + "\n"
            else:
                cur_bin, cur_ids = await asyncio.to_thread(
                    _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                prompt_len = len(cur_ids)
                gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                if gen_len <= 0:
                    try: cur_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"type": "error", "error": {"type": "invalid_request_error",
                         "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                        status_code=400)
                compression_fired = (cur_bin != prompt_bin)
                cmd_line, snap_prep = _build_cmd_line(
                    req, cur_bin, cur_ids, gen_len, prefix_cache,
                    prompt_ids, full_snap_prep_ref, compression_fired)

            try:
                _write_cmd(cmd_line)
            except RuntimeError as e:
                return JSONResponse({"type": "error", "error": {"type": "server_error",
                                     "message": str(e)}}, status_code=503)

            # FIX 6: use run_in_executor — same fix as OpenAI non-streaming path
            tokens = await _collect_tokens_sync(r_pipe, gen_len)

            full_snap_prep = full_snap_prep_ref[0]
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
            elif snap_prep:
                prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            try: prompt_bin.unlink()
            except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        cleaned, reasoning = parse_reasoning(
            text,
            thinking_enabled=_thinking_enabled(req.chat_template_kwargs),
            started_in_thinking=started_in_thinking,
        )
        content = [{"type": "text", "text": cleaned}]
        if reasoning:
            content.insert(0, {"type": "thinking", "thinking": reasoning})
        return JSONResponse({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": req.model or MODEL_NAME,
            "content": content,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_len,
                      "output_tokens": len(tokens)},
        })

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    default_ctx = 16384
    ap.add_argument("--max-ctx", type=int, default=default_ctx,
                    help=f"Maximum context length (default: {default_ctx}; "
                         "oversizing this — e.g. 131072 on short prompts — "
                         "can slow attention 20×+ until issue #10 is fixed)")
    ap.add_argument("--kv-f16", action="store_true",
                    help="Force F16 KV cache.")
    ap.add_argument("--cache-type-k", "--ctk", dest="cache_type_k", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"])
    ap.add_argument("--cache-type-v", "--ctv", dest="cache_type_v", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"])
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers. 0 = full attention.")
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--prefix-cache-slots", type=int, default=4)
    ap.add_argument("--prefill-cache-slots", type=int, default=4)
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--pflash", action="store_true",
                    help="Enable pFlash sparse-attention prefill in the daemon binary "
                         "(Gemma4 only; no-op for Qwen3).")
    add_cli_flags(ap)
    args = ap.parse_args()
    prefill_cfg = config_from_args(args)

    if args.cache_type_k:
        os.environ["DFLASH27B_KV_K"] = args.cache_type_k
    if args.cache_type_v:
        os.environ["DFLASH27B_KV_V"] = args.cache_type_v
    if args.max_ctx > 6144 and not args.kv_f16 and not args.cache_type_k and not args.cache_type_v:
        os.environ.setdefault("DFLASH27B_KV_TQ3", "1")

    if args.fa_window is not None:
        os.environ["DFLASH27B_FA_WINDOW"] = str(args.fa_window)

    if args.prefill_compression != "off":
        os.environ.setdefault("DFLASH27B_LM_HEAD_FIX", "0")
        os.environ.setdefault("DFLASH27B_FA_WINDOW", "0")
        os.environ.setdefault("DFLASH_FP_USE_BSA", "1")
        os.environ.setdefault("DFLASH_FP_ALPHA",   "0.85")
        if prefill_cfg.skip_park:
            os.environ["DFLASH_COMPRESS_NO_PARK"] = "1"

    if not args.target.is_file():
        raise SystemExit(f"target GGUF not found at {args.target}")

    # Detect architecture and select the right binary.
    # test_dflash itself dispatches qwen35/laguna by GGUF arch at main() entry,
    # but Gemma4 lives in a separate binary (test_gemma4_dflash) so we route
    # explicitly here.
    arch = _read_gguf_architecture(args.target)
    is_gemma4 = (arch == "gemma4")

    if args.bin != DEFAULT_BIN:
        # User explicitly specified a binary — use it as-is.
        bin_path = args.bin
    elif is_gemma4:
        bin_path = ROOT / "build" / ("test_gemma4_dflash" + (".exe" if sys.platform == "win32" else ""))
        print(f"[server] detected architecture=gemma4, using binary: {bin_path}")
    else:
        bin_path = DEFAULT_BIN

    if not bin_path.is_file():
        raise SystemExit(f"binary not found at {bin_path} (arch={arch})")

    if is_gemma4:
        # Gemma4 draft is a directory (safetensors dir), not a resolved file.
        if args.draft.is_dir():
            draft = args.draft
        elif args.draft.is_file():
            # User passed a file path inside the draft directory; use its parent.
            draft = args.draft.parent
            print(f"[server] note: --draft {args.draft} is a file; using parent {draft}", file=sys.stderr)
        else:
            raise SystemExit(f"draft path not found or not a directory: {args.draft}")
        if not draft.is_dir():
            raise SystemExit(f"draft directory not found: {draft} (from {args.draft})")
    elif arch in _LAGUNA_ARCHES:
        # No DFlash draft model exists for laguna yet; test_dflash's internal
        # arch dispatch reads general.architecture, accepts the no-draft argv
        # layout, and routes to run_laguna_daemon(). PFlash compression and
        # prefix-cache SNAPSHOT/RESTORE are both wired through the laguna
        # daemon now.
        draft = None
    else:
        draft = resolve_draft(args.draft) if args.draft.is_dir() else args.draft
        if not draft.is_file():
            raise SystemExit(f"draft safetensors not found at {args.draft}")

    tokenizer_id = args.tokenizer or _tokenizer_id_from_gguf(args.target)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    drafter_tokenizer = None
    if prefill_cfg.enabled:
        drafter_tokenizer = AutoTokenizer.from_pretrained(
            prefill_cfg.drafter_tokenizer_id, trust_remote_code=True)

    app = build_app(args.target, draft, bin_path, args.budget, args.max_ctx,
                    tokenizer, stop_ids,
                    prefill_cfg=prefill_cfg if prefill_cfg.enabled else None,
                    drafter_tokenizer=drafter_tokenizer,
                    prefix_cache_slots=args.prefix_cache_slots,
                    prefill_cache_slots=args.prefill_cache_slots,
                    arch=arch,
                    use_pflash=getattr(args, "pflash", False))

    import uvicorn
    print(f"Luce DFlash OpenAI server on http://{args.host}:{args.port}")
    print(f"  arch      = {arch}")
    print(f"  target    = {args.target}")
    print(f"  arch      = {arch or '(unknown)'}")
    print(f"  draft     = {draft}")
    print(f"  bin       = {bin_path}")
    print(f"  budget    = {args.budget}")
    print(f"  max_ctx   = {args.max_ctx}")
    print(f"  tokenizer = {tokenizer_id}")
    if prefill_cfg.enabled:
        print(f"  pflash    = {prefill_cfg.mode} · threshold={prefill_cfg.threshold} "
              f"keep={prefill_cfg.keep_ratio} drafter={prefill_cfg.drafter_gguf}")
    else:
        print("  pflash    = off")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
