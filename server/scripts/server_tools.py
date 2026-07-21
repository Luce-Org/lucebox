"""
OpenAI-compatible HTTP server on top of test_dflash, **with tool-calling support**.

Patched fork of scripts/server.py that:
  1. Accepts the OpenAI `tools` array in ChatRequest.
  2. Renders tools into the prompt via Qwen's chat template (`tools=...`).
  3. Parses `<tool_call><function=...><parameter=...></tool_call>` blocks out
     of the model output and returns them as proper OpenAI `tool_calls`.
  4. Supports `role: "tool"` and assistant `tool_calls` in input messages so
     multi-turn agent loops round-trip correctly.

Streaming behavior:
  - Content tokens are streamed as `delta.content` until a `<tool_call>` opener
    is detected; the rest of the response is then buffered, parsed at the end
    of generation, and emitted as a single final `delta.tool_calls` chunk with
    `finish_reason: "tool_calls"`.
  - If no tool call appears in the output, behavior is identical to the
    upstream server.

Greedy decoding still applies (verify path is greedy-only). `temperature` and
`top_p` are accepted but ignored, matching upstream.

When ``tools`` is non-empty, ``enable_thinking`` is forced off in the chat template
(Qwen3 thinking prefill otherwise dominates and the model rarely emits ``<tool_call>``).

Run:
  pip install fastapi uvicorn transformers
  python3 scripts/server_tools.py --port 8000
"""
import argparse
import asyncio
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
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from _prefill_hook import (
    PrefillConfig, add_cli_flags, config_from_args,
    compress_text_via_daemon,
)
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import iterate_in_threadpool
from transformers import AutoTokenizer

from prefix_cache import DaemonStdoutBus, PrefixCache
from tool_split.config import ToolSplitConfig, add_cli_flags as add_tool_split_flags
from tool_split.config import config_from_env_and_args as tool_split_config_from_args
from tool_split.base import ToolRequestContext
from tool_split.daemon_bridge import commit_pending_tool_snap
from tool_split.orchestrator import ToolSplitOrchestrator
from tool_split.registry import resolve_adapter as resolve_tool_split_adapter


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = Path(os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
))
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / ("test_dflash" + (".exe" if sys.platform == "win32" else ""))
DEFAULT_BUDGET = 22
MODEL_NAME = "luce-dflash"

# Passed through to apply_chat_template only (see server.py — avoid arbitrary kwargs).
_ALLOWED_TEMPLATE_KWARGS = frozenset({"enable_thinking", "add_generation_prompt", "tools"})


def _extra_daemon_has_target_sharding(extra: list[str] | None) -> bool:
    """True if we spawn test_dflash with multi-GPU target layer split."""
    if not extra:
        return False
    return any(tok.startswith("--target-gpus") for tok in extra)


# Architecture strings in `general.architecture` of the GGUF (see server.py).
_QWEN35_ARCHES = {"qwen35", "qwen36"}
_LAGUNA_ARCHES = {"laguna"}

_QWEN35_FAMILY_TOKENIZERS = {
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
}
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


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


# ─── pydantic schemas ──────────────────────────────────────────────

class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI spec


class ToolCall(BaseModel):
    id: str | None = None
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: Any | None = None  # str, list, or null when tool_calls present
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolDef(BaseModel):
    type: str = "function"
    function: dict  # {name, description, parameters: {...JSON schema...}}


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    # OpenAI-compatible alias (newer clients send this instead of max_tokens).
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ToolDef] | None = None
    tool_choice: Any | None = None  # "auto" | "none" | {"type":"function",...} | {"type":"required"}
    chat_template_kwargs: dict | None = None  # e.g. {"enable_thinking": false}
    stop: Any | None = None  # str or list[str]
    stream_options: dict | None = None  # e.g. {"include_usage": true}


class AnthropicMessage(BaseModel):
    role: str
    # Anthropic allows either a plain string or a list of content blocks.
    content: str | list[dict]


class AnthropicMessagesRequest(BaseModel):
    model: str = MODEL_NAME
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None


# ─── tool-call parser ──────────────────────────────────────────────

# Qwen3.6 chat template emits:
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>
#   VALUE
#   </parameter>
#   ...
#   </function>
#   </tool_call>
# Parsers ported from vLLM (Apache-2.0) for behavioral parity with
# `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`:
#   vllm/reasoning/qwen3_reasoning_parser.py
#   vllm/tool_parsers/qwen3coder_tool_parser.py
# Core algorithms reproduced without vLLM runtime dependencies.

TOOL_CALL_COMPLETE_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_CALL_FUNCTION_RE = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL,
)
# vLLM's improved parameter regex: tolerates unclosed </parameter> by using
# next <parameter= or </function> or end-of-string as a terminator.
TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)
TOOL_OPEN_TAG = "<tool_call>"

# Qwen3.6 chat template wraps the model's CoT inside <think>...</think>.
# The template typically prefills `<think>\n` into the prompt (headless mode)
# so only `</think>` appears in generated output; older templates emit both.
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


def normalize_stop(stop) -> list[str]:
    """Coerce OpenAI's stop field (str | list[str] | None) to list[str]."""
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if isinstance(s, str) and s]


def first_stop_match(text: str, stops: list[str]) -> int:
    """Return the earliest index where any stop sequence appears, or -1."""
    best = -1
    for s in stops:
        i = text.find(s)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def trim_at_stop(text: str, stops: list[str]) -> tuple[str, str | None]:
    """Trim *text* at the first stop sequence. Returns (trimmed, matched_stop)."""
    if not stops:
        return text, None
    best_i = -1
    matched: str | None = None
    for s in stops:
        i = text.find(s)
        if i != -1 and (best_i == -1 or i < best_i):
            best_i = i
            matched = s
    if best_i == -1:
        return text, None
    return text[:best_i], matched


def parse_reasoning(text: str, thinking_enabled: bool = True) -> tuple[str, str | None]:
    """Port of vLLM's Qwen3ReasoningParser.extract_reasoning.

    Handles the three Qwen3.x thinking flavors:
      1. Paired:   `<think>...</think>` both in generated output.
      2. Headless: template prefilled `<think>\\n` into the prompt, model
         only emits `...</think>...`.
      3. Disabled: user passed `chat_template_kwargs: {enable_thinking: false}`.
         Template still emits `<think>\\n\\n</think>\\n\\n` but into the prompt;
         the model output is pure content and contains no tags.

    If the output was truncated mid-thinking (no `</think>` seen and
    `thinking_enabled=True`), returns `("", full_output_as_reasoning)` —
    matching vLLM's convention.

    Returns (cleaned_content, reasoning_content).
    """
    # Strip <think> if the model emitted it itself (older templates).
    parts = text.partition(THINK_OPEN_TAG)
    rest = parts[2] if parts[1] else parts[0]
    if THINK_CLOSE_TAG not in rest:
        if thinking_enabled:
            # No close tag — assume truncated; everything is reasoning.
            return "", (rest.strip() or None)
        else:
            # Thinking disabled — output is pure content.
            return rest.strip(), None
    reasoning, _, content = rest.partition(THINK_CLOSE_TAG)
    return content.strip(), (reasoning.strip() or None)


def _find_tool_properties(tools, function_name):
    """Helper matching vLLM's `find_tool_properties`: returns the parameters
    dict for a given function name, or {} if not found.
    Accepts pydantic ToolDef instances or plain dicts.
    """
    for t in tools or []:
        fn = t.function if hasattr(t, "function") else t.get("function", {})
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump()
        if fn.get("name") == function_name:
            params = fn.get("parameters", {})
            if isinstance(params, dict):
                return params.get("properties", {})
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict,
                         func_name: str):
    """Port of vLLM's _convert_param_value. Coerces stringified XML values
    to their JSON-schema type (int/float/bool/object/array/string)."""
    import ast
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value
    cfg = param_config[param_name]
    if isinstance(cfg, dict) and "type" in cfg:
        ptype = str(cfg["type"]).strip().lower()
    elif isinstance(cfg, dict) and "anyOf" in cfg:
        ptype = "object"
    else:
        ptype = "string"
    if ptype in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value
    if any(ptype.startswith(p) for p in ("int", "uint", "long", "short", "unsigned")):
        try: return int(param_value)
        except (ValueError, TypeError): return param_value
    if ptype.startswith("num") or ptype.startswith("float"):
        try:
            f = float(param_value)
            return f if f - int(f) != 0 else int(f)
        except (ValueError, TypeError):
            return param_value
    if ptype in ("boolean", "bool", "binary"):
        return param_value.lower() == "true"
    # object / array / dict / list
    if (ptype in ("object", "array", "arr")
            or ptype.startswith("dict") or ptype.startswith("list")):
        try: return json.loads(param_value)
        except (json.JSONDecodeError, TypeError, ValueError): pass
    try: return ast.literal_eval(param_value)
    except (ValueError, SyntaxError, TypeError): return param_value


def parse_tool_calls(text: str, tools=None) -> tuple[str, list[dict]]:
    """Port of Qwen3CoderToolParser._parse_xml_function_call (non-streaming).

    Handles Qwen3.x's `<tool_call><function=NAME>...<parameter=KEY>VAL
    </parameter>...</function></tool_call>` XML. Uses vLLM's improved
    parameter regex that tolerates unclosed </parameter> tags. When `tools`
    is provided, each parameter value is coerced to its JSON-schema type.

    Returns (cleaned_content, tool_calls_list).
    """
    tool_calls: list[dict] = []
    cleaned_parts: list[str] = []
    cursor = 0
    for m in TOOL_CALL_COMPLETE_RE.finditer(text):
        cleaned_parts.append(text[cursor:m.start()])
        cursor = m.end()
        body = m.group(1)
        fn_match = TOOL_CALL_FUNCTION_RE.search(body)
        if not fn_match:
            continue
        fn_text = fn_match.group(1) or fn_match.group(2) or ""
        end_idx = fn_text.find(">")
        if end_idx == -1:
            continue
        function_name = fn_text[:end_idx].strip()
        params_region = fn_text[end_idx + 1:]
        param_config = _find_tool_properties(tools, function_name)
        args: dict = {}
        for match_text in TOOL_CALL_PARAMETER_RE.findall(params_region):
            eq_idx = match_text.find(">")
            if eq_idx == -1:
                continue
            k = match_text[:eq_idx].strip()
            v = match_text[eq_idx + 1:]
            if v.startswith("\n"): v = v[1:]
            if v.endswith("\n"): v = v[:-1]
            args[k] = _convert_param_value(v, k, param_config, function_name)
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), tool_calls


# ─── app ───────────────────────────────────────────────────────────

def build_app(target: Path, draft: Path | None, bin_path: Path, budget: int,
              max_ctx: int, tokenizer: AutoTokenizer, stop_ids: set[int],
              prefill_cfg: PrefillConfig | None = None,
              drafter_tokenizer: AutoTokenizer | None = None,
              prefix_cache_slots: int = 4,
              prefill_cache_slots: int = 4,
              arch: str = "qwen35",
              extra_daemon_args: list[str] | None = None,
              tool_split_cfg: ToolSplitConfig | None = None,
              tool_split: ToolSplitOrchestrator | None = None) -> FastAPI:
    import asyncio
    if _extra_daemon_has_target_sharding(extra_daemon_args):
        if prefix_cache_slots > 0 or prefill_cache_slots > 0:
            print(
                "  [cfg] target-gpus sharding: disabling prefix/full cache "
                "(daemon SNAPSHOT/RESTORE not implemented for this mode)",
                flush=True,
            )
            prefix_cache_slots = 0
            prefill_cache_slots = 0
        if tool_split is not None:
            print(
                "  [cfg] target-gpus sharding: disabling tool-split "
                "(SNAPSHOT_THIN / RESTORE_CHAIN unsupported)",
                flush=True,
            )
            tool_split = None
    app = FastAPI(title="Luce DFlash OpenAI server (tool-aware)")
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

    if arch in _LAGUNA_ARCHES:
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
        if extra_daemon_args:
            cmd.extend(extra_daemon_args)
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
    # Mirror server.py: resolve effective KV-K type + FA window from env so
    # they participate in the prefix-cache hash key.
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
    # Option 3: full-compress-result cache.  Only meaningful when pFlash
    # compression is enabled.  Uses a separate slot range [prefix_cap, ...).
    if prefill_cfg is not None and prefill_cache_slots > 0:
        prefix_cache.init_full_cache(prefill_cache_slots)

    async def _finish_inline_snap(
        snap_prep: tuple[int, int] | None,
        prompt_ids: list[int],
    ) -> None:
        if snap_prep:
            await bus.drain_inline_snap()
            prefix_cache.finish_inline_snap(
                snap_prep, prompt_ids, inline_slot=bus.inline_snap_slot())

    def _abort_full_snap_if_needed(full_snap_prep) -> None:
        if full_snap_prep is not None:
            fslot, _ = full_snap_prep
            prefix_cache.abort_full_snap(fslot)

    async def _finalize_request_snaps(
        *,
        full_snap_prep,
        snap_prep: tuple[int, int] | None,
        prompt_ids: list[int],
        cur_bin: Path,
        cur_ids: list[int] | None,
        success: bool,
    ) -> None:
        if full_snap_prep is not None:
            if success and cur_ids is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(
                    fslot, prompt_ids, cur_bin, len(cur_ids))
            else:
                _abort_full_snap_if_needed(full_snap_prep)
        elif success:
            await _finish_inline_snap(snap_prep, prompt_ids)
        elif snap_prep:
            prefix_cache.abort_inline_snap(snap_prep[0])

    def _build_usage(
        prompt_len: int,
        completion_tokens: int,
        *,
        conv_prefix_len: int | None = None,
    ) -> dict:
        usage = {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
        }
        timings: dict[str, float | int] = {}
        raw = bus.request_timings()
        for key, val in raw.items():
            if isinstance(val, (int, float)) and val > 0:
                timings[key] = val
        if conv_prefix_len is not None and conv_prefix_len > 0:
            timings["prefix_len"] = conv_prefix_len
        if timings:
            usage["timings"] = timings
        return usage

    def _compose_daemon_cmd(
        cur_bin: Path,
        gen_len: int,
        prompt_ids: list[int],
        *,
        full_hit,
        compression_fired: bool,
        full_snap_prep,
        cur_ids: list[int] | None,
        tool_ctx: ToolRequestContext | None,
    ) -> tuple[str, object | None, int | None]:
        """Build daemon stdin line, optional inline snap_prep, conv prefix len."""
        if full_hit is not None:
            slot, cached_cur_bin, cached_len = full_hit
            return f"RESTORE {slot} {cached_cur_bin} {gen_len}\n", None, cached_len
        if compression_fired:
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                return f"{cur_bin} {gen_len} snap={len(cur_ids)}:{fslot}\n", None, None
            return f"{cur_bin} {gen_len}\n", None, None

        hit = prefix_cache.lookup(prompt_ids)
        conv_prefix_len = hit[1] if hit else None
        reuse = hit[0] if hit else None
        snap_prep = prefix_cache.prepare_inline_snap(prompt_ids, reuse_slot=reuse)

        if tool_split and tool_ctx and tool_ctx.tool_slot_hit is not None:
            plan = tool_split.build_plan(
                split=tool_ctx.split,
                tools_fingerprint=tool_ctx.fingerprint,
                prompt_bin=cur_bin,
                prompt_len=len(prompt_ids),
                tool_slot_hit=tool_ctx.tool_slot_hit,
                conv_hit=hit,
                snap_prep=snap_prep,
                pending_tool_snap=tool_ctx.pending_tool_snap,
            )
            # On chain restore with a thick conv slot, always refresh that
            # slot in-place — never allocate a new prefix snapshot while thin
            # tool KV is also resident (OOM on 24GB cards).
            if (
                snap_prep
                and plan.conv_restore_slot is not None
                and plan.conv_restore_slot >= 0
            ):
                slot, cut = snap_prep
                if slot != plan.conv_restore_slot:
                    snap_prep = (plan.conv_restore_slot, cut)
            cmd = tool_split.format_daemon_command(plan, gen_len)
            print(
                f"  [tool-split] RESTORE_CHAIN thick={plan.conv_restore_slot} "
                f"thin={plan.thin_slot_ids} prompt_tokens={len(prompt_ids)}",
                flush=True,
            )
            return cmd, snap_prep, conv_prefix_len

        if hit:
            slot, _prefix_len = hit
            cmd = f"RESTORE {slot} {cur_bin} {gen_len}"
        else:
            cmd = f"{cur_bin} {gen_len}"
        if snap_prep:
            cmd += f" snap={snap_prep[1]}:{snap_prep[0]}"
        return cmd + "\n", snap_prep, conv_prefix_len

    async def _commit_tool_snap_if_needed(tool_ctx: ToolRequestContext | None) -> None:
        if not tool_split or not tool_ctx or not tool_ctx.pending_tool_snap:
            return
        if not tool_ctx.fingerprint:
            return
        slot, kv_end = tool_ctx.pending_tool_snap
        await commit_pending_tool_snap(
            orchestrator=tool_split,
            daemon_stdin=daemon_proc.stdin,
            await_reply=bus.await_reply,
            fingerprint=tool_ctx.fingerprint,
            slot=slot,
            kv_end=kv_end,
        )

    @app.on_event("startup")
    async def _startup():
        import asyncio
        bus.start(asyncio.get_running_loop())
        await prefix_cache.startup_sync()

    @app.get("/health")
    def health():
        alive = daemon_proc.poll() is None
        if not alive:
            return JSONResponse(
                {"status": "error", "detail": "daemon exited"},
                status_code=503,
            )
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "luce"}]}

    def _maybe_compress_tool_chat(req: "ChatRequest", prompt_bin: Path,
                                  prompt_len: int, started_in_thinking: bool
                                  ) -> tuple[Path, int, bool]:
        """If prefill is on and the request has no tools and the last user
        message is long, run the daemon compress + re-tokenise. Returns
        (bin, prompt_len, started_in_thinking) — the last is recomputed when
        compression fires, otherwise passed through.

        When ``tool_split`` is active and ``req.tools`` is non-empty, tool
        definitions stay in the template while only conversation text is
        eligible for PFlash (same last-user-message compress path).
        """
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_len, started_in_thinking
        tools_present = bool(req.tools)
        if tools_present and not tool_split:
            # Legacy: monolithic prompt — compressing mangles tool JSON in-stream.
            return prompt_bin, prompt_len, started_in_thinking

        compress_len = prompt_len
        if tools_present and tool_split:
            try:
                split = tool_split.split_request(
                    tokenizer, req.messages, req.tools,
                    chat_template_kwargs=req.chat_template_kwargs,
                )
                if tool_split.conversation_compressible(split):
                    compress_len = split.conversation_len
            except Exception as exc:
                print(f"  [tool-split] split failed, skipping conv compress: {exc}",
                      flush=True)

        if not prefill_cfg.should_compress(compress_len) or drafter_tokenizer is None:
            return prompt_bin, prompt_len, started_in_thinking

        last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
        if last_user is None or not isinstance(last_user.content, str):
            return prompt_bin, prompt_len, started_in_thinking

        compressed_text = compress_text_via_daemon(
            daemon_stdin=daemon_proc.stdin,
            r_pipe=r_pipe,
            drafter_tokenizer=drafter_tokenizer,
            cfg=prefill_cfg,
            prompt_text=last_user.content,
            skip_park=prefill_cfg.skip_park,
        )

        new_msgs = []
        compressed_emitted = False
        for m in req.messages:
            if m is last_user and not compressed_emitted:
                new_msgs.append({"role": "user", "content": compressed_text})
                compressed_emitted = True
            else:
                d = {"role": m.role}
                if m.content is not None:
                    d["content"] = m.content
                new_msgs.append(d)

        kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        kwargs.update(
            {k: v for k, v in (req.chat_template_kwargs or {}).items()
             if k in _ALLOWED_TEMPLATE_KWARGS},
        )
        if req.tools:
            kwargs["tools"] = [t.model_dump() for t in req.tools]
            kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(new_msgs, **kwargs)
        new_started_in_thinking = bool(re.search(r"<think>\s*$", prompt))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        try: prompt_bin.unlink()
        except Exception: pass
        return Path(path), len(ids), new_started_in_thinking

    def _tokenize_prompt(req: ChatRequest) -> tuple[Path, bool]:
        """Returns (prompt_bin_path, started_in_thinking). started_in_thinking
        is True when the chat template prefilled <think>\\n at the end of the
        prompt — the model's first emitted tokens are reasoning content."""
        # Convert pydantic messages to dicts the chat template expects.
        msgs: list[dict] = []
        for m in req.messages:
            d: dict = {"role": m.role}
            if m.content is not None:
                d["content"] = m.content
            if m.name is not None:
                d["name"] = m.name
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                # The Qwen template walks tool_calls[i].function.{name, arguments}
                d["tool_calls"] = []
                for tc in m.tool_calls:
                    args = tc.function.arguments
                    # Template expects arguments as a dict, not a JSON string.
                    if isinstance(args, str):
                        try:
                            args_obj = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args_obj = {"_raw": args}
                    else:
                        args_obj = args
                    d["tool_calls"].append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": args_obj},
                    })
            msgs.append(d)

        tools_arg = None
        if req.tools:
            # OpenAI-shaped tool defs (type + function{name,description,parameters}).
            tools_arg = [t.model_dump() for t in req.tools]

        # Mirror server.py: default enable_thinking=False. Qwen3's template default is
        # often True, which pre-fills <think> and the model rambles in
        # "reasoning" instead of emitting <tool_call> XML — looks like "ignores tools".
        kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        kwargs.update(
            {k: v for k, v in (req.chat_template_kwargs or {}).items()
             if k in _ALLOWED_TEMPLATE_KWARGS},
        )
        if tools_arg:
            kwargs["tools"] = tools_arg
            # Thinking + tool XML is a bad combo for Qwen3.x; never let merged kwargs
            # re-enable thinking while tools are mounted.
            kwargs["enable_thinking"] = False
        if req.tool_choice is not None:
            kwargs["tool_choice"] = req.tool_choice
        prompt = tokenizer.apply_chat_template(msgs, **kwargs)
        # Did the template prefill `<think>\n` at the end? Then streaming should
        # start in reasoning mode.
        started_in_thinking = bool(re.search(r"<think>\s*$", prompt))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp, started_in_thinking

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

    def _max_gen_tokens(req: ChatRequest) -> int:
        if req.max_completion_tokens is not None:
            return req.max_completion_tokens
        return req.max_tokens

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_bin, started_in_thinking = _tokenize_prompt(req)
        prompt_len = prompt_bin.stat().st_size // 4

        # Read back token ids for cache key (cheap — file is small).
        raw = prompt_bin.read_bytes()
        prompt_ids = [struct.unpack_from("<i", raw, i)[0]
                      for i in range(0, len(raw), 4)]

        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        # --- Option 3: check full-compress-result cache before compression ---
        # This avoids the drafter dance entirely when we have a cached result.
        full_hit = None
        full_snap_prep = None
        cur_bin = prompt_bin  # default: no compression
        cur_ids = prompt_ids
        compression_fired = False

        async with daemon_lock:
            allow_full_cache = (
                prefill_cfg is not None
                and prefill_cfg.enabled
                and (not req.tools or tool_split is not None)
            )
            if allow_full_cache:
                full_hit = prefix_cache.lookup_full(prompt_ids)

            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
                started_in_thinking = False  # cached result: no think prefill
            else:
                # pflash compress hook (no-op when off / has tools)
                new_bin, prompt_len, started_in_thinking = await asyncio.to_thread(
                    _maybe_compress_tool_chat, req, prompt_bin, prompt_len, started_in_thinking)
                compression_fired = (new_bin != prompt_bin)
                cur_bin = new_bin
                if compression_fired:
                    # Read cur_ids from compressed bin for full-cache confirm.
                    raw_compressed = cur_bin.read_bytes()
                    cur_ids = [struct.unpack_from("<i", raw_compressed, i)[0]
                               for i in range(0, len(raw_compressed), 4)]
                    full_snap_prep = prefix_cache.prepare_full_snap(prompt_ids)
                else:
                    cur_ids = prompt_ids

        available_gen = max_ctx - prompt_len - 20
        gen_len = min(_max_gen_tokens(req), available_gen)
        if gen_len <= 0:
            _abort_full_snap_if_needed(full_snap_prep)
            if full_hit is None:
                try: cur_bin.unlink()
                except Exception: pass
            else:
                # On full-cache hit, cur_bin points at the persistent cached file
                # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
                # the other hand, was never used (we hit before _maybe_compress) and
                # would otherwise leak.
                try: prompt_bin.unlink()
                except Exception: pass
            return JSONResponse(
                {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                status_code=400)

        tool_ctx: ToolRequestContext | None = None
        if tool_split and req.tools:
            try:
                tool_ctx = tool_split.prepare_request_context(
                    tokenizer, req.messages, req.tools,
                    chat_template_kwargs=req.chat_template_kwargs,
                )
            except Exception as exc:
                print(f"  [tool-split] prepare_request_context failed: {exc}", flush=True)

        if req.stream:
            return await _stream_response(req, cur_bin, prompt_ids, gen_len,
                                           completion_id, created,
                                           started_in_thinking, daemon_lock,
                                           full_hit=full_hit,
                                           full_snap_prep=full_snap_prep,
                                           compression_fired=compression_fired,
                                           cur_ids=cur_ids,
                                           tool_ctx=tool_ctx)

        # Non-streaming: collect, parse, return.
        async with daemon_lock:
            cmd_line, snap_prep, conv_prefix_len = _compose_daemon_cmd(
                cur_bin, gen_len, prompt_ids,
                full_hit=full_hit,
                compression_fired=compression_fired,
                full_snap_prep=full_snap_prep,
                cur_ids=cur_ids,
                tool_ctx=tool_ctx,
            )
            bus.begin_request()
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            snap_ok = False
            try:
                tokens = await asyncio.to_thread(list, _token_stream(r_pipe, gen_len))
                await bus.drain_timings()
                snap_ok = True
            except Exception:
                await _finalize_request_snaps(
                    full_snap_prep=full_snap_prep,
                    snap_prep=snap_prep,
                    prompt_ids=prompt_ids,
                    cur_bin=cur_bin,
                    cur_ids=cur_ids,
                    success=False,
                )
                raise
            await _finalize_request_snaps(
                full_snap_prep=full_snap_prep,
                snap_prep=snap_prep,
                prompt_ids=prompt_ids,
                cur_bin=cur_bin,
                cur_ids=cur_ids,
                success=snap_ok,
            )
            await _commit_tool_snap_if_needed(tool_ctx)
        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            # On full-cache hit, cur_bin points at the persistent cached file
            # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
            # the other hand, was never used (we hit before _maybe_compress) and
            # would otherwise leak.
            try: prompt_bin.unlink()
            except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        # User-supplied stop sequences: trim at first match.
        stops = normalize_stop(req.stop)
        if stops:
            i = first_stop_match(text, stops)
            if i != -1:
                text = text[:i]
        cleaned, tool_calls = parse_tool_calls(text, tools=req.tools)
        # Match the assistant prompt boundary (see _tokenize_prompt): if the template
        # did not open thinking there, missing `</think>` must not turn
        # the whole completion into reasoning-only.
        cleaned, reasoning = parse_reasoning(
            cleaned, thinking_enabled=started_in_thinking)

        msg: dict = {"role": "assistant"}
        finish_reason = "stop"
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["content"] = cleaned if cleaned else None
            msg["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            msg["content"] = cleaned

        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason,
            }],
            "usage": _build_usage(
                prompt_len, len(tokens), conv_prefix_len=conv_prefix_len),
        })

    async def _stream_response(req, prompt_bin, prompt_ids, gen_len, completion_id,
                                created, started_in_thinking, lock,
                                full_hit=None, full_snap_prep=None,
                                compression_fired=False, cur_ids=None,
                                tool_ctx: ToolRequestContext | None = None):
        # prompt_bin may be cur_bin (the compressed bin) when coming from the
        # compression or full-cache-hit path; prompt_len is derived from it.
        prompt_len = prompt_bin.stat().st_size // 4 if full_hit is None else (
            full_hit[2]  # cached_cur_ids_len
        )
        include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))
        def chunk(delta_obj, finish=None):
            return {"id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": delta_obj,
                                  "finish_reason": finish}]}

        async def sse() -> AsyncIterator[str]:
            async with lock:
                cmd_line, snap_prep, conv_prefix_len = _compose_daemon_cmd(
                    prompt_bin, gen_len, prompt_ids,
                    full_hit=full_hit,
                    compression_fired=compression_fired,
                    full_snap_prep=full_snap_prep,
                    cur_ids=cur_ids,
                    tool_ctx=tool_ctx,
                )
                bus.begin_request()
                daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                daemon_proc.stdin.flush()

                yield f"data: {json.dumps(chunk({'role': 'assistant'}))}\n\n"

                # State machine: mode ∈ {'reasoning', 'content', 'tool_buffer'}
                mode = "reasoning" if started_in_thinking else "content"
                window = ""           # holdback buffer for tag detection
                tool_buffer = ""
                stops = normalize_stop(req.stop)
                # Holdback must cover longest tag AND longest stop sequence.
                tag_holdback = max(len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG), len(TOOL_OPEN_TAG))
                stop_holdback = max((len(s) for s in stops), default=0)
                HOLDBACK = max(tag_holdback, stop_holdback)
                completion_tokens = 0
                stop_hit = False

                def emit_delta(text, kind):
                    """kind: 'content' or 'reasoning_content'"""
                    if not text:
                        return None
                    return f"data: {json.dumps(chunk({kind: text}))}\n\n"

                snap_ok = False
                try:
                    async for tok_id in iterate_in_threadpool(_token_stream(r_pipe, gen_len)):
                        completion_tokens += 1
                        piece = tokenizer.decode([tok_id])
                        window += piece

                        # Stop-sequence check on the visible (content/reasoning) stream.
                        if stops and mode != "tool_buffer":
                            si = first_stop_match(window, stops)
                            if si != -1:
                                window = window[:si]
                                stop_hit = True
                                # Flush truncated remainder per current mode.
                                kind = "reasoning_content" if mode == "reasoning" else "content"
                                out = emit_delta(window, kind)
                                if out: yield out
                                window = ""
                                break

                        # Process state transitions until no more tags found in window.
                        while True:
                            if mode == "tool_buffer":
                                tool_buffer += window
                                window = ""
                                break

                            # Look for the next tag of interest based on mode.
                            if mode == "reasoning":
                                idx = window.find(THINK_CLOSE_TAG)
                                if idx != -1:
                                    pre = window[:idx]
                                    out = emit_delta(pre, "reasoning_content")
                                    if out: yield out
                                    window = window[idx + len(THINK_CLOSE_TAG):]
                                    mode = "content"
                                    continue
                                # No close tag yet. Stream all but holdback.
                                if len(window) > HOLDBACK:
                                    safe = window[:-HOLDBACK]
                                    out = emit_delta(safe, "reasoning_content")
                                    if out: yield out
                                    window = window[-HOLDBACK:]
                                break  # need more tokens

                            else:  # mode == "content"
                                think_idx = window.find(THINK_OPEN_TAG)
                                tool_idx  = window.find(TOOL_OPEN_TAG)
                                # Pick the earliest tag that actually appears.
                                hits = [(i, t) for i, t in
                                        ((think_idx, "think"), (tool_idx, "tool")) if i != -1]
                                if hits:
                                    hits.sort()
                                    idx, which = hits[0]
                                    pre = window[:idx]
                                    out = emit_delta(pre, "content")
                                    if out: yield out
                                    if which == "think":
                                        window = window[idx + len(THINK_OPEN_TAG):]
                                        mode = "reasoning"
                                    else:  # tool
                                        tool_buffer = window[idx:]
                                        window = ""
                                        mode = "tool_buffer"
                                    continue
                                if len(window) > HOLDBACK:
                                    safe = window[:-HOLDBACK]
                                    out = emit_delta(safe, "content")
                                    if out: yield out
                                    window = window[-HOLDBACK:]
                                break  # need more tokens

                    if stop_hit:
                        finish_reason = "stop"
                        yield f"data: {json.dumps(chunk({}, finish=finish_reason))}\n\n"
                        if include_usage:
                            usage_chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                           "created": created, "model": MODEL_NAME, "choices": [],
                                           "usage": _build_usage(
                                               prompt_len, completion_tokens,
                                               conv_prefix_len=conv_prefix_len)}
                            yield f"data: {json.dumps(usage_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        await bus.drain_timings()
                        await _finalize_request_snaps(
                            full_snap_prep=full_snap_prep,
                            snap_prep=snap_prep,
                            prompt_ids=prompt_ids,
                            cur_bin=prompt_bin,
                            cur_ids=cur_ids,
                            success=True,
                        )
                        await _commit_tool_snap_if_needed(tool_ctx)
                        if full_hit is None:
                            try: prompt_bin.unlink()
                            except Exception: pass
                        return

                    # Generation done. Flush remaining window per current mode.
                    if mode == "reasoning" and window:
                        out = emit_delta(window, "reasoning_content")
                        if out: yield out
                    elif mode == "content" and window:
                        out = emit_delta(window, "content")
                        if out: yield out
                    elif mode == "tool_buffer":
                        tool_buffer += window
                    window = ""

                    finish_reason = "stop"
                    if mode == "tool_buffer":
                        cleaned_after, tool_calls = parse_tool_calls(tool_buffer, tools=req.tools)
                        if tool_calls:
                            if cleaned_after:
                                out = emit_delta(cleaned_after, "content")
                                if out: yield out
                            tc_delta_list = [{
                                "index": i, "id": tc["id"], "type": "function",
                                "function": {"name": tc["function"]["name"],
                                              "arguments": tc["function"]["arguments"]},
                            } for i, tc in enumerate(tool_calls)]
                            yield f"data: {json.dumps(chunk({'tool_calls': tc_delta_list}))}\n\n"
                            finish_reason = "tool_calls"
                        else:
                            # Unclosed <tool_call> — emit raw as content fallback.
                            out = emit_delta(tool_buffer, "content")
                            if out: yield out
                    snap_ok = True
                except Exception:
                    await _finalize_request_snaps(
                        full_snap_prep=full_snap_prep,
                        snap_prep=snap_prep,
                        prompt_ids=prompt_ids,
                        cur_bin=prompt_bin,
                        cur_ids=cur_ids,
                        success=False,
                    )
                    raise
                finally:
                    if full_hit is None:
                        try: prompt_bin.unlink()
                        except Exception: pass

                await bus.drain_timings()
                await _finalize_request_snaps(
                    full_snap_prep=full_snap_prep,
                    snap_prep=snap_prep,
                    prompt_ids=prompt_ids,
                    cur_bin=prompt_bin,
                    cur_ids=cur_ids,
                    success=snap_ok,
                )
                await _commit_tool_snap_if_needed(tool_ctx)

                yield f"data: {json.dumps(chunk({}, finish=finish_reason))}\n\n"
                if include_usage:
                    usage_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [],
                        "usage": _build_usage(
                            prompt_len, completion_tokens,
                            conv_prefix_len=conv_prefix_len),
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ── Anthropic Messages API ──────────────────────────────────────
    # Mirrors the OpenAI endpoint but formatted for the Anthropic SDK
    # (Claude Code, Anthropic clients). Tool calling NOT forwarded here
    # yet — agent CLIs that want tools should use /v1/chat/completions.

    def _anthropic_text_from_content(content) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)

    def _tokenize_anthropic(req: AnthropicMessagesRequest
                            ) -> tuple[Path, int, list[dict]]:
        msgs = []
        system_text = _anthropic_text_from_content(req.system) if req.system else None
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        for m in req.messages:
            msgs.append({"role": m.role,
                         "content": _anthropic_text_from_content(m.content)})
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # mkstemp returns (fd, path); discarding fd leaks 1 per request (#15).
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp, len(ids), msgs

    def _maybe_compress_anthropic(prompt_bin: Path, prompt_len: int,
                                  msgs: list[dict]) -> tuple[Path, int]:
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_len
        if not prefill_cfg.should_compress(prompt_len) or drafter_tokenizer is None:
            return prompt_bin, prompt_len
        last_user_idx = next((i for i in range(len(msgs) - 1, -1, -1)
                              if msgs[i]["role"] == "user"), None)
        if last_user_idx is None:
            return prompt_bin, prompt_len
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
        prompt = tokenizer.apply_chat_template(
            new_msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        try: prompt_bin.unlink()
        except Exception: pass
        return Path(path), len(ids)

    async def _astream_tokens(r, n_gen):
        """Yields one token at a time without blocking the event loop.
        Each 4-byte pipe read is dispatched to a worker thread."""
        generated = 0
        hit_stop = False
        while True:
            b = await asyncio.to_thread(os.read, r, 4)
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

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest):
        prompt_bin, prompt_len, raw_msgs = _tokenize_anthropic(req)

        # Read raw prompt_ids BEFORE compression (for full-cache key).
        raw = prompt_bin.read_bytes()
        prompt_ids = [struct.unpack_from("<i", raw, i)[0]
                      for i in range(0, len(raw), 4)]

        full_hit = None
        full_snap_prep = None
        cur_bin = prompt_bin
        cur_ids = prompt_ids
        compression_fired = False

        async with daemon_lock:
            full_hit = prefix_cache.lookup_full(prompt_ids)
            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
            else:
                new_bin, new_len = await asyncio.to_thread(
                    _maybe_compress_anthropic, prompt_bin, prompt_len, raw_msgs)
                compression_fired = (new_bin != prompt_bin)
                cur_bin = new_bin
                prompt_len = new_len
                if compression_fired:
                    raw_compressed = cur_bin.read_bytes()
                    cur_ids = [struct.unpack_from("<i", raw_compressed, i)[0]
                               for i in range(0, len(raw_compressed), 4)]
                    full_snap_prep = prefix_cache.prepare_full_snap(prompt_ids)
                else:
                    cur_ids = prompt_ids

        available_gen = max_ctx - prompt_len - 20
        gen_len = min(req.max_tokens, available_gen)
        if gen_len <= 0:
            _abort_full_snap_if_needed(full_snap_prep)
            if full_hit is None:
                try: cur_bin.unlink()
                except Exception: pass
            else:
                # On full-cache hit, cur_bin points at the persistent cached file
                # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
                # the other hand, was never used (we hit before _maybe_compress) and
                # would otherwise leak.
                try: prompt_bin.unlink()
                except Exception: pass
            return JSONResponse(
                {"type": "error",
                 "error": {"type": "invalid_request_error",
                           "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                status_code=400)

        msg_id = "msg_" + uuid.uuid4().hex[:24]
        user_stops = normalize_stop(req.stop_sequences)

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    if full_hit is not None:
                        slot, cached_cur_bin, _cached_len = full_hit
                        cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
                        snap_prep = None
                    elif compression_fired:
                        if full_snap_prep is not None:
                            fslot, _ = full_snap_prep
                            cmd_line = f"{cur_bin} {gen_len} snap={len(cur_ids)}:{fslot}\n"
                        else:
                            cmd_line = f"{cur_bin} {gen_len}\n"
                        snap_prep = None
                    else:
                        hit = prefix_cache.lookup(prompt_ids)
                        snap_prep = prefix_cache.prepare_inline_snap(prompt_ids)
                        if hit:
                            slot, _prefix_len = hit
                            cmd_line = f"RESTORE {slot} {cur_bin} {gen_len}"
                        else:
                            cmd_line = f"{cur_bin} {gen_len}"
                        if snap_prep:
                            cmd_line += f" snap={snap_prep[1]}:{snap_prep[0]}"
                        cmd_line += "\n"

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

                    cb_start = {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(cb_start)}\n\n"

                    bus.begin_request()
                    daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                    daemon_proc.stdin.flush()

                    out_tokens = 0
                    stop_reason = "end_turn"
                    matched_stop: str | None = None
                    holdback = max((len(s) for s in user_stops), default=0)
                    window = ""
                    snap_ok = False
                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            out_tokens += 1
                            piece = tokenizer.decode([tok_id])
                            if user_stops:
                                window += piece
                                si = first_stop_match(window, user_stops)
                                if si != -1:
                                    for s in user_stops:
                                        if window[si:si + len(s)] == s:
                                            matched_stop = s
                                            break
                                    emit = window[:si]
                                    stop_reason = "stop_sequence"
                                    if emit:
                                        delta = {
                                            "type": "content_block_delta", "index": 0,
                                            "delta": {"type": "text_delta", "text": emit},
                                        }
                                        yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                                    break
                                if len(window) > holdback:
                                    emit = window[:-holdback]
                                    window = window[-holdback:]
                                    if emit:
                                        delta = {
                                            "type": "content_block_delta", "index": 0,
                                            "delta": {"type": "text_delta", "text": emit},
                                        }
                                        yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                            else:
                                delta = {
                                    "type": "content_block_delta", "index": 0,
                                    "delta": {"type": "text_delta", "text": piece},
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                        else:
                            if user_stops and window:
                                trimmed, matched_stop = trim_at_stop(window, user_stops)
                                if matched_stop is not None:
                                    stop_reason = "stop_sequence"
                                    window = trimmed
                                if window:
                                    delta = {
                                        "type": "content_block_delta", "index": 0,
                                        "delta": {"type": "text_delta", "text": window},
                                    }
                                    yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                        snap_ok = True
                    except Exception:
                        await _finalize_request_snaps(
                            full_snap_prep=full_snap_prep,
                            snap_prep=snap_prep,
                            prompt_ids=prompt_ids,
                            cur_bin=cur_bin,
                            cur_ids=cur_ids,
                            success=False,
                        )
                        raise
                    finally:
                        if full_hit is None:
                            try: cur_bin.unlink()
                            except Exception: pass
                        else:
                            # On full-cache hit, cur_bin points at the persistent cached file
                            # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
                            # the other hand, was never used (we hit before _maybe_compress) and
                            # would otherwise leak.
                            try: prompt_bin.unlink()
                            except Exception: pass

                    await _finalize_request_snaps(
                        full_snap_prep=full_snap_prep,
                        snap_prep=snap_prep,
                        prompt_ids=prompt_ids,
                        cur_bin=cur_bin,
                        cur_ids=cur_ids,
                        success=snap_ok,
                    )

                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                    msg_delta = {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason,
                            "stop_sequence": matched_stop,
                        },
                        "usage": {"output_tokens": out_tokens},
                    }
                    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        async with daemon_lock:
            if full_hit is not None:
                slot, cached_cur_bin, _cached_len = full_hit
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
                snap_prep = None
            elif compression_fired:
                if full_snap_prep is not None:
                    fslot, _ = full_snap_prep
                    cmd_line = f"{cur_bin} {gen_len} snap={len(cur_ids)}:{fslot}\n"
                else:
                    cmd_line = f"{cur_bin} {gen_len}\n"
                snap_prep = None
            else:
                hit = prefix_cache.lookup(prompt_ids)
                snap_prep = prefix_cache.prepare_inline_snap(prompt_ids)
                if hit:
                    slot, _prefix_len = hit
                    cmd_line = f"RESTORE {slot} {cur_bin} {gen_len}"
                else:
                    cmd_line = f"{cur_bin} {gen_len}"
                if snap_prep:
                    cmd_line += f" snap={snap_prep[1]}:{snap_prep[0]}"
                cmd_line += "\n"
            bus.begin_request()
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            snap_ok = False
            try:
                tokens = [t async for t in _astream_tokens(r_pipe, gen_len)]
                snap_ok = True
            except Exception:
                await _finalize_request_snaps(
                    full_snap_prep=full_snap_prep,
                    snap_prep=snap_prep,
                    prompt_ids=prompt_ids,
                    cur_bin=cur_bin,
                    cur_ids=cur_ids,
                    success=False,
                )
                raise
            await _finalize_request_snaps(
                full_snap_prep=full_snap_prep,
                snap_prep=snap_prep,
                prompt_ids=prompt_ids,
                cur_bin=cur_bin,
                cur_ids=cur_ids,
                success=snap_ok,
            )

        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            # On full-cache hit, cur_bin points at the persistent cached file
            # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
            # the other hand, was never used (we hit before _maybe_compress) and
            # would otherwise leak.
            try: prompt_bin.unlink()
            except Exception: pass
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        text, matched_stop = trim_at_stop(text, user_stops)
        stop_reason = "stop_sequence" if matched_stop else "end_turn"
        return JSONResponse({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": req.model or MODEL_NAME,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": matched_stop,
            "usage": {"input_tokens": prompt_len,
                      "output_tokens": len(tokens)},
        })

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    # Attention compute currently scales with --max-ctx, not the actual
    # prompt+gen length (see issue #10). Default 16384 fits most API
    # workloads without the 20×+ slowdown users hit with --max-ctx=131072
    # on short requests. Bump via --max-ctx for long-context serving.
    ap.add_argument("--max-ctx", type=int, default=16384,
                    help="Maximum context length (default: 16384; oversizing "
                         "this, e.g. 131072 on short prompts, can slow "
                         "attention 20×+ until issue #10 is fixed)")
    ap.add_argument("--kv-f16", action="store_true",
                    help="Force F16 KV cache. When --max-ctx > 6144 the server "
                         "auto-enables TQ3_0 KV to fit; pass --kv-f16 to opt out.")
    ap.add_argument("--cache-type-k", "--ctk", dest="cache_type_k", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="K cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for K). "
                         "See kv_quant.cpp for supported (K,V) pairs.")
    ap.add_argument("--cache-type-v", "--ctv", dest="cache_type_v", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="V cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for V).")
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers (KV positions). 0 = full "
                         "attention. Default 2048 (set in C++); only kicks in "
                         "once kv_cache > window. Trades attention range for "
                         "long-context decode speed.")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="HF tokenizer id; default inferred from target GGUF.")
    ap.add_argument("--target-gpu", type=int, default=None,
                    help="Visible CUDA device id for test_dflash (sets DFLASH_TARGET_GPU)")
    ap.add_argument("--draft-gpu", type=int, default=None,
                    help="Visible CUDA device id for draft (sets DFLASH_DRAFT_GPU)")
    ap.add_argument("--target-gpus", type=str, default=None,
                    help="Comma-separated target GPU ids for target-layer sharding (passes --target-gpus)")
    ap.add_argument("--target-layer-split", nargs="?", const="", default=None,
                    metavar="WEIGHTS",
                    help="Optional comma-separated layer split weights for --target-gpus "
                         "(omit WEIGHTS after the flag to use defaults)")
    ap.add_argument("--draft-feature-mirror", action="store_true",
                    help="Pass --draft-feature-mirror to test_dflash (safe cross-GPU feature path)")
    ap.add_argument("--peer-access", action="store_true",
                    help="Pass --peer-access to test_dflash (prefer P2P memcpy when available)")
    ap.add_argument("--daemon", action="store_true",
                    help="No-op: accepted for parity with server.py / Compose; "
                         "this process always runs test_dflash with --daemon.")
    add_cli_flags(ap)
    add_tool_split_flags(ap)
    ap.add_argument("--prefix-cache-slots", type=int, default=4,
                    help="Number of prefix-cache snapshot slots (0 to disable)")
    ap.add_argument("--prefill-cache-slots", type=int, default=4,
                    help="Number of full-compress-result cache slots (Option 3). "
                         "Only active when --prefill-compression is enabled. "
                         "prefix-cache-slots + prefill-cache-slots must not exceed 8.")
    args = ap.parse_args()
    prefill_cfg = config_from_args(args)

    tool_split_cfg = tool_split_config_from_args(args)
    if getattr(args, "tool_split", None) == "auto":
        tool_split_cfg = ToolSplitConfig(
            enabled=True,
            profile=tool_split_cfg.profile,
            plugin_dir=tool_split_cfg.plugin_dir,
            pinned_tool_slots=tool_split_cfg.pinned_tool_slots,
            compress_conversation=tool_split_cfg.compress_conversation,
        )
    elif getattr(args, "tool_split", None) == "on":
        tool_split_cfg = ToolSplitConfig(
            enabled=True,
            profile=tool_split_cfg.profile or "auto",
            plugin_dir=tool_split_cfg.plugin_dir,
            pinned_tool_slots=tool_split_cfg.pinned_tool_slots,
            compress_conversation=tool_split_cfg.compress_conversation,
        )
    elif getattr(args, "tool_split", None) == "off":
        tool_split_cfg = ToolSplitConfig(
            enabled=False,
            profile=tool_split_cfg.profile,
            plugin_dir=tool_split_cfg.plugin_dir,
            pinned_tool_slots=tool_split_cfg.pinned_tool_slots,
            compress_conversation=tool_split_cfg.compress_conversation,
        )

    # Auto-enable TQ3_0 KV cache when the requested context exceeds what F16 fits.
    # setdefault so an explicit user DFLASH27B_KV_TQ3=0 still wins.
    if args.cache_type_k:
        os.environ["DFLASH27B_KV_K"] = args.cache_type_k
    if args.cache_type_v:
        os.environ["DFLASH27B_KV_V"] = args.cache_type_v
    if args.max_ctx > 6144 and not args.kv_f16 and not args.cache_type_k and not args.cache_type_v:
        os.environ.setdefault("DFLASH27B_KV_TQ3", "1")

    if args.fa_window is not None:
        os.environ["DFLASH27B_FA_WINDOW"] = str(args.fa_window)

    if args.target_gpu is not None:
        os.environ["DFLASH_TARGET_GPU"] = str(args.target_gpu)
    if args.draft_gpu is not None:
        os.environ["DFLASH_DRAFT_GPU"] = str(args.draft_gpu)

    # When pflash is on, daemon needs the same env the bench harness uses
    # (otherwise post-compress draft graph reserve OOMs at 64K+).
    if args.prefill_compression != "off":
        os.environ.setdefault("DFLASH27B_LM_HEAD_FIX", "0")
        os.environ.setdefault("DFLASH27B_FA_WINDOW", "0")
        # FlashPrefill bench-tuned defaults from pflash/README.md headline numbers
        # (10x TTFT @ 64K). Without these the drafter falls through to the WMMA
        # fallback at the default ALPHA=0.12, which roughly triples cold-start
        # TTFT. setdefault so explicit user overrides still win.
        os.environ.setdefault("DFLASH_FP_USE_BSA", "1")
        os.environ.setdefault("DFLASH_FP_ALPHA",   "0.85")
        if prefill_cfg.skip_park:
            os.environ["DFLASH_COMPRESS_NO_PARK"] = "1"

    if not args.target.is_file():
        raise SystemExit(f"target GGUF not found at {args.target}")

    arch = _arch_from_gguf(args.target)

    if not args.bin.is_file():
        raise SystemExit(f"binary not found at {args.bin} (arch={arch})")

    if arch in _LAGUNA_ARCHES:
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

    tool_split_orchestrator = None
    if tool_split_cfg.enabled:
        adapter = resolve_tool_split_adapter(
            tool_split_cfg, arch=arch, tokenizer_id=tokenizer_id)
        if adapter is not None:
            # Thick prefix snapshots live in system RAM (CPU snapshot backend),
            # so multiple conversation slots coexist with thin tool pins even
            # on 24GB cards. Agent stacks (Hermes) interleave 2+ prompt
            # families per turn — a single thick slot would thrash on every
            # request. No VRAM clamp needed anymore.
            if tool_split_cfg.pinned_tool_slots > 0:
                total_slots = (
                    args.prefix_cache_slots
                    + args.prefill_cache_slots
                    + tool_split_cfg.pinned_tool_slots
                )
                if total_slots > PrefixCache.DAEMON_MAX_SLOTS:
                    overflow = total_slots - PrefixCache.DAEMON_MAX_SLOTS
                    if args.prefill_cache_slots > 0:
                        cut = min(overflow, args.prefill_cache_slots)
                        args.prefill_cache_slots -= cut
                        overflow -= cut
                    if overflow > 0 and args.prefix_cache_slots > 0:
                        cut = min(overflow, args.prefix_cache_slots)
                        args.prefix_cache_slots -= cut
                        overflow -= cut
                    if overflow > 0:
                        tool_split_cfg = ToolSplitConfig(
                            enabled=tool_split_cfg.enabled,
                            profile=tool_split_cfg.profile,
                            plugin_dir=tool_split_cfg.plugin_dir,
                            pinned_tool_slots=max(
                                0,
                                tool_split_cfg.pinned_tool_slots - overflow,
                            ),
                            compress_conversation=tool_split_cfg.compress_conversation,
                        )
                    print(
                        f"  [cfg] tool-split slot budget: prefix={args.prefix_cache_slots} "
                        f"prefill={args.prefill_cache_slots} "
                        f"tool_pins={tool_split_cfg.pinned_tool_slots} "
                        f"(daemon max {PrefixCache.DAEMON_MAX_SLOTS})",
                        flush=True,
                    )
            from tool_split.orchestrator import ToolSlotCache
            tool_split_orchestrator = ToolSplitOrchestrator(
                adapter=adapter, config=tool_split_cfg)
            tool_split_orchestrator.tool_slots = ToolSlotCache(
                pinned_slots=tool_split_cfg.pinned_tool_slots,
                slot_base=args.prefix_cache_slots + args.prefill_cache_slots,
            )

    extra_daemon: list[str] = []
    if args.draft_feature_mirror:
        extra_daemon.append("--draft-feature-mirror")
    if args.peer_access:
        extra_daemon.append("--peer-access")
    if args.target_gpus:
        extra_daemon.append(f"--target-gpus={args.target_gpus}")
        if args.target_layer_split:
            extra_daemon.append(f"--target-layer-split={args.target_layer_split}")
        extra_daemon.append("--target-split-load-draft")
        extra_daemon.append("--target-split-dflash")
        if args.prefix_cache_slots > 0 or args.prefill_cache_slots > 0:
            print("  [cfg] target-gpus daemon mode disables prefix/full cache slots (snapshot protocol unsupported)")
            args.prefix_cache_slots = 0
            args.prefill_cache_slots = 0

    app = build_app(args.target, draft, args.bin, args.budget, args.max_ctx,
                    tokenizer, stop_ids,
                    prefill_cfg=prefill_cfg if prefill_cfg.enabled else None,
                    drafter_tokenizer=drafter_tokenizer,
                    prefix_cache_slots=args.prefix_cache_slots,
                    prefill_cache_slots=args.prefill_cache_slots,
                    arch=arch,
                    extra_daemon_args=extra_daemon or None,
                    tool_split_cfg=tool_split_cfg,
                    tool_split=tool_split_orchestrator)

    import uvicorn
    print(f"Luce DFlash OpenAI server (tool-aware) on http://{args.host}:{args.port}")
    print(f"  arch      = {arch}")
    print(f"  target    = {args.target}")
    print(f"  draft     = {draft}")
    print(f"  bin       = {args.bin}")
    print(f"  budget    = {args.budget}")
    print(f"  max_ctx   = {args.max_ctx}")
    print(f"  tokenizer = {tokenizer_id}")
    if prefill_cfg.enabled:
        print(f"  pflash = {prefill_cfg.mode} · threshold={prefill_cfg.threshold} "
              f"keep={prefill_cfg.keep_ratio} drafter={prefill_cfg.drafter_gguf}")
    else:
        print("  pflash = off")
    if tool_split_orchestrator is not None:
        print(f"  tool-split = on · profile={tool_split_orchestrator.profile} "
              f"pinned_slots={tool_split_cfg.pinned_tool_slots} "
              f"compress_conv={tool_split_cfg.compress_conversation}")
        if tool_split_cfg.plugin_dir:
            print(f"  tool-split plugins = {tool_split_cfg.plugin_dir}")
    elif tool_split_cfg.enabled:
        print("  tool-split = requested but no adapter matched (disabled)")
    else:
        print("  tool-split = off")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
