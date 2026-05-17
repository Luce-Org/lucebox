#!/usr/bin/env python3
"""Small protocol shim for client harness llama.cpp comparisons.

The proxy accepts the client-facing routes Lucebox supports and forwards the
model call to llama-server's OpenAI Chat Completions endpoint. It is intentionally
minimal: enough for harness benchmarks, not a full public API implementation.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def text_from_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                typ = part.get("type")
                if typ in ("text", "input_text", "output_text"):
                    out.append(part.get("text", ""))
                elif typ == "tool_result":
                    out.append(str(part.get("content", "")))
        return "".join(out)
    return str(content)


def normalize_anthropic_system(system_text: str | None) -> str | None:
    if not system_text:
        return None
    if os.environ.get("LLAMA_COMPAT_RAW_ANTHROPIC_SYSTEM", "0") == "1":
        return system_text
    if (
        "x-anthropic-billing-header:" in system_text
        or "Claude Agent SDK" in system_text
        or "Claude Code" in system_text
    ):
        return (
            "You are a concise coding assistant running behind an Anthropic "
            "Messages compatible client. Answer the user's request directly. "
            "Use the provided tools only when they are needed."
        )
    return system_text


def normalize_anthropic_user_text(text: str) -> str:
    if os.environ.get("LLAMA_COMPAT_RAW_ANTHROPIC_USER", "0") == "1":
        return text

    def replace_reminder(match: re.Match[str]) -> str:
        block = match.group(0)
        if "SKIP:" in block and ("- init:" in block or "- review:" in block):
            return ""
        return block

    return re.sub(
        r"<system-reminder>.*?</system-reminder>",
        replace_reminder,
        text,
        flags=re.DOTALL,
    )


def map_responses_input(req: dict) -> tuple[list[dict], list[dict] | None]:
    messages: list[dict] = []
    system_parts: list[str] = []
    if req.get("instructions"):
        system_parts.append(str(req["instructions"]))

    inp = req.get("input", "")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            typ = item.get("type", "message")
            if typ == "message":
                role = item.get("role", "user")
                content = text_from_content(item.get("content", ""))
                if role in ("developer", "system"):
                    system_parts.append(content)
                else:
                    messages.append({"role": role, "content": content})
            elif typ == "function_call":
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": item.get("call_id") or item.get("id") or "call_" + uuid.uuid4().hex[:12],
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }],
                })
            elif typ == "function_call_output":
                output = item.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output)
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": output,
                })

    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    tools = []
    for tool in req.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function":
            fn = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            }
            tools.append({"type": "function", "function": fn})
    return messages, tools or None


def map_anthropic_messages(req: dict) -> tuple[list[dict], list[dict] | None]:
    messages: list[dict] = []
    system = req.get("system")
    if system:
        normalized = normalize_anthropic_system(text_from_content(system))
        if normalized:
            messages.append({"role": "system", "content": normalized})

    for msg in req.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            tool_calls = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                typ = part.get("type")
                if typ == "text":
                    text_parts.append(normalize_anthropic_user_text(part.get("text", "")))
                elif typ == "tool_use":
                    tool_calls.append({
                        "id": part.get("id", "call_" + uuid.uuid4().hex[:12]),
                        "type": "function",
                        "function": {
                            "name": part.get("name", ""),
                            "arguments": json.dumps(part.get("input", {})),
                        },
                    })
                elif typ == "tool_result":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": part.get("tool_use_id", ""),
                        "content": text_from_content(part.get("content", "")),
                    })
            if tool_calls:
                messages.append({"role": "assistant", "content": "".join(text_parts), "tool_calls": tool_calls})
            elif text_parts:
                messages.append({"role": role, "content": "".join(text_parts)})
        else:
            messages.append({
                "role": role,
                "content": normalize_anthropic_user_text(text_from_content(content)),
            })

    tools = []
    for tool in req.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        }
        tools.append({"type": "function", "function": fn})
    return messages, tools or None


def upstream_chat(upstream: str, payload: dict) -> tuple[int, dict]:
    url = urlparse(upstream)
    conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    port = url.port or (443 if url.scheme == "https" else 80)
    conn = conn_cls(url.hostname, port, timeout=900)
    body = json.dumps(payload).encode("utf-8")
    path = (url.path.rstrip("/") or "") + "/v1/chat/completions"
    conn.request("POST", path, body, {
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    resp = conn.getresponse()
    data = resp.read()
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception:
        obj = {"error": data.decode("utf-8", errors="replace")}
    return resp.status, obj


def chat_text_and_tools(chat: dict) -> tuple[str, list[dict]]:
    choices = chat.get("choices") or []
    if not choices:
        return "", []
    msg = choices[0].get("message") or {}
    text = msg.get("content") or ""
    return text, msg.get("tool_calls") or []


def responses_json(req: dict, chat: dict, model: str) -> dict:
    text, tool_calls = chat_text_and_tools(chat)
    resp_id = "resp_" + uuid.uuid4().hex[:24]
    output = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        output.append({
            "type": "function_call",
            "id": tc.get("id", "call_" + uuid.uuid4().hex[:12]),
            "status": "completed",
            "call_id": tc.get("id", "call_" + uuid.uuid4().hex[:12]),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })
    if not output:
        output.append({
            "type": "message",
            "id": "msg_" + uuid.uuid4().hex[:24],
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    usage = chat.get("usage") or {}
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": text,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def anthropic_json(req: dict, chat: dict, model: str) -> dict:
    text, tool_calls = chat_text_and_tools(chat)
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for tc in tool_calls:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", "call_" + uuid.uuid4().hex[:12]),
            "name": fn.get("name", ""),
            "input": args,
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = chat.get("usage") or {}
    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def responses_sse(obj: dict) -> bytes:
    events = []
    shell = {k: v for k, v in obj.items() if k != "output"}
    shell["output"] = []
    events.append(("response.created", {"response": {**shell, "status": "in_progress"}}))
    for i, item in enumerate(obj.get("output") or []):
        events.append(("response.output_item.added", {"output_index": i, "item": item}))
        if item.get("type") == "message":
            content = (item.get("content") or [{"text": ""}])[0]
            text = content.get("text", "")
            events.append(("response.content_part.added", {
                "item_id": item["id"], "output_index": i, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }))
            if text:
                events.append(("response.output_text.delta", {
                    "item_id": item["id"], "output_index": i, "content_index": 0,
                    "delta": text,
                }))
            events.append(("response.output_text.done", {
                "item_id": item["id"], "output_index": i, "content_index": 0,
                "text": text,
            }))
            events.append(("response.content_part.done", {
                "item_id": item["id"], "output_index": i, "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            }))
        elif item.get("type") == "function_call":
            events.append(("response.function_call_arguments.delta", {
                "item_id": item["id"], "output_index": i, "delta": item.get("arguments", "{}"),
            }))
            events.append(("response.function_call_arguments.done", {
                "item_id": item["id"], "output_index": i, "arguments": item.get("arguments", "{}"),
                "name": item.get("name", ""),
            }))
        events.append(("response.output_item.done", {"output_index": i, "item": item}))
    events.append(("response.completed", {"response": obj}))
    out = []
    for event, data in events:
        data["type"] = event
        out.append(f"event: {event}\ndata: {json.dumps(data)}\n\n")
    return "".join(out).encode("utf-8")


def anthropic_sse(obj: dict) -> bytes:
    events = []
    start = {**obj, "content": [], "stop_reason": None, "usage": {**obj.get("usage", {}), "output_tokens": 0}}
    events.append(("message_start", {"type": "message_start", "message": start}))
    for i, block in enumerate(obj.get("content") or []):
        events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": block}))
        if block.get("type") == "text":
            events.append(("content_block_delta", {
                "type": "content_block_delta", "index": i,
                "delta": {"type": "text_delta", "text": block.get("text", "")},
            }))
        elif block.get("type") == "tool_use":
            events.append(("content_block_delta", {
                "type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))},
            }))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": obj.get("stop_reason", "end_turn"), "stop_sequence": None},
        "usage": {"output_tokens": obj.get("usage", {}).get("output_tokens", 0)},
    }))
    events.append(("message_stop", {"type": "message_stop"}))
    out = []
    for event, data in events:
        out.append(f"event: {event}\ndata: {json.dumps(data)}\n\n")
    return "".join(out).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    upstream = ""
    model = "luce-dflash"
    max_tokens_cap = 0

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)

    def send_json(self, status: int, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_sse(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/health"):
            self.send_json(200, {"status": "ok", "upstream": self.upstream})
        elif self.path.startswith("/v1/models"):
            self.send_json(200, {
                "object": "list",
                "data": [{"id": self.model, "object": "model", "created": 0, "owned_by": "llamacpp"}],
            })
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = self.read_json()
            if self.path.startswith("/v1/responses"):
                self.handle_responses(req)
            elif self.path.startswith("/v1/messages"):
                self.handle_anthropic(req)
            elif self.path.startswith("/v1/chat/completions"):
                status, obj = upstream_chat(self.upstream, {**req, "stream": False})
                self.send_json(status, obj)
            else:
                self.send_json(404, {"error": "not found"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def handle_responses(self, req: dict):
        messages, tools = map_responses_input(req)
        requested_max = req.get("max_output_tokens") or req.get("max_tokens") or 1024
        if self.max_tokens_cap > 0:
            requested_max = min(int(requested_max), self.max_tokens_cap)
        payload = {
            "model": req.get("model") or self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": requested_max,
            "temperature": req.get("temperature", 0),
            "top_p": req.get("top_p", 1),
        }
        if tools:
            payload["tools"] = tools
        status, chat = upstream_chat(self.upstream, payload)
        if status >= 400:
            self.send_json(status, chat)
            return
        obj = responses_json(req, chat, payload["model"])
        if req.get("stream"):
            self.send_sse(responses_sse(obj))
        else:
            self.send_json(200, obj)

    def handle_anthropic(self, req: dict):
        messages, tools = map_anthropic_messages(req)
        requested_max = req.get("max_tokens") or 1024
        if self.max_tokens_cap > 0:
            requested_max = min(int(requested_max), self.max_tokens_cap)
        payload = {
            "model": req.get("model") or self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": requested_max,
            "temperature": req.get("temperature", 0),
            "top_p": req.get("top_p", 1),
        }
        if tools:
            payload["tools"] = tools
        status, chat = upstream_chat(self.upstream, payload)
        if status >= 400:
            self.send_json(status, chat)
            return
        obj = anthropic_json(req, chat, payload["model"])
        if req.get("stream"):
            self.send_sse(anthropic_sse(obj))
        else:
            self.send_json(200, obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--upstream", default="http://127.0.0.1:18081")
    ap.add_argument("--model", default="luce-dflash")
    ap.add_argument("--max-tokens-cap", type=int, default=int(os.environ.get("LLAMA_COMPAT_MAX_TOKENS", "0")))
    args = ap.parse_args()
    Handler.upstream = args.upstream.rstrip("/")
    Handler.model = args.model
    Handler.max_tokens_cap = args.max_tokens_cap
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"llama.cpp compat proxy on http://{args.host}:{args.port} -> {Handler.upstream}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
