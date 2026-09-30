"""Request/decision types shared by routers, the gateway and eval scripts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


def content_text(content) -> str:
    """Text of an OpenAI message `content` (string or list of parts); non-text parts are dropped."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") in ("text", "input_text") and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    return ""


@dataclass
class RouteRequest:
    messages: list[dict]
    tools: list | None = None
    session_id: str | None = None
    max_tokens: int | None = None
    raw: dict = field(default_factory=dict)       # full request body, untouched
    headers: dict = field(default_factory=dict)   # lower-cased request headers (gateway only)

    @classmethod
    def from_openai(cls, body: dict, headers: dict | None = None) -> "RouteRequest":
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        session = (
            headers.get("x-session-id")
            or body.get("session_id")
            or meta.get("session_id")
            or body.get("user")
        )
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        return cls(
            messages=list(body.get("messages") or []),
            tools=body.get("tools") or None,
            session_id=str(session) if session is not None else None,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            raw=body,
            headers=headers,
        )

    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return content_text(m.get("content"))
        return ""

    def prompt_chars(self) -> int:
        n = sum(len(content_text(m.get("content"))) for m in self.messages)
        if self.tools:
            n += len(json.dumps(self.tools))
        return n


@dataclass
class RouteDecision:
    model: str            # a config model name, or an escalation target such as "swarm"
    router: str           # label of the router (outermost spec) that produced it
    reason: str
    scores: dict = field(default_factory=dict)    # router-specific scores (probs, predicted quality)
    signals: dict = field(default_factory=dict)   # extra inputs/diagnostics (margin, load, layer)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
