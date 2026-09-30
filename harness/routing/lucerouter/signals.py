"""Async clients for the signals routers use: Brick difficulty probs and backend load.

All parsers are separate pure functions so they can be unit-tested with fixtures.
"""

from __future__ import annotations

import asyncio
import math
import time

import httpx
import numpy as np

from .config import ModelBackend, root_url

LABELS = ("easy", "medium", "hard")

BRICK_SYSTEM = (
    "You are a query difficulty classifier for an LLM routing system.\n"
    "Classify each query as easy, medium, or hard based on the cognitive depth and domain "
    "expertise required to answer correctly.\n"
    "Respond with ONLY one word: easy, medium, or hard."
)

# Appended to the raw ChatML prompt when a model would otherwise open with a think block.
THINK_SUFFIX = "<think>\n\n</think>\n\n"


def truncate_query(text: str, limit: int = 3000) -> str:
    """Keep head and tail of long queries; the ask is usually at one end."""
    if len(text) <= limit:
        return text
    half = (limit - 5) // 2
    return text[:half] + "\n...\n" + text[-half:]


def brick_chatml_prompt(query: str, suffix: str = "") -> str:
    return (
        f"<|im_start|>system\n{BRICK_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nClassify: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n{suffix}"
    )


def _prob(entry: dict) -> float:
    if entry.get("prob") is not None:
        return float(entry["prob"])
    if entry.get("logprob") is not None:
        return math.exp(float(entry["logprob"]))
    return 0.0


def _candidate_positions(resp: dict) -> list[list[tuple[str, float]]]:
    """Per generated position, the list of (token_text, prob) alternatives.

    Accepts:
      * OpenAI chat:        choices[0].logprobs.content[i].top_logprobs[{token, logprob}]
      * OpenAI completions: choices[0].logprobs.top_logprobs[i] = {token: logprob}
      * llama-server /completion, newer: completion_probabilities[i].top_probs[{token, prob}]
        or .top_logprobs[{token, logprob}]; older: .probs[{tok_str, prob}]
    """
    positions: list[list[tuple[str, float]]] = []
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        lp = choices[0].get("logprobs") or {}
        if isinstance(lp.get("content"), list):
            for pos in lp["content"]:
                alts = pos.get("top_logprobs") or [pos]
                positions.append([(a.get("token", ""), _prob(a)) for a in alts])
        elif isinstance(lp.get("top_logprobs"), list):
            for pos in lp["top_logprobs"]:
                positions.append([(tok, math.exp(v)) for tok, v in (pos or {}).items()])
        return positions
    for pos in resp.get("completion_probabilities") or []:
        alts = pos.get("top_probs") or pos.get("top_logprobs") or pos.get("probs") or []
        positions.append([(a.get("token", a.get("tok_str", "")), _prob(a)) for a in alts])
    return positions


def token_label(token: str) -> str | None:
    """Label whose word starts with this token (leading spaces/case/punctuation ignored)."""
    tok = token.strip().strip(".,:;!\"'").lower()
    if not tok:
        return None
    for label in LABELS:
        if label.startswith(tok) or tok == label:
            return label
    return None


def parse_brick_probs(resp: dict) -> dict[str, float]:
    """Renormalised {"easy","medium","hard"} probs from the first position that mentions a label."""
    for alts in _candidate_positions(resp):
        mass = dict.fromkeys(LABELS, 0.0)
        for token, p in alts:
            label = token_label(token)
            if label:
                mass[label] += p
        total = sum(mass.values())
        if total > 0:
            return {k: v / total for k, v in mass.items()}
    raise ValueError("no easy/medium/hard token in brick response")


async def brick_probs(
    service_url: str,
    query_text: str,
    *,
    kind: str = "luce_server",
    model: str | None = None,
    prompt_suffix: str = "",
    client: httpx.AsyncClient | None = None,
    timeout: float = 10.0,
) -> dict[str, float]:
    """Difficulty probs from a Brick classifier.

    kind="llama-server" uses raw `/completion` with a ChatML prompt (+ optional
    `prompt_suffix`); anything else uses OpenAI `/v1/chat/completions` with
    logprobs and thinking disabled.
    """
    query = truncate_query(query_text)
    root = root_url(service_url)
    if kind == "llama-server":
        url = f"{root}/completion"
        body = {"prompt": brick_chatml_prompt(query, prompt_suffix),
                "n_predict": 1, "n_probs": 20, "temperature": 0}
    else:
        url = f"{root}/v1/chat/completions"
        body = {
            "messages": [{"role": "system", "content": BRICK_SYSTEM},
                         {"role": "user", "content": f"Classify: {query}"}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if model:
            body["model"] = model
    own = client is None
    client = client or httpx.AsyncClient()
    try:
        r = await client.post(url, json=body, timeout=timeout)
        r.raise_for_status()
        return parse_brick_probs(r.json())
    finally:
        if own:
            await client.aclose()


# ─── backend load ────────────────────────────────────────────────────────

def parse_luce_status(status: dict, backend: ModelBackend) -> dict | None:
    """{"in_flight","capacity"} from luce_server `/status/json`.

    Multi-model servers report `models[i].{id,in_flight,capacity}`; single-model
    servers report `active_requests`/`parked_requests`/`phase` and no capacity
    (config `capacity`, default 1, is used).
    """
    models = status.get("models")
    if isinstance(models, list) and models:
        wanted = {backend.name, backend.request_model}
        entry = next((m for m in models if m.get("id") in wanted), None)
        if entry is None and len(models) == 1:
            entry = models[0]
        if entry is not None and "in_flight" in entry:
            cap = entry.get("capacity") or backend.capacity or 1
            return {"in_flight": int(entry["in_flight"]), "capacity": int(cap)}
    if "active_requests" in status or "phase" in status:
        n = int(status.get("active_requests") or 0) + int(status.get("parked_requests") or 0)
        if n == 0 and status.get("phase") not in (None, "idle", "IDLE"):
            n = 1
        return {"in_flight": n, "capacity": int(backend.capacity or 1)}
    return None


def parse_llama_slots(slots, backend: ModelBackend) -> dict | None:
    if not isinstance(slots, list):
        return None
    busy = sum(1 for s in slots if s.get("is_processing") or s.get("state") == 1)
    return {"in_flight": busy, "capacity": len(slots) or int(backend.capacity or 1)}


def parse_llama_metrics(text: str, backend: ModelBackend) -> dict | None:
    for line in text.splitlines():
        if line.startswith("llamacpp:requests_processing"):
            return {"in_flight": int(float(line.split()[-1])), "capacity": int(backend.capacity or 1)}
    return None


async def backend_load(backend: ModelBackend, client: httpx.AsyncClient, timeout: float = 0.5) -> dict | None:
    try:
        if backend.kind == "llama-server":
            r = await client.get(f"{backend.root}/slots", timeout=timeout)
            if r.status_code == 200:
                return parse_llama_slots(r.json(), backend)
            r = await client.get(f"{backend.root}/metrics", timeout=timeout)
            return parse_llama_metrics(r.text, backend) if r.status_code == 200 else None
        r = await client.get(f"{backend.root}/status/json", timeout=timeout)
        return parse_luce_status(r.json(), backend) if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return None


class LoadMonitor:
    """Cached `{model: {"in_flight", "capacity"} | None}` for all configured models."""

    def __init__(self, models: dict[str, ModelBackend], client: httpx.AsyncClient | None = None,
                 ttl_s: float = 0.2):
        self.models = models
        self.client = client or httpx.AsyncClient()
        self.ttl_s = ttl_s
        self._cached: dict | None = None
        self._at = 0.0
        self._lock = asyncio.Lock()

    async def get(self) -> dict[str, dict | None]:
        async with self._lock:
            if self._cached is None or time.monotonic() - self._at > self.ttl_s:
                names = list(self.models)
                states = await asyncio.gather(*(backend_load(self.models[n], self.client) for n in names))
                self._cached = dict(zip(names, states))
                self._at = time.monotonic()
            return self._cached


async def load_state(models: dict[str, ModelBackend], client: httpx.AsyncClient | None = None) -> dict:
    """Uncached one-shot load query (see LoadMonitor for the cached version)."""
    if client is not None:
        return await LoadMonitor(models, client, ttl_s=0).get()
    async with httpx.AsyncClient() as own:
        return await LoadMonitor(models, own, ttl_s=0).get()
