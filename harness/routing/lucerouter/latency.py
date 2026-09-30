"""Expected end-to-end latency of a request on each arm (the latency term of brick_skill).

    Lat_m(x) = queue_wait_m + ttft_ms_m + ttft_ms_per_token_m * prompt_tokens(x)
               + tpot_ms_m * expected_output_tokens_m(x)

* prompt_tokens(x) = prompt characters / `chars_per_token` (no tokenizer in the router).
* expected_output_tokens_m(x) = sum over easy/medium/hard of p_class(x) * out_tokens_m[class],
  falling back to out_tokens_m["default"] when the profile has no per-class values or the
  complexity probs are missing, and capped at the request's max_tokens. For a thinking arm the
  profile's output tokens include the reasoning tokens (luce_server counts every generated
  token in usage.completion_tokens), so the think arm's long reasoning is priced in.
* queue_wait_m = in_flight / capacity * service_ms of the arm's *backend* (both 27B arms share
  one queue), from the live LoadMonitor. 0 when load is unknown, the backend is offline, or
  live load is off (offline eval).

Profile JSON (written by eval/fit_latency.py; `latency_defaults` in backends.json has the same
shape and is used for arms the file does not cover):

    {"chars_per_token": 3.8,
     "arms": {"<arm>": {"ttft_ms": 40, "ttft_ms_per_token": 0.05, "tpot_ms": 8,
                        "out_tokens": {"default": 300, "easy": 150, "medium": 350, "hard": 700}}},
     "backends": {"<model>": {"service_ms": 2500}}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CLASSES = ("easy", "medium", "hard")
DEFAULT_CHARS_PER_TOKEN = 4.0


@dataclass
class ArmLatency:
    ttft_ms: float
    ttft_ms_per_token: float
    tpot_ms: float
    out_tokens: dict[str, float]

    def expected_output_tokens(self, complexity: dict[str, float] | None, max_tokens: int | None = None) -> float:
        n = float(self.out_tokens.get("default", 0.0))
        if complexity and all(c in self.out_tokens for c in CLASSES):
            total = sum(complexity.get(c, 0.0) for c in CLASSES)
            if total > 0:
                n = sum(complexity.get(c, 0.0) * self.out_tokens[c] for c in CLASSES) / total
        return min(n, max_tokens) if max_tokens else n

    def service_ms(self, prompt_tokens: float = 0.0, out_tokens: float | None = None) -> float:
        out = self.out_tokens.get("default", 0.0) if out_tokens is None else out_tokens
        return self.ttft_ms + self.ttft_ms_per_token * prompt_tokens + self.tpot_ms * out


@dataclass
class LatencyProfile:
    arms: dict[str, ArmLatency]
    service_ms: dict[str, float]      # per backend model
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN

    @classmethod
    def from_dict(cls, data: dict, fallback: dict | None = None) -> "LatencyProfile":
        fallback = fallback or {}
        arms_raw = {**(fallback.get("arms") or {}), **(data.get("arms") or {})}
        backends_raw = {**(fallback.get("backends") or {}), **(data.get("backends") or {})}
        arms = {name: ArmLatency(ttft_ms=float(a.get("ttft_ms", 0.0)),
                                 ttft_ms_per_token=float(a.get("ttft_ms_per_token", 0.0)),
                                 tpot_ms=float(a["tpot_ms"]),
                                 out_tokens={k: float(v) for k, v in (a.get("out_tokens") or {}).items()})
                for name, a in arms_raw.items()}
        cpt = data.get("chars_per_token") or fallback.get("chars_per_token") or DEFAULT_CHARS_PER_TOKEN
        return cls(arms=arms, service_ms={m: float(b["service_ms"]) for m, b in backends_raw.items()
                                          if b.get("service_ms") is not None},
                   chars_per_token=float(cpt))

    @classmethod
    def load(cls, path: str | Path | None, fallback: dict | None = None) -> "LatencyProfile":
        data = json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
        return cls.from_dict(data, fallback)

    def prompt_tokens(self, prompt_chars: int) -> float:
        return prompt_chars / self.chars_per_token

    def backend_service_ms(self, backend: str, arms_on_backend: list[str]) -> float:
        """Mean service time of one request on ``backend``: fitted, else the mean of its arms."""
        if backend in self.service_ms:
            return self.service_ms[backend]
        vals = [self.arms[a].service_ms() for a in arms_on_backend if a in self.arms]
        return sum(vals) / len(vals) if vals else 0.0


def queue_wait_ms(load: dict | None, service_ms: float) -> float:
    """in_flight / capacity x mean service time; 0 when load is unknown."""
    if not load or not load.get("capacity"):
        return 0.0
    return max(0.0, load["in_flight"] / load["capacity"]) * service_ms


def expected_latency_ms(arm: ArmLatency, prompt_tokens: float, complexity: dict[str, float] | None,
                        max_tokens: int | None = None, queue_ms: float = 0.0) -> dict[str, float]:
    out = arm.expected_output_tokens(complexity, max_tokens)
    ttft = arm.ttft_ms + arm.ttft_ms_per_token * prompt_tokens
    decode = arm.tpot_ms * out
    return {"total_ms": queue_ms + ttft + decode, "queue_ms": queue_ms, "ttft_ms": ttft,
            "decode_ms": decode, "out_tokens": out}
