"""Forge tool-calling evaluation area for `--area forge`.

Wraps antoinezambelli/forge's scenario suite (vendored at
``lucebench/fixtures/forge_eval/``) and drives each scenario through a
*recording* subclass of ``AnthropicClient`` so we can intercept the raw
per-call API response (stop_reason, usage, usage.timings, raw content
blocks) before forge collapses it into its parsed ``LLMResponse``.

Each scenario row carries the same shape as ds4-eval rows
(http_status, finish_reason, prompt_tokens, completion_tokens,
timings, prompt, output, …) PLUS a per-call ``iterations[]``
breakdown for forensic re-grading.

The ``anthropic`` SDK is a hard dependency (as of v0.2.6); the import
guard below is kept for graceful failure on an old install.

The vendored ``_forge`` runtime + scenarios are MIT-licensed
(antoinezambelli/forge 0.7.1); see NOTICE for full attribution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Vendored forge_eval lives next to this module (one level up, under
# fixtures/). Insert the fixtures dir on sys.path so the package
# imports as ``forge_eval`` without polluting site-packages.
_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURES_DIR))


def _forge_anthropic_finish_reason(stop_reason: str | None) -> str | None:
    """Map Anthropic stop_reason → OpenAI-shape finish_reason.

    Lets forge rows share the ds4-eval row schema's finish_reason
    field. Anthropic's lexicon:
      end_turn       → stop
      max_tokens     → length
      tool_use       → tool_calls
      stop_sequence  → stop
    """
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason or "", stop_reason)


def _forge_extract_timings(raw_usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pluck usage.timings from a raw Anthropic-shape usage dict.

    dflash-style servers attach ``prefill_ms`` / ``decode_ms`` /
    ``decode_tokens_per_sec`` inside ``usage.timings``; native
    Anthropic does not. Returns None when the server doesn't surface
    them so downstream aggregation can no-op cleanly.
    """
    if not isinstance(raw_usage, dict):
        return None
    timings = raw_usage.get("timings")
    if not isinstance(timings, dict):
        return None
    out: dict[str, Any] = {}
    for k in ("prefill_ms", "decode_ms", "decode_tokens_per_sec", "prefill_tokens_per_sec"):
        if k in timings:
            out[k] = timings[k]
    return out or None


def _forge_aggregate_timings(per_call: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Sum per-iteration timings into a scenario-level summary.

    Each forge scenario makes N sequential ``send()`` calls. We add the
    per-call ``prefill_ms`` / ``decode_ms`` and recompute the
    tokens-per-sec from the totals so the scenario row carries a
    comparable timing block (rather than the last call's timings).
    """
    valid = [t for t in per_call if isinstance(t, dict) and t]
    if not valid:
        return None
    prefill_ms = sum(float(t.get("prefill_ms") or 0) for t in valid)
    decode_ms = sum(float(t.get("decode_ms") or 0) for t in valid)
    # Aggregate tok/s recomputed at top-level after we know total tokens.
    return {
        "prefill_ms": round(prefill_ms, 1) if prefill_ms else 0.0,
        "decode_ms": round(decode_ms, 1) if decode_ms else 0.0,
        "n_calls": len(valid),
    }


def run_forge_area(
    url: str,
    *,
    model: str,
    max_tokens: int,
    timeout_s: int,
    auth_header: str,
    tags: list[str] | None = None,
    names: list[str] | None = None,
    questions: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run vendored forge scenarios through a recording AnthropicClient.

    Returns ``(rows, summary)``:
      * ``rows``: ds4-eval-shaped row dicts (one per scenario), with
        a per-call ``iterations[]`` array.
      * ``summary``: forge-specific aggregate {n_scenarios, n_pass,
        pass_rate, ...}.

    Lazy-imports forge_eval — calling code that doesn't ``--area forge``
    can avoid the anthropic-SDK dependency entirely.
    """
    import asyncio
    import json as _json
    import time as _time

    try:
        from forge_eval._forge.clients.anthropic import (  # type: ignore[import-not-found]
            AnthropicClient,
        )
        from forge_eval._forge.core.workflow import (  # type: ignore[import-not-found]
            TextResponse,
        )
    except ImportError as exc:
        raise SystemExit(
            "[lucebench] --area forge: the `anthropic` SDK should have "
            "been pulled in as a runtime dep (v0.2.6+). If you're on an "
            "older install, `pip install -U luce-bench`. "
            f"(import failed: {exc})"
        ) from exc

    try:
        from forge_eval.eval_runner import (  # type: ignore[import-not-found]
            ALL_SCENARIOS,
            EvalConfig,
            RunResult,
            run_scenario,
        )
    except ImportError as exc:
        raise SystemExit(
            "[lucebench] forge_eval fixture tree is missing — wheel was "
            f"built without it? (import failed: {exc})"
        ) from exc

    api_key = "dummy"
    if auth_header:
        # The Anthropic SDK reads x-api-key from this string. Strip
        # ``Bearer `` if the caller used --auth-env.
        api_key = auth_header.removeprefix("Bearer ").strip() or "dummy"

    # ── Recording client ──────────────────────────────────────────────
    class _RecordingAnthropicClient(AnthropicClient):  # type: ignore[misc, valid-type]
        """AnthropicClient that records every send() into iteration_log."""

        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self.iteration_log: list[dict[str, Any]] = []

        def reset_log(self) -> None:
            self.iteration_log.clear()

        async def send(  # type: ignore[override]
            self,
            messages: list[dict[str, Any]],
            tools: Any = None,
            sampling: dict[str, Any] | None = None,
            passthrough: dict[str, Any] | None = None,
            inbound_anthropic_body: dict[str, Any] | None = None,
        ) -> Any:
            try:
                prompt_blob = _json.dumps(messages, ensure_ascii=False, default=str)
            except Exception:
                prompt_blob = str(messages)

            import anthropic as _anthropic  # type: ignore[import-not-found]
            from forge_eval._forge.errors import BackendError  # type: ignore[import-not-found]

            kwargs = self._build_kwargs(
                messages,
                tools,
                passthrough,
                inbound_anthropic_body,
            )
            t0 = _time.perf_counter()
            record: dict[str, Any] = {
                "wall_s": 0.0,
                "http_status": None,
                "finish_reason": None,
                "stop_reason": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "tool_calls": [],
                "prompt": prompt_blob,
                "output": "",
                "reasoning_content": "",
                "timings": None,
                "raw_usage": None,
                "error": None,
            }
            try:
                response = await self._client.messages.create(**kwargs)
            except _anthropic.APIError as exc:
                record["wall_s"] = round(_time.perf_counter() - t0, 4)
                record["http_status"] = getattr(exc, "status_code", 0) or 0
                record["error"] = f"{type(exc).__name__}: {exc}"
                self.iteration_log.append(record)
                raise BackendError(getattr(exc, "status_code", 0), str(exc)) from exc

            record["wall_s"] = round(_time.perf_counter() - t0, 4)
            record["http_status"] = 200
            try:
                record["prompt_tokens"] = int(response.usage.input_tokens)
                record["completion_tokens"] = int(response.usage.output_tokens)
            except (AttributeError, TypeError, ValueError):
                pass
            stop_reason = getattr(response, "stop_reason", None)
            record["stop_reason"] = stop_reason
            record["finish_reason"] = _forge_anthropic_finish_reason(stop_reason)
            try:
                dumped = response.model_dump()
                raw_usage = dumped.get("usage") if isinstance(dumped, dict) else None
            except Exception:
                raw_usage = None
            record["raw_usage"] = raw_usage
            record["timings"] = _forge_extract_timings(raw_usage)

            text_parts: list[str] = []
            tool_calls_out: list[dict[str, Any]] = []
            tool_uses_present = False
            for block in getattr(response, "content", None) or []:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(getattr(block, "text", "") or "")
                elif btype == "tool_use":
                    tool_uses_present = True
                    tool_calls_out.append(
                        {
                            "name": getattr(block, "name", None),
                            "arguments": getattr(block, "input", None),
                        }
                    )
            text_join = "\n".join(p for p in text_parts if p)
            if tool_uses_present:
                record["reasoning_content"] = text_join
                record["output"] = ""
            else:
                record["output"] = text_join
            record["tool_calls"] = tool_calls_out

            self.iteration_log.append(record)
            return TextResponse(text=text_join)

    # ── Scenario selection + runner ───────────────────────────────────
    scenarios = list(ALL_SCENARIOS)
    if tags:
        tagset = set(tags)
        scenarios = [s for s in scenarios if tagset & set(getattr(s, "tags", []))]
    if names:
        nameset = set(names)
        scenarios = [s for s in scenarios if s.name in nameset]
    if questions:
        scenarios = scenarios[:questions]

    if not scenarios:
        return [], {"n_scenarios": 0, "n_pass": 0, "pass_rate": 0.0}

    rows: list[dict[str, Any]] = []
    n_pass = 0
    # EvalConfig was refactored in the vendored forge_eval (eval_runner.py:73)
    # to drop the client_factory and sampling fields. Build the client
    # per-scenario in a local helper instead; run_scenario's signature is
    # (client, scenario, config) — positional order matters.
    cfg = EvalConfig()

    def _build_client() -> _RecordingAnthropicClient:
        return _RecordingAnthropicClient(
            api_key=api_key,
            base_url=url.rstrip("/"),
            model=model,
            max_tokens=max_tokens,
            timeout=timeout_s,
        )

    for sc in scenarios:
        client = _build_client()
        client.reset_log()
        t0 = _time.perf_counter()
        try:
            res: RunResult = asyncio.run(run_scenario(client, sc, cfg))
            err = None
        except Exception as exc:
            res = None
            err = f"{type(exc).__name__}: {exc}"
        wall = round(_time.perf_counter() - t0, 3)

        graded_pass = bool(res and not res.error_type)
        if graded_pass:
            n_pass += 1
        iterations = list(client.iteration_log)
        total_prompt = sum(int(it.get("prompt_tokens") or 0) for it in iterations)
        total_comp = sum(int(it.get("completion_tokens") or 0) for it in iterations)
        agg_timings = _forge_aggregate_timings([it.get("timings") for it in iterations])
        rows.append(
            {
                "case_id": sc.name,
                "source": "forge",
                "kind": "forge-scenario",
                "pass": graded_pass,
                "graded": {
                    "pass": graded_pass,
                    "given": getattr(res, "error_type", None) or "ok",
                    "correct": "no error_type",
                    "status": "passed" if graded_pass else "failed",
                },
                "wall_seconds": wall,
                "iterations": iterations,
                "prompt_tokens": total_prompt or None,
                "completion_tokens": total_comp or None,
                "timings": agg_timings,
                "error": err or (res and res.error_type),
                "http_status": 200 if graded_pass else None,
                "finish_reason": "tool_calls"
                if iterations and iterations[-1].get("tool_calls")
                else "stop",
            }
        )

    return rows, {
        "n_scenarios": len(rows),
        "n_pass": n_pass,
        "pass_rate": 100 * n_pass / len(rows) if rows else 0.0,
    }
