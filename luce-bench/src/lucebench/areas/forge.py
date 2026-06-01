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

import json
import re
import sys
from pathlib import Path
from typing import Any

# Pattern for ``call:<verb>{`` openers. The verb part allows snake_case,
# kebab-case, dotted, or namespaced (``ns:verb``) names — same alphabet
# as ``_CALL_INVOCATION`` in lucebench.areas.agent.
#
# The leading prefix accepts ``^`` (start), whitespace, common punctuation,
# OR an underscore. The underscore is required to handle a SentencePiece
# tokenizer-residual artifact post-bragi-channel-routing where the gemma
# server occasionally emits raw tokens like ``_call:foo{...}``. Matching
# ``\bcall:`` would miss these because ``_`` is a word char and the word-
# boundary ``\b`` does not fire between ``_`` and ``c``. Verified 2026-05-31
# against gemma-4-26b smoke test on lucebox-hub:cuda12 @ 8039911.
_CALL_OPEN = re.compile(r"(?:^|(?<=[\s,;:\(\[\{\}\)\]\>_]))call:([A-Za-z0-9_.:-]+)\s*\{")


def _balanced_braces_end(text: str, start: int) -> int | None:
    """Return the index *after* the closing ``}`` that matches ``text[start] == '{'``.

    Respects nesting and skips over string literals (single + double
    quoted, with backslash escapes). Returns ``None`` if no matching
    close brace is found.
    """
    depth = 0
    i = start
    n = len(text)
    in_str: str | None = None  # None or one of ", '
    while i < n:
        ch = text[i]
        if in_str is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _coerce_relaxed_json(payload: str) -> Any:
    """Parse a relaxed JSON5-ish arg block (unquoted keys, etc.).

    The plain-text tool emissions look like ``{country: "France"}`` —
    valid JSON5 but not strict JSON. Strategy:

    1. Try ``json.loads`` first (covers cases where the model happens to
       emit valid JSON).
    2. Else quote bare keys (``foo:`` → ``"foo":``) and retry.
    3. If parsing still fails, raise ``ValueError`` so the caller can
       drop the invocation without crashing the bench.
    """
    payload = payload.strip()
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        pass

    # Permissive pass: quote bare keys. The regex matches an identifier
    # followed by ``:`` only when it isn't already inside a string. We
    # walk the text and skip string contents to avoid mangling values.
    out: list[str] = []
    i = 0
    n = len(payload)
    in_str: str | None = None
    while i < n:
        ch = payload[i]
        if in_str is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(payload[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ('"', "'"):
            # Normalize single-quoted strings to double-quoted so json.loads accepts them.
            if ch == "'":
                out.append('"')
                in_str = "'"
            else:
                out.append(ch)
                in_str = ch
            i += 1
            continue
        # Try to match a bare identifier followed by optional whitespace + ':'
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\s*:)", payload[i:])
        if m and (not out or out[-1] not in ('"',)):
            out.append('"')
            out.append(m.group(1))
            out.append('"')
            out.append(m.group(2))
            i += m.end()
            continue
        out.append(ch)
        i += 1

    # Replace any single-quoted string close markers in the rewrite. We
    # already opened them as double quotes; close them as double quotes
    # too. This is a no-op for inputs that didn't use single quotes.
    rewritten = "".join(out).replace("'", '"')
    return json.loads(rewritten)


def _strip_plain_text_tool_calls(text: str) -> str:
    """Remove every full ``call:<verb>{...}`` span from *text*.

    Used to clean a model's narrative reasoning before echoing it back
    as the assistant message that precedes a synthesized tool call —
    without this, the conversation history accumulates duplicate signal
    (the structured tool_use AND its plain-text twin) and re-train the
    model toward the wrong shape inside a single scenario.
    """
    if not text:
        return text
    out: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _CALL_OPEN.search(text, pos)
        if m is None:
            out.append(text[pos:])
            break
        out.append(text[pos : m.start()])
        brace_open = m.end() - 1
        brace_end = _balanced_braces_end(text, brace_open)
        if brace_end is None:
            # Unbalanced — leave the rest as-is (we couldn't have
            # synthesized a ToolCall from this span anyway).
            out.append(text[m.start():])
            break
        pos = brace_end
    return "".join(out)


def _parse_plain_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract ``call:<verb>{args}`` invocations from a plain-text response.

    Returns a list of ``{"name": <verb>, "input": <dict>}`` dicts —
    same shape as Anthropic ``tool_use`` content blocks minus the SDK
    object overhead — preserving emission order. The caller wraps each
    entry in a ``ToolCall`` (forge-internal type) or a record dict
    (snapshot ``tool_calls`` field).

    Malformed args (unparseable even after the permissive pass) cause
    that single invocation to be dropped — no exception escapes, no
    placeholder tool_use is synthesized. This keeps a partially-mangled
    response from crashing the bench while still surfacing the
    correctly-formatted calls that precede or follow it.
    """
    if not text:
        return []
    results: list[dict[str, Any]] = []
    pos = 0
    while True:
        m = _CALL_OPEN.search(text, pos)
        if m is None:
            break
        name = m.group(1)
        brace_open = m.end() - 1  # index of the '{' itself
        brace_end = _balanced_braces_end(text, brace_open)
        if brace_end is None:
            # Unbalanced — stop scanning (the rest of the text can't
            # reliably contain more calls if we got the bracket count
            # wrong).
            break
        payload = text[brace_open + 1 : brace_end - 1]
        try:
            args = _coerce_relaxed_json("{" + payload + "}")
            if not isinstance(args, dict):
                raise ValueError("tool args must be a JSON object")
        except (ValueError, json.JSONDecodeError):
            # Drop this invocation, keep scanning past it.
            pos = brace_end
            continue
        results.append({"name": name, "input": args})
        pos = brace_end
    return results

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
            ToolCall,
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

            # If the server returned proper ``tool_use`` content blocks
            # we hand them to forge as-is. Otherwise — and this is the
            # gemma case (2026-05-30 bench), where the model emits
            # ``call:<verb>{...}`` as inline text instead of structured
            # blocks — scan the text for those invocations and
            # synthesize ToolCall entries so forge's validator sees the
            # tool calls it expects. This client-side synthesis
            # future-proofs the bench for any model that uses the same
            # plain-text tool serialization (codex-mini, DDX bead
            # executor, etc.) without requiring a server-side fix.
            synthesized: list[dict[str, Any]] = []
            if not tool_uses_present and text_join:
                synthesized = _parse_plain_text_tool_calls(text_join)
                for syn in synthesized:
                    tool_calls_out.append(
                        {"name": syn["name"], "arguments": syn["input"]}
                    )

            had_tool_calls = tool_uses_present or bool(synthesized)
            if had_tool_calls:
                record["reasoning_content"] = text_join
                record["output"] = ""
            else:
                record["output"] = text_join
            record["tool_calls"] = tool_calls_out

            self.iteration_log.append(record)

            # Build a forge-native LLMResponse. The contract
            # (forge_eval._forge.core.workflow.LLMResponse) is
            # ``list[ToolCall] | TextResponse``. We previously returned
            # ``TextResponse(text=...)`` which raised a pydantic
            # ValidationError every call (the field is named
            # ``content``, not ``text``) — that's the
            # ``error_type=ValidationError`` seen across the 2026-05-30
            # gemma full bench's forge rows.
            if tool_uses_present:
                reasoning = text_join or None
                return [
                    ToolCall(
                        tool=getattr(block, "name", ""),
                        args=dict(getattr(block, "input", {}) or {}),
                        reasoning=reasoning if i == 0 else None,
                    )
                    for i, block in enumerate(
                        b for b in (getattr(response, "content", None) or [])
                        if getattr(b, "type", None) == "tool_use"
                    )
                ]
            if synthesized:
                # Strip the synthesized call:<verb>{...} fragments out
                # of the reasoning text so it isn't echoed back to the
                # model as both a tool_call AND its plain-text twin.
                cleaned = _strip_plain_text_tool_calls(text_join).strip() or None
                return [
                    ToolCall(
                        tool=syn["name"],
                        args=dict(syn["input"]),
                        reasoning=cleaned if i == 0 else None,
                    )
                    for i, syn in enumerate(synthesized)
                ]
            return TextResponse(content=text_join)

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
