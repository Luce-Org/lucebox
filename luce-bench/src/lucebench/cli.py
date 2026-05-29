"""Command-line entry point: ``lucebench --area X --url Y --model Z``.

Minimal dispatcher around lucebench.runner — exposes parallelism,
forge / agent areas, sampling-from-card, and per-area max_tokens
defaults so external users can `pip install luce-bench` and benchmark
any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lucebench import __version__
from lucebench._thinking import verify_thinking_control
from lucebench.areas import (
    agent,
    agent_recorded,
    ds4_eval,
    gsm8k,
    hellaswag,
    humaneval,
    longctx,
    smoke,
    truthfulqa_mc1,
)
from lucebench.runner import run_case


def _summarize_injection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll the per-row ``_thinking_injection`` echoes into a single block.

    The runner stamps the same injection dict on every row (the resolution
    is identical across a run), so we just pick the first non-empty one
    and surface it at the top level. Returns the canonical inactive block
    when no row carries the field — e.g. a run with control_flag='off',
    or a sweep area that ran before the feature shipped.
    """
    for r in rows:
        info = r.get("_thinking_injection")
        if isinstance(info, dict):
            return info
    return {"active": False, "token": None, "source": "none"}

# Threshold below which we'll auto-pick the first model and surface the
# full list. Gateways with hundreds of models still need an explicit
# --model — silently picking from a long list masks user mistakes.
_SMALL_MODEL_LIST_THRESHOLD = 5


def resolve_model(url: str, auth_header: str = "", timeout_s: int = 10) -> str | None:
    """Pick a model id by probing the server's /v1/models endpoint.

    Returns:
      * the single model id if the server exposes exactly one
      * the first model id if the server exposes 2..4 (small list —
        likely a single-model server with aliases). The full list is
        printed by the caller via :func:`list_models` so the choice
        is visible.
      * None if the server exposes zero, 5+, or doesn't speak the
        OpenAI /v1/models shape.
    """
    chosen, _ = _list_models(url, auth_header=auth_header, timeout_s=timeout_s)
    return chosen


def list_models(
    url: str, auth_header: str = "", timeout_s: int = 10
) -> tuple[str | None, list[str]]:
    """Same as :func:`resolve_model` but also returns the full model id
    list (or an empty list on probe failure). Callers use this to surface
    the available models alongside the auto-pick.
    """
    return _list_models(url, auth_header=auth_header, timeout_s=timeout_s)


def _list_models(
    url: str, auth_header: str = "", timeout_s: int = 10
) -> tuple[str | None, list[str]]:
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/models", headers={"Accept": "application/json"}
    )
    if auth_header:
        req.add_header("Authorization", auth_header)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None, []
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None, []
    ids: list[str] = []
    for entry in models:
        if isinstance(entry, dict):
            mid = entry.get("id")
            if isinstance(mid, str) and mid:
                ids.append(mid)
    if not ids:
        return None, []
    # Auto-pick when the list is short enough to be useful — gateways
    # with 5+ models still require an explicit --model.
    if len(ids) < _SMALL_MODEL_LIST_THRESHOLD:
        return ids[0], ids
    return None, ids


AREAS = {
    "smoke": {
        "load": smoke.load_smoke_cases,
        "grade": smoke.grade_smoke_case,
        # Roomy. The prompts only need a few tokens of actual answer,
        # but servers with thinking on (ds4-server forces it, ignoring
        # the client's `thinking: disabled`) can spend thousands of
        # tokens on reasoning before emitting visible content. Most
        # servers will EOS naturally well before the cap on these
        # short prompts; the budget just keeps "model trips length
        # mid-think" out of the smoke failure modes.
        "default_max_tokens": 4096,
        "default_thinking": False,
    },
    "ds4-eval": {
        "load": ds4_eval.load_ds4_eval_cases,
        "grade": ds4_eval.grade_case,
        "default_max_tokens": ds4_eval.DS4_EVAL_MAX_TOKENS,
        "default_thinking": True,
    },
    "gsm8k": {
        "load": gsm8k.load_gsm8k_cases,
        "grade": gsm8k.grade_gsm8k_case,
        "default_max_tokens": gsm8k.GSM8K_MAX_TOKENS,
        # 0-shot, raw model behavior. Users who want CoT pass --think.
        "default_thinking": False,
    },
    "truthfulqa-mc1": {
        "load": truthfulqa_mc1.load_truthfulqa_mc1_cases,
        "grade": truthfulqa_mc1.grade_truthfulqa_mc1_case,
        "default_max_tokens": truthfulqa_mc1.TRUTHFULQA_MC1_MAX_TOKENS,
        "default_thinking": False,
    },
    "hellaswag": {
        "load": hellaswag.load_hellaswag_cases,
        "grade": hellaswag.grade_hellaswag_case,
        "default_max_tokens": hellaswag.HELLASWAG_MAX_TOKENS,
        "default_thinking": False,
    },
    "code": {
        "load": humaneval.load_humaneval_cases,
        "grade": humaneval.grade_humaneval_case,
        "default_max_tokens": 2048,
        "default_thinking": False,
    },
    "longctx": {
        "load": lambda: longctx.LONGCTX_CASES,
        "grade": longctx.grade_longctx_case,
        "default_max_tokens": 256,
        "default_thinking": False,
    },
    "agent": {
        "load": agent.load_agent_cases,
        "grade": agent.grade_agent_case,
        "default_max_tokens": 4096,
        "default_thinking": False,
    },
    "agent_recorded": {
        "load": agent_recorded.load_agent_recorded_cases,
        "grade": agent_recorded.grade_agent_recorded_case,
        "default_max_tokens": 4096,
        "default_thinking": False,
    },
}


def select_cases(
    cases: list[dict],
    *,
    questions: int | None = None,
    case_id: str | None = None,
    case_index: int | None = None,
    sources: list[str] | None = None,
) -> list[dict]:
    """Filter cases by id / index / source / count."""
    out = list(cases)
    if sources:
        out = [c for c in out if c.get("source") in sources]
    if case_id:
        out = [c for c in out if c.get("id") == case_id]
    if case_index is not None:
        out = out[case_index : case_index + 1] if 0 <= case_index < len(out) else []
    if questions:
        out = out[:questions]
    return out


def format_row(idx: int, row: dict, graded: dict) -> str:
    src = row.get("source") or "?"
    cid = row.get("case_id") or "?"
    verdict = "PASS" if graded.get("pass") else "FAIL"
    given = graded.get("given") or "?"
    correct = graded.get("correct") or "?"
    wall = row.get("wall_seconds") or 0
    timings = row.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}

    # ── Throughput. Prefer the server-reported decode rate (lucebox /
    # llama.cpp populate `decode_tokens_per_sec`); fall back to a wall-
    # clock estimate so OpenRouter / vLLM (which don't surface decode_tps)
    # don't always read "0tps". The fallback rolls prefill into the rate,
    # so mark it with a trailing `*` to keep the distinction visible.
    #
    # Two display refinements that prevent the noisy-but-useless "0tps*"
    # case (e.g. OpenRouter, smoke prompts emitting only 2 tokens — the
    # rate is then dominated by routing/first-token latency, not decode):
    #   1) When the fallback completion count is below 8 tokens, skip the
    #      rate entirely and show `out=N` only — the math measures
    #      router overhead, not decode.
    #   2) Sub-10tps values render with one decimal so 0.3 doesn't round
    #      down to 0.
    def _fmt_tps(v: float) -> str:
        if v < 10:
            return f"{v:.1f}"
        return f"{v:.0f}"

    tps_val = timings.get("decode_tokens_per_sec")
    completion_tokens = row.get("completion_tokens")
    _FALLBACK_MIN_TOKENS = 8
    if tps_val:
        tps_str = f"{_fmt_tps(tps_val)}tps"
    elif (
        completion_tokens
        and isinstance(completion_tokens, int)
        and completion_tokens >= _FALLBACK_MIN_TOKENS
        and wall
        and wall > 0
    ):
        tps_str = f"{_fmt_tps(completion_tokens / wall)}tps*"
    else:
        # Either no usable count or too few tokens to be meaningful — leave
        # the rate column off rather than print a number dominated by
        # prefill/router latency.
        tps_str = ""

    # ── Prefill / decode split. lucebox-server surfaces both in
    # `usage.timings` (prefill_ms + decode_ms); OpenRouter / vLLM
    # typically surface neither. Render whichever pair is available; if
    # both are missing fall back to the plain wall time.
    prefill_ms = timings.get("prefill_ms")
    decode_ms = timings.get("decode_ms")

    def _fmt_ms(ms: float) -> str:
        # Sub-second renders as e.g. "210ms"; >=1s as "3.5s" to keep the line tight.
        if ms < 1000:
            return f"{ms:.0f}ms"
        return f"{ms / 1000:.1f}s"

    # ── Time-to-first-token. Server-reported `prefill_ms` is the gold
    # standard (no network RTT, no SSE framing overhead). Streaming runs
    # also capture a wall-clock TTFT — useful for OpenRouter / vLLM where
    # the server doesn't ship prefill_ms. When both are present prefer
    # the server value and drop the wall-clock duplicate; when only the
    # streaming measurement is available mark it with `*` (same convention
    # as the tps fallback above).
    ttft_seconds = row.get("ttft_seconds")
    ttft_ms: float | None = ttft_seconds * 1000 if isinstance(ttft_seconds, int | float) else None

    time_parts: list[str] = []
    if prefill_ms is not None and decode_ms is not None:
        time_parts.append(f"prefill={_fmt_ms(prefill_ms)}")
        time_parts.append(f"decode={_fmt_ms(decode_ms)}")
    elif prefill_ms is not None:
        time_parts.append(f"prefill={_fmt_ms(prefill_ms)} wall={wall:.2f}s")
    elif ttft_ms is not None and decode_ms is not None:
        time_parts.append(f"ttft={_fmt_ms(ttft_ms)}* decode={_fmt_ms(decode_ms)}")
    elif ttft_ms is not None:
        time_parts.append(f"ttft={_fmt_ms(ttft_ms)}* wall={wall:.2f}s")
    elif decode_ms is not None:
        time_parts.append(f"decode={_fmt_ms(decode_ms)} wall={wall:.2f}s")
    else:
        time_parts.append(f"wall={wall:.2f}s")
    time_str = " ".join(time_parts)

    # ── Token breakdown: input / thinking / non-thinking. `reasoning_tokens`
    # is captured by runner.run_case from `usage.completion_tokens_details`
    # (OpenAI/OR) or the deprecated top-level `usage.reasoning_tokens`. We
    # do NOT count tokens ourselves — no tokenizer dep — so when the server
    # only ships `reasoning_content` text we leave `think` out and show `out`
    # as the full completion_tokens count.
    prompt_tokens = row.get("prompt_tokens")
    reasoning_tokens = row.get("reasoning_tokens")
    tok_bits: list[str] = []
    if prompt_tokens is not None:
        tok_bits.append(f"in={prompt_tokens}")
    if isinstance(reasoning_tokens, int) and isinstance(completion_tokens, int):
        non_thinking = max(completion_tokens - reasoning_tokens, 0)
        tok_bits.append(f"think={reasoning_tokens}")
        tok_bits.append(f"out={non_thinking}")
    elif completion_tokens is not None:
        tok_bits.append(f"out={completion_tokens}")
    tok_str = " ".join(tok_bits)

    tail_bits = [time_str]
    if tps_str:
        tail_bits.append(tps_str)
    if tok_str:
        tail_bits.append(tok_str)
    return (
        f"  {idx:3d} {verdict} {src:14s} {cid:24s} "
        f"given={given:20s} correct={correct:20s} " + " ".join(tail_bits)
    )


# Substrings in row["error"] that mean the server is unreachable — fail-fast
# triggers on the first row matching any of these unless --no-fail-fast is set.
_UNREACHABLE_ERRORS = (
    "ConnectionRefusedError",
    "ConnectionResetError",
    "Name or service not known",
    "Temporary failure in name resolution",
    "No route to host",
    "Connection refused",
    "URLError",
)


def _row_is_unreachable(row: dict) -> bool:
    """True if row["error"] looks like a connection-level failure.

    Used by the sweep's fail-fast guard. Timeouts and HTTP errors are
    deliberately excluded — those are per-request failures, not a
    server-down signal.
    """
    err = row.get("error") or ""
    return any(marker in err for marker in _UNREACHABLE_ERRORS)


def _format_models_inline(ids: list[str], selected: str, budget: int = 62) -> str:
    """Render a comma-separated `/v1/models` listing for the preflight grid.

    Marks the chosen id with a `*` prefix. If the full list fits in
    `budget` characters, it's shown verbatim. Otherwise the layout is:
    first model, then the selected model (if different), then sequential
    fillers until the budget is hit, ending with `… (+N more)`.
    """
    if not ids:
        return "(none)"

    def render(picked_idx: list[int], remaining: int) -> str:
        parts = [(f"*{ids[i]}" if ids[i] == selected else ids[i]) for i in picked_idx]
        s = ", ".join(parts)
        if remaining:
            s += f", … (+{remaining} more)"
        return s

    full = render(list(range(len(ids))), 0)
    if len(full) <= budget:
        return full

    picked = [0]
    if selected in ids and ids[0] != selected:
        picked.append(ids.index(selected))
    for i in range(1, len(ids)):
        if i in picked:
            continue
        candidate = sorted(picked + [i])
        remaining = len(ids) - len(candidate)
        if len(render(candidate, remaining)) > budget:
            break
        picked = candidate
    remaining = len(ids) - len(picked)
    return render(sorted(picked), remaining)


def _preflight(
    url: str,
    *,
    auth_header: str = "",
    timeout_s: int = 5,
    requested_model: str | None = None,
) -> tuple[bool, list[str], bool]:
    """Probe the server's liveness + OpenAI shape + lucebox /props endpoint.

    Returns ``(ok, lines, server_honors_api_flags)`` where ``lines`` is
    the printed grid (already formatted, one check per line), ``ok`` is
    False iff a HARD check failed — which is "liveness" or "/v1/models
    doesn't return a data list" — and ``server_honors_api_flags`` is True
    iff the server's /props response surfaces ``model_card_source`` (the
    marker that this is a lucebox stack which enforces thinking control
    server-side). The /props check is lucebox-specific: missing/404 prints
    a warning line but does NOT fail (OpenRouter, vLLM, stock ds4_server
    don't expose /props), and in those cases ``server_honors_api_flags``
    defaults to False so the client-side injection can take over.

    Designed to run before any case fires so a typo'd --url surfaces in
    ~50ms instead of after 92 timeouts. The CLI gates this behind
    ``--no-preflight`` for the rare case where preflight gets in the way
    (e.g. CI testing against a deliberately-flaky endpoint).
    """
    import time as _time

    base = url.rstrip("/")
    lines: list[str] = [f"[lucebench] preflight {url}"]

    def _line(name: str, ok: bool, detail: str) -> str:
        mark = "✓" if ok else "✗"  # ✓ / ✗
        return f"  {name:12s} {mark}  {detail}"

    # 1. Liveness — GET /v1/models with a tight timeout. Reusing the
    # /v1/models endpoint (rather than a bare TCP connect) gives us a
    # cheap two-for-one: if it returns JSON we already know the server
    # speaks the OpenAI shape, so check #2 reuses the response.
    req = urllib.request.Request(base + "/v1/models", headers={"Accept": "application/json"})
    if auth_header:
        req.add_header("Authorization", auth_header)
    t0 = _time.perf_counter()
    models_payload: Any = None
    liveness_ok = False
    liveness_detail = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
        liveness_ok = True
        liveness_detail = f"reached in {_time.perf_counter() - t0:.2f}s"
        try:
            models_payload = json.loads(body)
        except ValueError:
            models_payload = None
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        liveness_detail = (
            f"connection refused ({reason})" if "refused" in str(reason).lower() else str(reason)
        )
    except OSError as e:
        liveness_detail = f"{type(e).__name__}: {e}"
    except Exception as e:  # last-resort guard so preflight never raises
        liveness_detail = f"{type(e).__name__}: {e}"
    lines.append(_line("liveness", liveness_ok, liveness_detail))
    if not liveness_ok:
        return False, lines, False

    # 2. /v1/models shape — OpenAI-compat servers return {"data": [...]}.
    models_ok = False
    models_detail = ""
    if isinstance(models_payload, dict):
        data = models_payload.get("data")
        if isinstance(data, list):
            ids = [
                m.get("id") for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)
            ]
            if not ids:
                models_detail = "0 models exposed"
            else:
                models_ok = True
                # Selected = explicit --model if in the list; else first.
                # The `*` marker visualizes what the bench would send.
                if requested_model and requested_model != "default" and requested_model in ids:
                    selected = requested_model
                else:
                    selected = ids[0]
                models_detail = _format_models_inline(ids, selected)
        else:
            models_detail = "response missing 'data' list"
    else:
        models_detail = "response was not JSON"
    lines.append(_line("/v1/models", models_ok, models_detail))
    if not models_ok:
        return False, lines, False

    # 3. /props — lucebox-specific. Soft check: warn if absent, surface
    # the image identity + target GGUF basename + model_card_source when
    # the server is new enough to expose them (props_schema >= 3); fall
    # back to the schema-2 model_card + reply_budget display on older
    # servers.
    props_req = urllib.request.Request(base + "/props", headers={"Accept": "application/json"})
    if auth_header:
        props_req.add_header("Authorization", auth_header)
    try:
        with urllib.request.urlopen(props_req, timeout=timeout_s) as resp:
            props = json.loads(resp.read())
    except Exception:
        # Not a hard failure — OpenRouter, vLLM, ds4_server don't expose this.
        # ``server_honors_api_flags=False`` here is what flips the auto-mode
        # client-side thinking injection on by default for these stacks.
        lines.append(_line("/props", True, "absent (non-lucebox server) — skipped"))
        return True, lines, False

    bits: list[str] = []

    # `build` (schema 3+): image_tag + short git_sha → "image=<tag>@<sha7>"
    # so an operator scanning a bench log can pin the exact prebuilt image.
    # Fall back gracefully when the server is pre-schema-3 (no `build`
    # block) or when the fields are null (bare-metal / non-Docker builds).
    if isinstance(props, dict):
        build = props.get("build")
        if isinstance(build, dict):
            tag = build.get("image_tag")
            git_sha = build.get("git_sha")
            short_sha = git_sha[:7] if isinstance(git_sha, str) and git_sha else None
            if tag and short_sha:
                bits.append(f"image={tag}@{short_sha}")
            elif tag:
                bits.append(f"image={tag}")
            elif short_sha:
                bits.append(f"image=@{short_sha}")

        # `model.target` (schema 3+): GGUF basename + quant tag. Strips
        # the `.gguf` suffix so the line stays narrow.
        model = props.get("model")
        if isinstance(model, dict):
            target = model.get("target")
            if isinstance(target, dict):
                path = target.get("path")
                if isinstance(path, str) and path:
                    stem = path.rsplit("/", 1)[-1]
                    if stem.endswith(".gguf"):
                        stem = stem[: -len(".gguf")]
                    bits.append(f"target={stem}")

    # `budget_envelope` (schema 2+): card lookup hit + reply budget. Kept
    # in the line even when the schema-3 fields are present — operators
    # debugging budget-envelope bugs find this faster than digging through
    # the full `/props` body.
    env = props.get("budget_envelope") if isinstance(props, dict) else None
    env = env if isinstance(env, dict) else {}
    card = env.get("model_card_source") or (
        props.get("model_card_source") if isinstance(props, dict) else None
    )
    reply = env.get("hard_limit_reply_budget")
    if card:
        bits.append(f"model_card={card}")
    if reply is not None:
        bits.append(f"reply_budget={reply}")

    detail = "  ".join(bits) if bits else "present (no envelope fields)"
    lines.append(_line("/props", True, detail))
    # ``model_card_source`` is the lucebox-stack tell: a server that surfaces
    # which sidecar card it loaded is enforcing thinking control + reply
    # budget server-side via the chat template, so the auto-mode client-side
    # injection should stand down.
    server_honors = bool(card)
    return True, lines, server_honors


def _forge_available() -> tuple[bool, str | None]:
    """Probe whether the `[forge]` extra is installed without importing it eagerly.

    Returns (available, reason) where reason is a short string the
    sweep prints when forge is skipped. Lazy import keeps the default
    install free of the anthropic dep.
    """
    try:
        import anthropic  # noqa: F401

        return True, None
    except ImportError:
        return False, "anthropic SDK not installed — `pip install 'luce-bench[forge]'`"


def _run_forge_area_to_dir(
    *,
    out_root: Path,
    url: str,
    model: str,
    auth_header: str,
    timeout: int,
    max_tokens: int | None,
    questions: int | None,
) -> dict[str, Any] | None:
    """Drive the forge area + write ``<out_root>/forge.json``.

    Returns the per-area summary row (the dict appended to
    ``summary_areas``) or ``None`` if the forge runner raised
    ``SystemExit`` (e.g. no anthropic SDK installed).
    """
    from lucebench.areas.forge import run_forge_area

    max_tokens_forge = max_tokens if max_tokens is not None else 4096
    print(
        f"\n[lucebench] === area=forge max_tokens={max_tokens_forge} ===",
        flush=True,
    )
    try:
        forge_rows, forge_summary = run_forge_area(
            url=url,
            model=model,
            max_tokens=max_tokens_forge,
            timeout_s=timeout,
            auth_header=auth_header,
            questions=questions,
        )
    except SystemExit as exc:
        print(f"[lucebench] forge: {exc}", file=sys.stderr, flush=True)
        return None
    (out_root / "forge.json").write_text(
        json.dumps(
            {
                "lucebench_version": __version__,
                "area": "forge",
                "url": url,
                "model": model,
                **forge_summary,
                "rows": forge_rows,
            },
            indent=2,
            default=str,
        )
    )
    print(
        f"[lucebench] area=forge pass_rate={forge_summary.get('pass_rate', 0):.2f}% "
        f"({forge_summary.get('n_pass', 0)}/{forge_summary.get('n_scenarios', 0)})",
        flush=True,
    )
    return {
        "area": "forge",
        "n": forge_summary.get("n_scenarios", 0),
        "pass": forge_summary.get("n_pass", 0),
        "rate": forge_summary.get("pass_rate", 0.0),
        "wall_total": sum(r.get("wall_seconds") or 0 for r in forge_rows),
        "wall_median": (
            statistics.median([r.get("wall_seconds") or 0 for r in forge_rows])
            if forge_rows
            else 0
        ),
    }


def _run_standard_area_to_dir(
    area: str,
    *,
    out_root: Path,
    url: str,
    model: str,
    auth_header: str,
    timeout: int,
    max_tokens: int | None,
    think: bool | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    questions: int | None,
    no_fail_fast: bool,
    prompt_thinking_control: str,
    server_honors_api_flags: bool,
) -> tuple[dict[str, Any] | None, bool]:
    """Drive a single stdlib area into ``<out_root>/<area>.json``.

    Returns ``(summary_row, aborted)`` where ``aborted`` is ``True`` when
    the fail-fast guard tripped on the first case (server unreachable).
    """
    cfg = AREAS[area]
    cases = cfg["load"]()
    cases = select_cases(cases, questions=questions)
    chosen_max_tokens = max_tokens if max_tokens is not None else cfg["default_max_tokens"]
    chosen_think = think if think is not None else cfg["default_thinking"]
    print(
        f"\n[lucebench] === area={area} cases={len(cases)} think={chosen_think} "
        f"max_tokens={chosen_max_tokens} ===",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        row = run_case(
            url=url,
            case=case,
            timeout_s=timeout,
            max_tokens=chosen_max_tokens,
            think=chosen_think,
            model=model,
            auth_header=auth_header,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            thinking_control_flag=prompt_thinking_control,
            server_honors_api_flags=server_honors_api_flags,
        )
        graded = cfg["grade"](case, row)
        row["pass"] = graded.get("pass", False)
        row["graded"] = graded
        rows.append(row)
        print(format_row(idx, row, graded), flush=True)
        if idx == 1 and not no_fail_fast and _row_is_unreachable(row):
            print(
                f"\n[lucebench] sweep aborted — server at {url} appears "
                f"unreachable (case 1 raised {row.get('error')!r}). "
                "Pass --no-fail-fast to keep going anyway.",
                file=sys.stderr,
                flush=True,
            )
            return None, True

    pass_n = sum(1 for r in rows if r["pass"])
    rate = 100 * pass_n / len(rows) if rows else 0
    walls = [r.get("wall_seconds") or 0 for r in rows]
    wall_total = sum(walls)
    wall_median = statistics.median(walls) if walls else 0
    print(
        f"[lucebench] area={area} pass_rate={rate:.2f}% "
        f"({pass_n}/{len(rows)}) wall_total={wall_total:.0f}s",
        flush=True,
    )

    requested_mode = "think" if chosen_think else "nothink"
    honored, contradicting = verify_thinking_control(rows, requested_mode)
    injection_summary = _summarize_injection(rows)
    if not honored:
        host = url.split("://", 1)[1].split("/", 1)[0] if "://" in url else url
        print(
            f"[lucebench] WARNING: thinking control not honored at {host} — "
            f"{contradicting}/{len(rows)} rows in {requested_mode} mode have "
            f"non-empty reasoning. Consider --prompt-thinking-control=on or "
            f"pick a model card with an explicit thinking_control block.",
            file=sys.stderr,
            flush=True,
        )

    terse = [
        {k: v for k, v in r.items() if k not in {"_response", "_thinking_injection"}}
        for r in rows
    ]
    (out_root / f"{area}.json").write_text(
        json.dumps(
            {
                "lucebench_version": __version__,
                "area": area,
                "url": url,
                "model": model,
                "think": chosen_think,
                "max_tokens": chosen_max_tokens,
                "n": len(rows),
                "pass": pass_n,
                "pass_rate": rate,
                "wall_total": wall_total,
                "wall_median": wall_median,
                "thinking_control_requested": requested_mode,
                "thinking_control_honored": honored,
                "contradicting_rows": contradicting,
                "thinking_control_injection": injection_summary,
                "rows": terse,
            },
            indent=2,
        )
    )
    return (
        {
            "area": area,
            "n": len(rows),
            "pass": pass_n,
            "rate": rate,
            "wall_total": wall_total,
            "wall_median": wall_median,
        },
        False,
    )


def write_sweep_summary(
    out_root: Path,
    *,
    name: str,
    url: str,
    model: str,
    summary_areas: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``_summary.json`` + ``_summary.md`` to ``out_root`` and return the JSON payload.

    ``extra`` is shallow-merged into the JSON payload — used by the
    snapshot subcommand to record ``level`` next to the area roll-up so
    downstream tools (``submit-baseline``) can validate the snapshot
    against the requested tier.
    """
    summary: dict[str, Any] = {
        "lucebench_version": __version__,
        "name": name,
        "url": url,
        "model": model,
        "areas": summary_areas,
    }
    if extra:
        summary.update(extra)
    (out_root / "_summary.json").write_text(json.dumps(summary, indent=2))

    md_lines = [
        f"# luce-bench sweep — {name}",
        "",
        f"- url:   `{url}`",
        f"- model: `{model}`",
        f"- lucebench v{__version__}",
        "",
        "| area | n | pass | rate | wall_total | wall_median |",
        "|------|---|------|------|------------|-------------|",
    ]
    for a in summary_areas:
        md_lines.append(
            f"| {a['area']} | {a['n']} | {a['pass']} | "
            f"{a['rate']:.1f}% | {a['wall_total']:.0f}s | {a['wall_median']:.1f}s |"
        )
    (out_root / "_summary.md").write_text("\n".join(md_lines) + "\n")
    return summary


def _run_sweep(args) -> int:
    """Run every stdlib area in sequence, write per-area + combined JSON.

    Layout:
        <out_dir>/<name>/
            ds4-eval.json
            code.json
            longctx.json
            agent.json
            forge.json       # only when [forge] is installed; skipped with a hint otherwise
            _summary.json    # {areas: [{area, n, pass, rate, wall_s}, ...]}
            _summary.md
    """
    import datetime as _dt

    name = args.name or _dt.date.today().isoformat() + "-sweep"
    out_root = args.out_dir / name
    out_root.mkdir(parents=True, exist_ok=True)

    # The set of areas to run is supplied by main() in args.areas_list
    # (computed from --areas, with back-compat for --area).
    sweep_areas = list(args.areas_list)
    forge_ok, forge_reason = _forge_available()
    auth_header = ""
    if args.auth_env:
        token = os.environ.get(args.auth_env, "")
        if not token:
            print(f"--auth-env {args.auth_env}: env var is empty or unset", file=sys.stderr)
            return 2
        auth_header = f"Bearer {token}"

    print(
        f"[lucebench] sweep name={name} "
        f"areas={','.join(sweep_areas)} url={args.url} model={args.model} "
        f"out={out_root}",
        flush=True,
    )

    if "forge" in sweep_areas and not forge_ok:
        print(
            f"[lucebench] forge: skipped — {forge_reason}",
            file=sys.stderr,
            flush=True,
        )
        sweep_areas = [a for a in sweep_areas if a != "forge"]

    summary_areas: list[dict[str, Any]] = []
    for area in sweep_areas:
        if area == "forge":
            row = _run_forge_area_to_dir(
                out_root=out_root,
                url=args.url,
                model=args.model,
                auth_header=auth_header,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                questions=args.questions,
            )
            if row is not None:
                summary_areas.append(row)
            continue

        row, aborted = _run_standard_area_to_dir(
            area,
            out_root=out_root,
            url=args.url,
            model=args.model,
            auth_header=auth_header,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            think=args.think,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            questions=args.questions,
            no_fail_fast=args.no_fail_fast,
            prompt_thinking_control=getattr(args, "prompt_thinking_control", "off"),
            server_honors_api_flags=getattr(args, "server_honors_api_flags", False),
        )
        if aborted:
            return 3
        if row is not None:
            summary_areas.append(row)

    summary = write_sweep_summary(
        out_root,
        name=name,
        url=args.url,
        model=args.model,
        summary_areas=summary_areas,
    )

    md_text = (out_root / "_summary.md").read_text()
    print(f"\n[lucebench] sweep complete → {out_root}", flush=True)
    print(md_text.rstrip(), flush=True)
    del summary  # silence "assigned but never used" for the JSON payload
    return 0


def main() -> int:
    # ── Subcommand short-circuit. ``lucebench regrade ...`` and friends
    # (``snapshot``, ``report``, ``submit-baseline``) have their own
    # argparse trees; intercept the verb BEFORE the main bench-args
    # parser inspects sys.argv so the subcommand flags don't clash with
    # the bench parser's positional / option semantics. Keeps full
    # back-compat for plain ``lucebench --area X`` invocations.
    if len(sys.argv) >= 2 and sys.argv[1] == "regrade":
        from lucebench.regrade import main as regrade_main

        return regrade_main(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == "snapshot":
        from lucebench.snapshot import main as snapshot_main

        return snapshot_main(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == "report":
        from lucebench.report import main as report_main

        return report_main(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == "submit-baseline":
        from lucebench.submit_baseline import main as submit_baseline_main

        return submit_baseline_main(sys.argv[2:])

    ap = argparse.ArgumentParser(
        prog="lucebench",
        description="Capability benchmarks for chat-completion endpoints.",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument(
        "--url",
        "--base-url",
        dest="url",
        default="http://127.0.0.1:8080",
        help="Server base URL (default: http://127.0.0.1:8080).",
    )
    ap.add_argument(
        "--model",
        default="default",
        help="Model identifier sent in the request body. "
        "When left as the literal string 'default', "
        "the CLI queries `<base-url>/v1/models` and "
        "auto-picks the single exposed model. If the "
        "server exposes zero or multiple, it falls back "
        "to the literal 'default' (which most servers "
        "404 on — pass --model explicitly for gateways).",
    )
    ap.add_argument(
        "--areas",
        default=None,
        help="Comma-separated list of areas to run, OR the literal "
        "'all' to run every stdlib area (smoke, ds4-eval, code, "
        "longctx, agent, plus forge if [forge] extra is installed). "
        "Defaults to 'smoke' — a three-prompt sanity check that "
        "completes in seconds. Valid names: "
        + ", ".join(sorted(set(AREAS) | {"forge"}))
        + ". Examples: --areas smoke / --areas all / --areas ds4-eval,forge.",
    )
    # Back-compat aliases. Kept accepted (and forwarded into --areas) so
    # external scripts and docs that predate v0.2.5 don't break — a
    # deprecation note is printed when either is used.
    ap.add_argument(
        "--area",
        choices=sorted(set(AREAS) | {"forge"}),
        default=None,
        help="DEPRECATED (v0.2.5): use --areas <name>. Still accepted.",
    )
    ap.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip the pre-run liveness / /v1/models / /props checks. "
        "Use when running against a deliberately-degraded endpoint "
        "(chaos tests, CI fixtures) where the preflight would "
        "false-fail.",
    )
    ap.add_argument(
        "--list-models",
        action="store_true",
        help="Print the model ids exposed by --base-url/v1/models (one "
        "per line) and exit. Skips preflight, area validation, and the "
        "version banner — output is machine-readable so it can be piped "
        "to grep/head/fzf.",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="Label for snapshot directory under --out-dir. "
        "Common pattern: machine + model tag, e.g. "
        "`bragi-gemma4-26b-2026-05-26`.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./snapshots"),
        help="Root directory for sweep snapshots. Each area writes "
        "<out-dir>/<name>/<area>.json and a combined "
        "_summary.json. Default: ./snapshots",
    )
    ap.add_argument(
        "--questions", type=int, default=None, help="Limit to first N cases (after other filters)."
    )
    ap.add_argument("--case-id", default=None, help="Run only the case with this ID.")
    ap.add_argument(
        "--case-index",
        type=int,
        default=None,
        help="Run only the case at this position (after source filter).",
    )
    ap.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source filter (e.g. AIME2025,GPQA Diamond).",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Per-request decode cap (overrides area default).",
    )
    ap.add_argument("--think", dest="think", action="store_true", default=None)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument(
        "--prompt-thinking-control",
        choices=["auto", "on", "off"],
        default="auto",
        help="Client-side prompt-level thinking-control fallback. "
        "API-side flags (chat_template_kwargs.enable_thinking, "
        "thinking, reasoning_effort) keep firing regardless; this "
        "knob adds an in-band token (e.g. '/no_think' for Qwen3.x) "
        "to the last user turn as belt+suspenders against providers "
        "that strip the API flags. "
        "'auto' (default) injects only when the preflight cannot "
        "confirm a lucebox stack via /props; 'on' forces injection "
        "regardless; 'off' restores pre-feature behavior. Family "
        "tokens are picked from the model id (longest-prefix match) "
        "or from a model-card sidecar's thinking_control block.",
    )
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=300, help="Per-request wall timeout (s).")
    ap.add_argument(
        "--auth-env",
        default=None,
        help="Env var name to read auth bearer token from "
        "(e.g. OPENAI_API_KEY, OPENROUTER_API_KEY).",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the per-case rows as a JSON array to this path.",
    )
    ap.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="When running multiple areas (--areas all or a comma list), "
        "keep going even when the first case can't reach the server. "
        "Default behavior aborts on connection-refused-style errors to "
        "avoid burning ~92 timeouts per area on a typo'd URL.",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Run up to N cases concurrently. Default 1 "
        "(sequential). Safe to raise for stateless HTTP "
        "gateways (OpenRouter); leave at 1 for single-GPU "
        "local servers since concurrent requests just queue.",
    )

    args = ap.parse_args()

    if args.parallel < 1:
        ap.error("--parallel must be >= 1")

    # ── --list-models: machine-readable id dump + exit. Skips area
    # validation, the version banner, and preflight so the output is
    # safe to pipe (`lucebench --list-models | head -5`). Exits 0 when
    # one or more ids came back, 1 when /v1/models was empty / malformed.
    if args.list_models:
        auth_header = ""
        if args.auth_env:
            token = os.environ.get(args.auth_env, "")
            if token:
                auth_header = f"Bearer {token}"
        _chosen, models = list_models(args.url, auth_header=auth_header)
        if not models:
            print(
                f"no models exposed at {args.url.rstrip('/')}/v1/models",
                file=sys.stderr,
                flush=True,
            )
            return 1
        for mid in models:
            print(mid)
        return 0

    # ── Resolve --areas (canonical) + back-compat with --area.
    # Exactly one of {--areas, --area} can be supplied; if nothing is set
    # we default to the smoke area (the "is the server alive?" sanity
    # check). Both forms collapse to a single list of area names in
    # args.areas_list. --sweep was removed in v0.2.6 — use `--areas all`
    # (or the `snapshot` subcommand) for the equivalent multi-area run.
    if args.areas is not None and args.area:
        ap.error("--areas cannot be combined with --area — use --areas")

    all_areas = [
        "smoke",
        "ds4-eval",
        "gsm8k",
        "truthfulqa-mc1",
        "hellaswag",
        "code",
        "longctx",
        "agent",
        "agent_recorded",
        "forge",
    ]

    if args.area:
        print(
            f"[lucebench] note: --area is deprecated in v0.2.5; use --areas {args.area} instead.",
            file=sys.stderr,
            flush=True,
        )
        args.areas_list = [args.area]
    else:
        raw = args.areas if args.areas is not None else "smoke"
        if raw.strip().lower() == "all":
            args.areas_list = list(all_areas)
        else:
            wanted = [a.strip() for a in raw.split(",") if a.strip()]
            if not wanted:
                ap.error("--areas got an empty list")
            valid = set(AREAS) | {"forge"}
            bad = [a for a in wanted if a not in valid]
            if bad:
                ap.error(f"--areas: unknown area(s) {bad!r}. Valid: {sorted(valid)}")
            args.areas_list = wanted

    # First line out: which version of lucebench is actually running.
    # Surfaces stale uvx / pip caches at a glance — debugging "wait,
    # which lucebench is this?" used to require digging through the
    # area-header line buried after preflight + model resolution.
    print(f"[lucebench] v{__version__}", flush=True)

    # ── Preflight: bail fast on an unreachable / mis-shaped server BEFORE
    # firing case requests. The old behavior was to fall through to the
    # per-case loop and burn ~92 timeouts on a typo'd --url; preflight
    # surfaces "connection refused" in ~50ms with a one-line diagnostic.
    # Skip when --no-preflight is set (chaos tests, intentional-failure CI).
    auth_for_probe = ""
    if args.auth_env:
        token = os.environ.get(args.auth_env, "")
        if token:
            auth_for_probe = f"Bearer {token}"

    server_honors_api_flags = False
    if not args.no_preflight:
        ok, lines, server_honors_api_flags = _preflight(
            args.url,
            auth_header=auth_for_probe,
            timeout_s=5,
            requested_model=args.model,
        )
        for line in lines:
            print(line, flush=True)
        if not ok:
            print(
                f"abort: server not reachable. Did you forget to start it? "
                f"Or pass --url? (got {args.url})",
                file=sys.stderr,
                flush=True,
            )
            return 4
    args.server_honors_api_flags = server_honors_api_flags

    # /v1/models auto-resolution. Only fires when the user left --model
    # at the literal default; an explicit value (even if wrong) is
    # respected so gateways with hundreds of models stay predictable.
    # The preflight grid above already prints the list with `*` on the
    # selected id, so this stage only needs a terse one-liner.
    if args.model == "default":
        resolved, models = list_models(args.url, auth_header=auth_for_probe)
        if resolved:
            args.model = resolved
            print(f"[lucebench] --model default → '{resolved}'", flush=True)
        elif models:
            # Long list — refuse to guess; preflight already showed the list.
            print(
                f"[lucebench] --model default: {len(models)} models exposed at "
                f"{args.url}/v1/models — sending 'default' as-is. "
                "Pass --model explicitly to pick one.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[lucebench] --model default: /v1/models at {args.url} "
                "exposed no models — sending 'default' as-is. "
                "Most servers will 404 on this; pass --model explicitly.",
                file=sys.stderr,
                flush=True,
            )

    # ── Multi-area dispatch: anything > 1 area in args.areas_list runs
    # through the sweep path, which writes per-area JSON + a combined
    # summary under <out-dir>/<name>/. Single-area runs use the slimmer
    # in-place path below (single JSON-out, no snapshot dir).
    if len(args.areas_list) > 1:
        return _run_sweep(args)

    # Single area from here on — alias into args.area so the existing
    # forge / generic-area branches keep working unchanged.
    args.area = args.areas_list[0]

    # Forge takes a completely different path — it owns its own runner
    # (recording AnthropicClient + scenario driver) instead of using
    # run_case + a grader. Dispatch early.
    if args.area == "forge":
        from lucebench.areas.forge import run_forge_area

        max_tokens = args.max_tokens if args.max_tokens is not None else 4096
        auth_header = ""
        if args.auth_env:
            token = os.environ.get(args.auth_env, "")
            if not token:
                ap.error(f"--auth-env {args.auth_env}: env var is empty or unset")
            auth_header = f"Bearer {token}"
        rows, summary = run_forge_area(
            url=args.url,
            model=args.model,
            max_tokens=max_tokens,
            timeout_s=args.timeout,
            auth_header=auth_header,
            tags=None,
            names=None,
            questions=args.questions,
        )
        for idx, r in enumerate(rows, start=1):
            verdict = "PASS" if r.get("pass") else "FAIL"
            print(
                f"  {idx:3d} {verdict} forge   {r['case_id']:32s} "
                f"wall={r['wall_seconds']:.2f}s "
                f"calls={len(r.get('iterations') or [])}",
                flush=True,
            )
        print(
            f"\n[lucebench] forge pass_rate={summary['pass_rate']:.2f}% "
            f"({summary['n_pass']}/{summary['n_scenarios']})",
            flush=True,
        )
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(
                    {
                        "lucebench_version": __version__,
                        "area": "forge",
                        "url": args.url,
                        "model": args.model,
                        **summary,
                        "rows": rows,
                    },
                    indent=2,
                    default=str,
                )
            )
            print(f"[lucebench] wrote {len(rows)} rows to {args.json_out}", flush=True)
        return 0

    cfg = AREAS[args.area]
    cases = cfg["load"]()
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    selected = select_cases(
        cases,
        questions=args.questions,
        case_id=args.case_id,
        case_index=args.case_index,
        sources=sources,
    )
    if not selected:
        ap.error("no cases selected by the supplied filters")

    max_tokens = args.max_tokens if args.max_tokens is not None else cfg["default_max_tokens"]
    think = args.think if args.think is not None else cfg["default_thinking"]

    auth_header = ""
    if args.auth_env:
        token = os.environ.get(args.auth_env, "")
        if not token:
            ap.error(f"--auth-env {args.auth_env}: env var is empty or unset")
        auth_header = f"Bearer {token}"

    print(
        f"[lucebench] area={args.area} cases={len(selected)} "
        f"url={args.url} model={args.model} think={think} max_tokens={max_tokens}",
        flush=True,
    )

    def _do(idx_case):
        idx, case = idx_case
        row = run_case(
            url=args.url,
            case=case,
            timeout_s=args.timeout,
            max_tokens=max_tokens,
            think=think,
            model=args.model,
            auth_header=auth_header,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            thinking_control_flag=getattr(args, "prompt_thinking_control", "off"),
            server_honors_api_flags=getattr(args, "server_honors_api_flags", False),
        )
        graded = cfg["grade"](case, row)
        row["pass"] = graded.get("pass", False)
        row["graded"] = graded
        row["_idx"] = idx
        return row, graded

    rows: list[dict[str, Any]] = []
    if args.parallel > 1:
        # Parallel runner: stateless HTTP gateways (OpenRouter etc.) can
        # serve many concurrent requests. Local single-GPU servers just
        # queue them. Output streams "as completed" but the JSON-out rows
        # are sorted back to selection order so snapshots stay deterministic.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_do, (i, c)): (i, c) for i, c in enumerate(selected, start=1)}
            for fut in as_completed(futures):
                row, graded = fut.result()
                rows.append(row)
                print(format_row(row["_idx"], row, graded), flush=True)
        rows.sort(key=lambda r: r["_idx"])
    else:
        for idx, case in enumerate(selected, start=1):
            row, graded = _do((idx, case))
            rows.append(row)
            print(format_row(idx, row, graded), flush=True)
    for r in rows:
        r.pop("_idx", None)

    pass_n = sum(1 for r in rows if r["pass"])
    rate = 100 * pass_n / len(rows) if rows else 0
    walls = [r.get("wall_seconds") or 0 for r in rows]
    print(
        f"\n[lucebench] pass_rate={rate:.2f}% ({pass_n}/{len(rows)}) "
        f"wall_total={sum(walls):.0f}s wall_median={statistics.median(walls):.1f}s",
        flush=True,
    )

    # ── Post-run thinking-control verification. Counts rows whose
    # reasoning_tokens / reasoning_content contradict the requested
    # mode; flips honored=False when contradicting/n exceeds the 5%
    # slack. The block is written into the result JSON (canonical
    # schema fields) AND surfaced as a stderr warning so an operator
    # running `--no-think` against OpenRouter sees the failure at the
    # bottom of the bench output, not buried inside the result file.
    requested_mode = "think" if think else "nothink"
    honored, contradicting = verify_thinking_control(rows, requested_mode)
    injection_summary = _summarize_injection(rows)
    if not honored:
        host = (
            args.url.split("://", 1)[1].split("/", 1)[0]
            if "://" in args.url
            else args.url
        )
        print(
            f"[lucebench] WARNING: thinking control not honored at {host} — "
            f"{contradicting}/{len(rows)} rows in {requested_mode} mode have "
            f"non-empty reasoning. Consider --prompt-thinking-control=on or "
            f"pick a model card with an explicit thinking_control block.",
            file=sys.stderr,
            flush=True,
        )

    if args.json_out:
        # Drop the raw _response blob + the per-row _thinking_injection
        # echo (it's the same on every row; the top-level summary is what
        # consumers read) from JSON-out by default to keep file size sane.
        terse = [
            {k: v for k, v in r.items() if k not in {"_response", "_thinking_injection"}}
            for r in rows
        ]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "lucebench_version": __version__,
                    "area": args.area,
                    "url": args.url,
                    "model": args.model,
                    "think": think,
                    "max_tokens": max_tokens,
                    "n": len(rows),
                    "pass": pass_n,
                    "pass_rate": rate,
                    "thinking_control_requested": requested_mode,
                    "thinking_control_honored": honored,
                    "contradicting_rows": contradicting,
                    "thinking_control_injection": injection_summary,
                    "rows": terse,
                },
                indent=2,
            )
        )
        print(f"[lucebench] wrote {len(rows)} rows to {args.json_out}", flush=True)

    return 0 if pass_n == len(rows) or os.environ.get("LUCEBENCH_PASS_RATE_GATE") is None else 1


if __name__ == "__main__":
    sys.exit(main())
