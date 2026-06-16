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
from pathlib import Path
from typing import Any

from lucebench import __version__
from lucebench._display import format_row
from lucebench._preflight import (
    _preflight,
    list_models,
    resolve_model,  # noqa: F401  re-exported for back-compat (tests import from cli)
)
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
from lucebench.model_cards import (
    card_is_thinking_capable,
    card_sampling,
    normalize_model_card_stem,
    resolve_card,
)
from lucebench.runner import run_case

_AUTH_ENV_EMPTY_MSG = "--auth-env {name}: env var is empty or unset"


class _AuthEnvEmpty(Exception):
    """Raised by :func:`_resolve_auth_header` when an --auth-env var is empty.

    Carries the formatted, user-facing message so each call site can route it
    through its own error path (``ap.error`` vs ``print + return 2``).
    """


def _resolve_auth_header(args, *, on_empty: str = "error") -> str:
    """Build the ``Bearer <token>`` auth header from ``args.auth_env``.

    Returns ``""`` when ``--auth-env`` is unset. When it is set but the env var
    is empty:

    * ``on_empty="error"`` raises :class:`_AuthEnvEmpty` with a ready-made
      message (caller decides how to surface it / which exit code to use).
    * ``on_empty="ignore"`` tolerates the empty value and returns ``""`` (used
      by read-only probes like --list-models / preflight).
    """
    if not args.auth_env:
        return ""
    token = os.environ.get(args.auth_env, "")
    if not token:
        if on_empty == "ignore":
            return ""
        raise _AuthEnvEmpty(_AUTH_ENV_EMPTY_MSG.format(name=args.auth_env))
    return f"Bearer {token}"


_TERSE_DROP_KEYS = frozenset({"_response", "_thinking_injection"})


def _terse_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip the heavy per-row keys (raw ``_response`` blob, repeated
    ``_thinking_injection`` echo) before serializing to JSON-out, keeping
    file size sane. Returns fresh dicts; inputs are not mutated."""
    return [{k: v for k, v in r.items() if k not in _TERSE_DROP_KEYS} for r in rows]


def _write_area_json(
    out_path: Path,
    *,
    area: str,
    url: str,
    model: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Write the standard ``**summary``-style per-area JSON envelope.

    Used by areas (forge, agent_recorded) whose summary dict is spliced
    in wholesale; areas with a fixed-key envelope build theirs inline.
    """
    out_path.write_text(
        json.dumps(
            {
                "lucebench_version": __version__,
                "area": area,
                "url": url,
                "model": model,
                **summary,
                "rows": rows,
            },
            indent=2,
            default=str,
        )
    )


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

# Sampling fields threaded into run_case. The first three are also exposed as
# explicit CLI flags (per-field override); the rest are card-only knobs the
# CLI never surfaces but still forces from the card by default.
_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
)
# CLI-overridable subset — these map to --temperature / --top-p / --top-k.
_CLI_SAMPLING_FIELDS = ("temperature", "top_p", "top_k")


def resolve_sampling(
    *,
    card: dict[str, Any] | None,
    no_card_sampling: bool,
    cli_temperature: float | None,
    cli_top_p: float | None,
    cli_top_k: int | None,
) -> tuple[dict[str, Any], str]:
    """Compute the effective per-field sampling dict + its provenance source.

    Precedence, per field:
      1. explicit CLI flag (not None) → use it (per-field override), else
      2. resolved card's ``sampling[field]`` when a card resolved and has it
         AND ``--no-card-sampling`` was not passed → use it, else
      3. omit (None).

    Returns ``(sampling, source)`` where ``sampling`` carries only the fields
    that resolved (missing key = omit on the wire) and ``source`` is:
      * ``"none"``  — nothing resolved (no card / opted out, no CLI flags)
      * ``"cli"``   — every resolved field came from a CLI flag
      * ``"card"``  — every resolved field came from the card
      * ``"mixed"`` — a blend of CLI overrides and card values
    """
    card_block = {} if no_card_sampling else card_sampling(card)
    cli_values = {
        "temperature": cli_temperature,
        "top_p": cli_top_p,
        "top_k": cli_top_k,
    }
    sampling: dict[str, Any] = {}
    saw_cli = False
    saw_card = False
    for field_name in _SAMPLING_FIELDS:
        cli_val = cli_values.get(field_name)
        if cli_val is not None:
            sampling[field_name] = cli_val
            saw_cli = True
        elif field_name in card_block and card_block[field_name] is not None:
            sampling[field_name] = card_block[field_name]
            saw_card = True
    if saw_cli and saw_card:
        source = "mixed"
    elif saw_cli:
        source = "cli"
    elif saw_card:
        source = "card"
    else:
        source = "none"
    return sampling, source


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
    # Legacy single-turn area — kept for back-compat with archived
    # result snapshots. See lucebench.areas.agent_recorded module
    # docstring for why this was deprecated in favor of multi-turn.
    "agent_recorded_v1": {
        "load": agent_recorded.load_agent_recorded_v1_cases,
        "grade": agent_recorded.grade_agent_recorded_v1_case,
        "default_max_tokens": 4096,
        "default_thinking": False,
    },
    # ``agent_recorded`` (new default, multi-turn + LLM judge) does NOT
    # plug into _run_standard_area_to_dir's case-per-request loop — it
    # needs cold/warm passes around each case and a judge call. The CLI
    # dispatches it through ``_run_agent_recorded_to_dir`` (mirror of
    # ``_run_forge_area_to_dir``). The entry below is a placeholder for
    # introspection only; load/grade are unused on the new path.
    "agent_recorded": {
        "load": agent_recorded.load_agent_recorded_multi_turn_cases,
        "grade": agent_recorded.grade_multi_turn_replay_stub,
        "default_max_tokens": 512,
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
    ap.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="high",
        help="OpenAI/OpenRouter reasoning_effort tier sent in think mode "
        "(default: high — unchanged from pre-flag behavior). nothink always "
        "sends 'none'. 'low'/'medium' are a Tier-1 native budget hint: a "
        "provider that honors them yields shorter reasoning with zero "
        "client machinery. Reported as its own benchmark setting — do not "
        "pool medium/low runs with default-high think.",
    )
    ap.add_argument(
        "--thinking-budget-tokens",
        type=int,
        default=None,
        help="Tier-1 Anthropic-shape native budget hint. When set AND in "
        "think mode, adds thinking.budget_tokens=N to the request body "
        "(same shape lucebench-probe sends). No-op in nothink and when "
        "unset. Servers that don't understand it ignore it.",
    )
    ap.add_argument(
        "--client-thinking-budget",
        type=int,
        default=None,
        help="Tier-2 client-side thinking budget (opt-in, default off). When "
        "set AND in think mode, the client counts reasoning tokens as the "
        "stream arrives (char/4 estimate) and, once over N, aborts the read "
        "and issues a forced-</think> continuation (a fresh assistant-prefill "
        "request) whose answer is graded — bounding thinking even on backends "
        "that ignore the Tier-1 native hints (OpenRouter, MLX). client_abort "
        "is a SEPARATE benchmark mode: its scores are not pooled with "
        "single-pass think. No-op in nothink and when unset.",
    )
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument(
        "--no-card-sampling",
        action="store_true",
        help="Do not apply the model card's sampling block; use only explicit "
        "--temperature/--top-p/--top-k or omit.",
    )
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
        "--no-agent-recorded-restart",
        dest="agent_recorded_restart",
        action="store_false",
        default=True,
        help="Disable systemctl --user restart lucebox.service between "
        "cases in the agent_recorded area. Use on hosts without systemd "
        "(CI) or when the operator wants cache to accumulate across "
        "cases. Without restarts the first replay turn may already see "
        "warm cache state.",
    )
    ap.add_argument(
        "--agent-recorded-replay-mode",
        choices=("sequential", "exact-repeat"),
        default="sequential",
        help="Replay semantics for the agent_recorded area. sequential "
        "replays growing conversation prefixes and exercises prefix cache; "
        "exact-repeat sends the same full prompt twice to probe full-prompt "
        "prefill cache.",
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
        auth_header = _resolve_auth_header(args, on_empty="ignore")
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
        # ``agent_recorded`` is intentionally NOT in --areas all — it
        # has SIDE EFFECTS (systemctl restart, real Anthropic billing
        # for the judge) that don't belong in a CI smoke sweep. Run it
        # explicitly: ``--areas agent_recorded``.
        # ``agent_recorded_v1`` is similarly excluded — it's the
        # deprecated single-turn variant kept for back-compat only.
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
    auth_for_probe = _resolve_auth_header(args, on_empty="ignore")

    server_honors_api_flags = False
    props_model_card: dict[str, Any] | None = None
    if not args.no_preflight:
        ok, lines, server_honors_api_flags, props_model_card = _preflight(
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
    args.props_model_card = props_model_card

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

    # ── Card resolution + light preflight classification. Resolve the
    # model card now that --model is finalized: /props.model_card wins
    # (authoritative), else the bundled registry keyed by the normalized
    # stem. The resolved card drives the thinking-token resolver in
    # run_case (via model_card=), and its provenance is stamped per row.
    # Logging only — we record the resolution and whether the model is
    # thinking-capable; we do NOT build the full provider-capability matrix
    # (Tier 2, deferred).
    resolved_card, card_source = resolve_card(
        args.model, getattr(args, "props_model_card", None)
    )
    card_stem = normalize_model_card_stem(args.model)
    thinking_capable = card_is_thinking_capable(resolved_card)
    args.resolved_card = resolved_card
    args.card_source = card_source
    args.card_stem = card_stem
    print(
        f"[lucebench] model_card: source={card_source} "
        f"stem={card_stem or '(none)'} "
        f"thinking_capable={thinking_capable}",
        flush=True,
    )

    # ── Effective sampling. By default we FORCE the resolved card's sampling
    # block (so card-less servers — OpenRouter / MLX — run with the model's
    # recommended decode params instead of the provider's defaults); explicit
    # --temperature/--top-p/--top-k override per field, and --no-card-sampling
    # opts out entirely (CLI-flags-or-omit). Threaded into both run_case call
    # sites below. See resolve_sampling for the precedence.
    sampling, sampling_source = resolve_sampling(
        card=resolved_card,
        no_card_sampling=getattr(args, "no_card_sampling", False),
        cli_temperature=args.temperature,
        cli_top_p=args.top_p,
        cli_top_k=args.top_k,
    )
    args.sampling = sampling
    args.sampling_source = sampling_source
    print(
        f"[lucebench] sampling: source={sampling_source} "
        f"{sampling if sampling else '(server defaults)'}",
        flush=True,
    )

    # ── Multi-area dispatch: anything > 1 area in args.areas_list runs
    # through the sweep path, which writes per-area JSON + a combined
    # summary under <out-dir>/<name>/. Single-area runs use the slimmer
    # in-place path below (single JSON-out, no snapshot dir).
    if len(args.areas_list) > 1:
        from lucebench._orchestrate import _run_sweep

        return _run_sweep(args)

    # Single area from here on — alias into args.area so the existing
    # forge / generic-area branches keep working unchanged.
    args.area = args.areas_list[0]

    # agent_recorded multi-turn replay also owns its runner (cold/warm
    # passes around each case + LLM judge). Same dispatch pattern as
    # forge — early branch, write its own JSON.
    if args.area == "agent_recorded":
        from lucebench.areas.agent_recorded import run_agent_recorded_area

        max_tokens_eff = args.max_tokens if args.max_tokens is not None else 512
        try:
            auth_header = _resolve_auth_header(args, on_empty="error")
        except _AuthEnvEmpty as e:
            ap.error(str(e))
        try:
            rows, summary = run_agent_recorded_area(
                url=args.url,
                model=args.model,
                max_tokens=max_tokens_eff,
                auth_header=auth_header,
                timeout_s=args.timeout,
                restart_between_cases=getattr(args, "agent_recorded_restart", True),
                questions=args.questions,
                replay_mode=getattr(args, "agent_recorded_replay_mode", "sequential"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[lucebench] agent_recorded: {exc}", file=sys.stderr, flush=True)
            return 3
        for idx, r in enumerate(rows, start=1):
            verdict = "PASS" if r.get("pass") else "FAIL"
            if r.get("replay_mode") == "sequential":
                final = r.get("warm") or r.get("cold") or {}
                print(
                    f"  {idx:3d} {verdict} agent_rec  {r['case_id']:60s} "
                    f"bucket={r['bucket_tokens']:>6} turns={len(r.get('turns') or [])} "
                    f"cache_hits={r.get('cache_hit_turns', 0)}/{r.get('cache_eligible_turns', 0)} "
                    f"final_pf={final.get('prefill_s')}s judge={r['judge'].get('verdict')}",
                    flush=True,
                )
            else:
                cold = r.get("cold", {}) or {}
                warm = r.get("warm", {}) or {}
                speedup = r.get("cache_speedup_x")
                speedup_str = f"{speedup}x" if speedup is not None else "n/a"
                print(
                    f"  {idx:3d} {verdict} agent_rec  {r['case_id']:60s} "
                    f"bucket={r['bucket_tokens']:>6} "
                    f"cold_pf={cold.get('prefill_s')}s warm_pf={warm.get('prefill_s')}s "
                    f"speedup={speedup_str} judge={r['judge'].get('verdict')}",
                    flush=True,
                )
        print(
            f"\n[lucebench] agent_recorded pass_rate={summary.get('pass_rate', 0):.2f}% "
            f"({summary.get('n_pass', 0)}/{summary.get('n_judged', 0)}) "
            f"cache_speedup_p50={summary.get('cache_speedup_p50')} "
            f"cache_hit_rate={summary.get('cache_hit_rate')}%",
            flush=True,
        )
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            _write_area_json(
                args.json_out,
                area="agent_recorded",
                url=args.url,
                model=args.model,
                summary=summary,
                rows=rows,
            )
            print(f"[lucebench] wrote {len(rows)} rows to {args.json_out}", flush=True)
        return 0

    # Forge takes a completely different path — it owns its own runner
    # (recording AnthropicClient + scenario driver) instead of using
    # run_case + a grader. Dispatch early.
    if args.area == "forge":
        from lucebench.areas.forge import run_forge_area

        max_tokens = args.max_tokens if args.max_tokens is not None else 4096
        try:
            auth_header = _resolve_auth_header(args, on_empty="error")
        except _AuthEnvEmpty as e:
            ap.error(str(e))
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
            _write_area_json(
                args.json_out,
                area="forge",
                url=args.url,
                model=args.model,
                summary=summary,
                rows=rows,
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

    try:
        auth_header = _resolve_auth_header(args, on_empty="error")
    except _AuthEnvEmpty as e:
        ap.error(str(e))

    print(
        f"[lucebench] area={args.area} cases={len(selected)} "
        f"url={args.url} model={args.model} think={think} max_tokens={max_tokens}",
        flush=True,
    )

    # Capability gate: only inject think/nothink tokens when the resolved
    # card is thinking-capable. A non-thinking model (or unresolved card)
    # forces the flag off so neither the card nor the family-map fallback
    # injects a token into a model that has no thinking channel.
    effective_thinking_control = (
        getattr(args, "prompt_thinking_control", "off")
        if card_is_thinking_capable(getattr(args, "resolved_card", None))
        else "off"
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
            temperature=args.sampling.get("temperature"),
            top_p=args.sampling.get("top_p"),
            top_k=args.sampling.get("top_k"),
            min_p=args.sampling.get("min_p"),
            presence_penalty=args.sampling.get("presence_penalty"),
            repetition_penalty=args.sampling.get("repetition_penalty"),
            sampling_source=args.sampling_source,
            thinking_control_flag=effective_thinking_control,
            server_honors_api_flags=getattr(args, "server_honors_api_flags", False),
            reasoning_effort=getattr(args, "reasoning_effort", "high"),
            thinking_budget_tokens=getattr(args, "thinking_budget_tokens", None),
            client_thinking_budget=getattr(args, "client_thinking_budget", None),
            model_card=getattr(args, "resolved_card", None),
            card_source=getattr(args, "card_source", None),
            card_stem=getattr(args, "card_stem", None),
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
        terse = _terse_rows(rows)
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


# Run-orchestration functions live in lucebench._orchestrate to keep this
# module focused on arg-parse + dispatch. They depend on the shared helpers
# defined above, so _orchestrate imports back from cli — a two-way
# dependency. To avoid a circular import at module load we resolve the names
# lazily via PEP 562 __getattr__: ``from lucebench.cli import _run_sweep``
# (snapshot.py, tests, main()) still works, but the _orchestrate module is
# only pulled in on first access, after cli has finished initialising.
_ORCHESTRATION_EXPORTS = frozenset(
    {
        "_run_forge_area_to_dir",
        "_run_agent_recorded_to_dir",
        "_run_standard_area_to_dir",
        "write_sweep_summary",
        "_run_sweep",
    }
)


def __getattr__(name: str) -> Any:
    if name in _ORCHESTRATION_EXPORTS:
        import lucebench._orchestrate as _orch

        return getattr(_orch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(main())
