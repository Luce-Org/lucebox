"""Server probing: model auto-resolution + the pre-run liveness preflight.

Split out of cli.py. These talk to the target server's ``/v1/models`` and
``/props`` endpoints but build no benchmark state — they decide which model
id to send and surface a one-line-per-check liveness grid before any case
fires, so a typo'd --url fails fast instead of after dozens of timeouts.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from lucebench._display import _format_models_inline

# Auto-pick a model only when the server exposes fewer than this many ids;
# gateways with 5+ models still require an explicit --model.
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


def _preflight(
    url: str,
    *,
    auth_header: str = "",
    timeout_s: int = 5,
    requested_model: str | None = None,
) -> tuple[bool, list[str], bool, dict[str, Any] | None]:
    """Probe the server's liveness + OpenAI shape + lucebox /props endpoint.

    Returns ``(ok, lines, server_honors_api_flags, props_model_card)`` where
    ``lines`` is the printed grid (already formatted, one check per line),
    ``ok`` is False iff a HARD check failed — which is "liveness" or
    "/v1/models doesn't return a data list" — ``server_honors_api_flags`` is
    True iff the server's /props response surfaces ``model_card_source`` (the
    marker that this is a lucebox stack which enforces thinking control
    server-side), and ``props_model_card`` is the verbatim ``/props.model_card``
    dict (the authoritative card the server loaded) or None when /props is
    absent / carries no card. The /props check is lucebox-specific:
    missing/404 prints a warning line but does NOT fail (OpenRouter, vLLM,
    stock ds4_server don't expose /props), and in those cases
    ``server_honors_api_flags`` defaults to False so the client-side
    injection can take over.

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
        return False, lines, False, None

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
        return False, lines, False, None

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
        return True, lines, False, None

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
    # `/props.model_card` (props_schema 2+) is the verbatim sidecar JSON the
    # server loaded — the authoritative card. Capture it so the CLI can pass
    # it into the thinking resolver ahead of the bundled registry.
    props_model_card = props.get("model_card") if isinstance(props, dict) else None
    if not isinstance(props_model_card, dict):
        props_model_card = None
    return True, lines, server_honors, props_model_card
