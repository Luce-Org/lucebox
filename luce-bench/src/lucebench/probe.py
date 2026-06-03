"""Reproducible probe of a server's thinking-mode controls on a single case.

Runs N preset request shapes against one model and one ds4-eval case, then
dumps per-mode JSON (full response + usage + timings) and a summary row
table. Standard tool behind `docs/experiments/thinking-control-protocol.md`.

The point is comparability: same case, same model, same temperature, only
the thinking-control fields vary. Outputs land under a snapshot dir so
results from different (server, model, date) tuples can be diffed.

Modes (--modes csv, default = all five):

* ``think-default``   — ``thinking: {type: enabled}`` and ``chat_template_kwargs.enable_thinking = true``. No budget hint. Baseline for "what does the model do when allowed to think freely".
* ``nothink``         — ``thinking: {type: disabled}`` + ``enable_thinking = false``. Compare visible-completion length and content to ``think-default`` to answer "did it drop thinking or just hide it".
* ``think-low``       — Anthropic-shape ``thinking: {type: enabled, budget_tokens: 1024}``. Tests the server's soft/hard force-close at low budget.
* ``think-medium``    — same, ``budget_tokens: 4096``.
* ``think-raw-noprompt`` — ``thinking: {type: enabled}`` but the system message is replaced with an empty string and the chat template's thinking-opener is suppressed (``chat_template_kwargs.enable_thinking = false``). Probes whether the model self-thinks even without the template hint.

For each mode we capture:

* ``content``                — visible answer string
* ``reasoning_content``      — server-extracted reasoning block (if any)
* ``usage``                  — prompt/completion/thinking token counts
* ``finish_details``         — close reason: ``stop``, ``length``, ``hard_close``
* ``timings``                — prefill_ms, decode_ms, tok/s
* request_body / response_body — full envelope for forensic replay
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURE_PATH = ROOT / "fixtures" / "ds4_eval_cases.json"


def load_case(case_id: str) -> dict:
    payload = json.loads(FIXTURE_PATH.read_text())
    for c in payload.get("cases", []):
        if c.get("id") == case_id:
            return c
    raise SystemExit(f"case-id {case_id!r} not in ds4_eval_cases.json")


SYSTEM_DEFAULT = (
    "You are solving a hard benchmark question. Reason carefully. "
    "The final answer must follow the requested format exactly."
)

# Stronger no-reasoning prompt: explicitly forbids step-by-step.
SYSTEM_TERSE = (
    "Answer with ONLY the final answer (a single integer for AIME, "
    "or the requested format). Do not show any reasoning, work, "
    "intermediate steps, or explanation. Output the answer alone."
)


def build_body(
    model: str, case: dict, mode: str, max_tokens: int, temperature: float, top_p: float, top_k: int
) -> dict:
    user = case["question"]
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_DEFAULT},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if top_k > 0:
        body["top_k"] = top_k
    if mode == "think-default":
        body["thinking"] = {"type": "enabled"}
        body["chat_template_kwargs"] = {"enable_thinking": True}
    elif mode == "nothink":
        body["thinking"] = {"type": "disabled"}
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif mode == "think-low":
        body["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        body["chat_template_kwargs"] = {"enable_thinking": True}
    elif mode == "think-medium":
        body["thinking"] = {"type": "enabled", "budget_tokens": 4096}
        body["chat_template_kwargs"] = {"enable_thinking": True}
    elif mode == "think-raw-noprompt":
        body["thinking"] = {"type": "enabled"}
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["messages"][0]["content"] = ""
    # ── Prompt-side anti-reasoning experiments ─────────────────────
    # These test whether *behavioral* reasoning (model thinks in
    # plain content even when the channel-thought block is suppressed)
    # can be compelled away with stronger prompt-side controls. The
    # nothink mode above shows that hiding the tags doesn't change
    # what the model does — the question is whether anything in the
    # prompt actually does.
    elif mode == "nothink-terse":
        body["thinking"] = {"type": "disabled"}
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["messages"][0]["content"] = SYSTEM_TERSE
    elif mode == "nothink-prefill-answer":
        # Pre-seed an assistant turn so the model continues "The
        # answer is " directly — forces a commit before reasoning.
        # Many servers won't honor this without a special flag; this
        # tests whether ours does.
        body["thinking"] = {"type": "disabled"}
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["messages"].append({"role": "assistant", "content": "The answer is "})
    elif mode == "nothink-stop-after-answer":
        # Force a hard stop on tokens that signal "now let me explain".
        # If the model commits to an answer at all, stop before it
        # starts reasoning.
        body["thinking"] = {"type": "disabled"}
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["messages"][0]["content"] = SYSTEM_TERSE
        body["stop"] = [
            "\nReason",
            "\nLet",
            "\nFirst",
            "\nWe ",
            "\nTo ",
            "Reasoning:",
            "Explanation:",
            "Step 1",
        ]
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return body


def probe_one(url: str, body: dict, timeout_s: int) -> tuple[dict, dict, float]:
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read())
    dur = time.perf_counter() - t0
    return raw, dict(resp.headers) if hasattr(resp, "headers") else {}, dur


def extract_row(case_id: str, mode: str, body: dict, raw: dict, dur_s: float) -> dict:
    choice = (raw.get("choices") or [{}])[0]
    msg = choice.get("message", {}) if isinstance(choice, dict) else {}
    usage = raw.get("usage", {}) or {}
    timings = usage.get("timings", {}) or {}
    finish_details = choice.get("finish_details") or msg.get("finish_details") or {}
    return {
        "case_id": case_id,
        "mode": mode,
        "wall_s": round(dur_s, 3),
        "content_len_chars": len(msg.get("content") or ""),
        "reasoning_len_chars": len(msg.get("reasoning_content") or ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "thinking_tokens": usage.get("thinking_tokens") or usage.get("reasoning_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "finish_details": finish_details,
        "prefill_ms": timings.get("prefill_ms"),
        "decode_ms": timings.get("decode_ms"),
        "decode_tok_per_s": timings.get("decode_tokens_per_sec"),
    }


def render_summary(rows: list[dict]) -> str:
    cols = ["mode", "prompt", "comp", "think", "content", "reasoning", "finish", "wall_s", "tok/s"]
    out = [f"| {' | '.join(cols)} |", f"|{'|'.join(['---'] * len(cols))}|"]
    for r in rows:
        out.append(
            f"| {r['mode']} "
            f"| {r['prompt_tokens']} "
            f"| {r['completion_tokens']} "
            f"| {r['thinking_tokens']} "
            f"| {r['content_len_chars']} "
            f"| {r['reasoning_len_chars']} "
            f"| {r['finish_reason']} "
            f"| {r['wall_s']} "
            f"| {round(r['decode_tok_per_s'] or 0, 1)} |"
        )
    return "\n".join(out)


DEFAULT_MODES = "think-default,nothink,think-low,think-medium,think-raw-noprompt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default="dflash")
    ap.add_argument("--case-id", required=True, help="ds4-eval case id, e.g. aime2025-02")
    ap.add_argument(
        "--modes", default=DEFAULT_MODES, help=f"comma-separated subset of {DEFAULT_MODES}"
    )
    ap.add_argument("--max-tokens", type=int, default=8192, help="per-request decode cap")
    ap.add_argument("--timeout", type=int, default=600, help="per-request wall timeout (s)")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temp (default 0 for reproducibility; "
        "use the model card's recommended temp to expose "
        "sampling-driven failure modes)",
    )
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument(
        "--seed", type=int, default=0, help="sampler seed (only meaningful at temperature>0)"
    )
    ap.add_argument(
        "--out-dir", required=True, help="snapshot dir to write into (created if missing)"
    )
    args = ap.parse_args()

    case = load_case(args.case_id)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rows: list[dict] = []
    for mode in modes:
        body = build_body(
            args.model, case, mode, args.max_tokens, args.temperature, args.top_p, args.top_k
        )
        if args.seed and args.temperature > 0:
            body["seed"] = args.seed
        print(f"  [{mode}] POST ...", flush=True)
        try:
            raw, _hdrs, dur = probe_one(args.url, body, args.timeout)
        except Exception as e:
            print(f"  [{mode}] FAILED: {e}", flush=True)
            (out / f"{mode}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
            continue
        row = extract_row(args.case_id, mode, body, raw, dur)
        rows.append(row)
        full = {"case_id": args.case_id, "mode": mode, "request": body, "response": raw, "row": row}
        (out / f"{mode}.json").write_text(json.dumps(full, indent=2))
        print(
            f"  [{mode}] OK prompt={row['prompt_tokens']} "
            f"comp={row['completion_tokens']} think={row['thinking_tokens']} "
            f"content_chars={row['content_len_chars']} "
            f"reasoning_chars={row['reasoning_len_chars']} "
            f"finish={row['finish_reason']} {row['wall_s']}s",
            flush=True,
        )

    summary = {
        "case_id": args.case_id,
        "model": args.model,
        "url": args.url,
        "run_at": _dt.datetime.utcnow().isoformat() + "Z",
        "rows": rows,
    }
    (out / "_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "_summary.md").write_text(
        f"# probe_thinking_control — {args.case_id} on {args.model}\n\n"
        f"run_at: {summary['run_at']}\nurl: {args.url}\n\n" + render_summary(rows) + "\n"
    )
    print("\n" + render_summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
