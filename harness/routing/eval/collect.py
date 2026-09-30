"""Collect answers: every prompt x model x thinking mode, streamed, with timings.

Writes runs/<name>/answers.jsonl (append-only; resumable). One row per run:

    {id, model, thinking, text, reasoning, ttft_ms, total_ms, prompt_tokens,
     completion_tokens, finish_reason, error, max_tokens, concurrency}

``ttft_ms`` is time to the first streamed token of any kind (reasoning or content);
``total_ms`` is request start to end of stream. Keep ``--concurrency 1`` (default)
for clean per-request latency; raise it only for throughput runs (and use a
different run name, since latency under load is not comparable).

    python -m eval.collect --run pilot                      # every arm in backends.json
    python -m eval.collect --run pilot --arms qwen38-27b-think
    python -m eval.collect --run pilot --models qwen35-0.8b --thinking off   # ad-hoc model x mode

By default the jobs are the config's arms (2b off, 27b off, 27b on); the 27B thinking-on
answers double as judge references.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from .common import (
    PROMPTS_PATH,
    RUNS_DIR,
    append_jsonl,
    latest_by_key,
    load_arms,
    load_backends,
    read_jsonl,
)

DEFAULT_MAX_TOKENS: dict[str, int] = {
    "math_word": 1024,
    "knowledge_mc": 1024,
    "instruction_following": 1024,
    "math_hard": 2048,
    "code": 1024,
    "system_design": 2048,
    "debugging": 1536,
    "coding_open": 1536,
    "reasoning": 1024,
    "security": 1024,
    "default": 512,  # handwritten easy categories
}
THINKING_MULT = 4  # thinking-on budget = category budget x this


def max_tokens_for(category: str, thinking: bool, table: dict[str, int], think_mult: float) -> int:
    n = table.get(category, table.get("default", 512))
    return int(n * think_mult) if thinking else n


def _iter_sse(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


async def stream_chat(
    client: Any,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    thinking: bool,
    max_tokens: int,
    temperature: float,
    seed: int | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """One streamed OpenAI chat completion -> timing + text record (never raises)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if seed is not None:
        body["seed"] = seed
    out: dict[str, Any] = {
        "text": "", "reasoning": "", "ttft_ms": None, "total_ms": None,
        "prompt_tokens": None, "completion_tokens": None, "finish_reason": None, "error": None,
    }
    content: list[str] = []
    reasoning: list[str] = []
    n_chunks = 0
    t0 = time.perf_counter()
    try:
        async with client.stream(
            "POST", f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout_s
        ) as r:
            if r.status_code >= 400:
                err = (await r.aread()).decode(errors="replace")[:500]
                out["error"] = f"HTTP {r.status_code}: {err}"
                out["total_ms"] = (time.perf_counter() - t0) * 1000
                return out
            async for line in r.aiter_lines():
                chunk = _iter_sse(line)
                if chunk is None:
                    continue
                if chunk.get("usage"):
                    u = chunk["usage"]
                    out["prompt_tokens"] = u.get("prompt_tokens")
                    out["completion_tokens"] = u.get("completion_tokens")
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    rpiece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if (piece or rpiece) and out["ttft_ms"] is None:
                        out["ttft_ms"] = (time.perf_counter() - t0) * 1000
                    if piece:
                        content.append(piece)
                        n_chunks += 1
                    if rpiece:
                        reasoning.append(rpiece)
                        n_chunks += 1
                    if ch.get("finish_reason"):
                        out["finish_reason"] = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001 - recorded, row retried on the next resume
        out["error"] = f"{type(e).__name__}: {e}"[:500]
    out["total_ms"] = (time.perf_counter() - t0) * 1000
    out["text"] = "".join(content)
    out["reasoning"] = "".join(reasoning)
    if out["completion_tokens"] is None and n_chunks:
        out["completion_tokens"] = n_chunks  # fallback: ~1 token per streamed delta
        out["completion_tokens_estimated"] = True
    return out


def plan_jobs(
    prompts: list[dict[str, Any]],
    targets: list[tuple[str, bool]],
    done: set[tuple[str, str, bool]],
) -> list[tuple[dict[str, Any], str, bool]]:
    """Jobs still to run for each (model, thinking) target, target-major so each server stays warm."""
    return [(p, m, t) for m, t in targets for p in prompts if (p["id"], m, t) not in done]


def collect_targets(arms: dict[str, dict[str, Any]], arm_names: list[str] | None = None,
                    models: list[str] | None = None, thinking: str = "off") -> list[tuple[str, bool]]:
    """(model, thinking) pairs to collect: the given models x modes, else the arms (deduped)."""
    if models:
        modes = {"off": [False], "on": [True], "both": [False, True]}[thinking]
        return [(m, t) for m in models for t in modes]
    names = arm_names or list(arms)
    unknown = [a for a in names if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arms {unknown} (have {list(arms)})")
    return list(dict.fromkeys((arms[a]["model"], arms[a]["thinking"]) for a in names))


def completed_keys(answers_path: Path) -> set[tuple[str, str, bool]]:
    """Keys whose latest row succeeded; errored rows are retried."""
    return {k for k, row in latest_by_key(read_jsonl(answers_path)).items() if not row.get("error")}


async def run(args: argparse.Namespace) -> None:
    import httpx

    backends = load_backends(args.backends)
    targets = collect_targets(load_arms(args.backends),
                              [a.strip() for a in args.arms.split(",")] if args.arms else None,
                              [m.strip() for m in args.models.split(",")] if args.models else None,
                              args.thinking)
    for m, _ in targets:
        if m not in backends:
            raise SystemExit(f"model {m!r} not in backends ({list(backends)})")

    prompts = read_jsonl(args.prompts)
    if args.split != "all":
        prompts = [p for p in prompts if p["split"] == args.split]
    if args.categories:
        cats = set(args.categories.split(","))
        prompts = [p for p in prompts if p["category"] in cats or p["source"] in cats]
    if args.limit:
        prompts = prompts[: args.limit]

    table = dict(DEFAULT_MAX_TOKENS)
    if args.max_tokens_json:
        table.update(json.loads(Path(args.max_tokens_json).read_text()))
    for item in args.max_tokens:
        k, v = item.split("=", 1)
        table[k] = int(v)

    out_dir = RUNS_DIR / args.run if not args.out_dir else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers = out_dir / "answers.jsonl"
    jobs = plan_jobs(prompts, targets, completed_keys(answers))
    if args.shuffle:
        random.Random(0).shuffle(jobs)
    print(f"{len(jobs)} jobs to run -> {answers}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    t_start = time.perf_counter()
    limits = httpx.Limits(max_connections=max(4, args.concurrency * 2))
    async with httpx.AsyncClient(limits=limits, timeout=args.timeout) as client:

        async def one(p: dict[str, Any], model: str, thinking: bool) -> None:
            nonlocal done
            spec = backends[model]
            mt = max_tokens_for(p["category"], thinking, table, args.thinking_mult)
            async with sem:
                rec = await stream_chat(
                    client, spec["base_url"], spec.get("served_name", model), p["messages"],
                    thinking=thinking, max_tokens=mt, temperature=args.temperature,
                    seed=args.seed, timeout_s=args.timeout,
                )
            row = {"id": p["id"], "model": model, "thinking": thinking, **rec,
                   "max_tokens": mt, "concurrency": args.concurrency}
            append_jsonl(answers, row)
            done += 1
            if done % args.log_every == 0 or rec["error"]:
                rate = done / (time.perf_counter() - t_start)
                status = f"ERROR {rec['error'][:120]}" if rec["error"] else \
                    f"{rec['total_ms']:.0f} ms {rec['completion_tokens']} tok"
                print(f"[{done}/{len(jobs)} {rate:.2f}/s] {model} think={thinking} {p['id']}: {status}",
                      flush=True)

        await asyncio.gather(*(one(p, m, t) for p, m, t in jobs))
    print(f"done: {done} rows in {time.perf_counter() - t_start:.0f}s", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name -> runs/<name>/")
    ap.add_argument("--out-dir", default=None, help="override runs/<name>")
    ap.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--backends", default=None, help="backends.json (default config/backends.json)")
    ap.add_argument("--arms", default=None, help="comma list of arms (default: every arm in backends.json)")
    ap.add_argument("--models", default=None, help="ad-hoc comma list of models, instead of arms")
    ap.add_argument("--thinking", choices=["off", "on", "both"], default="off", help="modes for --models")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all")
    ap.add_argument("--categories", default=None, help="comma list of categories or sources")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tokens", action="append", default=[], metavar="CATEGORY=N")
    ap.add_argument("--max-tokens-json", default=None, help="JSON {category: N} overrides")
    ap.add_argument("--thinking-mult", type=float, default=THINKING_MULT)
    ap.add_argument("--shuffle", action="store_true", help="interleave jobs (throughput runs)")
    ap.add_argument("--log-every", type=int, default=10)
    return ap


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
