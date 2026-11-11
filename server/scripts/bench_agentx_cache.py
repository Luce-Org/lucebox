#!/usr/bin/env python3
"""Prefix-cache hit rate on recorded coding-agent sessions (the AgentX replay).

Replays the public Claude Code traces of SemiAnalysis's AgentX benchmark against a running
luce_server and reports how much of every prompt the server read from its cache:

    python3 server/scripts/bench_agentx_cache.py run --url http://127.0.0.1:8080 \\
        --model qwen3.8-27b --tokenizer Qwen/Qwen3.8-27B --sessions 1 --out results/agentx-1

    python3 server/scripts/bench_agentx_cache.py report results/agentx-1

The client is the AgentX harness (AIPerf) in the copy FlashStates publishes, which continues
every session from the server's own replies instead of the recorded ones, sent through
/v1/chat/completions. The traces store the shape of each session (token counts, which blocks
repeat, timing, sub-agents) and no text: the client fills that shape with filler text, and
asks for each reply at its recorded length with `ignore_eos`.

Start the server the way you serve it; nothing here changes its flags. `--max-context` must
not exceed the server's `--max-ctx`: sessions that grow past it are left out.

The first `run` sets the client up under --work (a git clone, a Python environment, the
tokenizer and the traces, about 2 GB); later runs reuse it. It needs git, Python 3.11 to 3.13
and network access. The numbers:

    hit_rate              cached prompt tokens / prompt tokens, over the window
    uncached_p50          median prompt tokens a request computed
    uncached split        first turn of a session, the previous reply computed again, new text
    hit_rate_reply_kept   the hit rate if no reply had been computed again
    ttft_p50_s / p90      time to first token
    decode_p50_tok_s      decode speed per request

The readings skip the first --warm seconds, while the sessions start cold.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HARNESS_REPO = "https://github.com/FutureMLS-Lab/FlashStates.git"
HARNESS_COMMIT = "e2ea9a7864b41081a0147a3968e73bd5c6875a7d"
DATASET = "semianalysis_cc_traces_weka_062126"
DATASET_REPO = "semianalysisai/cc-traces-weka-062126"
DATASET_REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"
TOKENIZER_FILES = ["tokenizer*", "vocab*", "merges*", "special_tokens*", "chat_template*", "added_tokens*",
                   "config.json"]
UNIT_S = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "sec": 1.0}


def sh(*cmd, env=None, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env, **kw)


def setup(work: Path, tokenizer: str) -> tuple[Path, Path]:
    """The client's Python environment and the tokenizer directory, built once under `work`."""
    work.mkdir(parents=True, exist_ok=True)
    repo, venv, hf = work / "flashstates", work / "venv", work / "hf"
    python = venv / "bin" / "python"
    if not (repo / ".git").exists():
        sh("git", "clone", HARNESS_REPO, repo)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != HARNESS_COMMIT:
        sh("git", "-C", repo, "fetch", "origin", HARNESS_COMMIT)
        sh("git", "-C", repo, "checkout", HARNESS_COMMIT)
    if not (venv / "bin" / "aiperf").exists():
        sh(sys.executable, "-m", "venv", venv)
        sh(python, "-m", "pip", "install", "--quiet", repo / "benchmarks" / "agentx_harness")
    env = {**os.environ, "HF_HOME": str(hf)}
    tok = Path(tokenizer)
    if not tok.is_dir():                      # a Hugging Face repository: keep its tokenizer files only
        tok = work / "tokenizers" / tokenizer.replace("/", "--")
        if not (tok / "tokenizer.json").exists():
            sh(python, "-c",
               "import sys; from huggingface_hub import snapshot_download; "
               "snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3:])",
               tokenizer, tok, *TOKENIZER_FILES, env=env)
    marker = hf / f".traces-{DATASET_REVISION}"
    if not marker.exists():
        sh("bash", repo / "benchmarks" / "tools" / "fetch_dataset.sh", env={**env, "PYTHON": str(python)})
        marker.touch()
    return venv, tok


def check_server(url: str, model: str) -> None:
    """The server must answer, and must honour ignore_eos: the replay asks for replies of a recorded length."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 48,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        usage = json.load(resp)["usage"]
    if usage["completion_tokens"] < 48:
        raise SystemExit(f"the server stopped after {usage['completion_tokens']} of 48 tokens: it does not honour "
                         "ignore_eos, so replies would not reach their recorded length")
    if "cached_tokens" not in (usage.get("prompt_tokens_details") or {}):
        raise SystemExit("the server does not report usage.prompt_tokens_details.cached_tokens")


def run(a) -> None:
    out = Path(a.out).resolve()
    if (out / "client").exists():
        raise SystemExit(f"{out}/client exists: choose another --out, or `report` it")
    venv, tok = setup(Path(a.work).expanduser(), a.tokenizer)
    check_server(a.url, a.model)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [venv / "bin" / "aiperf", "profile", "--scenario", "agentx", "--unsafe-override",
           "--endpoint-type", "chat", "--weka-live-assistant", "--public-dataset", DATASET,
           "--ui-type", "none", "--export-level", "raw", "--max-context-length", a.max_context,
           "--url", a.url, "--model", a.model, "--tokenizer", tok, "--concurrency", a.sessions,
           "--benchmark-duration", a.duration, "--output-artifact-dir", out / "client"]
    env = {**os.environ, "HF_HOME": str(Path(a.work).expanduser() / "hf"), "HF_DATASETS_OFFLINE": "1"}
    print("+", " ".join(str(c) for c in cmd), f"> {out}/client.log", flush=True)
    with open(out / "client.log", "w") as log:
        rc = subprocess.run([str(c) for c in cmd], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (out / "run.json").write_text(json.dumps({"url": a.url, "model": a.model, "sessions": a.sessions,
                                              "duration_s": a.duration, "warm_s": a.warm,
                                              "max_context": a.max_context, "client_rc": rc}, indent=1) + "\n")
    report(out, a.warm, a.duration)
    if rc:
        raise SystemExit(f"the client exited with {rc}: see {out}/client.log")


def _seconds(node):
    if not isinstance(node, dict) or node.get("value") is None:
        return None
    return float(node["value"]) * UNIT_S[str(node.get("unit", "ms")).lower()]


def _count(node):
    return int(node["value"]) if isinstance(node, dict) and node.get("value") is not None else None


def _percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def readings(out: Path, lo: float, hi: float) -> dict:
    """The readings of one run over [t0 + lo, t0 + hi), t0 = the first measured request's send time."""
    path = next((out / "client").rglob("profile_export.jsonl"))
    records = [json.loads(line) for line in path.open() if line.strip()]
    records = [r for r in records if (r.get("metadata") or {}).get("request_start_ns") is not None]
    def is_measured(r):
        return str(r["metadata"].get("benchmark_phase", "")).endswith("profiling")
    if not any(is_measured(r) for r in records):
        raise SystemExit("no measured request in the export")
    t0 = min(r["metadata"]["request_start_ns"] for r in records if is_measured(r)) / 1e9
    threads = defaultdict(list)                # one conversation thread (a session, or one of its sub-agents)
    errors = 0
    for r in records:
        meta, metrics = r["metadata"], r.get("metrics") or {}
        prompt = _count(metrics.get("usage_prompt_tokens"))
        sent = meta["request_start_ns"] / 1e9 - t0
        if prompt is None:
            errors += bool(is_measured(r) and r.get("error") and not meta.get("was_cancelled") and lo <= sent < hi)
            continue
        out_tokens = _count(metrics.get("usage_completion_tokens")) or 0
        ttft, e2e = _seconds(metrics.get("time_to_first_token")), _seconds(metrics.get("request_latency"))
        decode = (out_tokens - 1) / (e2e - ttft) if ttft is not None and e2e and out_tokens > 1 and e2e > ttft else None
        threads[meta.get("x_correlation_id") or meta.get("conversation_id")].append({
            "turn": meta.get("turn_index"), "sent": sent, "measured": is_measured(r), "prompt": prompt,
            "cached": _count(metrics.get("usage_prompt_cache_read_tokens")) or 0, "out": out_tokens,
            "ttft": ttft, "decode": decode})
    window, split = [], {"first_turn": 0, "reply_again": 0, "new_text": 0}
    for turns in threads.values():
        turns.sort(key=lambda t: (t["turn"], t["sent"]))
        prev = None
        for t in turns:
            if t["measured"] and lo <= t["sent"] < hi:
                uncached = t["prompt"] - t["cached"]
                if prev is None or prev["turn"] != t["turn"] - 1:
                    split["first_turn"] += uncached
                else:
                    # This prompt is the previous prompt, the previous reply, then new text. A server that
                    # resumes after the reply computes only the new text; one that resumes at the end of the
                    # previous prompt computes the reply again too.
                    new_text = max(0, t["prompt"] - prev["prompt"] - prev["out"])
                    again = min(prev["out"], max(0, uncached - new_text))
                    split["reply_again"] += again
                    split["new_text"] += uncached - again
                window.append(t)
            prev = t
    prompt, cached = sum(t["prompt"] for t in window), sum(t["cached"] for t in window)
    ttfts = [t["ttft"] for t in window if t["ttft"] is not None]
    decodes = [t["decode"] for t in window if t["decode"] is not None]
    return {
        "requests": len(window), "errors": errors, "threads": len(threads), "window_s": [lo, hi],
        "prompt_tokens": prompt, "cached_tokens": cached,
        "hit_rate": round(cached / prompt, 4) if prompt else None,
        "uncached_p50": statistics.median(t["prompt"] - t["cached"] for t in window) if window else None,
        "uncached_tokens": {**split, "total": prompt - cached},
        "hit_rate_reply_kept": round((cached + split["reply_again"]) / prompt, 4) if prompt else None,
        "ttft_p50_s": _percentile(ttfts, 0.5), "ttft_p90_s": _percentile(ttfts, 0.9),
        "decode_p50_tok_s": statistics.median(decodes) if decodes else None,
    }


def report(out: Path, lo: float, hi: float) -> None:
    result = readings(out, lo, hi)
    (out / "readings.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    r = sub.add_parser("run", help="replay the traces against a running server, then print the readings")
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--model", required=True, help="the server's --model-name")
    r.add_argument("--tokenizer", required=True,
                   help="the model's Hugging Face repository or a directory with its tokenizer.json")
    r.add_argument("--sessions", type=int, default=1, help="agent sessions at once (sub-agents run inside them)")
    r.add_argument("--duration", type=int, default=1800, help="seconds of replay (the scenario's minimum is 900)")
    r.add_argument("--warm", type=int, default=900, help="seconds left out of the readings at the start")
    r.add_argument("--max-context", type=int, default=131072, help="the server's --max-ctx; longer sessions are left out")
    r.add_argument("--out", required=True, help="a new directory for the run")
    r.add_argument("--work", default="~/.cache/luce-agentx", help="where the client is set up")
    s = sub.add_parser("report", help="the readings of a finished run")
    s.add_argument("out")
    s.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"), help="seconds; default: the run's own")
    a = p.parse_args()
    if a.what == "run":
        if a.warm >= a.duration:
            p.error("--warm must be shorter than --duration")
        if not shutil.which("git"):
            p.error("git is needed to fetch the client")
        run(a)
    else:
        out = Path(a.out).resolve()
        meta = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
        lo, hi = a.window or (meta.get("warm_s", 900), meta.get("duration_s", 1800))
        report(out, lo, hi)


if __name__ == "__main__":
    main()
