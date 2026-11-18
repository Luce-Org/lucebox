#!/usr/bin/env python3
"""Prefix-cache hit rate on recorded coding-agent sessions, under the AgentX benchmark's rules.

Replays the public Claude Code traces of SemiAnalysis's AgentX benchmark against a running
luce_server and reports how much of every prompt the server read from its cache:

    python3 server/scripts/bench_agentx_cache.py run --url http://127.0.0.1:8080 \\
        --model qwen3.8-27b --tokenizer Qwen/Qwen3.8-27B --sessions 1 --out results/agentx-1

    python3 server/scripts/bench_agentx_cache.py report results/agentx-1

The client is the official AgentX harness (`agentx-harness` on PyPI, pinned below) running its
`agentx` scenario unchanged: no `--unsafe-override`, the recorded assistant replies, the
scenario's own timing, cache bust and warm-up. The harness checks those rules itself and
stamps the run `submission_valid`; `report` prints its verdict next to the readings.

The traces store the shape of each session (token counts, which blocks repeat, timing,
sub-agents) and no text: the client fills that shape with filler text and asks for each reply
at its recorded length with `ignore_eos`, which the server must honour.

Start the server the way you serve it; nothing here changes its flags. `--corpus 256k` (the
default) is the official corpus for servers with a 256K context, with every request of at most
256,000 tokens, so serve it with `--max-ctx 262144`. `--corpus full` keeps every request and
needs a server that takes the longest sessions (about 1M tokens).

The first `run` sets the client up under --work (a Python environment, the tokenizer and the
traces); later runs reuse it. It needs Python 3.11 to 3.13 and network access. The numbers,
over the measured (profiling) phase:

    hit_rate              cached prompt tokens / prompt tokens
    perfect_cache         the harness's hit rate for a cache that never misses, same requests
    submission_valid      the harness's verdict on the run's rules, with its reasons when false
    uncached_p50          median prompt tokens a request computed
    uncached split        the first turn of a session, and every later turn
    ttft_p50_s / p90      time to first token
    decode_p50_tok_s      decode speed per request
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HARNESS = "agentx-harness==1.0.13"
CORPUS = {"256k": "semianalysis_cc_traces_weka_062126_256k", "full": "semianalysis_cc_traces_weka_062126"}
MIN_DURATION_S = 900                    # the scenario's floor; its default is 3600
TOKENIZER_FILES = ["tokenizer*", "vocab*", "merges*", "special_tokens*", "chat_template*", "added_tokens*",
                   "config.json"]
UNIT_S = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "sec": 1.0}


def sh(*cmd, env=None, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env, **kw)


def setup(work: Path, tokenizer: str) -> tuple[Path, Path]:
    """The client's Python environment and the tokenizer directory, built once under `work`."""
    work.mkdir(parents=True, exist_ok=True)
    venv = work / f"venv-{HARNESS.replace('==', '-')}"
    python = venv / "bin" / "python"
    if not (venv / "bin" / "agentx").exists():
        sh(sys.executable, "-m", "venv", venv)
        sh(python, "-m", "pip", "install", "--quiet", HARNESS)
    tok = Path(tokenizer)
    if not tok.is_dir():                      # a Hugging Face repository: keep its tokenizer files only
        tok = work / "tokenizers" / tokenizer.replace("/", "--")
        if not (tok / "tokenizer.json").exists():
            sh(python, "-c",
               "import sys; from huggingface_hub import snapshot_download; "
               "snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3:])",
               tokenizer, tok, *TOKENIZER_FILES, env={**os.environ, "HF_HOME": str(work / "hf")})
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
    work = Path(a.work).expanduser()
    venv, tok = setup(work, a.tokenizer)
    check_server(a.url, a.model)
    out.mkdir(parents=True, exist_ok=True)
    # Only the run's own settings: everything the scenario fixes is left to it.
    cmd = [venv / "bin" / "agentx", "profile", "--url", a.url, "--endpoint", "/v1/chat/completions",
           "--model", a.model, "--tokenizer", tok, "--tokenizer-trust-remote-code",
           "--concurrency", a.sessions, "--public-dataset", CORPUS[a.corpus],
           "--benchmark-duration", a.duration, "--ui-type", "none", "--export-level", "raw",
           "--output-artifact-dir", out / "client"]
    env = {**os.environ, "HF_HOME": str(work / "hf")}
    print("+", " ".join(str(c) for c in cmd), f"> {out}/client.log", flush=True)
    with open(out / "client.log", "w") as log:
        rc = subprocess.run([str(c) for c in cmd], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (out / "run.json").write_text(json.dumps({"url": a.url, "model": a.model, "sessions": a.sessions,
                                              "duration_s": a.duration, "corpus": CORPUS[a.corpus],
                                              "harness": HARNESS, "client_rc": rc}, indent=1) + "\n")
    report(out)
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


def verdict(out: Path) -> dict:
    """The harness's own summary: its perfect-cache hit rate and whether the run kept the rules."""
    path = next((out / "client").rglob("profile_export_aiperf.json"), None)
    if path is None:
        return {}
    summary = json.loads(path.read_text())
    meta = summary.get("metadata") or {}
    perfect = (summary.get("theoretical_prefix_cache_hit") or {}).get("avg")
    return {"perfect_cache": round(perfect / 100, 4) if perfect is not None else None,
            "submission_valid": meta.get("submission_valid"),
            "submission_invalid_reasons": meta.get("submission_invalid_reasons") or []}


def readings(out: Path) -> dict:
    """The readings of the measured (profiling) phase of one run."""
    path = next((out / "client").rglob("profile_export.jsonl"))
    records = [json.loads(line) for line in path.open() if line.strip()]
    records = [r for r in records if (r.get("metadata") or {}).get("request_start_ns") is not None]
    def is_measured(r):
        return str(r["metadata"].get("benchmark_phase", "")).endswith("profiling")
    if not any(is_measured(r) for r in records):
        raise SystemExit("no measured request in the export")
    threads = defaultdict(list)                # one conversation thread (a session, or one of its sub-agents)
    errors = 0
    for r in records:
        meta, metrics = r["metadata"], r.get("metrics") or {}
        prompt = _count(metrics.get("usage_prompt_tokens"))
        if prompt is None:
            errors += bool(is_measured(r) and r.get("error") and not meta.get("was_cancelled"))
            continue
        out_tokens = _count(metrics.get("usage_completion_tokens")) or 0
        ttft, e2e = _seconds(metrics.get("time_to_first_token")), _seconds(metrics.get("request_latency"))
        decode = (out_tokens - 1) / (e2e - ttft) if ttft is not None and e2e and out_tokens > 1 and e2e > ttft else None
        threads[meta.get("x_correlation_id") or meta.get("conversation_id")].append({
            "turn": meta.get("turn_index"), "sent": meta["request_start_ns"], "measured": is_measured(r),
            "prompt": prompt, "cached": _count(metrics.get("usage_prompt_cache_read_tokens")) or 0,
            "ttft": ttft, "decode": decode})
    window, split = [], {"first_turn": 0, "later_turns": 0}
    for turns in threads.values():
        turns.sort(key=lambda t: (t["turn"], t["sent"]))
        prev = None
        for t in turns:
            if t["measured"]:
                first = prev is None or prev["turn"] != t["turn"] - 1
                split["first_turn" if first else "later_turns"] += t["prompt"] - t["cached"]
                window.append(t)
            prev = t
    prompt, cached = sum(t["prompt"] for t in window), sum(t["cached"] for t in window)
    ttfts = [t["ttft"] for t in window if t["ttft"] is not None]
    decodes = [t["decode"] for t in window if t["decode"] is not None]
    return {
        "requests": len(window), "errors": errors, "threads": len(threads),
        "prompt_tokens": prompt, "cached_tokens": cached,
        "hit_rate": round(cached / prompt, 4) if prompt else None,
        **verdict(out),
        "uncached_p50": statistics.median(t["prompt"] - t["cached"] for t in window) if window else None,
        "uncached_tokens": {**split, "total": prompt - cached},
        "ttft_p50_s": _percentile(ttfts, 0.5), "ttft_p90_s": _percentile(ttfts, 0.9),
        "decode_p50_tok_s": statistics.median(decodes) if decodes else None,
    }


def report(out: Path) -> None:
    result = readings(out)
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
    r.add_argument("--duration", type=int, default=3600,
                   help=f"seconds of measured replay (the scenario's default; at least {MIN_DURATION_S})")
    r.add_argument("--corpus", choices=sorted(CORPUS), default="256k",
                   help="256k: requests of at most 256,000 tokens (serve with --max-ctx 262144); full: all of them")
    r.add_argument("--out", required=True, help="a new directory for the run")
    r.add_argument("--work", default="~/.cache/luce-agentx", help="where the client is set up")
    s = sub.add_parser("report", help="the readings of a finished run")
    s.add_argument("out")
    a = p.parse_args()
    if a.what == "run":
        if a.duration < MIN_DURATION_S:
            p.error(f"--duration must be at least {MIN_DURATION_S}: shorter runs break the benchmark's rules")
        run(a)
    else:
        report(Path(a.out).resolve())


if __name__ == "__main__":
    main()
