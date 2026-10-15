#!/usr/bin/env python3
"""Model-backed end-to-end check of luce_server on one GPU.

Starts the server, waits for the model to load, sends the fixed prompt suite in
prompts.json with greedy decoding, stops the server, and compares the results
with the last good run on main (the baseline).

Verdict:
  fail  the server crashed, hung, did not stop, logged GPU errors, never
        loaded, a request failed or timed out, the suite ran out of time, two
        or more checks that passed on the baseline now fail, or fewer than
        half of the checks pass;
  warn  output text differs from the baseline, one check regressed, a check
        fails that also failed on the baseline (or there is no baseline), a
        repeated request gave a different answer, decode speed dropped more
        than 15%, loading got 50% slower, or the kernel log could not be read
        or lost messages during the run;
  pass  otherwise.

Text differences and a single regressed check only warn: kernel changes
legitimately flip near-tie tokens, and some short answers sit on a near tie (on
the Strix Halo, DS4 answers 17 * 23 correctly with exact prefill and not with
sparse prefill). A real bug usually breaks several checks at once.
The exit status is 1 on fail and 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_PASS_RATE = 0.5
# Checks that passed on the baseline and now fail; fewer than this only warn.
REGRESSIONS_TO_FAIL = 2
PERF_DROP = 0.15
LOAD_GROWTH = 0.5
# Prompts with at least this many generated tokens feed the decode-speed median.
SPEED_MIN_TOKENS = 32
GPU_ERROR = re.compile(
    r"amdgpu.*(fault|timeout|reset|hang)|kfd.*(fault|error)|ring \S+ timeout", re.IGNORECASE
)
# dmesg's "[seconds since boot]" prefix.
KERNEL_TIME = re.compile(r"^\[\s*(\d+\.\d+)\]")


# ─── Prompts and checks ───────────────────────────────────────────────


def needle_messages() -> list[dict]:
    colors = ["red", "blue", "green", "amber", "silver", "violet"]
    things = ["crate", "ladder", "lantern", "toolbox", "barrel", "rope"]
    lines = []
    for i in range(180):
        lines.append(
            f"Note {i}: the {colors[i % 6]} {things[(i * 5) % 6]} was moved to shelf {(i * 7) % 97}."
        )
        if i == 110:
            lines.append("Remember this: the secret code is 7429.")
    content = (
        "Read these warehouse notes.\n\n"
        + "\n".join(lines)
        + "\n\nWhat is the secret code mentioned in the notes? Reply with only the code."
    )
    return [{"role": "user", "content": content}]


def load_prompts(path: Path) -> list[dict]:
    prompts = json.loads(path.read_text())
    for prompt in prompts:
        # chat() reassembles streamed text only, not streamed tool calls.
        if prompt.get("stream") and prompt.get("tools"):
            raise ValueError(f"prompt {prompt['id']}: streaming with tools is not supported")
        if prompt.get("generate") == "needle":
            prompt["messages"] = needle_messages()
    return prompts


def request_body(prompt: dict) -> dict:
    body = {
        "model": "luce",
        "messages": prompt["messages"],
        "max_tokens": prompt["max_tokens"],
        "temperature": 0,
        "stream": bool(prompt.get("stream")),
        "chat_template_kwargs": {"enable_thinking": bool(prompt.get("thinking"))},
    }
    if prompt.get("tools"):
        body["tools"] = prompt["tools"]
    return body


def observed_text(message: dict) -> str:
    parts = []
    if message.get("reasoning_content"):
        parts.append(f"[reasoning] {message['reasoning_content']}")
    if message.get("content"):
        parts.append(message["content"])
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        parts.append(f"[tool] {fn.get('name')} {fn.get('arguments')}")
    return "\n".join(parts)


def first_json_object(text: str) -> object:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])


def run_check(check: dict, message: dict) -> tuple[bool, str]:
    content = message.get("content") or ""
    if check.get("field") == "any":
        text = f"{message.get('reasoning_content') or ''}\n{content}"
    else:
        text = content
    lowered = text.lower()
    problems = []

    if "contains" in check:
        missing = [s for s in check["contains"] if s.lower() not in lowered]
        if missing:
            problems.append(f"missing {missing}")
    if "any" in check and not any(s.lower() in lowered for s in check["any"]):
        problems.append(f"none of {check['any']}")
    if "regex" in check and not re.search(check["regex"], text, re.IGNORECASE | re.DOTALL):
        problems.append(f"no match for /{check['regex']}/")
    if "number" in check:
        numbers = []
        for raw in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
            try:
                numbers.append(float(raw.replace(",", "")))
            except ValueError:
                continue
        if float(check["number"]) not in numbers:
            problems.append(f"no {check['number']}")
    if "sequence" in check:
        want = list(range(1, check["sequence"] + 1))
        if [int(n) for n in re.findall(r"\d+", text)][: len(want)] != want:
            problems.append(f"not 1..{check['sequence']}")
    if "json" in check:
        try:
            data = first_json_object(text)
            if not isinstance(data, dict) or any(
                data.get(k) != v for k, v in check["json"].items()
            ):
                problems.append(f"JSON {data!r} does not match")
        except ValueError as exc:
            problems.append(f"invalid JSON ({exc})")
    if "tool_call" in check:
        want = check["tool_call"]
        found = False
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            arguments = fn.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {}
            if fn.get("name") == want["name"] and all(
                str(v).lower() in str(arguments.get(k, "")).lower()
                for k, v in want.get("arguments_contain", {}).items()
            ):
                found = True
        if not found:
            problems.append(f"no {want['name']} tool call")
    if "min_words" in check and len(text.split()) < check["min_words"]:
        problems.append(f"fewer than {check['min_words']} words")
    if check.get("no_loop"):
        words = text.split()
        grams = Counter(tuple(words[i : i + 4]) for i in range(len(words) - 3))
        if grams and max(grams.values()) > 4:
            problems.append("repeats itself")
    return (not problems, "; ".join(problems))


# ─── HTTP ─────────────────────────────────────────────────────────────


def http_json(url: str, body: dict | None, timeout: float) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def chat(base_url: str, body: dict, timeout: float) -> tuple[dict, dict]:
    """Send one chat request; return (message, usage). Streams are reassembled."""
    url = f"{base_url}/v1/chat/completions"
    if not body.get("stream"):
        reply = http_json(url, body, timeout)
        return reply["choices"][0]["message"], reply.get("usage") or {}

    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    deadline = time.monotonic() + timeout
    message: dict = {"content": "", "reasoning_content": ""}
    usage: dict = {}
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            if time.monotonic() > deadline:
                raise TimeoutError("stream exceeded the request timeout")
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                message["content"] += delta.get("content") or ""
                message["reasoning_content"] += delta.get("reasoning_content") or ""
    if not done:
        raise RuntimeError("stream ended without [DONE]")
    return message, usage


# ─── Server lifecycle ─────────────────────────────────────────────────


def rocm_version() -> str | None:
    try:
        return Path("/opt/rocm/.info/version").read_text().strip() or None
    except OSError:
        return None


def read_kernel_log() -> list[str] | None:
    for cmd in (["sudo", "-n", "dmesg"], ["dmesg"]):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            return proc.stdout.splitlines()
    return None


def stamped(lines: list[str]) -> list[tuple[float | None, str]] | None:
    """Pair each line with its timestamp; continuation lines take the one before.

    None when the log has lines but no timestamps (printk.time=0)."""
    out: list[tuple[float | None, str]] = []
    now = None
    for line in lines:
        match = KERNEL_TIME.match(line)
        if match:
            now = float(match.group(1))
        out.append((now, line))
    return out if now is not None or not lines else None


def kernel_lines_since(before: list[str], after: list[str]) -> tuple[list[str], bool] | None:
    """The lines of `after` logged after `before` was read, and whether none can be missing.

    Lines are matched by timestamp, not position: the kernel's ring buffer drops
    its oldest lines as new ones arrive, so line counts shift. When `after` no
    longer reaches back to the last line of `before`, the buffer wrapped or was
    cleared during the run and messages in between may be lost. None when the
    log has no timestamps to compare.
    """
    old, new = stamped(before), stamped(after)
    if old is None or new is None:
        return None
    if not old:
        return after, True
    last = old[-1][0]
    first = next((t for t, _ in new if t is not None), None)
    continuous = first is not None and first <= last
    # Lines stamped exactly `last` may be old or new; the old ones are known.
    old_at_last = Counter(line for t, line in old if t == last)
    lines = []
    for t, line in new:
        if t is None or t < last:
            continue
        if t == last and old_at_last[line] > 0:
            old_at_last[line] -= 1
            continue
        lines.append(line)
    return lines, continuous


def stop_server(proc: subprocess.Popen) -> bool:
    """Stop the server's process group. Returns False if it would not die."""
    if proc.poll() is not None:
        return True
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return True
        try:
            proc.wait(timeout=wait)
            return True
        except subprocess.TimeoutExpired:
            continue
    return False


def run_suite(args: argparse.Namespace, prompts: list[dict], out_dir: Path) -> dict:
    cmd = [args.server, args.target]
    if args.draft:
        cmd += ["--draft", args.draft]
    cmd += ["--host", "127.0.0.1", "--port", str(args.port), *shlex.split(args.server_args)]
    base_url = f"http://127.0.0.1:{args.port}"
    result: dict = {
        "model": args.model,
        "device": args.device,
        "host": socket.gethostname(),
        "rocm": rocm_version(),
        "commit": args.commit,
        "config": {
            "target": Path(args.target).name,
            "draft": Path(args.draft).name if args.draft else None,
            "server_args": args.server_args,
        },
        "command": shlex.join(cmd),
        "load_seconds": None,
        "server": {"crashed": False, "returncode": None, "stuck": False, "loaded": False},
        "budget": args.budget,
        "over_budget": 0,  # prompts not sent because the suite ran out of time
        "gpu_errors": [],
        # checked, incomplete (messages may be lost) or unreadable (or no timestamps).
        "kernel_log": "unreadable",
        "prompts": [],
    }

    kernel_before = read_kernel_log()
    log = (out_dir / "server.log").open("wb")
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        # Load: wait for /health, then a tiny request forces the (lazy) model load.
        load_deadline = started + args.load_timeout
        while time.monotonic() < load_deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=5):
                    break
            except OSError:
                time.sleep(0.5)
        if proc.poll() is None and time.monotonic() < load_deadline:
            try:
                warmup = {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8}
                chat(base_url, {**warmup, "temperature": 0}, load_deadline - time.monotonic())
                result["server"]["loaded"] = True
                result["load_seconds"] = round(time.monotonic() - started, 1)
            except Exception as exc:
                result["load_error"] = f"{type(exc).__name__}: {exc}"

        by_id: dict[str, dict] = {}
        budget_deadline = time.monotonic() + args.budget
        for prompt in prompts:
            source = by_id.get(prompt["repeat_of"]) if prompt.get("repeat_of") else None
            spec = source["prompt"] if source else prompt
            entry = {
                "id": prompt["id"],
                "kind": "determinism" if prompt.get("repeat_of") else "check",
                "status": "error",
                "detail": "",
                "observed": "",
                "completion_tokens": None,
                "seconds": None,
            }
            result["prompts"].append(entry)
            # Follow-on skips: the load or crash failure is reported once, not per prompt.
            if not result["server"]["loaded"]:
                entry.update(status="skipped", detail="server did not load")
                continue
            if proc.poll() is not None:
                entry.update(status="skipped", detail="server is not running")
                continue
            if time.monotonic() > budget_deadline:
                entry.update(status="skipped", detail="suite budget used up")
                result["over_budget"] += 1
                continue
            if prompt.get("repeat_of") and source is None:
                entry.update(status="skipped", detail=f"{prompt['repeat_of']} got no answer")
                continue
            t0 = time.monotonic()
            # A request never outlasts the suite budget, so the budget bounds the step.
            timeout = min(args.request_timeout, budget_deadline - t0)
            try:
                message, usage = chat(base_url, request_body(spec), timeout)
            except Exception as exc:
                if time.monotonic() >= budget_deadline:
                    entry.update(status="skipped", detail="suite budget used up")
                    result["over_budget"] += 1
                    continue
                entry["detail"] = f"{type(exc).__name__}: {exc}"[:300]
                # A failed request often means the server just died; let it finish
                # exiting so the next prompts are skipped rather than failing too.
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                continue
            seconds = time.monotonic() - t0
            entry["seconds"] = round(seconds, 2)
            entry["completion_tokens"] = usage.get("completion_tokens")
            entry["observed"] = observed_text(message)
            if prompt.get("repeat_of"):
                same = entry["observed"] == source["entry"]["observed"]
                entry["status"] = "pass" if same else "differs"
                entry["detail"] = "" if same else f"differs from {prompt['repeat_of']}"
            else:
                ok, detail = run_check(prompt["check"], message)
                entry["status"] = "pass" if ok else "fail"
                entry["detail"] = detail
                by_id[prompt["id"]] = {"prompt": prompt, "entry": entry}
    finally:
        if proc.poll() is not None:
            result["server"]["crashed"] = True
            result["server"]["returncode"] = proc.returncode
        result["server"]["stuck"] = not stop_server(proc)
        log.close()

    kernel_after = read_kernel_log()
    since = None
    if kernel_before is not None and kernel_after is not None:
        since = kernel_lines_since(kernel_before, kernel_after)
    if since is not None:
        new, continuous = since
        result["kernel_log"] = "checked" if continuous else "incomplete"
        result["gpu_errors"] = [line for line in new if GPU_ERROR.search(line)][:20]
    return result


# ─── Evaluation and report ────────────────────────────────────────────


def first_divergence(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def decode_speed(result: dict) -> float | None:
    rates = [
        p["completion_tokens"] / p["seconds"]
        for p in result["prompts"]
        if p["kind"] == "check"
        and p["status"] in ("pass", "fail")
        and (p["completion_tokens"] or 0) >= SPEED_MIN_TOKENS
        and p["seconds"]
    ]
    return statistics.median(rates) if rates else None


def evaluate(result: dict, baseline: dict | None) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    server = result["server"]
    if not server["loaded"]:
        failures.append(
            f"the server did not finish loading ({result.get('load_error', 'no reply')})"
        )
    if server["crashed"]:
        failures.append(f"the server exited during the run (exit code {server['returncode']})")
    if server["stuck"]:
        failures.append("the server did not stop after SIGKILL; the GPU driver may be wedged")
    if result["gpu_errors"]:
        failures.append(f"{len(result['gpu_errors'])} GPU error(s) in the kernel log")
    if result.get("over_budget"):
        failures.append(
            f"the suite used up its {result['budget']:.0f}s budget; "
            f"{result['over_budget']} prompt(s) did not run"
        )
    kernel_log = result.get("kernel_log", "unreadable")
    if kernel_log == "incomplete":
        warnings.append(
            "the kernel log wrapped or was cleared during the run; GPU errors may be missing"
        )
    elif kernel_log == "unreadable":
        warnings.append("the kernel log could not be read or compared; GPU errors were not checked")

    base_prompts = {p["id"]: p for p in (baseline or {}).get("prompts", [])}
    checks = [p for p in result["prompts"] if p["kind"] == "check" and p["status"] != "skipped"]
    regressions = []
    for p in result["prompts"]:
        before = base_prompts.get(p["id"])
        if p["status"] == "error":
            failures.append(f"`{p['id']}`: request failed: {p['detail']}")
        elif p["status"] == "fail":
            if before and before["status"] == "pass":
                regressions.append(f"`{p['id']}`: passed on the baseline, now fails: {p['detail']}")
            else:
                warnings.append(f"`{p['id']}`: check fails: {p['detail']}")
        elif p["status"] == "differs":
            warnings.append(f"`{p['id']}`: the same request gave a different answer")
        if (
            before
            and p["kind"] == "check"
            and p["status"] in ("pass", "fail")
            and before["observed"] != p["observed"]
        ):
            at = first_divergence(before["observed"], p["observed"])
            warnings.append(f"`{p['id']}`: output differs from the baseline from character {at}")

    (failures if len(regressions) >= REGRESSIONS_TO_FAIL else warnings).extend(regressions)

    passed = sum(p["status"] == "pass" for p in checks)
    if checks and server["loaded"] and passed / len(checks) < MIN_PASS_RATE:
        failures.append(f"only {passed}/{len(checks)} checks pass")

    if baseline:
        if baseline.get("config") != result.get("config"):
            warnings.append("the baseline used a different model or server configuration")
        # A ROCm upgrade since the baseline explains drift.
        if baseline.get("rocm") != result.get("rocm"):
            warnings.append(
                f"the baseline ran on ROCm {baseline.get('rocm') or '?'}, "
                f"this run on {result.get('rocm') or '?'}"
            )
        now, then = decode_speed(result), decode_speed(baseline)
        if now and then and now < then * (1 - PERF_DROP):
            warnings.append(f"decode speed {now:.1f} tok/s vs {then:.1f} on the baseline")
        load_now, load_then = result.get("load_seconds"), baseline.get("load_seconds")
        if load_now and load_then and load_now > load_then * (1 + LOAD_GROWTH):
            warnings.append(f"load took {load_now:.0f}s vs {load_then:.0f}s on the baseline")
    return failures, warnings


def render_report(result: dict, baseline: dict | None, title: str, log_tail: str) -> str:
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[result["verdict"]]
    base_prompts = {p["id"]: p for p in (baseline or {}).get("prompts", [])}
    speed = decode_speed(result)
    lines = [
        f"## {icon} {title}: {result['verdict']}",
        "",
        f"- Model: `{result['config']['target']}`"
        + (f" + draft `{result['config']['draft']}`" if result["config"]["draft"] else ""),
        f"- Server args: `{result['config']['server_args'] or '(none)'}`",
        f"- ROCm: {result.get('rocm') or 'unknown'}",
        f"- Load: {result['load_seconds']}s · decode median: "
        + (f"{speed:.1f} tok/s" if speed else "n/a"),
        "- Baseline: "
        + (
            f"main @ `{(baseline.get('commit') or '?')[:10]}` on {baseline.get('host', '?')}"
            if baseline
            else "none yet"
        ),
        "- Kernel log: "
        + {
            "checked": "checked",
            "incomplete": "checked, but messages were lost during the run",
            "unreadable": "not readable (or has no timestamps)",
        }[result["kernel_log"]],
        "",
    ]
    if result["failures"]:
        lines += ["**Failures**", "", *[f"- {f}" for f in result["failures"]], ""]
    if result["warnings"]:
        lines += ["**Warnings**", "", *[f"- {w}" for w in result["warnings"]], ""]
    lines += ["| Prompt | Result | Tokens | Seconds | vs baseline |", "|---|---|---:|---:|---|"]
    for p in result["prompts"]:
        before = base_prompts.get(p["id"])
        if not before:
            versus = ""
        elif before["observed"] == p["observed"]:
            versus = "same text"
        else:
            versus = f"differs @ {first_divergence(before['observed'], p['observed'])}"
        detail = f" ({p['detail']})" if p["detail"] else ""
        lines.append(
            f"| `{p['id']}` | {p['status']}{detail} | {p['completion_tokens'] or ''} "
            f"| {p['seconds'] or ''} | {versus} |"
        )
    if result["gpu_errors"]:
        lines += ["", "**GPU errors in the kernel log**", "", "```", *result["gpu_errors"], "```"]
    if result["verdict"] == "fail" and log_tail:
        lines += [
            "",
            "<details><summary>Server log (tail)</summary>",
            "",
            "```",
            log_tail,
            "```",
            "",
            "</details>",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", required=True, help="label, e.g. qwen or ds4")
    parser.add_argument("--device", required=True, help="label, e.g. r9700")
    parser.add_argument("--server", required=True, help="luce_server binary")
    parser.add_argument("--target", required=True, help="target GGUF")
    parser.add_argument("--draft", help="draft GGUF")
    parser.add_argument("--server-args", default="", help="extra server flags, shell-quoted")
    parser.add_argument("--port", type=int, default=18200)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompts", default=str(HERE / "prompts.json"))
    parser.add_argument("--baseline", help="result JSON of the last good main run")
    parser.add_argument("--write-baseline", help="store this run here unless it fails")
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--load-timeout", type=float, default=420)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--budget", type=float, default=600, help="seconds for the whole suite")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which(args.server):
        print(f"::error::server binary not found: {args.server}")
        return 2
    for label, path in (("target", args.target), ("draft", args.draft)):
        if path and not Path(path).is_file():
            print(f"::error::{label} model not found: {path}")
            return 2

    baseline = None
    if args.baseline and Path(args.baseline).is_file():
        baseline = json.loads(Path(args.baseline).read_text())

    result = run_suite(args, load_prompts(Path(args.prompts)), out_dir)
    failures, warnings = evaluate(result, baseline)
    result["failures"], result["warnings"] = failures, warnings
    result["verdict"] = "fail" if failures else "warn" if warnings else "pass"

    log_lines = (out_dir / "server.log").read_text(errors="replace").splitlines()
    report = render_report(
        result,
        baseline,
        f"{args.model} on {args.device} @ {result['host']}",
        "\n".join(log_lines[-60:]),
    )
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "report.md").write_text(report)
    print(report)
    for failure in failures:
        print(f"::error title=Model e2e ({args.model})::{failure}")
    for warning in warnings:
        print(f"::warning title=Model e2e ({args.model})::{warning}")

    if args.write_baseline and result["verdict"] != "fail":
        target = Path(args.write_baseline)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(target)
        print(f"Baseline updated: {target}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
