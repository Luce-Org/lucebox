"""Tests for the model e2e scripts. A small fake server stands in for luce_server."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import run_model_e2e
from find_baseline import add_baselines, keep_changed
from run_model_e2e import evaluate, kernel_lines_since, load_prompts, main, run_check
from select_models import matrix, models_for

HERE = Path(__file__).resolve().parent

# The fake server's answer to each prompt in prompts.json, found by a phrase in
# the prompt's last message (the tool-call prompt about the weather is handled
# separately). test_the_fake_answers_every_prompt keeps this in step with
# prompts.json.
ANSWERS = [
    ("17 * 23", "391"),
    ("capital of France", "Paris"),
    ("prime numbers", "2, 3, 5, 7, 11"),
    ("train travels", "40"),
    ("Count from 1 to 40", " ".join(str(i) for i in range(1, 41))),
    ("is_even", "def is_even(n):\n    return n % 2 == 0"),
    ("JSON object", '{"name": "lucebox", "version": 3}'),
    ("good morning", "Buongiorno"),
    ("Repeat this text", "Città 東京 🚀"),
    ("which city", "Lisbon."),
    ("secret code", "7429"),
    ("capital of Japan", "Tokyo"),
    ("12 + 30", "42"),
    ("lighthouse", " ".join(f"word{i}" for i in range(60))),
    ("Say hello", "Hello"),
]

FAKE_SERVER = textwrap.dedent(
    """
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    ANSWERS = json.loads(os.environ["FAKE_ANSWERS"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            question = body["messages"][-1]["content"]
            crash = os.environ.get("FAKE_CRASH_ON")
            slow = os.environ.get("FAKE_SLOW_ON")
            if slow and slow in question:
                time.sleep(30)
            if os.environ.get("FAKE_WRONG") and "train travels" in question:
                question = "Say hello"
            if crash and crash in question:
                sys.stderr.write("fake server: simulated crash\\n")
                sys.stderr.flush()
                os._exit(3)
            message = {"role": "assistant", "content": ""}
            if "weather" in question:
                message["tool_calls"] = [{"type": "function", "function": {
                    "name": "get_weather", "arguments": json.dumps({"city": "Rome"})}}]
            else:
                message["content"] = next(a for q, a in ANSWERS if q in question)
                if os.environ.get("FAKE_WRONG") and "17 * 23" in question:
                    message["content"] = "392"
            usage = {"prompt_tokens": 10, "completion_tokens": len(message["content"].split())}
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in message["content"].split(" "):
                    chunk = {"choices": [{"delta": {"content": word + " "}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\\n\\n".encode())
                self.wfile.write(b"data: [DONE]\\n\\n")
                return
            data = json.dumps({"choices": [{"message": message}], "usage": usage}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    port = int(sys.argv[sys.argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def quiet_kernel_log(monkeypatch: pytest.MonkeyPatch) -> None:
    # The machine running the tests may not let us read its kernel log.
    monkeypatch.setattr(run_model_e2e, "read_kernel_log", lambda: ["[    1.000000] boot"])


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FAKE_ANSWERS", json.dumps(ANSWERS))
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER)
    return server


def test_the_fake_answers_every_prompt() -> None:
    # The healthy-run tests send the real prompts.json to the fake server. A new
    # or reworded prompt needs an entry in ANSWERS (and an answer that passes
    # its check), or those tests fail with a confusing request error.
    prompts = load_prompts(HERE / "prompts.json")
    for prompt in prompts:
        if prompt.get("repeat_of"):
            continue
        question = prompt["messages"][-1]["content"]
        assert "weather" in question or any(q in question for q, _ in ANSWERS), (
            f"prompt {prompt['id']!r} has no answer in ANSWERS"
        )
    # test_crash_fails_and_skips_the_rest crashes on `primes`.
    ids = [p["id"] for p in prompts]
    assert ids.index("capital") < ids.index("primes") < ids.index("count")


def run(fake: Path, out: Path, *extra: str) -> tuple[int, dict]:
    # The fake is launched as `python fake_server.py <flags>`: python is the
    # "server binary" and the script sits where the target GGUF would.
    code = main([
        "--model", "fake", "--device", "cpu", "--server", sys.executable,
        "--target", str(fake), "--port", str(free_port()), "--out-dir", str(out),
        "--load-timeout", "20", "--request-timeout", "10", *extra,
    ])  # fmt: skip
    return code, json.loads((out / "result.json").read_text())


def test_healthy_server_passes_and_writes_a_baseline(fake: Path, tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    code, result = run(fake, tmp_path / "a", "--write-baseline", str(baseline))
    assert code == 0, result["failures"]
    assert result["verdict"] == "pass"
    assert all(p["status"] == "pass" for p in result["prompts"])
    assert baseline.is_file()

    code, again = run(fake, tmp_path / "b", "--baseline", str(baseline))
    assert code == 0 and again["verdict"] == "pass"
    assert "same text" in (tmp_path / "b" / "report.md").read_text()


def test_regressions_against_the_baseline_fail(
    fake: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    run(fake, tmp_path / "a", "--write-baseline", str(baseline))
    monkeypatch.setenv("FAKE_WRONG", "1")
    code, result = run(fake, tmp_path / "b", "--baseline", str(baseline))
    assert code == 1
    assert any("`arith`: passed on the baseline" in f for f in result["failures"])
    assert any("`word_problem`: passed on the baseline" in f for f in result["failures"])


def test_a_single_regression_only_warns() -> None:
    base = result(prompt("a", "pass", "x"), prompt("b", "pass", "y"))
    now = result(prompt("a", "fail", "x2"), prompt("b", "pass", "y"))
    failures, warnings = evaluate(now, base)
    assert failures == []
    assert any("`a`: passed on the baseline" in w for w in warnings)


def test_crash_fails_and_skips_the_rest(
    fake: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CRASH_ON", "prime numbers")
    code, result = run(fake, tmp_path / "a")
    assert code == 1
    assert result["server"]["crashed"] and result["server"]["returncode"] == 3
    statuses = {p["id"]: p["status"] for p in result["prompts"]}
    assert statuses["capital"] == "pass"
    assert statuses["primes"] == "error"
    assert statuses["count"] == "skipped"
    assert not any("checks pass" in f for f in result["failures"])
    assert "simulated crash" in (tmp_path / "a" / "report.md").read_text()


def test_missing_server_is_a_setup_error(tmp_path: Path) -> None:
    code = main([
        "--model", "x", "--device", "y", "--server", str(tmp_path / "luce_server"),
        "--target", str(tmp_path), "--out-dir", str(tmp_path),
    ])  # fmt: skip
    assert code == 2


def test_missing_target_is_a_setup_error(tmp_path: Path) -> None:
    code = main([
        "--model", "x", "--device", "y", "--server", "true",
        "--target", str(tmp_path / "missing.gguf"), "--out-dir", str(tmp_path),
    ])  # fmt: skip
    assert code == 2


def test_missing_draft_is_a_setup_error(fake: Path, tmp_path: Path) -> None:
    # Running without the draft would test a different configuration.
    code = main([
        "--model", "x", "--device", "y", "--server", "true", "--target", str(fake),
        "--draft", str(tmp_path / "missing.gguf"), "--out-dir", str(tmp_path),
    ])  # fmt: skip
    assert code == 2


def test_running_out_of_budget_fails_as_such(fake: Path, tmp_path: Path) -> None:
    code, result = run(fake, tmp_path / "a", "--budget", "0")
    assert code == 1
    assert result["over_budget"] == len(result["prompts"])
    assert any("used up its 0s budget" in f for f in result["failures"])
    assert not any("request failed" in f for f in result["failures"])


def test_a_slow_request_stops_at_the_budget(
    fake: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The request timeout (10 s) is longer than what is left of the budget.
    monkeypatch.setenv("FAKE_SLOW_ON", "prime numbers")
    started = time.monotonic()
    code, result = run(fake, tmp_path / "a", "--budget", "3")
    assert time.monotonic() - started < 10
    assert code == 1
    primes = next(p for p in result["prompts"] if p["id"] == "primes")
    assert primes["status"] == "skipped"
    assert any("budget" in f for f in result["failures"])
    assert not any("request failed" in f for f in result["failures"])


def test_streaming_prompts_cannot_use_tools(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"id": "x", "stream": True, "tools": [{}]}]))
    with pytest.raises(ValueError, match="streaming with tools"):
        load_prompts(prompts)


@pytest.mark.parametrize(
    ("check", "message", "ok"),
    [
        ({"number": 391}, {"content": "The answer is 391."}, True),
        ({"number": 391}, {"content": "392"}, False),
        ({"sequence": 5}, {"content": "1 2 3 4 5"}, True),
        ({"sequence": 5}, {"content": "1 2 4 5"}, False),
        ({"json": {"v": 3}}, {"content": '```json\n{"v": 3}\n```'}, True),
        ({"json": {"v": 3}}, {"content": "{v: 3}"}, False),
        ({"any": ["buongiorno", "buon giorno"]}, {"content": "Buon giorno!"}, True),
        ({"number": 42, "field": "any"}, {"reasoning_content": "so 42", "content": ""}, True),
        ({"number": 42}, {"reasoning_content": "so 42", "content": ""}, False),
        (
            {"tool_call": {"name": "f", "arguments_contain": {"city": "rome"}}},
            {"tool_calls": [{"function": {"name": "f", "arguments": '{"city": "Rome"}'}}]},
            True,
        ),
        ({"no_loop": True}, {"content": "a b c d " * 6}, False),
        ({"min_words": 3}, {"content": "one two"}, False),
    ],
)
def test_checks(check: dict, message: dict, ok: bool) -> None:
    assert run_check(check, message)[0] is ok


def prompt(pid: str, status: str, observed: str, tokens: int = 64, seconds: float = 2.0) -> dict:
    return {
        "id": pid, "kind": "check", "status": status, "detail": "", "observed": observed,
        "completion_tokens": tokens, "seconds": seconds,
    }  # fmt: skip


def result(*prompts: dict, load: float = 10.0) -> dict:
    return {
        "config": {"target": "m.gguf", "draft": None, "server_args": ""},
        "load_seconds": load,
        "server": {"crashed": False, "returncode": None, "stuck": False, "loaded": True},
        "gpu_errors": [],
        "kernel_log": "checked",
        "prompts": list(prompts),
    }


def test_evaluate_warns_on_drift_speed_and_load() -> None:
    base = result(prompt("a", "pass", "hello world", seconds=2.0), load=10)
    now = result(prompt("a", "pass", "hello there", seconds=4.0), load=30)
    failures, warnings = evaluate(now, base)
    assert failures == []
    assert any("differs from the baseline from character 6" in w for w in warnings)
    assert any("decode speed" in w for w in warnings)
    assert any("load took" in w for w in warnings)


def test_evaluate_only_warns_on_checks_that_already_failed() -> None:
    base = result(prompt("a", "fail", "x"), prompt("b", "pass", "y"))
    now = result(prompt("a", "fail", "x"), prompt("b", "pass", "y"))
    failures, warnings = evaluate(now, base)
    assert failures == [] and any("`a`: check fails" in w for w in warnings)


def test_evaluate_fails_on_gpu_errors_and_low_pass_rate() -> None:
    now = result(prompt("a", "fail", "x"), prompt("b", "fail", "y"), prompt("c", "pass", "z"))
    now["gpu_errors"] = ["amdgpu: ring gfx_0.0.0 timeout"]
    failures, _ = evaluate(now, None)
    assert any("GPU error" in f for f in failures)
    assert any("only 1/3 checks pass" in f for f in failures)


def test_evaluate_warns_when_the_kernel_log_is_incomplete_or_unreadable() -> None:
    now = result(prompt("a", "pass", "x"))
    now["kernel_log"] = "incomplete"
    assert any("wrapped or was cleared" in w for w in evaluate(now, None)[1])
    now["kernel_log"] = "unreadable"
    assert any("could not be read" in w for w in evaluate(now, None)[1])


def test_run_reports_gpu_errors_logged_during_the_run(
    fake: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs = iter([
        ["[    1.000000] boot"],
        ["[    1.000000] boot", "[   99.000000] amdgpu: ring gfx_0.0.0 timeout"],
    ])  # fmt: skip
    monkeypatch.setattr(run_model_e2e, "read_kernel_log", lambda: next(logs))
    code, result = run(fake, tmp_path / "a")
    assert code == 1
    assert result["kernel_log"] == "checked"
    assert result["gpu_errors"] == ["[   99.000000] amdgpu: ring gfx_0.0.0 timeout"]


def test_kernel_log_new_lines_survive_ring_buffer_rollover() -> None:
    before = [f"[   {i}.000000] old {i}" for i in range(1, 6)]
    # Two old lines rolled out while two new ones, one a GPU fault, arrived:
    # the line count is unchanged, but the new lines are still found.
    after = before[2:] + ["[   10.000000] new", "[   11.000000] amdgpu: ring gfx timeout"]
    assert kernel_lines_since(before, after) == (
        ["[   10.000000] new", "[   11.000000] amdgpu: ring gfx timeout"],
        True,
    )


def test_kernel_log_reports_lost_continuity() -> None:
    before = ["[    1.000000] a", "[    2.000000] b"]
    # Nothing from before is left: the buffer wrapped or was cleared.
    after = ["[    7.000000] x", "[    8.000000] y"]
    assert kernel_lines_since(before, after) == (after, False)
    assert kernel_lines_since(before, []) == ([], False)


def test_kernel_log_same_timestamp_and_continuation_lines() -> None:
    before = ["[    1.000000] a", "[    2.000000] b", "  b continued"]
    after = [*before, "[    2.000000] c", "  c continued", "[    3.000000] d"]
    assert kernel_lines_since(before, after) == (
        ["[    2.000000] c", "  c continued", "[    3.000000] d"],
        True,
    )


def test_kernel_log_without_timestamps_cannot_be_compared() -> None:
    assert kernel_lines_since(["a", "b"], ["a", "b", "c"]) is None
    assert kernel_lines_since([], ["[    1.000000] a"]) == (["[    1.000000] a"], True)


def gpu_wait(procs: Path) -> subprocess.CompletedProcess:
    """Run gpu_wait.sh with no wait against a fake KFD process list."""
    return subprocess.run(
        ["bash", str(HERE / "gpu_wait.sh"), "0"],
        env={**os.environ, "KFD_PROC_DIR": str(procs)}, capture_output=True, text=True, timeout=30,
    )  # fmt: skip


def test_gpu_wait_reads_the_kernel_process_list(tmp_path: Path) -> None:
    procs = tmp_path / "procs"
    procs.mkdir()
    assert gpu_wait(procs).stdout.splitlines()[-1] == "state=free"
    # Any user's process counts, even one that is still being torn down.
    (procs / "999999999").mkdir()
    assert gpu_wait(procs).stdout.splitlines()[-1] == "state=busy"


def test_gpu_wait_fails_when_it_cannot_see_every_user(tmp_path: Path) -> None:
    assert gpu_wait(tmp_path / "missing").returncode == 1
    procs = tmp_path / "procs"
    (procs / "123").mkdir(parents=True)
    procs.chmod(0)
    try:
        if os.access(procs, os.R_OK):
            pytest.skip("running as root: permissions are not enforced")
        assert gpu_wait(procs).returncode == 1
    finally:
        procs.chmod(0o755)


@pytest.mark.parametrize(
    ("paths", "models"),
    [
        (["server/src/deepseek4/ds4_graph.cpp"], ["ds4"]),
        (["server/src/qwen35/qwen35_target_graph.cpp"], ["qwen"]),
        (["server/src/draft/dflash_draft.cpp", "docs/x.md"], ["qwen"]),
        (["server/src/server/http_server.cpp"], ["ds4", "qwen"]),
        (["server/deps/llama.cpp/ggml/src/ggml.c"], ["ds4", "qwen"]),
        (["server/src/laguna/laguna.cpp", "README.md", "server/scripts/x.py"], []),
        ([".github/ci/e2e/prompts.json"], ["ds4", "qwen"]),
    ],
)
def test_select_models(paths: list[str], models: list[str]) -> None:
    assert models_for(paths) == models


def test_matrix_carries_each_models_gpu_and_files() -> None:
    entries = {e["model"]: e for e in matrix(["ds4", "qwen"])["include"]}
    assert (entries["qwen"]["arch"], entries["qwen"]["hip_index"]) == ("gfx1201", 0)
    assert (entries["ds4"]["arch"], entries["ds4"]["hip_index"]) == ("gfx1151", 1)
    assert entries["qwen"]["draft"] and not entries["ds4"]["draft"]
    assert "--ds4-prefill sparse" in entries["ds4"]["server_args"]


REPO = "Luce-Org/lucebox"


def fake_api(
    artifacts: list[dict], runs: dict[int, dict], diffs: dict[str, list[str]] | None = None
):
    def get(path: str) -> dict:
        if "/actions/artifacts?" in path:
            name = path.split("name=")[1].split("&")[0]
            return {"artifacts": [a for a in artifacts if a["name"] == name]}
        if "/compare/" in path:
            base = path.rsplit("/", 1)[1].split("...")[0]
            return {"files": [{"filename": f} for f in (diffs or {})[base]]}
        run_id = int(path.rsplit("/", 1)[1])
        return {**runs[run_id], "id": run_id}

    return get


def main_run(event: str = "push", sha: str = "base", **overrides: object) -> dict:
    return {
        "event": event,
        "head_sha": sha,
        "head_branch": "main",
        "head_repository": {"full_name": REPO},
        "path": ".github/workflows/model-e2e.yml",
        **overrides,
    }


def artifact(run_id: int, created: str, name: str = "model-e2e-baseline-qwen-r9700") -> dict:
    return {"name": name, "expired": False, "created_at": created, "workflow_run": {"id": run_id}}


def test_baseline_is_the_newest_from_a_trusted_main_run() -> None:
    artifacts = [
        artifact(1, "2026-09-01T03:00:00Z"),
        artifact(2, "2026-09-03T03:00:00Z"),
        # Newer, but uploaded by pull request code or from a fork's main.
        artifact(3, "2026-09-04T03:00:00Z"),
        artifact(4, "2026-09-05T03:00:00Z"),
        artifact(5, "2026-09-06T03:00:00Z"),
        {**artifact(6, "2026-09-07T03:00:00Z"), "expired": True},
    ]
    runs = {
        1: main_run(),
        2: main_run("workflow_dispatch"),
        3: main_run("pull_request", head_branch="feature"),
        4: main_run(head_repository={"full_name": "someone/lucebox"}),
        5: main_run(path=".github/workflows/ci.yml"),
        6: main_run(),
    }
    got = add_baselines(matrix(["ds4", "qwen"]), fake_api(artifacts, runs), REPO)
    assert {e["model"]: e["baseline_run"] for e in got["include"]} == {"ds4": "", "qwen": "2"}


@pytest.mark.parametrize(
    ("changed", "kept"),
    [
        (["server/src/qwen35/qwen35_target_graph.cpp"], ["ds4", "qwen"]),
        (["docs/x.md"], ["ds4"]),
        ([f"docs/{i}.md" for i in range(300)], ["ds4", "qwen"]),
    ],
)
def test_merges_rerun_models_changed_since_their_baseline(
    changed: list[str], kept: list[str]
) -> None:
    # qwen has a baseline at commit "base"; ds4 has none yet, so it always runs.
    artifacts = [artifact(1, "2026-09-01T03:00:00Z")]
    api = fake_api(artifacts, {1: main_run()}, {"base": changed})
    got = keep_changed(add_baselines(matrix(["ds4", "qwen"]), api, REPO), api, REPO, "head")
    assert [e["model"] for e in got["include"]] == kept


def test_evaluate_warns_when_the_baseline_ran_on_another_rocm() -> None:
    base, now = result(prompt("a", "pass", "x")), result(prompt("a", "pass", "x"))
    base["rocm"], now["rocm"] = "7.1.0", "7.2.0"
    assert any("ROCm 7.1.0" in w for w in evaluate(now, base)[1])
    now["rocm"] = "7.1.0"
    assert evaluate(now, base) == ([], [])
