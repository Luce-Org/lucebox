"""Shared pytest fixtures for the lucebox test suite.

Test tiers (see ``[tool.pytest.ini_options].markers`` in pyproject.toml):

- unmarked unit tests run anywhere with plain ``pytest``
- ``server`` tests need a live luce_server — point pytest at one with
  ``--base-url http://host:port`` (or ``LUCE_TEST_SERVER_URL``) or let it
  spawn one via ``--launch path/to/model.gguf``
- ``model`` tests need local GGUF files and/or built binaries
- ``slow`` marks long-running cases

``spawn_luce_server`` is a factory fixture for tests that must run against a
server started with bespoke flags (e.g. the prefix/prefill cache tests); it
picks a free port and waits for readiness. Callers stop their server with
``ServerHandle.stop()`` when their module is done; anything still running at
session end is stopped as a backstop.
"""

import json
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pytest

ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER_BIN = ROOT / "server" / "build" / "luce_server"
MAX_CONCURRENCY_RANGE = range(1, 65)


def pytest_addoption(parser):
    group = parser.getgroup("luce", "lucebox server tests")
    group.addoption(
        "--base-url",
        default=None,
        help="Base URL of a running luce_server (env: LUCE_TEST_SERVER_URL).",
    )
    group.addoption(
        "--launch",
        metavar="MODEL_GGUF",
        default=None,
        help="Spawn luce_server on this model for the test session "
        "(env: LUCE_TEST_MODEL).",
    )
    group.addoption(
        "--server-bin",
        default=None,
        help="luce_server binary used by spawned servers and CLI tests "
        "(env: LUCE_SERVER_BIN, default: server/build/luce_server).",
    )
    group.addoption(
        "--server-extra-args",
        default="",
        help="Extra CLI flags appended to a --launch'ed server "
        '(e.g. --server-extra-args="--max-ctx 8192"; '
        "env: LUCE_SERVER_EXTRA_ARGS).",
    )
    group.addoption(
        "--max-concurrency",
        type=int,
        default=None,
        help="N concurrent slots of the server under test, in [1, 64]. "
        "A --launch'ed server is started with --paged-attention "
        "--max-concurrency N. The parallel-serving tests skip without it "
        "(env: LUCE_MAX_CONCURRENCY).",
    )


def _max_concurrency(config) -> int | None:
    n = (config.getoption("--max-concurrency")
         or os.environ.get("LUCE_MAX_CONCURRENCY"))
    if n is None or n == "":
        return None
    try:
        n = int(n)
    except ValueError:
        raise pytest.UsageError(
            f"--max-concurrency must be an integer, got {n!r}") from None
    if n not in MAX_CONCURRENCY_RANGE:
        raise pytest.UsageError(
            f"--max-concurrency must be in [1, 64], got {n}")
    return n


def pytest_configure(config):
    _max_concurrency(config)  # fail fast on a bad value


# ─── HTTP client ─────────────────────────────────────────────────────────


class HttpResponse:
    """Status, headers and body of a response, whatever its status code."""

    def __init__(self, status_code: int, headers, fp):
        self.status_code = status_code
        self.headers = headers
        self._fp = fp
        self._body: bytes | None = None

    @property
    def content(self) -> bytes:
        if self._body is None:
            self._body = self._fp.read()
        return self._body

    @property
    def text(self) -> str:
        return self.content.decode(errors="replace")

    def json(self):
        return json.loads(self.content)

    def iter_lines(self):
        """Yield decoded lines as they arrive (for SSE streams)."""
        for raw in self._fp:
            yield raw.decode().rstrip("\r\n")


class LuceHttpClient:
    """Minimal urllib-based HTTP client for the luce_server OpenAI API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _build(self, method: str, path: str, body, data, headers):
        hdrs = dict(headers or {})
        if body is not None:
            data = json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        elif isinstance(data, str):
            data = data.encode()
        return urllib.request.Request(self.base_url + path, data=data,
                                      headers=hdrs, method=method)

    def send(self, method: str, path: str, body: dict | None = None, *,
             data: bytes | str | None = None, headers: dict | None = None,
             timeout: float = 120.0) -> HttpResponse:
        """Send a request and return the response without raising on 4xx/5xx.
        ``body`` is JSON-encoded; ``data`` is sent as-is."""
        req = self._build(method, path, body, data, headers)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            return HttpResponse(e.code, e.headers, e)
        return HttpResponse(resp.status, resp.headers, resp)

    def request(self, method: str, path: str, body: dict | None = None,
                stream: bool = False, timeout: float = 120.0):
        """Send a request; raise ``urllib.error.HTTPError`` on 4xx/5xx.
        Returns the parsed JSON body, or the open response when ``stream``."""
        req = self._build(method, path, body, None, None)
        resp = urllib.request.urlopen(req, timeout=timeout)
        if stream:
            return resp
        return json.loads(resp.read().decode())

    def get(self, path: str, timeout: float = 30.0):
        return self.request("GET", path, timeout=timeout)

    def post(self, path: str, body: dict, timeout: float = 120.0):
        return self.request("POST", path, body=body, timeout=timeout)

    def post_stream(self, path: str, body: dict, timeout: float = 120.0):
        return self.request("POST", path, body=body, stream=True,
                            timeout=timeout)


# ─── Server processes ────────────────────────────────────────────────────


def wait_for_server(base_url: str, proc: subprocess.Popen | None = None,
                    timeout: float = 240.0) -> bool:
    """Poll /v1/models until the server answers or the deadline passes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"{base_url}/v1/models", timeout=1).read()
            return True
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            time.sleep(1)
    return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class ServerHandle:
    base_url: str
    proc: subprocess.Popen | None = None
    log_path: Path | None = None
    log_file: IO | None = field(default=None, repr=False)

    @property
    def client(self) -> LuceHttpClient:
        return LuceHttpClient(self.base_url)

    def read_log(self) -> str:
        """The spawned server's log, or "" for an external server."""
        if self.log_path and self.log_path.exists():
            return self.log_path.read_text(errors="replace")
        return ""

    def stop(self) -> None:
        """SIGINT the spawned server (kill after 10 s) and close its log.
        No-op for an external server; safe to call twice."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.close()


def resolve_server_bin(config) -> Path:
    return Path(config.getoption("--server-bin")
                or os.environ.get("LUCE_SERVER_BIN")
                or DEFAULT_SERVER_BIN)


@pytest.fixture(scope="session")
def server_bin(request) -> Path:
    """The luce_server binary (--server-bin / LUCE_SERVER_BIN); skips the
    requesting test when it has not been built."""
    path = resolve_server_bin(request.config)
    if not path.is_file():
        pytest.skip(f"luce_server binary not found at {path} "
                    "(set --server-bin or LUCE_SERVER_BIN)")
    return path


@pytest.fixture(scope="session")
def spawn_luce_server(tmp_path_factory):
    """Factory: launch luce_server with custom argv; auto-selects the port
    and waits for /v1/models. Callers ``stop()`` the returned handle; any
    server still running at session end is stopped here.

    ``argv`` must NOT contain --port (the fixture allocates a free one).
    """
    log_dir = tmp_path_factory.mktemp("luce-servers")
    handles: list[ServerHandle] = []

    def _spawn(argv: list[str], *, name: str = "luce_server",
               timeout: float = 240.0) -> ServerHandle:
        port = _free_port()
        log_path = log_dir / f"{name}-{port}.log"
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            [*argv, "--port", str(port)],
            stdout=log_file, stderr=subprocess.STDOUT)
        handle = ServerHandle(base_url=f"http://127.0.0.1:{port}", proc=proc,
                              log_path=log_path, log_file=log_file)
        handles.append(handle)
        if not wait_for_server(handle.base_url, proc=proc, timeout=timeout):
            handle.stop()
            tail = handle.read_log()[-4000:]
            pytest.fail(f"luce_server '{name}' did not become ready within "
                        f"{timeout:.0f}s; log tail:\n{tail}")
        return handle

    yield _spawn

    for handle in handles:
        handle.stop()


@pytest.fixture(scope="session")
def server_handle(request, spawn_luce_server) -> ServerHandle:
    """The luce_server under test: --base-url/LUCE_TEST_SERVER_URL for an
    external server, or --launch MODEL_GGUF to spawn one for the session."""
    config = request.config
    url = (config.getoption("--base-url")
           or os.environ.get("LUCE_TEST_SERVER_URL"))
    if url:
        return ServerHandle(base_url=url.rstrip("/"))
    model = config.getoption("--launch") or os.environ.get("LUCE_TEST_MODEL")
    if not model:
        pytest.skip("no luce_server configured — pass --base-url URL or "
                    "--launch MODEL_GGUF")
    server_bin = resolve_server_bin(config)
    if not server_bin.is_file():
        pytest.skip(f"luce_server binary not found at {server_bin} "
                    "(set --server-bin or LUCE_SERVER_BIN)")
    if not Path(model).is_file():
        pytest.skip(f"model not found at {model}")
    argv = [str(server_bin), str(model), "--max-ctx", "4096",
            "--max-tokens", "64"]
    n = _max_concurrency(config)
    if n is not None:
        argv += ["--paged-attention", "--max-concurrency", str(n)]
    argv += shlex.split(config.getoption("--server-extra-args")
                        or os.environ.get("LUCE_SERVER_EXTRA_ARGS") or "")
    return spawn_luce_server(argv, name="launched")


@pytest.fixture(scope="session")
def max_concurrency(request) -> int:
    """N slots of the server under test; skips when it was not given."""
    n = _max_concurrency(request.config)
    if n is None:
        pytest.skip("parallel-serving tests need --max-concurrency N "
                    "(or LUCE_MAX_CONCURRENCY) matching the server under test")
    return n
