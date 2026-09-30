"""Gateway end to end: real HTTP gateway in front of mock luce_server backends."""

import json

import httpx
import pytest
from mocks import make_config, serve
from starlette.responses import JSONResponse

from lucerouter.gateway import create_app
from lucerouter.routers import build

S, M, L = "qwen35-0.8b", "qwen35-2b", "qwen38-27b"
SPEC = "brick:service=brick-max"


@pytest.fixture
def gw(mock, tmp_path):
    cfg = make_config(mock.base)
    log = tmp_path / "decisions.jsonl"
    app = create_app(cfg, build(SPEC, cfg), str(log))
    with serve(app) as url:
        with httpx.Client(base_url=url, timeout=10) as client:
            client.log = log
            client.mock = mock
            yield client


def log_lines(client):
    return [json.loads(line) for line in client.log.read_text().splitlines()]


def test_routed_non_streaming(gw):
    body = {"model": "route", "messages": [{"role": "user", "content": "HARDQ prove it"}],
            "chat_template_kwargs": {"enable_thinking": True}, "route_extra": 1}
    r = gw.post("/v1/chat/completions", json=body, headers={"x-session-id": "s1"})
    assert r.status_code == 200
    assert r.headers["x-lucerouter-model"] == L and r.headers["x-lucerouter-router"] == SPEC
    data = r.json()
    assert data["choices"][0]["message"]["content"] == f"hello {L}"
    assert data["lucerouter"]["model"] == L and data["lucerouter"]["scores"]["hard"] > 0.5

    model, sent = gw.mock.requests[-1]
    assert model == L
    assert sent["model"] == "qwen3.6-27b"  # served_name rewrite
    assert sent["chat_template_kwargs"] == {"enable_thinking": True} and sent["route_extra"] == 1

    [rec] = log_lines(gw)
    assert rec["routed"] and rec["decision"]["model"] == L and rec["session_id"] == "s1"
    assert rec["usage"]["total_tokens"] == 10 and rec["status"] == 200
    assert rec["router_latency_ms"] > 0 and rec["backend_ttft_ms"] == rec["backend_total_ms"] > 0


def test_streaming_passthrough(gw):
    body = {"messages": [{"role": "user", "content": "EASYQ hi"}], "stream": True}
    with gw.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200 and r.headers["x-lucerouter-model"] == S
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(r.iter_raw())
    events = [e for e in raw.decode().split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    text = "".join(json.loads(e[5:])["choices"][0]["delta"]["content"]
                   for e in events[:-1] if json.loads(e[5:])["choices"])
    assert text == f"hello {S}"
    [rec] = log_lines(gw)
    assert rec["stream"] and rec["decision"]["model"] == S
    assert rec["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert 0 < rec["backend_ttft_ms"] <= rec["backend_total_ms"]


def test_named_model_passthrough_and_unknown(gw):
    body = {"model": M, "messages": [{"role": "user", "content": "HARDQ"}]}
    r = gw.post("/v1/chat/completions", json=body)
    assert r.headers["x-lucerouter-model"] == M and r.headers["x-lucerouter-router"] == "passthrough"
    assert gw.mock.brick_requests == []
    r = gw.post("/v1/chat/completions", json={"model": "qwen3.6-27b", "messages": []})  # served_name alias
    assert r.headers["x-lucerouter-model"] == L
    assert gw.post("/v1/chat/completions", json={"model": "gpt-9", "messages": []}).status_code == 404


def test_arm_sets_thinking_only_on_routed_requests(mock, tmp_path):
    """Both 27B arms hit one backend; the arm, not the client, decides thinking on routed requests."""
    cfg = make_config(mock.base, arms=True)
    log = tmp_path / "d.jsonl"
    client_kwargs = {"enable_thinking": False, "other": 1}
    with serve(create_app(cfg, build(SPEC, cfg), str(log))) as url, httpx.Client(base_url=url) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "HARDQ"}],
                                                  "chat_template_kwargs": client_kwargs})
        assert r.headers["x-lucerouter-model"] == "qwen38-27b-think" and r.headers["x-lucerouter-backend"] == L
        c.post("/v1/chat/completions", json={"model": L, "messages": [{"role": "user", "content": "HARDQ"}],
                                             "chat_template_kwargs": client_kwargs})
        c.post("/v1/chat/completions", json={"model": "qwen38-27b-think", "messages": []})
    sent = [(m, b.get("chat_template_kwargs")) for m, b in mock.requests]
    assert sent == [(L, {"enable_thinking": True, "other": 1}),   # routed: arm overrides the client
                    (L, client_kwargs),                              # passthrough: untouched
                    (L, {"enable_thinking": True})]                  # arm named: its overrides
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["arm"], r["backend"], r["routed"]) for r in recs] == [
        ("qwen38-27b-think", L, True), (L, L, False), ("qwen38-27b-think", L, False)]


def test_models_health_stats(gw):
    ids = [m["id"] for m in gw.get("/v1/models").json()["data"]]
    assert ids == ["route", S, M, L]
    assert gw.get("/health").json() == {"status": "ok", "router": SPEC}
    gw.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "EASYQ"}]})
    gw.post("/v1/chat/completions", json={"model": L, "messages": [{"role": "user", "content": "x"}]})
    stats = gw.get("/router/stats").json()
    assert stats["requests"] == 2 and stats["routed"] == 1 and stats["by_arm"] == {S: 1, L: 1}
    assert stats["by_backend"] == {S: 1, L: 1}


def test_backend_down_is_502_and_logged(mock, tmp_path):
    cfg = make_config(mock.base, {L: {"base_url": "http://127.0.0.1:1/v1"}})
    log = tmp_path / "d.jsonl"
    with serve(create_app(cfg, build(f"fixed:model={L}", cfg), str(log))) as url:
        r = httpx.post(f"{url}/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 502 and r.headers["x-lucerouter-model"] == L
    rec = json.loads(log.read_text())
    assert rec["status"] == 502 and "ConnectError" in rec["error"]


def test_escalation_target_hook(mock, tmp_path):
    """A router may return a non-backend target (future swarm); the gateway hands it to extra_targets."""
    cfg = make_config(mock.base)

    async def swarm(body, decision, request):
        return JSONResponse({"swarm": True, "n": len(body["messages"])})

    router = build("fixed:model=swarm", cfg)
    with serve(create_app(cfg, router, None, extra_targets={"swarm": swarm})) as url:
        r = httpx.post(f"{url}/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        assert r.json() == {"swarm": True, "n": 1}
    with serve(create_app(cfg, build("fixed:model=swarm", cfg), None)) as url:
        assert httpx.post(f"{url}/v1/chat/completions", json={"messages": []}).status_code == 502
