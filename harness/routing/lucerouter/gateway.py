"""OpenAI-compatible routing gateway.

    python -m lucerouter.gateway --config config/backends.json \
        --router "load_aware>brick:service=brick-max" --port 8400 --log runs/gateway/decisions.jsonl

Requests with model "route"/"auto"/absent are routed; a request naming a
configured model (or its served_name) is passed through. Every request appends
one JSON line to the decision log.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .config import Config, load_config
from .routers import Router, build
from .types import RouteDecision, RouteRequest

ROUTE_ALIASES = {None, "", "route", "auto"}
# Request fields the gateway consumes; everything else (incl. chat_template_kwargs) passes through.
GATEWAY_FIELDS = ("route",)
HOP_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding", "keep-alive",
               "date", "server"}


class DecisionLog:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict):
        if not self.path:
            return
        line = json.dumps(record, default=str)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")


class Stats:
    def __init__(self):
        self.started = time.time()
        self.requests = 0
        self.routed = 0
        self.errors = 0
        self.by_model: Counter = Counter()
        self.router_ms_total = 0.0

    def to_dict(self, router_label: str) -> dict:
        return {"router": router_label, "uptime_s": round(time.time() - self.started, 1),
                "requests": self.requests, "routed": self.routed, "errors": self.errors,
                "by_model": dict(self.by_model),
                "router_ms_avg": round(self.router_ms_total / self.routed, 3) if self.routed else None}


def _usage_from_sse(buffer: bytes) -> dict | None:
    """Last `usage` object found in an SSE stream (backends send it in the final chunk)."""
    usage = None
    for line in buffer.split(b"\n"):
        if b'"usage"' not in line or not line.startswith(b"data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("usage"):
            usage = obj["usage"]
    return usage


def create_app(config: Config, router: Router, log_path: str | None = None,
               extra_targets: dict | None = None, backend_timeout: float | None = 600.0) -> Starlette:
    """`extra_targets` maps non-backend decision targets (e.g. "swarm") to
    `async (body, decision, request) -> starlette Response` handlers."""
    log = DecisionLog(log_path)
    stats = Stats()
    extra_targets = extra_targets or {}
    served = {m.request_model: m.name for m in config.models.values()}
    client = httpx.AsyncClient(timeout=httpx.Timeout(backend_timeout, connect=5.0))

    def headers_for(decision: RouteDecision, request_id: str) -> dict:
        return {"x-lucerouter-model": decision.model, "x-lucerouter-router": decision.router,
                "x-lucerouter-request-id": request_id}

    def record(request_id, decision, routed, stream, session_id, **extra):
        stats.requests += 1
        stats.by_model[decision.model] += 1
        if routed:
            stats.routed += 1
            stats.router_ms_total += decision.latency_ms
        if extra.get("error"):
            stats.errors += 1
        log.write({"ts": datetime.now(timezone.utc).isoformat(), "request_id": request_id,
                   "session_id": session_id, "routed": routed, "stream": stream,
                   "decision": decision.to_dict(), "router_latency_ms": round(decision.latency_ms, 3),
                   **extra})

    async def chat_completions(request: Request):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
        req = RouteRequest.from_openai(body, dict(request.headers))
        asked = body.get("model")
        routed = asked in ROUTE_ALIASES
        if routed:
            decision = await router.route(req)
        elif asked in config.models or asked in served:
            name = asked if asked in config.models else served[asked]
            decision = RouteDecision(model=name, router="passthrough", reason="model named in request")
        else:
            return JSONResponse({"error": {"message": f"unknown model {asked!r}; use 'route' or one of "
                                                      f"{sorted(config.models)}"}}, status_code=404)
        stream = bool(body.get("stream"))

        if decision.model not in config.models:
            handler = extra_targets.get(decision.model)
            if handler is None:
                record(request_id, decision, routed, stream, req.session_id, status=502,
                       error=f"no handler for target {decision.model!r}")
                return JSONResponse({"error": {"message": f"route target {decision.model!r} unavailable"}},
                                    status_code=502, headers=headers_for(decision, request_id))
            record(request_id, decision, routed, stream, req.session_id, status=None, handler=decision.model)
            return await handler(body, decision, request)

        backend = config.models[decision.model]
        fwd = {k: v for k, v in body.items() if k not in GATEWAY_FIELDS}
        fwd["model"] = backend.request_model
        url = f"{backend.base_url.rstrip('/')}/chat/completions"
        hdrs = headers_for(decision, request_id)
        auth = request.headers.get("authorization")
        out_headers = {"authorization": auth} if auth else {}
        t0 = time.perf_counter()

        try:
            upstream = await client.send(client.build_request("POST", url, json=fwd, headers=out_headers),
                                         stream=True)
        except httpx.HTTPError as e:
            record(request_id, decision, routed, stream, req.session_id, status=502,
                   backend_total_ms=round((time.perf_counter() - t0) * 1000, 2), error=f"{type(e).__name__}: {e}")
            return JSONResponse({"error": {"message": f"backend {decision.model} unreachable: {e}"}},
                                status_code=502, headers=hdrs)

        passthrough = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS}

        if stream and upstream.status_code == 200:
            async def relay():
                ttft = None
                tail = b""
                error = None
                try:
                    async for chunk in upstream.aiter_raw():
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        tail = (tail + chunk)[-65536:]  # usage arrives in the last chunks
                        yield chunk
                except httpx.HTTPError as e:
                    error = f"{type(e).__name__}: {e}"
                finally:
                    await upstream.aclose()
                    record(request_id, decision, routed, True, req.session_id, status=200,
                           backend_ttft_ms=round(ttft, 2) if ttft is not None else None,
                           backend_total_ms=round((time.perf_counter() - t0) * 1000, 2),
                           usage=_usage_from_sse(tail), error=error)

            return StreamingResponse(relay(), status_code=200, headers={**passthrough, **hdrs},
                                     media_type=upstream.headers.get("content-type", "text/event-stream"))

        raw = await upstream.aread()
        await upstream.aclose()
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        usage = None
        content = raw
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                usage = payload.get("usage")
                if upstream.status_code == 200:
                    payload["lucerouter"] = decision.to_dict()
                content = json.dumps(payload).encode()
        except ValueError:
            pass
        record(request_id, decision, routed, stream, req.session_id, status=upstream.status_code,
               backend_ttft_ms=total_ms, backend_total_ms=total_ms, usage=usage,
               error=None if upstream.status_code < 400 else raw[:500].decode(errors="replace"))
        return Response(content, status_code=upstream.status_code, headers={**passthrough, **hdrs},
                        media_type=upstream.headers.get("content-type", "application/json"))

    async def models(request: Request):
        now = int(time.time())
        data = [{"id": "route", "object": "model", "created": now, "owned_by": "lucerouter"}]
        data += [{"id": m.name, "object": "model", "created": now, "owned_by": m.kind}
                 for m in config.by_rank()]
        return JSONResponse({"object": "list", "data": data})

    async def health(request: Request):
        return JSONResponse({"status": "ok", "router": router.label})

    async def router_stats(request: Request):
        return JSONResponse(stats.to_dict(router.label))

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        await client.aclose()
        await router.aclose()

    app = Starlette(routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/router/stats", router_stats, methods=["GET"]),
    ], lifespan=lifespan)
    app.state.stats = stats
    app.state.router = router
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="lucerouter OpenAI-compatible routing gateway")
    ap.add_argument("--config", required=True, help="backends.json")
    ap.add_argument("--router", required=True, help='router spec, e.g. "load_aware>brick:service=brick-max"')
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--log", default="runs/gateway/decisions.jsonl", help="decision log (JSONL)")
    args = ap.parse_args(argv)

    import uvicorn

    config = load_config(args.config)
    router = build(args.router, config)
    app = create_app(config, router, args.log)
    print(f"lucerouter: router={router.label} models={[m.name for m in config.by_rank()]} log={args.log}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
