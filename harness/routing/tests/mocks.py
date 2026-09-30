"""Test support: mock luce_server-style backends on a real local HTTP port (no GPU, no model servers).

One mock server hosts every fake model under a path prefix, so a model's
base_url is http://127.0.0.1:<port>/<name>/v1 and its root serves /status/json.
"""

import asyncio
import contextlib
import json
import math
import socket
import threading
import time

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from lucerouter.config import config_from_dict


@contextlib.contextmanager
def serve(app):
    """Run an ASGI app with uvicorn in a thread; yield its base URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("mock server did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# Brick answers keyed on a marker word in the query.
BRICK_DISTS = {
    "EASYQ": {"easy": 0.9, "medium": 0.08, "hard": 0.02},
    "MIDQ": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    "HARDQ": {"easy": 0.05, "medium": 0.15, "hard": 0.8},
}


class MockBackends:
    def __init__(self):
        self.load = {}          # model -> {"in_flight", "capacity"} served by /status/json
        self.requests = []      # (model, body) received on chat/completions
        self.brick_requests = []

    def app(self):
        async def chat(request: Request):
            model = request.path_params["m"]
            body = await request.json()
            self.requests.append((model, body))
            usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            if body.get("stream"):
                async def gen():
                    for piece in ("hel", "lo ", model):
                        chunk = {"choices": [{"delta": {"content": piece}}]}
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                        await asyncio.sleep(0.01)
                    yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                return StreamingResponse(gen(), media_type="text/event-stream")
            return JSONResponse({"id": "cmpl-1", "model": body.get("model"),
                                 "choices": [{"message": {"role": "assistant", "content": f"hello {model}"}}],
                                 "usage": usage})

        async def status(request: Request):
            s = self.load.get(request.path_params["m"])
            if s is None:
                return JSONResponse({"error": "down"}, status_code=500)
            # multi-model luce_server shape
            return JSONResponse({"routing": "primary-first", "waiting": 0,
                                 "models": [{"id": request.path_params["m"], **s}]})

        async def brick(request: Request):
            body = await request.json()
            self.brick_requests.append(body)
            text = body["messages"][-1]["content"]
            dist = next((d for k, d in BRICK_DISTS.items() if k in text), BRICK_DISTS["MIDQ"])
            tops = [{"token": tok, "logprob": math.log(dist[label])}
                    for tok, label in (("easy", "easy"), (" medium", "medium"), ("Hard", "hard"))]
            tops.append({"token": "The", "logprob": math.log(0.001)})
            return JSONResponse({"choices": [{"message": {"content": "easy"},
                                              "logprobs": {"content": [{"token": tops[0]["token"],
                                                                        "logprob": tops[0]["logprob"],
                                                                        "top_logprobs": tops}]}}]})

        return Starlette(routes=[
            Route("/brick/v1/chat/completions", brick, methods=["POST"]),
            Route("/{m}/v1/chat/completions", chat, methods=["POST"]),
            Route("/{m}/status/json", status, methods=["GET"]),
        ])


ARMS = {  # the production arm layout: 2b off, 27b off, 27b on (same backend)
    "qwen35-2b": {"model": "qwen35-2b", "thinking": False, "rank": 0},
    "qwen38-27b": {"model": "qwen38-27b", "thinking": False, "rank": 1},
    "qwen38-27b-think": {"model": "qwen38-27b", "thinking": True, "rank": 2},
}


def make_config(base: str, overrides: dict | None = None, arms: bool = False, extra: dict | None = None):
    models = {
        "qwen35-0.8b": {"base_url": f"{base}/qwen35-0.8b/v1", "kind": "luce_server", "rank": 0},
        "qwen35-2b": {"base_url": f"{base}/qwen35-2b/v1", "kind": "luce_server", "rank": 1},
        "qwen38-27b": {"base_url": f"{base}/qwen38-27b/v1", "kind": "luce_server", "rank": 2,
                       "served_name": "qwen3.6-27b"},
    }
    for name, extra in (overrides or {}).items():
        models[name].update(extra)
    return config_from_dict({"models": models, "services": {
        "brick-max": {"base_url": f"{base}/brick", "kind": "luce_server"},
        "embed": {"base_url": f"{base}/embed", "kind": "luce_server"},
    }, **({"arms": ARMS} if arms else {}), **(extra or {})})


def run(coro):
    return asyncio.run(coro)
