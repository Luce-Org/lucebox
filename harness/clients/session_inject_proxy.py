#!/usr/bin/env python3
"""Thin proxy that injects extra_body.session_id into /v1/messages requests.

Run between the claude CLI and the dflash server when PFLASH_SESSION_ID is set.
All other paths and methods are forwarded verbatim.

Usage:
    python3 session_inject_proxy.py \\
        --host 127.0.0.1 --port 18081 \\
        --upstream http://127.0.0.1:18080 \\
        --session-id <id>

The proxy listens on --port and forwards to --upstream, injecting
extra_body.session_id on every POST /v1/messages request.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class Handler(BaseHTTPRequestHandler):
    upstream: str = ""
    session_id: str = ""
    force_temperature: float | None = None
    dump_path: str = ""
    think_budget: int = 0

    def log_message(self, fmt, *args):
        print("[session-proxy] %s" % (fmt % args), flush=True)

    def _upstream_conn(self) -> tuple[http.client.HTTPConnection, str]:
        url = urlparse(self.upstream)
        port = url.port or (443 if url.scheme == "https" else 80)
        cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        return cls(url.hostname, port, timeout=900), url.path.rstrip("/")

    def _forward_raw(self, body: bytes):
        """Forward request verbatim (no injection needed)."""
        conn, base = self._upstream_conn()
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        headers["Content-Length"] = str(len(body))
        conn.request(self.command, base + self.path, body, headers)
        resp = conn.getresponse()
        self._relay_response(resp)

    def _relay_response(self, resp: http.client.HTTPResponse):
        """Relay upstream response back to client, handling SSE streaming."""
        content_type = resp.getheader("Content-Type", "")
        is_sse = "text/event-stream" in content_type

        self.send_response(resp.status)
        skip_headers = {"transfer-encoding", "content-length"}
        for k, v in resp.getheaders():
            if k.lower() not in skip_headers:
                self.send_header(k, v)

        if is_sse:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # Stream chunk by chunk
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    # Write terminal chunk
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    break
                size = f"{len(chunk):X}\r\n"
                self.wfile.write(size.encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        else:
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return b""
        return self.rfile.read(n)

    def do_GET(self):
        conn, base = self._upstream_conn()
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        conn.request("GET", base + self.path, None, headers)
        resp = conn.getresponse()
        self._relay_response(resp)

    def do_POST(self):
        body = self._read_body()
        path = self.path

        if self.dump_path and path.startswith("/v1/"):
            try:
                with open(self.dump_path, "a") as fh:
                    fh.write(f"\n===== POST {path} ({len(body)} bytes) =====\n")
                    fh.write(body.decode("utf-8", "replace"))
                    fh.write("\n")
            except Exception:
                pass

        is_api_path = path.startswith("/v1/messages") or path.startswith("/v1/chat/completions")
        if is_api_path and (self.session_id or self.force_temperature is not None or self.think_budget):
            try:
                obj = json.loads(body.decode("utf-8"))
                if self.session_id:
                    if "extra_body" not in obj:
                        obj["extra_body"] = {}
                    if "session_id" not in obj["extra_body"]:
                        obj["extra_body"]["session_id"] = self.session_id
                if self.force_temperature is not None:
                    obj["temperature"] = self.force_temperature
                if self.think_budget and path.startswith("/v1/messages"):
                    obj["thinking"] = {"type": "enabled", "budget_tokens": self.think_budget}
                body = json.dumps(obj).encode("utf-8")
            except Exception as exc:
                print(f"[session-proxy] JSON parse error, forwarding raw: {exc}", flush=True)

        self._forward_raw(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--upstream", default="http://127.0.0.1:18080")
    ap.add_argument("--session-id", default=os.environ.get("PFLASH_SESSION_ID", ""))
    ap.add_argument("--force-temperature", type=float, default=None)
    args = ap.parse_args()

    if not args.session_id and args.force_temperature is None:
        print("[session-proxy] WARNING: no session_id or force_temperature set; proxy is pass-through only", flush=True)

    Handler.upstream = args.upstream.rstrip("/")
    Handler.session_id = args.session_id
    Handler.force_temperature = args.force_temperature
    Handler.dump_path = os.environ.get("PROXY_DUMP", "")
    Handler.think_budget = int(os.environ.get("THINK_BUDGET", "0") or "0")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[session-proxy] listening on http://{args.host}:{args.port} "
        f"-> {Handler.upstream} "
        f"(session_id={Handler.session_id!r} force_temperature={Handler.force_temperature!r})",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
