#!/usr/bin/env python3
"""Sandboxed read-only executor for speculative coding-agent tools."""

from __future__ import annotations

import json
import os
import sys

from coding_tools import PROTOCOL, execute_tool


def _verify_affinity(request: dict) -> None:
    expected = request.get("cpu_affinity", [])
    if not isinstance(expected, list) or any(
        isinstance(cpu, bool) or not isinstance(cpu, int) for cpu in expected
    ):
        raise ValueError("cpu_affinity must contain integers")
    if not expected:
        raise ValueError("a non-empty isolated CPU affinity is required")
    if not hasattr(os, "sched_getaffinity"):
        raise ValueError("configured CPU affinity cannot be verified on this platform")
    if sorted(os.sched_getaffinity(0)) != sorted(set(expected)):
        raise ValueError("executor CPU affinity does not match the server contract")


def main() -> int:
    if sys.argv[1:] != ["--dflash-tool-spec-v1"]:
        print("expected --dflash-tool-spec-v1", file=sys.stderr)
        return 2
    try:
        request_line = sys.stdin.readline()
        request = json.loads(request_line) if request_line else None
        if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
            raise ValueError("invalid executor request")
        _verify_affinity(request)
        call = request.get("call")
        if not isinstance(call, dict):
            raise ValueError("missing tool call")
        result = execute_tool(
            call.get("name"),
            call.get("arguments"),
            os.environ.get("DFLASH_TOOL_WORKSPACE"),
        )
        if result.get("ok") is not True:
            envelope = {"ok": False, "error": result.get("error", "tool failed")}
        else:
            envelope = {"ok": True, "result": result}
        sys.stdout.write(json.dumps(envelope, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:  # noqa: BLE001 - protocol boundary fails closed
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
