#!/usr/bin/env python3
"""Read-only executor for end-to-end BFCL tool-speculation replay.

BFCL functions are specifications rather than deployable APIs. This adapter
therefore performs no external action: it waits for a fixed, documented API
latency and returns a deterministic digest of the canonical call. The engine's
own allowlist remains the authority for which predictions may reach it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any


PROTOCOL = "dflash.tool-speculation.v1"
LATENCY_MS = 2_000
REFERENCE_WORDS = (
    "amber", "azure", "cedar", "coral", "gold", "ivory", "maple", "olive",
    "pearl", "plum", "sable", "silver", "teal", "violet", "willow", "jade",
)


def canonical_call(call: dict[str, Any]) -> str:
    return json.dumps(call, sort_keys=True, separators=(",", ":"))


def call_sha256(call: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_call(call).encode()).hexdigest()


def call_ref(call: dict[str, Any]) -> str:
    return REFERENCE_WORDS[int(call_sha256(call)[:8], 16) % len(REFERENCE_WORDS)]


def execute(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("protocol") != PROTOCOL:
        raise ValueError("unsupported protocol")
    call = request.get("call")
    if not isinstance(call, dict):
        raise ValueError("call must be an object")
    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError("tool name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")

    expected = request.get("cpu_affinity", [])
    if not isinstance(expected, list) or not all(
        isinstance(cpu, int) and cpu >= 0 for cpu in expected
    ):
        raise ValueError("cpu_affinity must contain non-negative integers")
    observed = sorted(os.sched_getaffinity(0)) if hasattr(
        os, "sched_getaffinity"
    ) else []
    if expected and observed != sorted(set(expected)):
        raise ValueError("observed CPU affinity does not match request")

    canonical = {"name": name, "arguments": arguments}
    digest = call_sha256(canonical)
    reference = call_ref(canonical)
    started = time.perf_counter()
    time.sleep(LATENCY_MS / 1_000.0)
    elapsed = (time.perf_counter() - started) * 1_000.0
    return {
        "ok": True,
        "result": {
            "call_sha256": digest,
            "call_ref": reference,
            "tool_name": name,
            "latency_ms": LATENCY_MS,
            "elapsed_ms": elapsed,
            "cpu_affinity": observed,
            "side_effects": False,
        },
    }


def main() -> int:
    if sys.argv[1:] != ["--dflash-tool-spec-v1"]:
        print("expected --dflash-tool-spec-v1", file=sys.stderr)
        return 2
    try:
        line = sys.stdin.readline()
        if not line:
            raise ValueError("missing request")
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        print(json.dumps(execute(request), separators=(",", ":")), flush=True)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
