#!/usr/bin/python3
"""dflash tool-speculation executor for the public-REST-API benchmark.

`<executor> --dflash-tool-spec-v1`, one JSON request on stdin -> performs the real
HTTPS call (same code path as the client, rest_tools.run_tool) -> one JSON envelope
on stdout. Runs on the reserved CPU lane under taskset; env is minimal.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rest_tools  # noqa: E402


def main() -> int:
    if sys.argv[1:] != ["--dflash-tool-spec-v1"]:
        print("expected --dflash-tool-spec-v1", file=sys.stderr)
        return 2
    try:
        line = sys.stdin.readline()
        request = json.loads(line) if line else None
        if not isinstance(request, dict) or request.get("protocol") != rest_tools.PROTOCOL:
            raise ValueError("bad request")
        call = request.get("call") or {}
        name, args = call.get("name"), call.get("arguments")
        if name not in rest_tools.READ_ONLY_TOOLS or not isinstance(args, dict):
            raise ValueError("tool not eligible")
        expected = request.get("cpu_affinity", [])
        if (not isinstance(expected, list) or not expected or
                any(isinstance(cpu, bool) or not isinstance(cpu, int) for cpu in expected)):
            raise ValueError("non-empty integer cpu affinity required")
        if not hasattr(os, "sched_getaffinity"):
            raise ValueError("cpu affinity inspection unavailable")
        if sorted(os.sched_getaffinity(0)) != sorted(set(expected)):
            raise ValueError("cpu affinity mismatch")
        raw = rest_tools.run_tool(name, args)
        if not isinstance(raw, dict) or raw.get("ok") is not True:
            sys.stdout.write(json.dumps({
                "ok": False,
                "error": raw.get("value") if isinstance(raw, dict) else "invalid tool result",
            }, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            return 0
        sys.stdout.write(json.dumps({"ok": True, "result": raw}, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
