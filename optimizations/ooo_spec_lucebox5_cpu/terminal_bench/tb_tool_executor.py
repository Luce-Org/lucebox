#!/usr/bin/python3
"""dflash tool-speculation executor for terminal-bench containers.

Invoked by dflash_server as `<executor> --dflash-tool-spec-v1` (under taskset on
the reserved CPU lane) with one JSON request on stdin. Runs the predicted
read-only tool inside the CURRENT task container (docker exec) and prints one
JSON envelope on stdout. Only read-only tools are accepted here; the server's
allowlist is the outer authority.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tb_tools  # noqa: E402


def execute(request: dict) -> dict:
    if request.get("protocol") != tb_tools.PROTOCOL:
        raise ValueError("unsupported protocol")
    call = request.get("call")
    if not isinstance(call, dict):
        raise ValueError("call must be an object")
    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(name, str) or name not in tb_tools.READ_ONLY_TOOLS:
        raise ValueError("tool is not read-only eligible")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")

    expected = request.get("cpu_affinity", [])
    if isinstance(expected, list) and expected and hasattr(os, "sched_getaffinity"):
        observed = sorted(os.sched_getaffinity(0))
        if observed != sorted(set(int(c) for c in expected)):
            raise ValueError("observed CPU affinity does not match request")

    state = tb_tools.load_state()
    cid = state["cid"]
    raw = tb_tools.run_tool_sync(cid, state.get("workdir"), state.get("user"), name, arguments)
    raw["container"] = cid
    return {"ok": True, "result": raw}


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
        sys.stdout.write(json.dumps(execute(request), separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    except Exception as error:  # noqa: BLE001
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

