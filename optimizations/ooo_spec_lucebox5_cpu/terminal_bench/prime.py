#!/usr/bin/env python3
"""Prime the server prefix cache with a task's turn-0 prompt (system + instruction + tools).

Sent twice with max_tokens=1 so the snapshot policy stores the full turn-0 prefix.
Used before BOTH arms so neither arm pays the cold turn-0 prefill (which would
otherwise bias the pair toward whichever arm ran second).
"""

import json
import sys
import time
import urllib.request

sys.path.insert(0, "/home/lucebox5/tbspec")
import tb_agent  # noqa: E402
import tb_tools  # noqa: E402

task = sys.argv[1]
instruction = open(f"/home/lucebox5/tbspec/datasets/terminal-bench/{task}/instruction.md", encoding="utf-8").read()
messages = [
    {"role": "system", "content": tb_agent.SYSTEM_PROMPT},
    {"role": "user", "content": instruction},
]
for i in range(2):
    body = {
        "model": "deepseek-v4-flash", "messages": messages, "tools": tb_tools.TOOLS,
        "tool_choice": "auto", "temperature": 0, "max_tokens": 1, "stream": False,
        "automatic_tool_speculation": False,
    }
    started = time.perf_counter()
    req = urllib.request.Request(
        "http://127.0.0.1:18145/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    data = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    timings = (data.get("usage") or {}).get("timings") or {}
    print(f"prime {task} #{i}: {time.perf_counter() - started:.1f}s prompt={data.get('usage', {}).get('prompt_tokens')} "
          f"cache_hit={timings.get('cache_hit')} cached={timings.get('cached_prefix_tokens')} prefill_ms={timings.get('prefill_ms')}", flush=True)

