#!/usr/bin/env python3
"""Exit 0 if the server's native tool predictor is alive, 1 if dead/unavailable."""
import json, sys, urllib.request
sys.path.insert(0, "/home/lucebox5/tbspec")
import tb_tools
body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "List the files in /tmp using list_dir."}],
        "tools": tb_tools.TOOLS, "tool_choice": "auto", "temperature": 0, "max_tokens": 48, "stream": False, "automatic_tool_speculation": True}
req = urllib.request.Request("http://127.0.0.1:18145/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=600).read())
sp = d.get("dflash_tool_speculation") or {}
print("probe:", sp.get("status"), sp.get("reason"), sp.get("detail"), "pred_ms", sp.get("predictor_wall_ms"))
dead = str(sp.get("detail") or "").startswith("native_predictor_not_initialized") or str(sp.get("detail") or "").startswith("native_predictor_timeout")
sys.exit(1 if dead else 0)
