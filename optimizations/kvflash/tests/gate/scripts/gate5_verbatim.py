"""Like gate4_live.py, but every turn's prompt is the previous prompt + the generated tokens verbatim,
so live restore can reach the generation-end rung G (a chat template re-render strips the think block and never does).
One user message carries the whole transcript as ChatML text. A: LUCE_KVFLASH_SESSION=0 (cold). B: =1, replays A's prompts.
Thinking off only: the chain copies the empty think block the template pre-fills, so a turn that streams reasoning exits.
B exits nonzero when live restore ran but no turn resumed at the generation-end checkpoint (C == G).
Writes gate4_<mode>.json/.log so cmp_gate.py works unchanged."""
import sys, os, json, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlib import *

mode = sys.argv[1]
T1 = int(os.environ.get("G4_T1", "12500")); TURNS = int(os.environ.get("G4_TURNS", "10"))
env = {"LUCE_KVFLASH_SESSION": "0" if mode == "A" else "1"}
log = f"{ROOT}/gate4_{mode}.log"
ref = json.load(open(f"{ROOT}/gate4_A.json")) if mode == "B" else None
rec = []

def upd(t):
    extra = " ".join(f"Update {t}.{i}: record R{t}{i:03d} shows alpha-{(t*31+i*7)%97} and bravo-{(t*17+i*5)%89}; status ok." for i in range(6 + (t % 4) * 3))
    return f"{extra}\nQuestion {t}: summarise update {t}.0 in one sentence."

def seg_since(pos):
    time.sleep(0.7)
    with open(log, errors="replace") as f:
        f.seek(pos); return f.read().splitlines()

def consumed(gen, live):
    """This turn's live restore resumed at the generation-end checkpoint (C == G) and did not fall back to a cold prefill."""
    g = {m.group(1) for l in gen for m in [re.search(r"gen ckpt check: G=(\d+)", l)] if m}
    c = {m.group(1) for l in live for m in [re.search(r"live restore: L=\d+ C=(\d+)", l)] if m}
    return bool(g & c) and not any("failed" in l for l in live)

def chat_raw(content, mt):
    body = json.dumps({"model": "luce", "messages": [{"role": "user", "content": content}], "temperature": 0, "seed": 1,
                       "max_tokens": mt, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); reas, text, usage, fin = [], [], {}, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            for c in j.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("reasoning_content"): reas.append(dl["reasoning_content"])
                if dl.get("content"): text.append(dl["content"])
                fin = c.get("finish_reason") or fin
    return dict(text="".join(text), reasoning="".join(reas), finish=fin, ct=usage.get("completion_tokens"), pt=usage.get("prompt_tokens"), tps=None)

import urllib.request
with Server(log, env=env):
    chat([{"role": "user", "content": filler(1500, salt=99)}], max_tokens=4)  # warm-up
    content = filler(T1, salt=5) + "\nQuestion 1: what is the secret code? Answer in one sentence."
    for t in range(1, TURNS + 1):
        if ref: content = ref[t - 1]["content"]
        pos = os.path.getsize(log)
        o = chat_raw(content, 80)
        seg = seg_since(pos)
        live, gen = [l for l in seg if "live restore" in l], [l for l in seg if "gen ckpt" in l]
        r = dict(turn=t, content=content, answer=o["text"], reasoning=o["reasoning"], finish=o["finish"], prompt_tokens=o["pt"], ct=o["ct"], tps=None,
                 live=live, gen=gen, consumed=consumed(gen, live))
        rec.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "content"}), flush=True)
        if o["reasoning"]:
            sys.exit(f"turn {t} streamed reasoning; the verbatim chain needs thinking off")
        # next prompt = this prompt + the pre-filled empty think block + answer + EOS + the next user turn, as raw ChatML
        content = content + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + o["text"] + "<|im_end|>\n<|im_start|>user\n" + upd(t + 1)
json.dump(rec, open(f"{ROOT}/gate4_{mode}.json", "w"), indent=1)
if mode == "B" and any(r["live"] for r in rec) and not any(r["consumed"] for r in rec):
    sys.exit("live restore ran but no turn resumed at the generation-end checkpoint (C == G)")
