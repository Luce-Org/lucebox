"""Gate 4 (S8a): 10-turn chat.
  A: python3 -I gate4_live.py A     LUCE_KVFLASH_SESSION=0 (cold re-prefill); writes gate4_A.json (prompts + answers)
  B: python3 -I gate4_live.py B     LUCE_KVFLASH_SESSION=1 (live restore), replays A's prompts verbatim
  C: python3 -I gate4_live.py C     negative: B setup, turns 1-5 as A, turn 6 with a middle message edited
Env: G4_T1 (turn-1 filler tokens, default 12500), G4_TURNS (10)."""
import sys, os, json, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlib import *

mode = sys.argv[1]
T1 = int(os.environ.get("G4_T1", "12500"))
TURNS = int(os.environ.get("G4_TURNS", "10"))
SYS = {"role": "system", "content": "You are a careful assistant who reads long record dumps. Keep answers short (two sentences)."}


def user_msg(t):
    if t == 1:
        return filler(T1, salt=5) + "\nQuestion 1: what is the secret code? Answer in one sentence."
    extra = " ".join(f"Update {t}.{i}: record R{t}{i:03d} shows alpha-{(t*31+i*7)%97} and bravo-{(t*17+i*5)%89}; status ok." for i in range(6 + (t % 4) * 3))
    return f"{extra}\nQuestion {t}: summarise update {t}.0 in one sentence."


# One switch: LUCE_KVFLASH_SESSION is on by default with --kvflash; =0 is the cold re-prefill reference.
env = {"LUCE_KVFLASH_SESSION": "0" if mode == "A" else "1"}
log = f"{ROOT}/gate4_{mode}.log"
ref = json.load(open(f"{ROOT}/gate4_A.json")) if mode in ("B", "C") else None
rec = []


def seg_since(pos):
    time.sleep(0.7)
    with open(log, errors="replace") as f:
        f.seek(pos)
        return f.read().splitlines()


with Server(log, env=env):
    chat([{"role": "user", "content": filler(1500, salt=99)}], max_tokens=4)  # warm-up
    msgs = [SYS]
    for t in range(1, TURNS + 1):
        if ref:
            msgs = json.loads(json.dumps(ref[t - 1]["messages"]))
            if False:  # q38spec: C = live arm without drafter, replays A verbatim
                # edit the user message of turn 3 (a middle message); keep everything else
                idx = [i for i, m in enumerate(msgs) if m["role"] == "user"][2]
                msgs[idx]["content"] = msgs[idx]["content"].replace("Update 3.0", "Update 3.0 (EDITED)", 1)
        else:
            msgs = msgs + [{"role": "user", "content": user_msg(t)}]
        pos = os.path.getsize(log)
        o = chat(msgs, max_tokens=80)
        seg = seg_since(pos)
        r = dict(turn=t, messages=msgs, answer=o["text"], prompt_tokens=o["prompt_tokens"], ct=o["completion_tokens"],
                 ttft=o["ttft"], total=o["total"], tps=o["tps"],
                 live=[l for l in seg if "live restore" in l], cache=[l for l in seg if "chat CACHE" in l],
                 pooled=[l for l in seg if "pooled prefill:" in l], ladder=[l for l in seg if "state ladder" in l],
                 perf=[l for l in seg if "prefill=" in l])
        rec.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "messages"}), flush=True)
        if not ref:
            msgs = msgs + [{"role": "assistant", "content": o["text"]}]
            rec[-1]["messages"] = rec[-1]["messages"]  # messages up to and including this turn's user msg
json.dump(rec, open(f"{ROOT}/gate4_{mode}.json", "w"), indent=1)
print("median decode tok/s:", statistics.median([r["tps"] for r in rec if r["tps"]]))
