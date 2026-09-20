import sys, json, time, requests
port=int(sys.argv[1]); nt=int(sys.argv[2]); mt=int(sys.argv[3]) if len(sys.argv)>3 else 1
base=f"http://127.0.0.1:{port}"
mid=requests.get(base+"/v1/models",timeout=30).json()["data"][0]["id"]
filler=("Lucebox is a GPU inference engine for local language models. It runs on a single integrated GPU with unified memory. ")
reps=max(1,(nt-40)//13)
plant="OPERATIONS NOTE 47-B: the calibration key for the Strix Halo test rig is QUINCE-AMBER-7731, and the rig thermal ceiling is recorded as 54 degrees Celsius. "
prompt=plant+filler*reps+"\nQuestion: What is the calibration key? Answer verbatim."
payload={"model":mid,"messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}
t0=time.perf_counter(); first=None; usage={}
with requests.post(base+"/v1/chat/completions",json=payload,stream=True,timeout=1800) as r:
    for line in r.iter_lines():
        if not line: continue
        s=line.decode("utf-8","replace")
        if not s.startswith("data:"): continue
        d=s[5:].strip()
        if d=="[DONE]": break
        try: o=json.loads(d)
        except: continue
        if o.get("usage"): usage=o["usage"]
        if o.get("choices"):
            tk=(o["choices"][0].get("delta") or {}).get("content") or ""
            if tk and first is None: first=time.perf_counter()
print("pt=%s ttft=%.2fs" % (usage.get("prompt_tokens"), (first-t0) if first else -1))
