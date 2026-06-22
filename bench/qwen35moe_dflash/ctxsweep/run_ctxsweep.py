#!/usr/bin/env python3
"""Context-length sweep runner: hit a LIVE server with ctx_*.json prompts,
parse per-ctx prefill_s / decode_tps / accept% / action from the server log."""
import json, os, re, sys, time, glob, urllib.request
LOG=sys.argv[1]; LABEL=sys.argv[2] if len(sys.argv)>2 else "run"
HERE=os.path.dirname(__file__); URL="http://127.0.0.1:18080/v1/chat/completions"
DONE=re.compile(r'prefill=([\d.]+)s decode=([\d.]+)s\(([\d.]+)tok/s\)')
SPEC=re.compile(r'accepted=\d+/\d+ \(([\d.]+)%\) avg_commit=([\d.]+)')
FLOOR=re.compile(r'\[spec-gate\] floor reason=(\w+)')
def post(p):
    body=open(p,'rb').read()
    r=urllib.request.Request(URL,data=body,headers={'content-type':'application/json'})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=600) as x: o=json.loads(x.read())
    return time.time()-t0,o
prev=sum(1 for _ in open(LOG)) if os.path.exists(LOG) else 0
rows=[]
for p in sorted(glob.glob(os.path.join(HERE,'ctx_*.json'))):
    ctx=int(os.path.basename(p)[4:10])
    wall,o=post(p); time.sleep(0.3)
    with open(LOG) as f: lines=f.readlines()
    blob=''.join(lines[prev:]); prev=len(lines)
    d=DONE.findall(blob); sp=SPEC.findall(blob); fl=FLOOR.findall(blob)
    prefill=d[-1][0] if d else 'NA'; decode=d[-1][2] if d else 'NA'
    accept=sp[-1][0] if sp else '-'; commit=sp[-1][1] if sp else '-'
    action='floor:'+fl[-1] if fl else ('spec' if sp else 'ar')
    rows.append((ctx,prefill,decode,accept,commit,action))
    print(f"ctx={ctx:6d}  prefill={prefill:>6}s  decode={decode:>6}tok/s  accept%={accept:>5}  commit={commit:>5}  {action}")
notes=os.path.join(HERE,'NOTES.md')
with open(notes,'a') as f:
    f.write(f"\n### {LABEL}\n\n| ctx | prefill_s | decode tok/s | accept% | commit | action |\n|--:|--:|--:|--:|--:|---|\n")
    for r in rows: f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n")
print("appended to",notes)
