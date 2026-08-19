#!/usr/bin/env python3
"""Join per-request server timings (prefill/decode/prefix) with per-arm traces, in order."""
import glob, json, re, sys, statistics
log = sys.argv[1]; tag = sys.argv[2]
done = re.compile(r'chat DONE (\S+) ok=(\S+) in=(\d+) effective_in=(\d+) out=(\d+) ([\d.]+)s .*?prefix_len=(\d+) prefill=([\d.]+)s decode=([\d.]+)s\(([\d.]+)tok/s\)')
reqs = []
for line in open(log, errors='replace'):
    m = done.search(line)
    if m:
        reqs.append(dict(id=m.group(1), inp=int(m.group(3)), out=int(m.group(5)), wall=float(m.group(6)), prefix=int(m.group(7)), prefill=float(m.group(8)), decode=float(m.group(9)), dtps=float(m.group(10))))
# order trials by agent start time
trials = []
for rp in glob.glob(f'jobs/{tag}__*__*/*/result.json'):
    r = json.load(open(rp)); job = rp.split('/')[-3]; _, task, arm = job.rsplit('__', 2)
    tp = rp.replace('result.json', 'agent/tbspec_trace.json')
    try: t = json.load(open(tp))
    except Exception: continue
    trials.append((r['agent_execution']['started_at'], task, arm, t))
trials.sort()
i = 0
rows = []
for started, task, arm, t in trials:
    for turn in t['turns']:
        if turn.get('prompt_tokens') is None: continue
        # find next request with matching prompt tokens (allow skipping priming reqs)
        j = i
        while j < len(reqs) and reqs[j]['inp'] != turn['prompt_tokens']: j += 1
        if j >= len(reqs): print('unmatched', task, arm, turn['turn'], turn['prompt_tokens']); continue
        r = reqs[j]; i = j + 1
        new = r['inp'] - r['prefix']
        rows.append(dict(task=task, arm=arm, turn=turn['turn'], inp=r['inp'], new=new, out=r['out'], prefill=r['prefill'], decode=r['decode'], dtps=r['dtps'], wall=r['wall'], client=turn['model_wall_ms']/1000, pred=((turn.get('spec') or {}).get('predictor_wall_ms') or 0)/1000))
for arm in ('control', 'spec'):
    rs = [x for x in rows if x['arm'] == arm]
    pf = [x['new']/x['prefill'] for x in rs if x['prefill'] > 0.5 and x['new'] > 50]
    dt = [x['dtps'] for x in rs if x['out'] >= 20]
    ov = [x['client'] - x['wall'] for x in rs]
    print(f"{arm}: n={len(rs)} prefill tok/s median={statistics.median(pf):.1f} mean={statistics.mean(pf):.1f} | decode tok/s median={statistics.median(dt):.1f} mean={statistics.mean(dt):.1f} | client-server overhead median={statistics.median(ov):.2f}s (pred median={statistics.median([x['pred'] for x in rs]):.2f}s) | new tokens total={sum(x['new'] for x in rs)} out total={sum(x['out'] for x in rs)} prefill total={sum(x['prefill'] for x in rs):.0f}s decode total={sum(x['decode'] for x in rs):.0f}s")
if len(sys.argv) > 3:
    for x in rows: print(x)
