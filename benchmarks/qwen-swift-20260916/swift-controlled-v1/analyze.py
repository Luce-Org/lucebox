import json,pathlib,statistics,collections,datetime,csv
root=pathlib.Path(__file__).resolve().parent;app=root.parent.parent/'outputs/swift-controlled-report';raw=json.loads((root/'results.json').read_text());suite=json.loads((root/'suite.json').read_text())
assert len({(x['backend'],x['rep'],x['case']) for x in raw})==len(raw)
labels={'dflash':'Lucebox + DFlash2','mtp':'llama.cpp + MTP'}
for row in raw:
 assert row['request']['reasoning_effort']=='xhigh' and row['request']['max_tokens']==64000
 assert row['prefill_ms'] is not None and row['decode_ms'] is not None
runs=[]
for (backend,rep,group),items in __import__('itertools').groupby(sorted(raw,key=lambda x:(x['backend'],x['rep'],x['suite'])),key=lambda x:(x['backend'],x['rep'],x['suite'])):
 a=list(items);required=6 if group=='quality' else 10
 if len(a)!=required:continue
 tokens=sum(x['total_tokens'] for x in a);decode=sum(x['decode_ms'] for x in a)/1000;prefill=sum(x['prefill_ms'] for x in a)/1000;wall=sum(x['wall_seconds'] for x in a)
 runs.append({'backend':labels[backend],'suite':group,'rep':rep,'wall':wall,'prefill':prefill,'decode':decode,'tokens':tokens,'thinking':sum(x['reasoning_text_tokens'] for x in a) if all('reasoning_text_tokens' in x for x in a) else None,'tps':tokens/decode,'prefill_tokens':sum(x['prompt_tokens'] for x in a),'prefill_tps':sum(x['prompt_tokens'] for x in a)/prefill})
summary=[]
for backend in labels.values():
 for group in ['quality','speed']:
  a=[x for x in runs if x['backend']==backend and x['suite']==group]
  if not a:continue
  z={'backend':backend,'suite':group,'repetitions':len(a)}
  for field in ['wall','prefill','decode','tokens','thinking','tps','prefill_tokens','prefill_tps']:
   values=[x[field] for x in a if x[field] is not None];z['median_'+field]=round(statistics.median(values),2) if values else None
  for field in ['wall','tps']:z['min_'+field]=round(min(x[field] for x in a),2);z['max_'+field]=round(max(x[field] for x in a),2)
  summary.append(z)
detail=[];quality=[]
for case in suite['cases']:
 for backend in labels:
  a=[x for x in raw if x['case']==case['id'] and x['backend']==backend]
  if not a:continue
  row={'backend':labels[backend],'suite':case['suite'],'prompt':case['id'],'repetitions':len(a)}
  for out,field,mul in [('wall','wall_seconds',1),('prefill','prefill_ms',.001),('decode','decode_ms',.001),('tokens','total_tokens',1),('thinking','reasoning_text_tokens',1)]:
   vals=[x[field]*mul for x in a if field in x];row[out]=round(statistics.median(vals),2) if vals else None
  detail.append(row)
  if case['suite']=='quality':quality.append({'backend':labels[backend],'prompt':case['id'],'answers':sum(x['score']['answer_pass'] for x in a),'format':sum(x['score']['format_pass'] for x in a),'passed':sum(x['score']['pass'] for x in a),'repetitions':len(a)})
matched={}
for case in suite['cases']:
 counts={b:sorted(set(x['prompt_tokens'] for x in raw if x['backend']==b and x['case']==case['id'])) for b in labels}
 matched[case['id']]=counts
(root/'analysis.json').write_text(json.dumps({'summary':summary,'runs':runs,'quality':quality,'detail':detail,'prompt_counts':matched},indent=2))
for name,rows in [('measurements', [{k:v for k,v in x.items() if k not in ('request','response','reasoning_text','score')}|x['score'] for x in raw]),('summary',summary),('runs',runs),('quality',quality),('detail',detail)]:
 if rows:
  with (root/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
p=app/'src/data.json';d=json.loads(p.read_text());d['generatedAt']=datetime.datetime.now(datetime.timezone.utc).isoformat();d['buildStatus']='updating';d['report']={'asOf':'2026-09-16'}
for name,rows in [('summary',summary),('token_summary',summary),('quality',quality),('detail',detail),('runs',runs)]:
 d['queries'][name]={'rows':rows,'source':{'name':'Frozen Swift controlled benchmark v1','description':'Measured on linuxmacan. Raw results.json, suite.json, launch.json, and analysis.py retained in work/swift-controlled-v1.','evidenceFlow':[{'title':'Measurement','detail':'96 requests planned: 16 frozen prompts × 2 backends × 3 repetitions. No prior exploratory runs included.'},{'title':'Aggregation','detail':'Sum times and tokens within each suite/repetition; report medians of three run totals. Quality passes require exact answers and JSON types, no repair.'}],'metricDefinitions':[{'label':'Thinking text tokens','definition':'Returned reasoning text retokenized using the same Swift tokenizer with special-token addition disabled. Excludes wrapper markers; may differ from native generated reasoning-token counts.'},{'label':'Generation rate','definition':'Total completion tokens, including thinking, divided by measured generation seconds. Prefill and HTTP overhead excluded.'}]}}
d['queries']['protocol']['rows']=[{'suite':c['suite'],'prompt':c['id'],'criterion':c['criterion'],'input':c['messages'][0]['content'],'expected':json.dumps(c.get('expected'),indent=2) if 'expected' in c else 'Not scored for coding correctness','tools':json.dumps(c.get('tools',[]),indent=2)} for c in suite['cases']]
p.write_text(json.dumps(d,indent=2))
print(json.dumps({'completed':len(raw),'summary':summary,'quality':quality,'prompt_counts':matched},indent=2))
