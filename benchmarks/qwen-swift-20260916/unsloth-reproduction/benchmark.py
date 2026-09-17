import ast,json,pathlib,time,urllib.request,os
root=pathlib.Path('/opt/lucebox/work/unsloth-reproduction-20260915')
prefix=os.environ.get('RESULT_PREFIX','benchmark')
source=(pathlib.Path('/opt/lucebox-concurrent/server/scripts')/os.environ.get('PROMPT_SOURCE','bench_he_http.py')).read_text()
module=ast.parse(source)
prompts=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPTS' for t in n.targets))
rows=[]
for name,prompt in prompts:
 body={'model':'qwen3.8-27b','messages':[{'role':'user','content':prompt}],'max_tokens':256,'temperature':0}
 if os.environ.get('CACHE_MODE'):body['extra_body']={'lucebox_cache':{'mode':os.environ['CACHE_MODE']}}
 start=time.monotonic();req=urllib.request.Request('http://127.0.0.1:8216/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as r:reply=json.load(r)
 wall=time.monotonic()-start;usage=reply['usage'];timing=usage.get('timings',{})
 row={'name':name,'wall_seconds':wall,'usage':usage,'response':reply['choices']};rows.append(row)
 print(json.dumps({'name':name,'wall_seconds':wall,'usage':usage}),flush=True)
 (root/(prefix+'-results.json')).write_text(json.dumps(rows,indent=2))
tokens=sum(x['usage']['completion_tokens'] for x in rows);wall=sum(x['wall_seconds'] for x in rows)
decode=sum(x['usage']['timings']['decode_ms']/1000 for x in rows)
summary={'requests':len(rows),'tokens':tokens,'wall_seconds':wall,'end_to_end_tps':tokens/wall,'decode_seconds':decode,'decode_tps':tokens/decode}
(root/(prefix+'-summary.json')).write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
