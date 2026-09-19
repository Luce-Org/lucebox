import json,time,urllib.request,pathlib
root=pathlib.Path('/opt/lucebox/work/unsloth-reproduction-20260915');owner='reproduction-compression'
def call(path,body=None):
 req=urllib.request.Request('http://127.0.0.1:8216'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
doc='Reference ledger:\n'+('ordinary filler text. '*2600)+'\nThe ALPHA access value is 507831.\n'+('ordinary filler text. '*2600)+'\nThe BETA access value is 926104.\n'+('ordinary filler text. '*2600)+'\nThe GAMMA access value is 318762.\n'+('ordinary filler text. '*200)
body={'model':'qwen3.8-27b','messages':[{'role':'user','content':doc+'\nReturn the ALPHA, BETA and GAMMA access values in that order, separated only by commas.'}],'max_tokens':32768,'reasoning_effort':'none','temperature':0,'extra_body':{'lucebox_cache':{'owner':owner,'mode':'auto','bulk_regions':[{'message':0,'start':0,'end':len(doc.encode()),'text':doc}]}}}
rows=[]
try:
 for name in ['cold','repeat']:
  t=time.monotonic();response=call('/v1/chat/completions',body);wall=time.monotonic()-t
  answer=response['choices'][0]['message']['content']
  assert ''.join(answer.split())=='507831,926104,318762',answer
  trace=call('/cache/status')['last_request'];row={'name':name,'wall_seconds':wall,'answer':answer,'usage':response['usage'],'trace':trace};rows.append(row);print(json.dumps(row),flush=True)
  (root/'eligible-compression-results.json').write_text(json.dumps(rows,indent=2))
  assert trace['selected_path'] in ('compress','frozen','full'),trace
  assert trace['scored_regions']==(1 if name=='cold' else 0),trace
finally:call('/cache/release',{'owner':owner})
