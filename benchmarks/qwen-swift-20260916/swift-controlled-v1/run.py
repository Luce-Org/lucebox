import json,pathlib,subprocess,urllib.request,time,os,hashlib,datetime
r=pathlib.Path('/opt/lucebox/work/swift-controlled-v1');s=json.loads((r/'suite.json').read_text());expected_hash=(r/'suite.sha256').read_text().split()[0];assert hashlib.sha256((r/'suite.json').read_bytes()).hexdigest()==expected_hash
port=18217
def call(path,body=None,port=port,timeout=600):
 req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as f:return json.load(f)
def shape(a,b):
 if type(a)!=type(b):return False
 if isinstance(a,dict):return a.keys()==b.keys() and all(shape(a[k],b[k]) for k in b)
 if isinstance(a,list):return len(a)==len(b) and all(shape(x,y) for x,y in zip(a,b))
 return True
def score(case,response):
 ch=response['choices'][0];m=ch['message'];ok=ch['finish_reason'] in ('stop','tool_calls')
 if case['suite']=='speed':return {'completed':ok and bool(m.get('content','').strip()),'format_pass':None,'answer_pass':None,'pass':None}
 value=None;fmt=False
 try:
  if case['id']=='tool_call':
   tc=m.get('tool_calls',[]);assert len(tc)==1 and tc[0]['function']['name']=='lookup_weather' and ch['finish_reason']=='tool_calls'
   value=json.loads(tc[0]['function']['arguments'])
  else:value=json.loads(m.get('content',''))
  fmt=shape(value,case['expected'])
 except (ValueError,AssertionError,KeyError,TypeError):pass
 return {'completed':ok,'format_pass':fmt,'answer_pass':value==case['expected'],'pass':ok and fmt and value==case['expected']}
model='/code/models/swift-qwen/ukisai_Swift-Qwen3.8-27b-IQ4_XS.gguf';template='/opt/lucebox/work/swift-mtp/frozen-template.jinja'
args={
'dflash':['/opt/lucebox-concurrent/server/build-hip/dflash_server',model,'--draft','/code/models/qwen38/qwen38-dflash2-q8_0.gguf','--draft-block-size','16','--prefix-cache-slots','0','--max-ctx','65536','--cache-type-k','q8_0','--cache-type-v','q8_0','--host','127.0.0.1','--port',str(port),'--model-name','swift-qwen','--chat-template-file',template,'--default-max-tokens','64000','--think-max-tokens','64000','--reasoning-effort-x-high','64000','--reasoning-effort-max','64000','--hard-limit-reply-budget','0'],
'mtp':['/opt/lucebox/work/swift-mtp/build/bin/llama-server','-m',model,'--host','127.0.0.1','--port',str(port),'--alias','swift-qwen','-ngl','99','-c','65536','-np','1','-b','512','-ub','512','-fa','on','-ctk','q8_0','-ctv','q8_0','--jinja','--chat-template-file',template,'--metrics','--spec-type','draft-mtp','--spec-draft-n-max','3','--reasoning-budget','-1','--no-context-shift','--cache-ram','0']}
env=dict(os.environ,LD_LIBRARY_PATH='/opt/rocm/core-10.0/lib',DFLASH_IDLE_PREFILL_TOKENS='512',DFLASH_MIXED_PREFILL_TOKENS='64',DFLASH_LONG_MIXED_PREFILL_TOKENS='64',DFLASH_HYBRID_CACHE='off')
(r/'launch.json').write_text(json.dumps(args,indent=2));h=call('/health',port=8216,timeout=5)
if h.get('busy') or h.get('pending_requests'):raise RuntimeError(f'Live requests present: {h}')
def drain_gpu():
 for _ in range(150):
  if int(pathlib.Path('/sys/class/drm/card5/device/mem_info_vram_used').read_text()) < 1536*1024**2:return
  time.sleep(0.2)
 raise RuntimeError('Previous GPU allocations did not drain')
def wait_production(timeout=180):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  try:
   h=call('/health',port=8216,timeout=2)
   if h.get('status')=='ok' and not h.get('busy') and not h.get('pending_requests'):return h
  except Exception:pass
  time.sleep(1)
 raise RuntimeError('Production Lucebox did not become healthy and idle')
guard=f'lucebox-benchmark-restore-{os.getpid()}'
def arm_guard():subprocess.run(['systemd-run','--unit',guard,'--on-active=45m','--timer-property=AccuracySec=1s','/bin/systemctl','start','lucebox.service'],check=True)
def disarm_guard():subprocess.run(['systemctl','stop',guard+'.timer'],check=False)
rows=[];reps={'dflash':0,'mtp':0};p=None
arm_guard();subprocess.run(['systemctl','stop','lucebox.service'],check=True)
try:
 drain_gpu()
 for backend in s['order']:
  reps[backend]+=1;rep=reps[backend];name=f'{backend}-{rep}'
  with (r/(name+'.log')).open('w') as log:
   p=subprocess.Popen(args[backend],env=env,stdout=log,stderr=subprocess.STDOUT)
   for _ in range(180):
    if p.poll() is not None:raise RuntimeError(name+' startup failed')
    try:
     if call('/health',timeout=2).get('status')=='ok':break
    except Exception:pass
    time.sleep(1)
   else:raise RuntimeError('startup timeout')
   call('/v1/chat/completions',{'model':'swift-qwen','messages':[{'role':'user','content':'Reply only: ready'}],'temperature':0,'max_tokens':16,'reasoning_effort':'none','cache_prompt':False})
   for case in s['cases']:
    body={'model':'swift-qwen','messages':case['messages'],'temperature':0,'seed':42,'max_tokens':64000,'reasoning_effort':'xhigh','cache_prompt':False}
    if 'tools' in case:body['tools']=case['tools']
    t=time.monotonic();response=call('/v1/chat/completions',body);wall=time.monotonic()-t
    usage=response['usage'];timing=response.get('timings',usage.get('timings',{}));m=response['choices'][0]['message'];reason=m.get('reasoning_content') or m.get('reasoning') or ''
    row={'backend':backend,'rep':rep,'suite':case['suite'],'case':case['id'],'wall_seconds':wall,'prefill_ms':timing.get('prefill_ms',timing.get('prompt_ms')),'decode_ms':timing.get('decode_ms',timing.get('predicted_ms')),'prompt_tokens':usage['prompt_tokens'],'total_tokens':usage['completion_tokens'],'reasoning_tokens_native':usage.get('completion_tokens_details',{}).get('reasoning_tokens'),'reasoning_text':reason,'score':score(case,response),'request':body,'response':response}
    rows.append(row);(r/'results.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({k:v for k,v in row.items() if k not in ('request','response','reasoning_text')}),flush=True)
   if backend=='mtp':
    for row in rows:
     if 'reasoning_text_tokens' not in row:row['reasoning_text_tokens']=len(call('/tokenize',{'content':row['reasoning_text'],'add_special':False})['tokens']) if row['reasoning_text'] else 0
    (r/'results.json').write_text(json.dumps(rows,indent=2))
   p.terminate()
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   p=None
   drain_gpu()
finally:
 if p and p.poll() is None:p.kill();p.wait()
 drain_gpu()
 subprocess.run(['systemctl','start','lucebox.service'],check=True)
 wait_production();disarm_guard()
