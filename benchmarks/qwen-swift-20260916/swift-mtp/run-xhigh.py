import ast,json,pathlib,time,urllib.request,subprocess,os
r=pathlib.Path('/opt/lucebox/work/swift-mtp');source=pathlib.Path('/opt/lucebox-concurrent/server/scripts/bench_he.py').read_text();module=ast.parse(source)
prompts=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPTS' for t in n.targets))
def call(path,body=None,port=18217):
 q=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=120) as f:return json.load(f)
h=call('/health',port=8216);assert not h.get('busy') and not h.get('pending_requests'),h
model='/code/models/swift-qwen/ukisai_Swift-Qwen3.8-27b-IQ4_XS.gguf'
env=dict(os.environ,LD_LIBRARY_PATH='/opt/rocm/core-10.0/lib',DFLASH_IDLE_PREFILL_TOKENS='512',DFLASH_MIXED_PREFILL_TOKENS='64',DFLASH_LONG_MIXED_PREFILL_TOKENS='64',DFLASH_HYBRID_CACHE='off')
variants=[('article-swift-dflash-xhigh', ['/opt/lucebox-concurrent/server/build-hip/dflash_server',model,'--draft','/code/models/qwen38/qwen38-dflash2-q8_0.gguf','--draft-block-size','16','--prefix-cache-slots','2','--max-ctx','65536','--cache-type-k','q8_0','--cache-type-v','q8_0','--host','127.0.0.1','--port','18217','--model-name','swift-qwen'])]
p=None;subprocess.run(['systemctl','stop','lucebox.service'],check=True)
try:
 for label,args in variants:
  (r/(label+'-args.json')).write_text(json.dumps(args,indent=2))
  with (r/(label+'-server.log')).open('w') as log:
   p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
   for i in range(120):
    if p.poll() is not None:raise RuntimeError(label+' failed to start')
    try:
     if call('/health').get('status')=='ok':break
    except Exception:pass
    time.sleep(1)
   else:raise RuntimeError('load timeout')
   call('/v1/chat/completions',{'model':'swift-qwen','messages':[{'role':'user','content':'Reply only: ready'}],'max_tokens':8,'temperature':0,'reasoning_effort':'none'})
   props=call('/props');(r/'xhigh-props.json').write_text(json.dumps(props,indent=2))
   subprocess.run(['/opt/lucebox-python/bin/python','-u',str(r/'evaluate-xhigh.py'),'http://127.0.0.1:18217','swift-qwen','quality-swift-dflash-xhigh',str(r)],check=True)
   rows=[]
   for name,prompt in prompts:
    body={'model':'swift-qwen','messages':[{'role':'user','content':prompt}],'max_tokens':32768,'temperature':0,'reasoning_effort':'xhigh','seed':42}
    t=time.monotonic();response=call('/v1/chat/completions',body);elapsed=time.monotonic()-t
    rows.append({'name':name,'request':body,'wall_seconds':elapsed,'response':response});(r/(label+'.json')).write_text(json.dumps(rows,indent=2))
    print(json.dumps({'label':label,'name':name,'wall':elapsed,'usage':response.get('usage'),'finish':response['choices'][0]['finish_reason']}),flush=True)
   p.terminate()
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   p=None
finally:
 if p and p.poll() is None:p.kill();p.wait()
 subprocess.run(['systemctl','start','lucebox.service'],check=True)
