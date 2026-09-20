import os,sys,time,json,subprocess,statistics,urllib.request,fcntl
from pathlib import Path
root=Path('/home/duster/lucebox-qwen4exp'); out=Path('/tmp/qwen1100');out.mkdir(exist_ok=True)
lock=open('/tmp/qwen-perf/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
warmups=int(os.environ.get('QWEN_BENCH_WARMUPS','1'))
name=sys.argv[1]; variants=json.loads(sys.argv[2]); rounds=json.loads(sys.argv[3]) if len(sys.argv)>3 else [0,1,1,0]
base=dict(os.environ,HIP_VISIBLE_DEVICES='1',DFLASH_HIP_NO_AUTO_UMA='1',GGML_CUDA_MMB='1',QWEN4EXP_QSA='1',QWEN4EXP_MMB_CUBLAS='5',DFLASH_MMB_SHADOW='1',LLAMA_MMB_HC16='2',QWEN4EXP_LAST_TOKEN_FFN='1')
model='/home/duster/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf';port='8877';records=[]
if len(sys.argv)>4 and sys.argv[4]=='resume':records=json.loads((out/f'{name}.json').read_text())
round_offset=max((r['round'] for r in records),default=-1)+1
for step,v in enumerate(rounds):
 round_id=round_offset+step
 label,settings=variants[v]; settings=dict(settings);chunk=settings.pop('chunk','16384');env=dict(base)
 for k,value in settings.items():
  if value is None:env.pop(k,None)
  else:env[k]=value
 log=open(out/f'{name}-{round_id}-{label}.log','w')
 print('START',name,round_id,label,settings,'chunk',chunk,flush=True)
 p=subprocess.Popen([str(root/'server/build-hip/dflash_server'),model,'--host','127.0.0.1','--port',port,'--target-device','hip:0','--max-ctx','40000','--chunk',chunk],env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(240):
   if p.poll() is not None:raise RuntimeError('server died; see '+log.name)
   try:urllib.request.urlopen('http://127.0.0.1:'+port+'/v1/models',timeout=2);break
   except Exception:time.sleep(2)
  else:raise RuntimeError('load timeout')
  for i in range(warmups+int(os.environ.get("QWEN_BENCH_SAMPLES", "4"))):
   t=time.perf_counter();s=subprocess.check_output(['python3','/tmp/pf.py',port,'8875','1'],text=True)
   pt=int(s.split('pt=')[1].split()[0]);tt=float(s.split('ttft=')[1].strip().rstrip('s'))
   assert pt==16366 and tt>0,s
   print(name,label,'warmup' if i<warmups else i-warmups+1,s.strip(),'tps=%.3f'%(pt/tt),flush=True)
   if i>=warmups:records.append(dict(label=label,round=round_id,ttft=tt,tps=pt/tt,wall_time=time.time()))
   (out/f'{name}.json').write_text(json.dumps(records,indent=2))
 finally:
  p.terminate()
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:p.kill();p.wait()
  log.close()
for label,_ in variants:
 vals=[r['tps'] for r in records if r['label']==label]
 if vals:print('RESULT',label,'n',len(vals),'min',min(vals),'median',statistics.median(vals),'best',max(vals),flush=True)
