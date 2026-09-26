import os,sys,time,json,subprocess,statistics,urllib.request,fcntl
from pathlib import Path
root=Path('/home/duster/lucebox-qwen4exp');out=Path('/tmp/qwen1100');name=sys.argv[1];settings=json.loads(sys.argv[2]);chunk=settings.pop('chunk','16384');port='8877'
lock=open('/tmp/qwen-perf/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,HIP_VISIBLE_DEVICES='1',DFLASH_HIP_NO_AUTO_UMA='1',GGML_CUDA_MMB='1',QWEN4EXP_QSA='1',QWEN4EXP_MMB_CUBLAS='5',DFLASH_MMB_SHADOW='1',LLAMA_MMB_HC16='2')
for k,v in settings.items():
 if v is None:env.pop(k,None)
 else:env[k]=v
model='/home/duster/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf';log=open(out/f'{name}-server.log','w')
p=subprocess.Popen([str(root/'server/build-hip/dflash_server'),model,'--host','127.0.0.1','--port',port,'--target-device','hip:0','--max-ctx','40000','--chunk',chunk],env=env,stdout=log,stderr=subprocess.STDOUT)
try:
 for _ in range(240):
  if p.poll() is not None:raise RuntimeError('server died')
  try:urllib.request.urlopen('http://127.0.0.1:'+port+'/v1/models',timeout=2);break
  except Exception:time.sleep(2)
 else:raise RuntimeError('load timeout')
 print('QUALITY START',name,settings,'chunk',chunk,flush=True)
 qlog=open(out/f'{name}-quality.log','w')
 q=subprocess.Popen(['python3','-u',str(root/'harness/client_test_runner.py'),'bench','--url','http://127.0.0.1:'+port,'--suite','he,gsm,math,recall','--model','dflash','--json-out',str(out/f'{name}-quality.json')],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in q.stdout:qlog.write(line);qlog.flush();print(line,end='',flush=True)
 q.wait();qlog.close()
 baseline=json.load(open('/tmp/qwen-fa-iq4-final-quality.json'))['suites'];actual=json.load(open(out/f'{name}-quality.json'))['suites'];ok=q.returncode==0
 for suite in ['he','gsm','math','recall']:
  previous={r['id'] for r in baseline[suite]['results'] if r.get('correct') is True};current={r['id'] for r in actual[suite]['results'] if r.get('correct') is True}
  lost=previous-current;ok &= not lost
  print('QUALITY RESULT',suite,actual[suite].get('accuracy'),'regressed_cases',sorted(lost),flush=True)
 subprocess.run(['python3','-u','/tmp/qwen1100/longcheck.py',port,str(out/f'{name}-longcheck.json')],check=True)
 times=[]
 for i in range(9):
  s=subprocess.check_output(['python3','/tmp/pf.py',port,'8875','1'],text=True);pt=int(s.split('pt=')[1].split()[0]);tt=float(s.split('ttft=')[1].strip().rstrip('s'));assert pt==16366 and tt>0,s
  print('FINAL PREFILL','warmup' if i==0 else i,s.strip(),'tps=%.3f'%(pt/tt),flush=True)
  if i:times.append(tt)
 summary={'quality_ok':ok,'settings':settings,'chunk':chunk,'ttft':times,'min_tps':16366/max(times),'median_tps':statistics.median(16366/t for t in times),'best_tps':16366/min(times)}
 (out/f'{name}-summary.json').write_text(json.dumps(summary,indent=2));print('FINAL SUMMARY',summary,flush=True)
finally:
 p.terminate()
 try:p.wait(timeout=15)
 except subprocess.TimeoutExpired:p.kill();p.wait()
 log.close()
sys.exit(0 if ok else 1)
