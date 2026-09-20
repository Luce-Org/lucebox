import os,sys,json,subprocess,shutil,fcntl
from pathlib import Path
root=Path('/home/duster/lucebox-qwen4exp');out=Path('/tmp/qwen1100')
lock=open('/tmp/qwen-perf/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
kind=sys.argv[1]; settings=json.loads(sys.argv[2]);dest=out/('reference-'+kind);dest.mkdir(exist_ok=True)
models={'gsq':'/home/duster/models/qwen4exp/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf','iq4':'/home/duster/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf'}
pairs=[('L00.gnorm','final_output-0'),('L03.faout','attn_pregate-3'),('L03.att','attn_output-3'),('logits','result_output')]
env=dict(os.environ,**settings,QWEN4EXP_DUMP_BIN=','.join(a for a,b in pairs),QWEN4EXP_UP_DUMP_BIN=','.join(b for a,b in pairs))
for a,b in pairs:
 for f in [Path('/tmp/our_'+a+'.bin'),Path('/tmp/up_'+b+'.bin')]:f.unlink(missing_ok=True)
log=open(dest/'comparison.log','w')
p=subprocess.run(['python3','-u',str(root/'server/scripts/qwen4exp_upstream_diff.py'),'--model',models[kind],'--seq','16','--reference','--output-dir',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
print('REFERENCE',kind,'exit',p.returncode,flush=True)
ok=p.returncode==0
for a,b in pairs:
 x=Path('/tmp/our_'+a+'.bin');y=Path('/tmp/up_'+b+'.bin')
 same=x.exists() and y.exists() and x.read_bytes()==y.read_bytes();ok &= same
 print('BINARY',kind,a,b,'EXACT' if same else 'FAIL','bytes',x.stat().st_size if x.exists() else -1,flush=True)
 for f in [x,y]:
  if f.exists():shutil.copy2(f,dest/f.name)
(dest/'result.json').write_text(json.dumps({'ok':ok,'settings':settings,'harness_exit':p.returncode},indent=2))
sys.exit(0 if ok else 1)
