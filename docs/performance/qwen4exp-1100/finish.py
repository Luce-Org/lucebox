import os,sys,json,time,subprocess,statistics
from pathlib import Path
out=Path('/tmp/qwen1100')
for i in range(300):
 p=out/'reference-iq4/result.json'
 if p.exists():
  assert json.loads(p.read_text())['ok'], 'reference failed'
  break
 time.sleep(2)
else:raise RuntimeError('reference did not complete')
subprocess.run(['flock','/tmp/qwen-perf/gpu.lock','true'],check=True)
variants=[['table',{'QWEN4EXP_DENSE_TABLE':'1'}],['table_hc',{'QWEN4EXP_DENSE_TABLE':'1','QWEN4EXP_HC_TILE16':'1'}]]
subprocess.run(['python3','-u',str(out/'run.py'),'combined-ab',json.dumps(variants)],env=dict(os.environ,QWEN_BENCH_WARMUPS='2'),check=True)
rows=json.loads((out/'combined-ab.json').read_text())
a=[r['tps'] for r in rows if r['label']=='table'];b=[r['tps'] for r in rows if r['label']=='table_hc']
assert len(a)==len(b)==8
settings={'QWEN4EXP_LAST_TOKEN_FFN':'1','QWEN4EXP_DENSE_TABLE':'1'}
if statistics.median(b)>statistics.median(a)*1.002:settings['QWEN4EXP_HC_TILE16']='1'
print('FINAL CONFIG SELECTION',settings,'medians',statistics.median(a),statistics.median(b),flush=True)
(out/'final-settings.json').write_text(json.dumps(settings,indent=2))
subprocess.run(['python3','-u',str(out/'quality.py'),'final',json.dumps(settings)],check=True)
