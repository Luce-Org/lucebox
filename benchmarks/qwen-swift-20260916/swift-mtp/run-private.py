import pathlib,subprocess,os,urllib.request,json,time
r=pathlib.Path('/opt/lucebox/work/swift-mtp');env=dict(os.environ,LD_LIBRARY_PATH='/opt/rocm/core-10.0/lib')
guard=f'lucebox-benchmark-restore-{os.getpid()}'
def arm_guard():
 subprocess.run(['systemd-run','--unit',guard,'--on-active=2h','--timer-property=AccuracySec=1s','/bin/systemctl','start','lucebox.service'],check=True)
def wait_production(timeout=180):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  try:
   with urllib.request.urlopen('http://127.0.0.1:8216/health',timeout=2) as f:h=json.load(f)
   if h.get('status')=='ok' and not h.get('busy') and not h.get('pending_requests'):return h
  except Exception:pass
  time.sleep(1)
 raise RuntimeError('Production Lucebox did not become healthy and idle')
def disarm_guard():
 subprocess.run(['systemctl','stop',guard+'.timer'],check=False)
with urllib.request.urlopen('http://127.0.0.1:8216/health',timeout=5) as f:h=json.load(f)
if h.get('busy') or h.get('pending_requests'):raise RuntimeError(f'Live requests present: {h}')
# Capture deployed baseline before any service interruption.
subprocess.run(['/opt/lucebox-python/bin/python','-u',str(r/'evaluate.py'),'http://127.0.0.1:8216','qwen3.8-27b','current',str(r)],check=True)
with urllib.request.urlopen('http://127.0.0.1:8216/health',timeout=5) as f:h=json.load(f)
if h.get('busy') or h.get('pending_requests'):raise RuntimeError(f'Live requests present: {h}')
arm_guard();subprocess.run(['systemctl','stop','lucebox.service'],check=True)
p=None
try:
 for label,extra in [('swift-plain',[]),('swift-mtp',['--spec-type','draft-mtp','--spec-draft-n-max','3'])]:
  args=[str(r/'build/bin/llama-server'),'-m','/code/models/swift-qwen/ukisai_Swift-Qwen3.8-27b-IQ4_XS.gguf','--host','127.0.0.1','--port','18217','--alias','swift-qwen','-ngl','99','-c','8192','-np','1','-b','512','-ub','512','-fa','on','-ctk','q8_0','-ctv','q8_0','--jinja','--metrics']+extra
  (r/(label+'-args.json')).write_text(json.dumps(args,indent=2))
  with (r/(label+'-server.log')).open('w') as log:
   p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
   ready=False
   for _ in range(120):
    if p.poll() is not None:raise RuntimeError(label+' server exited; inspect log')
    try:
     with urllib.request.urlopen('http://127.0.0.1:18217/health',timeout=2) as f:ready=json.load(f).get('status')=='ok'
    except Exception:pass
    if ready:break
    time.sleep(1)
   if not ready:raise RuntimeError('load timeout')
   subprocess.run(['/opt/lucebox-python/bin/python','-u',str(r/'evaluate.py'),'http://127.0.0.1:18217','swift-qwen',label,str(r)],check=True)
   with urllib.request.urlopen('http://127.0.0.1:18217/metrics',timeout=10) as f:(r/(label+'-metrics.txt')).write_bytes(f.read())
   p.terminate()
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   p=None
finally:
 if p and p.poll() is None:p.kill();p.wait()
 subprocess.run(['systemctl','start','lucebox.service'],check=True)
 wait_production();disarm_guard()
