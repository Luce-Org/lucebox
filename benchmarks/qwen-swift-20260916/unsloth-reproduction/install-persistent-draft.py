import json,pathlib,urllib.request,subprocess,shutil
root=pathlib.Path('/opt/lucebox')
with urllib.request.urlopen('http://127.0.0.1:8216/health') as r:h=json.load(r)
assert not h.get('busy') and not h.get('pending_requests'),h
p=root/'models.json';c=json.loads(p.read_text())
for v in c['models'].values():
 args=v.get('extra_args',[]);out=[];i=0
 while i<len(args):
  if args[i]=='--draft-residency':i+=2;continue
  if args[i]=='--lazy-draft':i+=1;continue
  out.append(args[i]);i+=1
 if 'extra_args' in v:v['extra_args']=out
subprocess.run(['systemctl','stop','lucebox.service'],check=True)
try:
 dest=pathlib.Path('/opt/lucebox-concurrent/server/build-hip/dflash_server')
 shutil.copy2('/opt/lucebox/work/concurrency-20260913/alternating-candidate/dflash_server',str(dest)+'.new')
 pathlib.Path(str(dest)+'.new').replace(dest)
 p.write_text(json.dumps(c,indent=2)+'\n')
finally:subprocess.run(['systemctl','start','lucebox.service'],check=True)
print('Installed persistent decoding drafter build and removed obsolete flags')
