import json
import os
import pathlib
import shutil
import subprocess
import time
import urllib.request

root=pathlib.Path('/opt/lucebox')
config_path=root/'models.json'
dest=pathlib.Path('/opt/lucebox-concurrent/server/build-hip/dflash_server')
source=pathlib.Path('/opt/lucebox/work/concurrency-20260913/alternating-candidate/dflash_server')

def health(timeout=5):
 with urllib.request.urlopen('http://127.0.0.1:8216/health',timeout=timeout) as response:return json.load(response)

def wait_ready(timeout=180):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  try:
   state=health(timeout=2)
   if state.get('status')=='ok' and not state.get('busy') and not state.get('pending_requests'):return state
  except Exception:pass
  time.sleep(1)
 raise RuntimeError('Production Lucebox did not become healthy and idle')

def atomic_write(path,data):
 temporary=path.with_name(path.name+'.new')
 temporary.write_bytes(data);os.replace(temporary,path)

state=health()
if state.get('busy') or state.get('pending_requests'):raise RuntimeError(f'Live requests present: {state}')
if not source.is_file() or source.stat().st_size==0:raise RuntimeError(f'Invalid replacement executable: {source}')
config_before=config_path.read_bytes();config=json.loads(config_before)
for value in config['models'].values():
 args=value.get('extra_args',[]);out=[];index=0
 while index<len(args):
  if args[index]=='--draft-residency':index+=2;continue
  if args[index]=='--lazy-draft':index+=1;continue
  out.append(args[index]);index+=1
 if 'extra_args' in value:value['extra_args']=out
config_after=(json.dumps(config,indent=2)+'\n').encode()
backup_dir=root/'backups'/('persistent-draft-'+time.strftime('%Y%m%d-%H%M%S'))
backup_dir.mkdir(parents=True)
binary_backup=backup_dir/'dflash_server';config_backup=backup_dir/'models.json'
shutil.copy2(dest,binary_backup);config_backup.write_bytes(config_before)
staged_binary=dest.with_name(dest.name+'.new');shutil.copy2(source,staged_binary)
guard=f'lucebox-persistent-draft-restore-{os.getpid()}'
subprocess.run(['systemd-run','--unit',guard,'--on-active=15m','--timer-property=AccuracySec=1s','/bin/systemctl','start','lucebox.service'],check=True)
try:
 subprocess.run(['systemctl','stop','lucebox.service'],check=True)
 os.replace(staged_binary,dest);atomic_write(config_path,config_after)
 subprocess.run(['systemctl','start','lucebox.service'],check=True);wait_ready()
 subprocess.run(['systemctl','stop',guard+'.timer'],check=False)
except BaseException:
 subprocess.run(['systemctl','stop','lucebox.service'],check=False)
 rollback_binary=dest.with_name(dest.name+'.rollback');shutil.copy2(binary_backup,rollback_binary);os.replace(rollback_binary,dest)
 atomic_write(config_path,config_before)
 subprocess.run(['systemctl','start','lucebox.service'],check=True);wait_ready()
 subprocess.run(['systemctl','stop',guard+'.timer'],check=False)
 raise
finally:
 if staged_binary.exists():staged_binary.unlink()
print('Installed persistent decoding drafter build; production is healthy; rollback:',backup_dir)
