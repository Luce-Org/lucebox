import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

root=pathlib.Path('/opt/lucebox/work/unsloth-reproduction-20260915');root.mkdir(exist_ok=True)
models=pathlib.Path('/code/models/qwen38');prod=pathlib.Path('/opt/lucebox');api=HfApi()
target_repo='unsloth/Qwen3.8-27B-GGUF';draft_repo='incoai/Qwen3.8-27B-DFlash2'
target_rev=api.model_info(target_repo).sha;draft_rev=api.model_info(draft_repo).sha

def health(timeout=5):
 try:
  with urllib.request.urlopen('http://127.0.0.1:8216/health',timeout=timeout) as response:return json.load(response)
 except urllib.error.HTTPError as error:return json.load(error)

def require_idle():
 state=health()
 if state.get('busy') or state.get('pending_requests'):raise RuntimeError(f'Live requests present: {state}')

def wait_ready(timeout=180):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  try:
   state=health(timeout=2)
   if state.get('status')=='ok' and not state.get('busy') and not state.get('pending_requests'):return state
  except Exception:pass
  time.sleep(1)
 raise RuntimeError('Production Lucebox did not become healthy and idle')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def atomic_write(path,data):
 temporary=path.with_name(path.name+'.new')
 temporary.write_bytes(data);os.replace(temporary,path)

def start_and_wait():
 subprocess.run(['systemctl','start','lucebox.service'],check=True);wait_ready()

require_idle()
stamp=time.strftime('%Y%m%d-%H%M%S')
stage=models/f'.unsloth-stage-{stamp}'
backup=models/f'.unsloth-rollback-{stamp}'
stage.mkdir();backup.mkdir()
config_path=prod/'models.json';config_before=config_path.read_bytes()
manifest={'target_repo':target_repo,'target_revision':target_rev,'draft_repo':draft_repo,'draft_revision':draft_rev,'replaced':[],'stage':str(stage),'rollback':str(backup)}
(root/'models.before.json').write_bytes(config_before)

# Download and convert while production remains available. Nothing in the live
# model directory is changed until every staged output has been validated.
try:
 print('Downloading target',target_rev,flush=True)
 hf_hub_download(target_repo,'Qwen3.8-27B-UD-IQ4_XS.gguf',revision=target_rev,local_dir=stage,force_download=True)
 print('Downloading source drafter',draft_rev,flush=True)
 snapshot_download(draft_repo,revision=draft_rev,local_dir=stage/'dflash2',force_download=True,allow_patterns=['*.safetensors','config.json','README.md'])
 py='/opt/lucebox-python/bin/python';scripts=pathlib.Path('/opt/lucebox-concurrent/server/scripts')
 subprocess.run([py,str(scripts/'convert_dflash_to_gguf.py'),str(stage/'dflash2/model.safetensors'),str(stage/'qwen38-dflash2-f16.gguf')],check=True)
 subprocess.run([py,str(scripts/'quantize_dflash_draft.py'),str(stage/'qwen38-dflash2-f16.gguf'),str(stage/'qwen38-dflash2-q8_0.gguf'),'--scheme','q8_0'],check=True)
 installed_names=['Qwen3.8-27B-UD-IQ4_XS.gguf','qwen38-dflash2-q8_0.gguf']
 for name in installed_names:
  path=stage/name
  if not path.is_file() or path.stat().st_size==0:raise RuntimeError(f'Invalid staged model: {path}')
 manifest['installed']=[{'path':str(models/name),'bytes':(stage/name).stat().st_size,'sha256':sha(stage/name)} for name in installed_names]
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
except BaseException:
 shutil.rmtree(stage,ignore_errors=True);shutil.rmtree(backup,ignore_errors=True);raise

config=json.loads(config_before);entry=config['models'][config['default']]
entry.update({'path':str(models/'Qwen3.8-27B-UD-IQ4_XS.gguf'),'draft':str(models/'qwen38-dflash2-q8_0.gguf'),'draft_block_size':16,'context':131072,'max_concurrency':1,'hybrid_cache':'off','hybrid_projections':False,'extra_args':[],'cache_default_policy':'exact'})
config_after=(json.dumps(config,indent=2)+'\n').encode()
live_names=installed_names+['dflash-Qwen3.8-27B-Q8_0.gguf']
moved=[];published=[];service_stopped=False
guard=f'lucebox-model-replacement-restore-{os.getpid()}'
subprocess.run(['systemd-run','--unit',guard,'--on-active=45m','--timer-property=AccuracySec=1s','/bin/systemctl','start','lucebox.service'],check=True)
try:
 require_idle();subprocess.run(['systemctl','stop','lucebox.service'],check=True);service_stopped=True
 for name in live_names:
  live=models/name
  if live.exists():
   saved=backup/name;os.replace(live,saved);moved.append((live,saved))
   manifest['replaced'].append({'path':str(live),'bytes':saved.stat().st_size,'sha256':sha(saved)})
 for name in installed_names:
  live=models/name;os.replace(stage/name,live);published.append(live)
 atomic_write(config_path,config_after)
 start_and_wait();service_stopped=False
 subprocess.run(['systemctl','stop',guard+'.timer'],check=False)
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print('Conversion complete; service healthy; rollback retained at',backup,flush=True)
except BaseException:
 # Reconstitute the exact previous config and model bytes before recovery.
 if not service_stopped:
  subprocess.run(['systemctl','stop','lucebox.service'],check=False);service_stopped=True
 atomic_write(config_path,config_before)
 for live in published:
  if live.exists():live.unlink()
 for live,saved in reversed(moved):
  if saved.exists():os.replace(saved,live)
 start_and_wait();service_stopped=False
 subprocess.run(['systemctl','stop',guard+'.timer'],check=False)
 raise
finally:
 if service_stopped:
  subprocess.run(['systemctl','start','lucebox.service'],check=False)
 shutil.rmtree(stage,ignore_errors=True)
