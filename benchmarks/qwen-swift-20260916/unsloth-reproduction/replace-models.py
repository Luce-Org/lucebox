import json,pathlib,subprocess,hashlib,shutil,time,urllib.request
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
root=pathlib.Path('/opt/lucebox/work/unsloth-reproduction-20260915');root.mkdir(exist_ok=True)
models=pathlib.Path('/code/models/qwen38');prod=pathlib.Path('/opt/lucebox');api=HfApi()
target_repo='unsloth/Qwen3.8-27B-GGUF';draft_repo='incoai/Qwen3.8-27B-DFlash2'
target_rev=api.model_info(target_repo).sha;draft_rev=api.model_info(draft_repo).sha
try:
 with urllib.request.urlopen('http://127.0.0.1:8216/health') as r:health=json.load(r)
except urllib.error.HTTPError as e:health=json.load(e)
if health.get('busy') or health.get('pending_requests'):raise RuntimeError('Live requests present; no files removed')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
manifest={'target_repo':target_repo,'target_revision':target_rev,'draft_repo':draft_repo,'draft_revision':draft_rev,'removed':[]}
shutil.copy2(prod/'models.json',root/'models.before.json')
subprocess.run(['systemctl','stop','lucebox.service'],check=True)
for name in ['Qwen3.8-27B-UD-IQ4_XS.gguf','qwen38-dflash2-q8_0.gguf','dflash-Qwen3.8-27B-Q8_0.gguf']:
 p=models/name
 if p.exists():
  manifest['removed'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)});p.unlink();print('Removed',p,flush=True)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Downloading target',target_rev,flush=True)
hf_hub_download(target_repo,'Qwen3.8-27B-UD-IQ4_XS.gguf',revision=target_rev,local_dir=models,force_download=True)
print('Downloading source drafter',draft_rev,flush=True)
snapshot_download(draft_repo,revision=draft_rev,local_dir=models/'dflash2',force_download=True,allow_patterns=['*.safetensors','config.json','README.md'])
py='/opt/lucebox-python/bin/python';scripts=pathlib.Path('/opt/lucebox-concurrent/server/scripts')
subprocess.run([py,str(scripts/'convert_dflash_to_gguf.py'),str(models/'dflash2/model.safetensors'),str(models/'qwen38-dflash2-f16.gguf')],check=True)
subprocess.run([py,str(scripts/'quantize_dflash_draft.py'),str(models/'qwen38-dflash2-f16.gguf'),str(models/'qwen38-dflash2-q8_0.gguf'),'--scheme','q8_0'],check=True)
manifest['installed']=[{'path':str(models/n),'bytes':(models/n).stat().st_size,'sha256':sha(models/n)} for n in ['Qwen3.8-27B-UD-IQ4_XS.gguf','qwen38-dflash2-q8_0.gguf']]
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
config=json.loads((prod/'models.json').read_text());m=config['models'][config['default']]
m.update({'path':str(models/'Qwen3.8-27B-UD-IQ4_XS.gguf'),'draft':str(models/'qwen38-dflash2-q8_0.gguf'),'draft_block_size':16,'context':131072,'max_concurrency':1,'hybrid_cache':'off','hybrid_projections':False,'extra_args':[],'cache_default_policy':'exact'})
(prod/'models.json').write_text(json.dumps(config,indent=2)+'\n')
subprocess.run(['systemctl','start','lucebox.service'],check=True)
print('Conversion complete; service started',flush=True)
