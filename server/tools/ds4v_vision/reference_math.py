#!/usr/bin/env python3
"""Parent BF16 SDPA Math sensitivity, never a replacement fixture or parity gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from safetensors import safe_open
from torch.nn.attention import sdpa_kernel, SDPBackend

p=argparse.ArgumentParser()
p.add_argument('source',type=Path)
p.add_argument('reference',type=Path)
p.add_argument('native',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.bfloat16)
sys.path.insert(0,str(a.source/'inference'))
from vision import ViT, Aligner
manifest=json.loads((a.reference/'manifest.json').read_text())
for name,digest in manifest['source_hashes'].items():
    assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==digest
config=json.loads((a.source/'config.json').read_text())
config['dim']=config['hidden_size']
vit,aligner=ViT(SimpleNamespace(**config)).eval(),Aligner(SimpleNamespace(**config)).eval()
index=json.loads((a.source/'model.safetensors.index.json').read_text())['weight_map']
for prefix,module in [('vision.',vit),('aligner.',aligner)]:
    state={}
    for shard in sorted({v for k,v in index.items() if k.startswith(prefix)}):
        with safe_open(a.source/shard,framework='pt',device='cpu') as f:
            for name in f.keys():
                if name.startswith(prefix): state[name[len(prefix):]]=f.get_tensor(name)
    module.load_state_dict(state,strict=True)
    del state

def metrics(x,y):
    assert x.shape==y.shape
    x,y=x.astype(np.float64),y.astype(np.float64)
    delta=x-y
    return dict(shape=list(x.shape),finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),max_abs=float(np.abs(delta).max()),rmse=float(np.sqrt(np.mean(delta**2))),cosine=float(np.dot(x.ravel(),y.ravel())/(np.linalg.norm(x)*np.linalg.norm(y))),exact_fraction=float(np.mean(x==y)))

results={'torch':torch.__version__,'backend':'SDPBackend.MATH','precision':'original BF16 modules, math SDPA default F32 reduction','images':{}}
a.output.mkdir(parents=True,exist_ok=True)
for label in ('corn','carrots'):
    entry=manifest['images'][label]
    patches=torch.from_numpy(np.fromfile(a.reference/entry['patches']['file'],np.float32).reshape(entry['patches']['shape'])).to(torch.bfloat16)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        features=vit(patches,*entry['vit_grid'])
        embeddings=aligner(features,*entry['vit_grid'])
    results['images'][label]={}
    for name,tensor in [('features',features),('embeddings',embeddings)]:
        math=tensor.float().contiguous().numpy()
        math.tofile(a.output/f'{label}-{name}.f32')
        original=np.fromfile(a.reference/entry[name]['file'],np.float32).reshape(entry[name]['shape'])
        native=np.fromfile(a.native/f'{label}-{name}.f32',np.float32).reshape(entry[name]['shape'])
        results['images'][label][name]={'math_vs_original_fixture':metrics(math,original),'native_vs_math':metrics(native,math)}
    (a.output/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print(label,json.dumps(results['images'][label]),flush=True)
