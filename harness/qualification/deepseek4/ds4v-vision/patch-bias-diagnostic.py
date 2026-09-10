#!/usr/bin/env python3
"""Independent exact dyadic patch-projection rounding diagnostic; CPU only."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import numpy as np
import torch
from safetensors import safe_open

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
from vision import PatchEmbed
manifest=json.loads((a.reference/'manifest.json').read_text())
for name,digest in manifest['source_hashes'].items():
    assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==digest
config=json.loads((a.source/'config.json').read_text())
config['dim']=config['hidden_size']
module=PatchEmbed(SimpleNamespace(**config)).eval()
index=json.loads((a.source/'model.safetensors.index.json').read_text())['weight_map']
state={}
for leaf in ('weight','bias'):
    name='vision.patch_embed.proj.'+leaf
    with safe_open(a.source/index[name],framework='pt',device='cpu') as f:
        state['proj.'+leaf]=f.get_tensor(name)
module.load_state_dict(state,strict=True)
weights=module.proj.weight.detach().float().numpy()
bias=module.proj.bias.detach().float().numpy()

report={'torch':torch.__version__,'threads':2,'images':{}}
a.output.mkdir(parents=True,exist_ok=True)
for label in ('corn','carrots'):
    meta=manifest['images'][label]['patches']
    patches=torch.from_numpy(np.fromfile(a.reference/meta['file'],np.float32).reshape(meta['shape'])).to(torch.bfloat16)
    with torch.inference_mode():
        fused=module(patches)
        unbiased=torch.nn.functional.linear(patches.flatten(1),module.proj.weight,None)
        separate=unbiased+module.proj.bias
    original=fused.float().numpy()
    split=separate.float().numpy()
    native=np.fromfile(a.native/f'{label}-patch_embed.f32',np.float32).reshape(original.shape)
    mismatch=native!=original
    report['images'][label]={'shape':list(original.shape),'fused_dtype':str(fused.dtype),'unbiased_dtype':str(unbiased.dtype),'separate_dtype':str(separate.dtype),'elements':int(original.size),'fused_vs_separate_differing':int(np.sum(original!=split)),'native_vs_fused_differing':int(np.sum(mismatch)),'native_vs_separate_differing':int(np.sum(native!=split)),'on_native_fused_disagreements_separate_matches_native':int(np.sum(mismatch & (split==native))),'on_native_fused_disagreements_separate_matches_fused':int(np.sum(mismatch & (split==original))),'fused_vs_separate_max_abs':float(np.max(np.abs(original-split))),'first_differing_element_fused':float(original.ravel()[np.flatnonzero(mismatch)[0]]),'first_differing_element_native':float(native.ravel()[np.flatnonzero(mismatch)[0]]),'first_differing_element_separate':float(split.ravel()[np.flatnonzero(mismatch)[0]])}
(a.output/'patch-bias.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
