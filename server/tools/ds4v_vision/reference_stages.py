#!/usr/bin/env python3
"""Run the unmodified parent modules with observation hooks and compare stages.

Use only with the original CPU fixture environment. Two Torch threads; no GPU.
Outputs metrics, and confirms instrumentation reproduces original final fixtures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from safetensors import safe_open

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('reference', type=Path)
p.add_argument('native', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.bfloat16)
sys.path.insert(0, str(a.source / 'inference'))
import vision
manifest = json.loads((a.reference / 'manifest.json').read_text())
for name, expected in manifest['source_hashes'].items():
    assert hashlib.sha256((a.source / name).read_bytes()).hexdigest() == expected, name
config = json.loads((a.source / 'config.json').read_text())
config['dim'] = config['hidden_size']
args = SimpleNamespace(**config)
vit, aligner = vision.ViT(args).eval(), vision.Aligner(args).eval()
index = json.loads((a.source / 'model.safetensors.index.json').read_text())['weight_map']
for prefix, module in [('vision.', vit), ('aligner.', aligner)]:
    state = {}
    for shard in sorted({v for k, v in index.items() if k.startswith(prefix)}):
        with safe_open(a.source / shard, framework='pt', device='cpu') as f:
            for name in f.keys():
                if name.startswith(prefix):
                    state[name[len(prefix):]] = f.get_tensor(name)
    module.load_state_dict(state, strict=True)
    del state
results = {}
label = None
mode = "original"

def compare(stage, value):
    reference = value.detach().float().contiguous().numpy()
    native_path = a.native / f'{label}-{stage}.f32'
    actual = np.fromfile(native_path, np.float32).reshape(reference.shape)
    x, y = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    delta = x-y
    metrics = dict(shape=list(reference.shape), finite=bool(np.isfinite(x).all()), max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta**2))), cosine=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))), exact_fraction=float(np.mean(x==y)))
    # Monotone BF16 bit ordering gives ULP distances, including negatives.
    def ordered_bf16(value):
        bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32) >> 16
        return np.where(bits & 0x8000, 0x8000 - (bits & 0x7fff), 0x8000 + bits).astype(np.int32)
    ulps = np.abs(ordered_bf16(x)-ordered_bf16(y))
    metrics.update(bf16_ulp_max=int(ulps.max()), bf16_ulp_p99=float(np.percentile(ulps,99)), bf16_within_one_ulp=float(np.mean(ulps<=1)))
    results[label][stage if mode == 'original' else mode + '.' + stage] = metrics
    print(label, mode, stage, json.dumps(metrics), flush=True)
    a.output.write_text(json.dumps(results, indent=2)+'\n')

hooks=[]
for module, name in [(vit.patch_embed,'patch_embed'), (vit.blocks[0].norm1,'block0.norm1'),
                     (vit.blocks[0].attn.wqkv,'block0.qkv'), (vit.norm,'features'),
                     (aligner.w1,'aligner.w1'), (aligner.w2,'embeddings')]:
    hooks.append(module.register_forward_hook(lambda m, inp, out, name=name: compare(name,out)))
for i, block in enumerate(vit.blocks):
    hooks.append(block.register_forward_hook(lambda m, inp, out, i=i: compare(f'block{i}',out)))
hooks.append(aligner.w1.register_forward_pre_hook(lambda m, inp: compare('unfold',inp[0])))
hooks.append(aligner.w2.register_forward_pre_hook(lambda m, inp: compare('aligner.gelu',inp[0])))
original_rotary = vision.apply_rotary
original_sdpa = vision.F.scaled_dot_product_attention
rotary_count = 0
attention_count = 0

def rotary(*args, **kwargs):
    global rotary_count
    out = original_rotary(*args, **kwargs)
    if rotary_count < 2:
        compare('block0.q' if rotary_count == 0 else 'block0.k', out)
    rotary_count += 1
    return out

def sdpa(*args, **kwargs):
    global attention_count
    out = original_sdpa(*args, **kwargs)
    if attention_count == 0:
        compare('block0.attention',out.transpose(0,1).reshape(out.shape[1],-1))
    attention_count += 1
    return out

vision.apply_rotary = rotary
vision.F.scaled_dot_product_attention = sdpa
for label in ('corn','carrots'):
    results[label] = {}
    mode = 'original'
    rotary_count = attention_count = 0
    meta = manifest['images'][label]
    patches = torch.from_numpy(np.fromfile(a.reference / meta['patches']['file'],np.float32).reshape(meta['patches']['shape'])).to(torch.bfloat16)
    with torch.inference_mode():
        features = vit(patches,*meta['vit_grid'])
        embeddings = aligner(features,*meta['vit_grid'])
    for name,value in [('features',features),('embeddings',embeddings)]:
        original = np.fromfile(a.reference / meta[name]['file'],np.float32).reshape(meta[name]['shape'])
        assert np.array_equal(original,value.float().numpy()), f'{label} instrumented parent {name} changed'
    results[label]['original_fixture_bitwise_match'] = True
    a.output.write_text(json.dumps(results,indent=2)+'\n')

    # Isolate kernel/rounding effects: every parent block receives the exact
    # native preceding residual, avoiding cumulative differences from earlier blocks.
    mode = 'same_input'
    cos, sin = vision.get_vision_cos_sin(*meta['vit_grid'], vit.rope_dim, vit.rope_theta)
    with torch.inference_mode():
        for i, block in enumerate(vit.blocks):
            previous = 'patch_embed' if i == 0 else f'block{i-1}'
            native_input = torch.from_numpy(np.fromfile(a.native / f'{label}-{previous}.f32', np.float32).reshape(-1,1024)).to(torch.bfloat16)
            rotary_count = attention_count = 0 if i == 0 else 100
            block(native_input,cos,sin)
        native_last = torch.from_numpy(np.fromfile(a.native / f'{label}-block31.f32',np.float32).reshape(-1,1024)).to(torch.bfloat16)
        vit.norm(native_last)
        native_features = torch.from_numpy(np.fromfile(a.native / f'{label}-features.f32',np.float32).reshape(-1,1024)).to(torch.bfloat16)
        aligner(native_features,*meta['vit_grid'])
    a.output.write_text(json.dumps(results,indent=2)+'\n')
