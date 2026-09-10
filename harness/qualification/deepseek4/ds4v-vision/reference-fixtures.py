#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--encode', action='store_true')
args = parser.parse_args()
import torch
from safetensors import safe_open

torch.set_num_threads(2)
torch.set_default_dtype(torch.bfloat16)
sys.path.insert(0, str(args.source / 'inference'))
from image_processor import load_image, build_image_block
from vision import ViT, Aligner

config = json.loads((args.source / 'config.json').read_text())
config['dim'] = config['hidden_size']
model_args = SimpleNamespace(**config)
args.output.mkdir(parents=True, exist_ok=True)
manifest = {'torch': torch.__version__, 'config': config, 'images': {}, 'source_hashes': {}}
for name in ['config.json', 'inference/vision.py', 'inference/image_processor.py']:
    manifest['source_hashes'][name] = hashlib.sha256((args.source / name).read_bytes()).hexdigest()

def save(name, tensor):
    path = args.output / name
    tensor.detach().to(torch.float32 if tensor.is_floating_point() else torch.int64).contiguous().numpy().tofile(path)
    return {'file': name, 'shape': list(tensor.shape), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

if args.encode:
    vit, aligner = ViT(model_args).eval(), Aligner(model_args).eval()
    index = json.loads((args.source / 'model.safetensors.index.json').read_text())['weight_map']
    for prefix, module in [('vision.', vit), ('aligner.', aligner)]:
        state = {}
        for shard in sorted({v for k,v in index.items() if k.startswith(prefix)}):
            with safe_open(args.source / shard, framework='pt', device='cpu') as f:
                for name in f.keys():
                    if name.startswith(prefix):
                        state[name[len(prefix):]] = f.get_tensor(name)
        module.load_state_dict(state, strict=True)
        del state

for label in ['carrots', 'corn']:
    path = args.source / 'inference/examples/images' / f'{label}.jpeg'
    patches, vh, vw, lh, lw = load_image({'url': str(path)}, model_args)
    item = {'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'vit_grid': [vh,vw], 'aligner_grid': [lh,lw], 'patches': save(f'{label}-patches.f32', patches), 'layouts': {}}
    for start in [0,1,2,3,127]:
        types, perm = build_image_block(lh,lw,start)
        item['layouts'][str(start)] = {'types': save(f'{label}-{start}-types.i64', types), 'perm': save(f'{label}-{start}-perm.i64', perm)}
    if args.encode:
        with torch.inference_mode():
            features = vit(patches,vh,vw)
            embeddings = aligner(features,vh,vw)
        assert torch.isfinite(features).all() and torch.isfinite(embeddings).all()
        item['features'] = save(f'{label}-features.f32',features)
        item['embeddings'] = save(f'{label}-embeddings.f32',embeddings)
    manifest['images'][label] = item
    (args.output / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(label, 'PASS', 'ViT grid', vh,vw, 'aligner grid',lh,lw, flush=True)
