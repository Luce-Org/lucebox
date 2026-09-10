#!/usr/bin/env python3
"""Compare native raster tensors against independently generated parent fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

GATES = {
    'features': dict(max_abs=0.25, rmse=0.03, cosine=0.9995),
    'embeddings': dict(max_abs=0.75, rmse=0.08, cosine=0.9990),
}

p = argparse.ArgumentParser()
p.add_argument('reference', type=Path)
p.add_argument('native', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = json.loads((a.reference / 'manifest.json').read_text())
results = {}
passed = True
for label, entry in manifest['images'].items():
    results[label] = {}
    patches = entry['patches']
    assert hashlib.sha256((a.reference / patches['file']).read_bytes()).hexdigest() == patches['sha256']
    for stage in ('features', 'embeddings'):
        meta = entry[stage]
        reference_path = a.reference / meta['file']
        assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == meta['sha256']
        ref = np.fromfile(reference_path, np.float32)
        actual = np.fromfile(a.native / f'{label}-{stage}.f32', np.float32)
        assert actual.size == ref.size == np.prod(meta['shape']), (label, stage, 'shape mismatch')
        finite = bool(np.isfinite(actual).all())
        delta = actual.astype(np.float64) - ref.astype(np.float64)
        cosine = np.dot(actual.astype(np.float64), ref.astype(np.float64)) / (np.linalg.norm(actual.astype(np.float64)) * np.linalg.norm(ref.astype(np.float64)))
        results[label][stage] = dict(shape=meta['shape'], finite=finite, max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta ** 2))), cosine=float(cosine), exact_fraction=float(np.mean(actual==ref)))
        measured = results[label][stage]
        gate = GATES[stage]
        measured['gate'] = gate
        measured['pass'] = (finite and measured['max_abs'] <= gate['max_abs'] and
                            measured['rmse'] <= gate['rmse'] and measured['cosine'] >= gate['cosine'])
        passed = passed and measured['pass']
a.output.write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
raise SystemExit(0 if passed else 3)
