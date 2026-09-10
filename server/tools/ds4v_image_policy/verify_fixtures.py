#!/usr/bin/env python3
"""CPU-only comparison against immutable parent routing and raw-mask fixtures."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('probe', type=Path)
p.add_argument('fixtures', type=Path)
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
manifest = json.loads((a.fixtures / 'manifest.json').read_text())
source_file = a.source / 'inference/model.py'
source = source_file.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == manifest['source_sha256']
config = json.loads((a.source / 'inference/config.json').read_text())
assert config['score_func'] not in ('softmax', 'sigmoid')
assert config['route_scale'] == 1.5 and config['n_activated_experts'] == 6

def verify_tree(value):
    if isinstance(value, dict):
        if 'file' in value and 'sha256' in value:
            assert hashlib.sha256((a.fixtures / value['file']).read_bytes()).hexdigest() == value['sha256'], value['file']
        for child in value.values():
            verify_tree(child)

verify_tree(manifest)

def read(meta):
    return np.fromfile(a.fixtures / meta['file'], dtype=meta['dtype']).reshape(meta['shape'])

tree = ast.parse(source)
linear = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'linear')
gate = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Gate')
forward = next(n for n in gate.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
boundary = next(i for i, n in enumerate(forward.body)
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'original_scores' for t in n.targets))
score_function = ast.FunctionDef(name='source_scores',
    args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self'), ast.arg(arg='x')],
                       kwonlyargs=[], kw_defaults=[], defaults=[]),
    body=forward.body[:boundary] + [ast.Return(value=ast.Name(id='scores', ctx=ast.Load()))],
    decorator_list=[])
namespace = {'torch': torch, 'F': F, 'Optional': Optional}
exec(compile(ast.fix_missing_locations(ast.Module(body=[linear, score_function], type_ignores=[])),
             str(source_file), 'exec'), namespace)

report = {'source_sha256': manifest['source_sha256'], 'fixture_hashes': 'PASS',
          'weight_max_absolute_error_limit': 1e-6, 'routing': {}, 'masks': {},
          'integration': 'Unintegrated pure image policy; no text/hash graph or compressed-row changes'}
for layer, entry in manifest['routing'].items():
    router = torch.from_numpy(read(entry['router']).copy())
    inputs = torch.from_numpy(read(entry['input']).copy())
    with torch.inference_mode():
        scores = namespace['source_scores'](SimpleNamespace(weight=router, score_func=config['score_func']), inputs)
    scores_path = a.output / f'layer-{layer}-scores.f32'
    scores.numpy().tofile(scores_path)
    prefix = a.output / f'layer-{layer}'
    subprocess.run([str(a.probe), 'route', str(scores_path), str(a.fixtures / entry['bias_vl']['file']),
                    str(prefix), str(inputs.shape[0]), str(router.shape[0]), '6'], check=True)
    actual_ids = np.fromfile(str(prefix) + '-indices.i32', np.int32).reshape(-1, 6)
    actual_weights = np.fromfile(str(prefix) + '-weights.f32', np.float32).reshape(-1, 6)
    ids = read(entry['ids'])
    image_rows = np.flatnonzero(ids >= config['vocab_size'])
    assert np.array_equal(ids[image_rows], np.arange(config['vocab_size'], config['vocab_size'] + 5))
    expected_ids, expected_weights = read(entry['indices']), read(entry['weights'])
    assert np.array_equal(actual_ids[image_rows], expected_ids[image_rows]), ('indices', layer)
    difference = float(np.max(np.abs(actual_weights[image_rows] - expected_weights[image_rows])))
    assert np.isfinite(actual_weights).all() and difference <= 1e-6, ('weights', layer, difference)
    assert np.max(np.abs(actual_weights[image_rows].sum(1) - 1.5)) <= 1e-6
    assert np.array_equal(actual_ids[image_rows], np.repeat(actual_ids[image_rows[:1]], 5, axis=0))
    hash_difference = None
    if 'hash_rows' in entry:
        hash_rows = read(entry['hash_rows'])
        assert np.array_equal(expected_ids[:3], hash_rows), ('original text hash fixture', layer)
        hash_difference = bool(np.all(np.any(hash_rows != actual_ids[image_rows[0]], axis=1)))
        assert hash_difference, ('fixture must distinguish learned image selection from text hash', layer)
    report['routing'][layer] = {'all_five_image_kinds_exact_indices': True,
        'max_weight_absolute_error': difference, 'image_selection_differs_from_text_hash_rows': hash_difference,
        'scores_sha256': hashlib.sha256(scores_path.read_bytes()).hexdigest()}
    print('routing', layer, 'PASS', difference, flush=True)

for name, entry in manifest['masks'].items():
    ids = read(entry['ids']).reshape(-1)
    count = ids.size
    ranges = np.full((count, 2), -1, np.int64)
    starts = np.flatnonzero(ids == config['vocab_size'])
    ends = np.flatnonzero(ids == config['vocab_size'] + 4)
    assert len(starts) == len(ends)
    for begin, end in zip(starts, ends):
        assert begin <= end and (ranges[begin:end+1] == -1).all()
        ranges[begin:end+1] = [begin, end+1]
    ranges_path = a.output / f'{name}-ranges.i64'
    ranges.tofile(ranges_path)
    output = a.output / f'{name}-mask.u8'
    subprocess.run([str(a.probe), 'mask', str(ranges_path), str(count), str(config['window_size']), str(output)], check=True)
    actual = np.fromfile(output, np.uint8).reshape(count, count)
    expected = np.zeros((count, count), np.uint8)
    raw = read(entry['raw_indices']).reshape(count, -1)
    for query, keys in enumerate(raw):
        valid = keys[keys >= 0]
        assert (valid < count).all()
        expected[query, valid] = 1
    assert np.array_equal(actual, expected), ('raw visibility mismatch', name, np.count_nonzero(actual != expected))
    report['masks'][name] = {'all_key_query_pairs_exact': True, 'comparisons': count*count,
        'longer_than_window': bool(np.any(ends-starts+1 > config['window_size'])),
        'native_mask_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    print('mask', name, 'PASS', count*count, flush=True)

report['verdict'] = 'PASS'
(a.output / 'verdict.json').write_text(json.dumps(report, indent=2) + '\n')
