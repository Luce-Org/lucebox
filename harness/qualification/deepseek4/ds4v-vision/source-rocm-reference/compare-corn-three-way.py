#!/usr/bin/env python3
"""CPU-only three-way comparison of already completed corn output files."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

assert sys.flags.isolated == 1
for key in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'):
    assert os.environ.get(key) == '-1'
reference, source, native, output = map(Path, sys.argv[1:])
assert not output.exists()
manifest_bytes = (reference/'manifest.json').read_bytes()
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(manifest_bytes) == '38d12f30b9a10ed2f0e99bf4d40a07357ab5e8c4880151c58cb90414e4a53f4f'
manifest = json.loads(manifest_bytes)
source_report = json.loads((source/'report.json').read_text())
assert source_report['script_sha256'] == 'cd34fd3df07963bea9014ccc6f405ec95400a7b8b70e4233fe5d3ead7dc9468c'
assert source_report['device']['name'] == 'Radeon RX 7900 XT'
assert source_report['device']['gcn_arch'].split(':')[0] == 'gfx1100'
assert source_report['image'] == 'corn'
native_sha = {'features':'59bd19a13750d07f7f1018c32c5a43c4ae2cd7a7ff132da3f6201b08c400cc4e',
              'embeddings':'a398c9c10a7b2bbeb63f4b910bf5f71bb9fefd0aa388b276a3bf06ca3e280a06'}
gates = {'features':{'max_abs':.25,'rmse':.03,'cosine':.9995},
         'embeddings':{'max_abs':.75,'rmse':.08,'cosine':.9990}}
report = {'diagnostic_only':True, 'native_acceptance':'NOT_QUALIFIED', 'image':'corn',
          'script_sha256':sha(Path(__file__).read_bytes()), 'hashes':{}, 'comparisons':{}}
for stage,gate in gates.items():
    expected = manifest['images']['corn'][stage]
    raw_cpu = (reference/expected['file']).read_bytes()
    raw_source = (source/source_report['outputs'][stage]['file']).read_bytes()
    raw_native = (native/('corn-'+stage+'.f32')).read_bytes()
    assert sha(raw_cpu) == expected['sha256']
    assert sha(raw_source) == source_report['outputs'][stage]['sha256']
    assert sha(raw_native) == native_sha[stage]
    values = {name:np.frombuffer(raw,np.float32).reshape(expected['shape'])
              for name,raw in [('original_cpu',raw_cpu),('source_hip',raw_source),('native_hip',raw_native)]}
    report['hashes'][stage] = {'original_cpu':sha(raw_cpu),'source_hip':sha(raw_source),'native_hip':sha(raw_native)}
    report['comparisons'][stage] = {}
    for name,left,right in [('source_hip_vs_original_cpu','source_hip','original_cpu'),
                            ('native_hip_vs_source_hip','native_hip','source_hip'),
                            ('native_hip_vs_original_cpu','native_hip','original_cpu')]:
        a,b=values[left].astype(np.float64).ravel(),values[right].astype(np.float64).ravel()
        d=a-b
        row={'shape':expected['shape'],'finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),
             'max_abs':float(np.abs(d).max()),'rmse':float(np.sqrt(np.mean(d*d))),
             'cosine':float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),
             'exact_fraction':float(np.mean(a==b)),
             'byte_identical':report['hashes'][stage][left]==report['hashes'][stage][right], 'gate':gate}
        row['pass']=row['finite'] and row['max_abs']<=gate['max_abs'] and row['rmse']<=gate['rmse'] and row['cosine']>=gate['cosine']
        report['comparisons'][stage][name]=row
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
