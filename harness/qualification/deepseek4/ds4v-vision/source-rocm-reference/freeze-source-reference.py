#!/usr/bin/env python3
"""Freeze first source-HIP outputs after exact repeat validation; CPU-only file work."""
import datetime
import filecmp
import hashlib
import json
from pathlib import Path
import shutil

home = Path.home()
root = home/'ds4v-work/source-rocm210-reference'
cpu = home/'lucebox-ds4v-mix-fix/artifacts/vision-reference'
canonical = root/'source-hip-reference'
freeze_path = root/'source-hip-reference-freeze.json'
assert not canonical.exists() and not freeze_path.exists()

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text())

policy = {'reviewed':(root/'reference-policy-reviewed.md','7edde20ee70b804cc903b827dbea1dbc9b8d43d9d352e22f82672758430f9682'),
          'adopted':(root/'reference-policy-adopted.md','62b3daef05bcaaac175f6907b8748d64952a4bf96baab316bab9b2140451929f')}
for path, expected in policy.values():
    assert digest(path) == expected
assert [x for x in policy['reviewed'][0].read_text().splitlines() if not x.startswith('Status:')] == \
       [x for x in policy['adopted'][0].read_text().splitlines() if not x.startswith('Status:')]
assert digest(cpu/'manifest.json') == '38d12f30b9a10ed2f0e99bf4d40a07357ab5e8c4880151c58cb90414e4a53f4f'
original = read(cpu/'manifest.json')
runner = root/'source-forward-radeon-name.py'
assert digest(runner) == 'cd34fd3df07963bea9014ccc6f405ec95400a7b8b70e4233fe5d3ead7dc9468c'
pairs = {'corn':('hip-corn-confirmed','hip-supervision-confirmed','source-corn-repeat','source-corn-repeat-supervision'),
         'carrots':('source-carrots-first','source-carrots-first-supervision','source-carrots-repeat','source-carrots-repeat-supervision')}
gates = {'features':{'max_abs':.25,'rmse':.03,'cosine':.9995},
         'embeddings':{'max_abs':.75,'rmse':.08,'cosine':.9990}}
freeze = {'status':'SOURCE_REFERENCE_STABILITY_PASS', 'candidate_acceptance':'NOT_EVALUATED',
          'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'policy':{k:{'file':str(p),'sha256':h} for k,(p,h) in policy.items()},
          'original_cpu_manifest':{'file':str(cpu/'manifest.json'),'sha256':digest(cpu/'manifest.json')},
          'runner':{'file':str(runner),'sha256':digest(runner)}, 'gates':gates, 'images':{},
          'reference_selection':'FIRST completed source corn and FIRST source carrots outputs; repeats only validate stability',
          'scope':'Original source on this RX7900XT/software configuration; CPU portability remains separate'}
manifest = {'torch':'2.10.0+rocm7.2.4.git3d3aa833', 'reference_kind':'original_source_hip_7900xt',
            'config':original['config'], 'source_hashes':original['source_hashes'], 'images':{},
            'policy_sha256':policy['adopted'][1], 'original_cpu_manifest_sha256':freeze['original_cpu_manifest']['sha256']}
all_libraries = set()
for image,(first_name,first_supervision,repeat_name,repeat_supervision) in pairs.items():
    first,repeat = root/first_name,root/repeat_name
    reports = [read(first/'report.json'),read(repeat/'report.json')]
    runs = [read(root/first_supervision/'run.json'),read(root/repeat_supervision/'run.json')]
    for report,run in zip(reports,runs):
        assert run['exit'] == 0 and not run.get('timed_out') and not run.get('error')
        assert report['script_sha256'] == freeze['runner']['sha256']
        assert report['source_hashes'] == original['source_hashes'] and report['image'] == image
        assert report['reference_manifest_sha256'] == freeze['original_cpu_manifest']['sha256']
        assert report['weight_inventory_sha256'] == 'b0556c40a8bff3f4c2c262d57137a97123cbdbf7444a7fae495ef17cd28469ee'
        assert report['device']['name'] == 'Radeon RX 7900 XT' and report['device']['gcn_arch'].split(':')[0] == 'gfx1100'
        assert report['device']['rocr_visible_device'] == 'GPU-93a97448a27aeff3'
        assert report['torch_git'] == '3d3aa833db84eed6b7f5595cb5f162c2f78300a4' and report['torch_hip'] == '7.2.53211'
        assert report['threads'] == [2,2] and report['default_dtype'] == 'torch.bfloat16'
        all_libraries.update(report['loaded_libraries'])
    item = {key:original['images'][image][key] for key in ('image_sha256','vit_grid','aligner_grid','patches')}
    assert digest(cpu/item['patches']['file']) == item['patches']['sha256']
    evidence = {'first_directory':str(first),'repeat_directory':str(repeat),
                'first_report_sha256':digest(first/'report.json'),'repeat_report_sha256':digest(repeat/'report.json'),
                'first_run':runs[0],'repeat_run':runs[1], 'hardware':reports[0]['device'],
                'outputs':{}, 'cpu_portability':{'first':reports[0]['comparisons'],'repeat':reports[1]['comparisons']}}
    for stage in gates:
        a,b = reports[0]['outputs'][stage],reports[1]['outputs'][stage]
        assert a['shape'] == b['shape'] == original['images'][image][stage]['shape']
        assert digest(first/a['file']) == a['sha256'] and digest(repeat/b['file']) == b['sha256']
        assert a['sha256'] == b['sha256'] and filecmp.cmp(first/a['file'],repeat/b['file'],shallow=False)
        for report in reports:
            assert report['comparisons'][stage]['gate'] == gates[stage] and report['comparisons'][stage]['finite']
        item[stage] = a.copy()
        evidence['outputs'][stage] = {'first_sha256':a['sha256'],'repeat_sha256':b['sha256'],'byte_identical':True}
    freeze['images'][image] = evidence
    manifest['images'][image] = item

# These hashes capture the actual loaded shared-library set, not a guessed loader path.
freeze['loaded_libraries'] = [{'file':name,'sha256':digest(Path(name)),'bytes':Path(name).stat().st_size}
                              for name in sorted(all_libraries)]
freeze['provenance_files'] = {name:{'file':str(root/name),'sha256':digest(root/name)} for name in (
    'requirements.lock','constraints.txt','evidence/wheels.json','evidence/install-report.json',
    'evidence/cpu-runtime-private.json','evidence/miopen-package.json','evidence/miopen-library.json',
    'evidence/freeze.txt','reference-policy-reviewed.md','reference-policy-adopted.md',
    'run-hip-confirmed-supervised.py','run-source-stability-supervised.py','compare-corn-three-way.py')}
freeze['wheel_inventory'] = read(root/'evidence/wheels.json')
freeze['miopen_package'] = read(root/'evidence/miopen-package.json')
freeze['miopen_library'] = read(root/'evidence/miopen-library.json')
freeze['source_weights'] = {'inventory_sha256':reports[0]['weight_inventory_sha256'],
                            'index_sha256':reports[0]['index_sha256'],'source_hashes':original['source_hashes']}
freeze['freeze_script_sha256'] = digest(Path(__file__))
canonical.mkdir()
for image,item in manifest['images'].items():
    first = Path(freeze['images'][image]['first_directory'])
    for stage in ('patches','features','embeddings'):
        entry = item[stage]
        src = cpu/entry['file'] if stage == 'patches' else first/entry['file']
        dst = canonical/entry['file']
        shutil.copyfile(src,dst)
        assert digest(dst) == entry['sha256']
        dst.chmod(0o444)
(canonical/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(canonical/'manifest.json').chmod(0o444)
freeze['canonical_reference'] = {'directory':str(canonical),'manifest_sha256':digest(canonical/'manifest.json')}
freeze_path.write_text(json.dumps(freeze,indent=2)+'\n')
freeze_path.chmod(0o444)
print(json.dumps({'canonical_directory':str(canonical),'manifest_sha256':digest(canonical/'manifest.json'),
                  'freeze_manifest':str(freeze_path),'freeze_manifest_sha256':digest(freeze_path),
                  'status':freeze['status'],'frozen_utc':freeze['frozen_utc']},indent=2))
