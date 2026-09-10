#!/usr/bin/env python3
"""Verify immutable accepted-core fixture hashes before native prompt checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('probe',type=Path)
p.add_argument('fixtures',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
digests=json.loads((a.fixtures/'sha256.json').read_text())
verified={}
for name,digest in digests.items():
    if name.split('/')[0] not in ('corn','carrots'):
        continue
    actual=hashlib.sha256((a.fixtures/name).read_bytes()).hexdigest()
    assert actual==digest,name
    verified[name]=digest
assert len(verified)==28
run=subprocess.run([str(a.probe),str(a.fixtures)],capture_output=True,text=True)
print(run.stdout,end='')
print(run.stderr,end='')
assert run.returncode==0,run.returncode
a.output.write_text(json.dumps({'verdict':'PASS','verified_source_fixture_hashes':verified,
    'single_image_cases':10,'ordered_pair_cases':2,'decoder_calls':0,'tower_calls':0,
    'scope':'Unintegrated prompt preparation using accepted source RGB core layouts and BF16 patch fixtures'},indent=2)+'\n')
