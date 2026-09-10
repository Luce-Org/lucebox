#!/usr/bin/env python3
"""Hash source fixtures before executing the CPU-only composition probe."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('binary',type=Path)
p.add_argument('tokenizer_gguf',type=Path)
p.add_argument('fixtures',type=Path)
p.add_argument('verdict',type=Path)
a=p.parse_args()

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

expected=json.loads((a.fixtures/'sha256.json').read_text())
verified={}
for name,sha in expected.items():
    if name.split('/')[0] in ('corn','carrots'):
        assert digest(a.fixtures/name)==sha,name
        verified[name]=sha
assert len(verified)==28
run=subprocess.run([str(a.binary),str(a.tokenizer_gguf),str(a.fixtures)],capture_output=True,text=True)
print(run.stdout,end='')
print(run.stderr,end='')
assert run.returncode==0,run.returncode
a.verdict.write_text(json.dumps({'verdict':'PASS','source_fixture_hashes':verified,
    'binary_sha256':digest(a.binary),'tokenizer_smoke_gguf_sha256':digest(a.tokenizer_gguf),
    'scope':'CPU transport/decode/preprocess/render/tokenize/prompt composition with explicit probe adapter; no HTTP, backend, or tower'},indent=2)+'\n')
