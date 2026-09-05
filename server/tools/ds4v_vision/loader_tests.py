#!/usr/bin/env python3
"""Malformed GGUF table tests; rejected before allocating the sparse payload."""
import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import export_ds4v_mmproj as exporter
p=argparse.ArgumentParser()
p.add_argument('probe',type=Path)
p.add_argument('projector',type=Path)
a=p.parse_args()
tensors=[exporter.SourceTensor(name,Path('/unused'),shape,'BF16',0,2*__import__('math').prod(shape),0,0,0,0)
         for name,shape in sorted(exporter.expected_shapes().items())]
original_metadata=exporter.metadata
cases=[('schema', 'deepseek4.vision.schema_version',2,'unsupported metadata'),
       ('recipe','deepseek4.vision.image.layout_recipe','unknown','unsupported metadata'),
       ('vocabulary','deepseek4.vision.vocabulary_size',1,'unsupported metadata'),
       ('dimension','deepseek4.vision.language_embedding_length',1,'unsupported metadata'),
       ('rope','deepseek4.vision.attention.rope_layout','adjacent','unsupported metadata'),
       ('gelu','deepseek4.vision.aligner.activation','gelu-tanh','unsupported metadata')]
with tempfile.TemporaryDirectory(prefix='ds4v-loader-') as temporary:
    root=Path(temporary)
    for label,key,value,error in cases+[('missing',None,None,'wrong projector tensor count'),
                                      ('shape',None,None,'wrong tensor shape'),
                                      ('unknown',None,None,'unknown or duplicate tensor'),
                                      ('truncated',None,None,'tensor outside file'),
                                      ('dtype',None,None,'wrong tensor dtype'),
                                      ('metadata_type',None,None,'missing or wrong metadata type'),
                                      ('missing_metadata',None,None,'missing or wrong metadata type'),
                                      ('overlap',None,None,'could not parse vision GGUF')]:
        exporter.metadata=lambda: [(k,t,value if k==key else v) for k,t,v in original_metadata()]
        if label=='metadata_type':
            exporter.metadata=lambda: [(k,'string' if k=='deepseek4.vision.schema_version' else t,v) for k,t,v in original_metadata()]
        if label=='missing_metadata':
            exporter.metadata=lambda: [(k,t,v) for k,t,v in original_metadata() if k!='deepseek4.vision.block_count']
        selected=list(tensors)
        if label=='missing': selected.pop()
        if label=='shape': selected[0]=replace(selected[0],shape=(2048,),nbytes=4096)
        if label=='unknown': selected[0]=replace(selected[0],name='unknown.tensor')
        header,out=exporter._build_header(selected)
        if label in ('dtype','overlap'):
            import struct
            header=bytearray(header)
            tensor=selected[0 if label=='dtype' else 1]
            name=exporter._pack_string(tensor.name)
            type_offset=header.index(name)+len(name)+4+8*len(tensor.shape)
            if label=='dtype': struct.pack_into('<I',header,type_offset,1)
            else: struct.pack_into('<Q',header,type_offset+4,0)
        path=root/(label+'.gguf')
        with path.open('wb') as f:
            f.write(header)
            if label!='truncated': f.truncate(len(header)+sum(exporter._align(t.nbytes) for t in selected))
        run=subprocess.run([str(a.probe),str(path),'--load-only','4096','129280'],text=True,capture_output=True)
        assert run.returncode==1 and error in run.stderr,(label,run.returncode,run.stderr)
        print('PASS',label,flush=True)
        path.unlink()
    for dim,vocab in [(4095,129280),(4096,129279)]:
        run=subprocess.run([str(a.probe),str(a.projector),'--load-only',str(dim),str(vocab)],text=True,capture_output=True)
        assert run.returncode==1 and 'language model dimension/vocabulary mismatch' in run.stderr,run.stderr
        print('PASS','language contract',dim,vocab,flush=True)
