import hashlib
import json
from pathlib import Path
import struct
import sys

repo, source, output, evidence = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / 'server/deps/llama.cpp/gguf-py'))
import gguf

weight_map = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
selected = {name: shard for name, shard in weight_map.items()
            if name.startswith(('vision.', 'aligner.', 'image_'))}
reader = gguf.GGUFReader(output)
assert set(selected) == {tensor.name for tensor in reader.tensors}
assert len(selected) == 267
assert reader.get_field('general.architecture').contents() == 'deepseek4_vision'
headers = {}
for shard in set(selected.values()):
    with (source / shard).open('rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        headers[shard] = (size + 8, json.loads(handle.read(size)))
results = []
for tensor in reader.tensors:
    shard = selected[tensor.name]
    data_offset, header = headers[shard]
    entry = header[tensor.name]
    assert entry['dtype'] == 'BF16' and tensor.tensor_type.name == 'BF16'
    assert list(reversed(tensor.shape.tolist())) == entry['shape']
    begin, end = entry['data_offsets']
    with (source / shard).open('rb') as handle:
        handle.seek(data_offset + begin)
        original = handle.read(end - begin)
    exported = tensor.data.tobytes()
    assert original == exported, tensor.name
    results.append({'name': tensor.name, 'shape': entry['shape'], 'bytes': len(original),
                    'sha256': hashlib.sha256(exported).hexdigest()})
assert sum(row['bytes'] for row in results) == 932786176
digest = hashlib.sha256()
with output.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
manifest = {'verdict': 'PASS', 'tensor_count': len(results), 'payload_bytes': 932786176,
            'output': str(output), 'output_sha256': digest.hexdigest(),
            'reader': 'vendored gguf.GGUFReader', 'tensors': results}
(evidence / 'byte-proof.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('PASS all 267 source tensor names, shapes, BF16 types, and payload bytes')
print('output_sha256=' + digest.hexdigest())
