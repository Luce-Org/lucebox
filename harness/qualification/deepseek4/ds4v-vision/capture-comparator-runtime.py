"""Read-only CPU comparator runtime inventory; run only on soulf."""
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy

assert sys.flags.isolated
root = pathlib.Path(numpy.__file__).parent
files = {pathlib.Path(sys.executable).resolve()}
files.update(root.rglob('*.py'))
files.update(root.rglob('*.so'))
files.update((root.parent / 'numpy.libs').glob('*'))
for path in list(files):
    if path.suffix == '.so' or path.name.startswith('python'):
        result = subprocess.run(['/usr/bin/ldd', str(path)], text=True, capture_output=True, check=True)
        assert 'not found' not in result.stdout
        for line in result.stdout.splitlines():
            parts = line.split()
            name = parts[2] if len(parts) > 2 and parts[1] == '=>' else parts[0] if parts else ''
            if name.startswith('/'):
                files.add(pathlib.Path(name).resolve())
print(json.dumps({'python': sys.version, 'numpy': numpy.__version__, 'files': {
    str(path): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    for path in sorted(files) if path.is_file()
}}, indent=2))
