#!/usr/bin/env python3
"""CPU-only import metadata: deliberately makes no torch.cuda calls."""
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

assert sys.flags.isolated == 1
assert sys.prefix != sys.base_prefix
for key in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'):
    assert os.environ.get(key) == '-1'
private_lib = Path.home()/'ds4v-work/source-rocm210-reference/private-miopen/opt/rocm-7.2.4/lib'
assert os.environ.get('LD_LIBRARY_PATH') == str(private_lib)
with (private_lib/'libMIOpen.so.1').open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest() == 'bad776611dcee04ec70ca17674999e1af606ab9bfa0c6c6309ccf933ab1cdbbd'
sys.dont_write_bytecode = True
import torch
import numpy
import PIL
import safetensors
assert importlib.metadata.version('torch') == '2.10.0+rocm7.2.4.lw.git3d3aa833'
assert torch.__version__ == '2.10.0+rocm7.2.4.git3d3aa833'
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
libs = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
               if '.so' in line and line.split()[-1].startswith('/')})
report = dict(torch=torch.__version__, torch_git=torch.version.git_version,
              hip_version=torch.version.hip, rocm_version=torch.version.rocm, torch_module=torch.__file__, torch_extension=torch._C.__file__,
              numpy=numpy.__version__, pillow=PIL.__version__, safetensors=safetensors.__version__,
              python=sys.version, prefix=sys.prefix, base_prefix=sys.base_prefix, path=sys.path,
              platform=platform.platform(), system_rocm=Path('/opt/rocm/.info/version').read_text().strip(),
              venv_config=(Path(sys.prefix)/'pyvenv.cfg').read_text(),
              environment={k:os.environ.get(k) for k in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES',
                           'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH','LD_PRELOAD')},
              packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
              loaded_libraries=libs, torch_config=torch.__config__.show(),
              parallel_info=torch.__config__.parallel_info(), gpu_api_called=False)
print(json.dumps(report,indent=2,sort_keys=True))
