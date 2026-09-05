#!/usr/bin/env python3
"""Record actual PyTorch attention dispatch for source-shaped 3D Q/K/V."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
native=Path(sys.argv[1])
q=torch.from_numpy(np.fromfile(native/'corn-block0.q.f32',np.float32).reshape(782,16,64)).to(torch.bfloat16).transpose(0,1)
k=torch.from_numpy(np.fromfile(native/'corn-block0.k.f32',np.float32).reshape(782,16,64)).to(torch.bfloat16).transpose(0,1)
qkv=torch.from_numpy(np.fromfile(native/'corn-block0.qkv.f32',np.float32).reshape(782,3072)).to(torch.bfloat16)
v=qkv.chunk(3,dim=-1)[2].reshape(782,16,64).transpose(0,1)
with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
    result=torch.nn.functional.scaled_dot_product_attention(q,k,v)
print(json.dumps({'torch':torch.__version__,'q_shape':list(q.shape),'q_stride':list(q.stride()),'k_stride':list(k.stride()),'v_stride':list(v.stride()),'output_dtype':str(result.dtype),'operations':[e.key for e in profile.key_averages()]},indent=2))
