#!/usr/bin/env bash
set -euo pipefail
[ "$(hostname)" = soulf ]
root="$HOME/lucebox-ds4v-runtime"
proof="$root/artifacts/vision-runtime"
reference="$HOME/lucebox-ds4v-mix-fix/artifacts/vision-reference"
python="$HOME/lucebox-ds4v-mix-fix/.venv-vision-reference/bin/python"
probe=/tmp/ds4v-runtime-build/ds4v_vision_probe
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
set +e
"$python" "$root/server/tools/ds4v_vision/compare.py" "$reference" "$proof/native" --output "$proof/comparison.json" > "$proof/comparison.log" 2>&1
comparison_exit=$?
set -e
printf '%s\n' "$comparison_exit" > "$proof/comparison.exit"
[ "$comparison_exit" = 3 ]
"$python" - "$proof" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1])
best=(0,0,0)
for h in range(1,1153):
    for w in range(1,1153):
        a,b=(h+2)//3,(w+2)//3
        rows=a+a%2
        block=rows*(b+1)+2+(rows//2*(b+1)%2)*2
        if block+3<=384 and h*w>best[0]:
            best=(h*w,h,w)
assert best==(3366,6,561),best
values=((np.arange(best[0]*588,dtype=np.int32)%31)-15).astype(np.float32)/16
values.tofile(p/'maximum-patches.f32')
(p/'maximum-grid.json').write_text(json.dumps(dict(patches=best[0],height=best[1],width=best[2],purpose='largest grid permitted by full block budget, not a resize aspect-policy fixture'))+'\n')
PY
for stages in 0 1; do
    /usr/bin/time -v "$probe" "$HOME/ds4v-work/ds4v-mmproj.gguf" "$proof/maximum-patches.f32" 6 561 "$proof/maximum-$stages" maximum "$stages" > "$proof/maximum-$stages.log" 2>&1
done
"$python" - "$proof" <<'PY'
from pathlib import Path
import hashlib,json,sys
import numpy as np
p=Path(sys.argv[1]); result={}
for stage in ('features','embeddings'):
    paths=[p/f'maximum-{i}'/f'maximum-{stage}.f32' for i in (0,1)]
    arrays=[np.fromfile(path,np.float32) for path in paths]
    assert all(np.isfinite(a).all() for a in arrays)
    assert arrays[0].size==(3366*1024 if stage=='features' else 374*4096)
    assert np.array_equal(*arrays)
    result[stage]=dict(finite=True,observer_invariance='PASS',elements=arrays[0].size,sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest())
(p/'maximum-verdict.json').write_text(json.dumps(result,indent=2)+'\n')
PY
date -u +%FT%TZ > "$proof/qualification.finished"
