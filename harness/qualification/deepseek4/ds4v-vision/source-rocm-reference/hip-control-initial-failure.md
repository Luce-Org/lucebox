# Original-source corn HIP control

**Initial released attempt stopped at the device-name guard before weight loading or a tower forward. No source-HIP tensors were produced.** The frozen script was not changed for this attempt, and no numerical threshold was changed.

The parent's explicit GPU release authorized the single original-source corn lane. `run-hip-supervised.py` verified source-forward SHA256 `17ba9d66260d25c056ecc56a35192748a662b040605bb0bf8864bd704a66267b` and private MIOpen SHA256 `bad776611dcee04ec70ca17674999e1af606ab9bfa0c6c6309ccf933ab1cdbbd`, then checked operator inactive/MainPID0, no listeners8016/8217, empty KFD process directory, and at least8 GiB host and discrete VRAM available. Actual preflight:36102742016 host bytes and21430087680 discrete bytes free.

The Python argv/environment matched the prepared command; direct child supervision with `os.wait4` supplied actual PID/resource timing instead of GNU time. The finite deadline was300 seconds, with termination restricted to the recorded unreaped direct child. No timeout or signal was needed.

- Actual Python PID3359202, exit1, elapsed2.091728805 seconds.
- User1.455116 seconds, system0.239513 seconds, peak RSS744712 KiB.
- Error:`RuntimeError: actual device is not RX 7900 XT` at the exact `props.name == 'AMD Radeon RX 7900 XT'` guard.
- The frozen script constructs an identity dictionary before that guard but does not print it on this failure path, so the actual Torch name was not captured. No weight loading, forward or output directory creation occurred.
- After exit:operator inactive/PID0, ports free, KFD empty, discrete VRAM unchanged at21430087680 free bytes.

The completed native lane's existing `device-check.log` reports Device0 as `Radeon RX 7900 XT` without the `AMD` prefix, gfx1100,20464 MiB. This is a plausible display-name mismatch, not proof of the source attempt's actual selected device. No further GPU identity query or retry was performed in this initial attempt.

Evidence:remote `~/ds4v-work/source-rocm210-reference/hip-supervision/`; local copied `hip-supervision/{run.json,hip-corn.log,memory.jsonl,python.pid,exit}`. The wrapper source is in this report's directory. Source-HIP/CPU and native-HIP/source-HIP comparisons remain unavailable until a source forward completes.
