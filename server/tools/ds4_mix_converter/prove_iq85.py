#!/usr/bin/env python3
"""Bounded CPU-only original-source IQ85 pilot; never converts a full model."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=pathlib.Path, required=True)
    p.add_argument("--input", type=pathlib.Path, required=True)
    p.add_argument("--imatrix", type=pathlib.Path, required=True)
    p.add_argument("--imatrix-provenance", choices=["uniform-unvalidated", "activation-derived", "transferred-text-calibration"], required=True)
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--experts", type=int, default=8, choices=range(1, 18))
    p.add_argument("--reference-threads", type=int, default=1, choices=[1, 8])
    p.add_argument("--parallel-threads", type=int, default=8, choices=[8, 16])
    p.add_argument("--timeout", type=int, default=14400)
    a = p.parse_args()
    if sys.platform != "linux":
        p.error("run only on the authorized Linux CPU host")
    if not 1 <= a.timeout <= 28800:
        p.error("timeout must be 1..28800 seconds per lane")
    for name in ("binary", "input", "imatrix", "output_dir"):
        value = getattr(a, name)
        if not value.is_absolute():
            p.error(f"--{name.replace('_','-')} must be absolute")
    a.output_dir.mkdir(parents=False, exist_ok=False)
    manifest = {"recipe": "iq85-v1", "quality_validated": False,
                "imatrix_provenance": a.imatrix_provenance,
                "binary_sha256": sha256(a.binary), "imatrix_sha256": sha256(a.imatrix),
                "source_index_sha256": sha256(a.input / "model.safetensors.index.json"), "lanes": []}
    common = [str(a.binary), "--input", str(a.input), "--recipe", "iq85",
              "--imatrix", str(a.imatrix), "--imatrix-provenance", a.imatrix_provenance,
              "--layer-count", "1", "--expert-limit", str(a.experts), "--experts-only"]
    def save():
        (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    reference_label = "serial" if a.reference_threads == 1 else "reference8"
    for label, threads in ((reference_label, a.reference_threads),
                           (f"parallel{a.parallel_threads}", a.parallel_threads),
                           (f"repeat{a.parallel_threads}", a.parallel_threads)):
        artifact = a.output_dir / (label + ".gguf")
        command = common + ["--output", str(artifact), "--encode-threads", str(threads)]
        start = time.monotonic()
        lane = {"label": label, "command": command}
        manifest["lanes"].append(lane)
        save()
        try:
            with (a.output_dir / (label + ".plan.json")).open("w") as out, (a.output_dir / (label + ".log")).open("w") as err:
                result = subprocess.run(command, stdout=out, stderr=err, timeout=a.timeout, check=False)
            lane.update(exit_code=result.returncode, seconds=time.monotonic()-start)
            if result.returncode:
                save()
                raise RuntimeError(f"{label} failed: exit {result.returncode}")
            lane.update(bytes=artifact.stat().st_size, sha256=sha256(artifact))
            plan = json.loads((a.output_dir / (label + ".plan.json")).read_text())
            if lane["bytes"] != plan["file_bytes"]:
                raise RuntimeError("plan byte count differs from output")
        finally:
            save()
    manifest["byte_identical"] = len({lane["sha256"] for lane in manifest["lanes"]}) == 1
    save()
    if not manifest["byte_identical"]:
        raise RuntimeError("reference/parallel/repeat whole-file hashes differ")
    print("PASS: bounded pilot exact plan sizes and reference/parallel/repeat whole-file identity; quality remains unvalidated")


if __name__ == "__main__":
    main()
