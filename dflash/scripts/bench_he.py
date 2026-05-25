"""
Bench DFlash test_dflash over multiple HumanEval-style prompts to get a stable
average acceptance length. Single-prompt measurements are noisy — z-lab's 8.09
AL on humaneval is averaged over 164 samples.

Usage on lucebox:
    python3 bench_he.py                 # run all 10 prompts with --fast-rollback
    python3 bench_he.py --mode batched  # run without --fast-rollback for A/B
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# Bench-prompt data centralized in bench_humaneval.py (the data
# source-of-truth for HE_CASES used by --area code and other entry
# points). Re-export here so legacy callers keep working.
from bench_humaneval import (  # noqa: E402,F401
    AUTOTUNE_PREFLIGHT_PROMPTS,
    PROMPTS,
)


from placement.backend_device import apply_backend_visible_devices
from placement.test_dflash_args import TestDflashLaunchArgs

ROOT = Path(__file__).resolve().parent.parent
BIN_SUFFIX = ".exe" if os.name == "nt" else ""
TARGET = os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
)
_LOCAL_DRAFT_FILE = ROOT / "models" / "draft" / "dflash-draft-3.6-q4_k_m.gguf"
_LOCAL_DRAFT_ROOT = ROOT / "models" / "draft"
DRAFT = None
TEST_DFLASH = os.environ.get(
    "DFLASH_BIN",
    str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"),
)
TMPDIR = Path(tempfile.gettempdir()) / "dflash_bench"
TMPDIR.mkdir(parents=True, exist_ok=True)

def _find_draft_file(root: Path) -> str | None:
    if root.is_file():
        return str(root) if root.suffix in (".safetensors", ".gguf") else None
    if not root.is_dir():
        return None
    for pattern in ("dflash-draft-*.gguf", "*.gguf", "model.safetensors"):
        matches = sorted(root.rglob(pattern))
        if matches:
            return str(matches[0])
    return None


def _resolve_draft() -> str:
    env = os.environ.get("DFLASH_DRAFT")
    if env:
        found = _find_draft_file(Path(env))
        if found:
            return found
        raise FileNotFoundError(f"DFLASH_DRAFT does not point to a draft file: {env}")

    for candidate in (_LOCAL_DRAFT_FILE, _LOCAL_DRAFT_ROOT):
        found = _find_draft_file(candidate)
        if found:
            return found

    raise FileNotFoundError(
        "draft model file not found. Expected one of:\n"
        f"  - {_LOCAL_DRAFT_FILE}\n"
        "Download it as documented in the README, or set DFLASH_DRAFT to an explicit "
        ".safetensors/.gguf file or directory."
    )


def _require_file(path: str, label: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _tokenizer_slug(tokenizer_id: str) -> str:
    """Filesystem-safe slug for tokenizer cache keying."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tokenizer_id)


def _prompt_path(i: int, tokenizer_slug: str) -> Path:
    return TMPDIR / f"he_prompt_{tokenizer_slug}_{i:02d}.bin"


def tokenize_prompt(prompt: str, out_path: Path, tokenizer) -> int:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with open(out_path, "wb") as f:
        for tid in ids:
            f.write(struct.pack("<i", int(tid)))
    return len(ids)


def run_test_dflash(prompt_path: Path, n_gen: int, fast_rollback: bool,
                    ddtree_budget: int | None = None,
                    ddtree_temp: float | None = None,
                    ddtree_no_chain_seed: bool = False,
                    extra_args: list[str] | None = None,
                    extra_env: dict[str, str] | None = None) -> dict:
    out_bin = TMPDIR / "he_bench_out.bin"
    cmd = [
        TEST_DFLASH, TARGET, DRAFT, str(prompt_path), str(n_gen), str(out_bin),
    ]
    if fast_rollback:
        cmd.append("--fast-rollback")
    if ddtree_budget is not None:
        cmd.append("--ddtree")
        cmd.append(f"--ddtree-budget={ddtree_budget}")
    if ddtree_temp is not None:
        cmd.append(f"--ddtree-temp={ddtree_temp}")
    if ddtree_no_chain_seed:
        cmd.append("--ddtree-no-chain-seed")
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        raise RuntimeError(f"test_dflash exited {r.returncode}")

    # Parse output. The target layer-split harness prints both prefill and
    # decode lines, so avoid the older "first tok/s wins" regexp there.
    out = r.stdout
    m_prefill = re.search(
        r"\[target-split\] prefill tokens=(\d+) time=(\d+(?:\.\d+)?) s speed=(\d+(?:\.\d+)?) tok/s",
        out,
    )
    m_decode_split = re.search(
        r"\[target-split-dflash\] decode tokens=(\d+) time=(\d+(?:\.\d+)?) s "
        r"speed=(\d+(?:\.\d+)?) tok/s",
        out,
    )
    m_decode_default = re.search(
        r"\[dflash\] generated \d+ tokens in \d+(?:\.\d+)? s\s+->\s+(\d+(?:\.\d+)?) tok/s",
        out,
    )
    m_tps = re.search(r"(\d+(?:\.\d+)?)\s+tok/s", out)
    m_commit = re.search(r"avg commit/step=(\d+(?:\.\d+)?)", out)
    m_accept = re.search(r"accepted=(\d+)/(\d+) \((\d+(?:\.\d+)?)%", out)
    m_steps = re.search(r"(\d+) draft steps", out)
    if not ((m_decode_split or m_decode_default or m_tps) and m_commit and m_accept and m_steps):
        print("STDOUT tail:", out[-2000:])
        raise RuntimeError("failed to parse output")
    if m_decode_split:
        tok_s = float(m_decode_split.group(3))
    elif m_decode_default:
        tok_s = float(m_decode_default.group(1))
    else:
        tok_s = float(m_tps.group(1))
    return {
        "tok_s": tok_s,
        "prefill_tok_s": float(m_prefill.group(3)) if m_prefill else None,
        "commit_per_step": float(m_commit.group(1)),
        "accepted": int(m_accept.group(1)),
        "total_draft_pos": int(m_accept.group(2)),
        "pct": float(m_accept.group(3)),
        "steps": int(m_steps.group(1)),
    }


def main():
    global DRAFT
    DRAFT = _resolve_draft()
    _require_file(TARGET, "target GGUF")
    _require_file(TEST_DFLASH, "test_dflash binary")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--mode", choices=["fast", "batched"], default="fast")
    ap.add_argument("--skip-tokenize", action="store_true")
    ap.add_argument("--ddtree-budget", type=int, default=None,
                    help="Enable DDTree mode with this node budget (e.g. 15, 32, 64)")
    ap.add_argument("--ddtree-temp", type=float, default=None,
                    help="Sharpen draft logits with this temperature (T<1 widens top-1/top-2 gap)")
    ap.add_argument("--ddtree-no-chain-seed", action="store_true",
                    help="Use paper's pure best-first (no chain pre-seed)")
    ap.add_argument("--draft-feature-mirror", action="store_true",
                    help="Use the draft-side target feature mirror path")
    ap.add_argument("--peer-access", action="store_true",
                    help="Prefer CUDA P2P memcpy between GPUs when available "
                         "(else host-staged copy)")
    ap.add_argument("--target-gpu", type=int, default=None,
                    help="Visible CUDA device id for the target backend")
    ap.add_argument("--draft-gpu", type=int, default=None,
                    help="Visible CUDA device id for the draft backend")
    ap.add_argument("--target-gpus", default=None,
                    help="Comma-separated target GPU ids for the layer-split harness")
    ap.add_argument("--target-layer-split", default=None,
                    help="Comma-separated layer split weights matching --target-gpus")
    ap.add_argument("--target-split-load-draft", action="store_true",
                    help="Load the draft alongside the target layer-split harness")
    ap.add_argument("--target-split-dflash", action="store_true",
                    help="Run chain DFlash decode through the target layer-split harness")
    ap.add_argument("--draft-ipc-bin", default=None,
                    help="Path to a different-backend test_dflash used as the remote draft daemon")
    ap.add_argument("--draft-ipc-gpu", type=int, default=None,
                    help="GPU id passed to the remote draft daemon")
    ap.add_argument("--draft-ipc-work-dir", default=None,
                    help="Work directory for host-file IPC with the remote draft daemon")
    ap.add_argument("--draft-ipc-ring-cap", type=int, default=None,
                    help="Feature-ring capacity for the remote draft daemon")
    ap.add_argument("--max-ctx", type=int, default=None,
                    help="Forward --max-ctx=N to test_dflash")
    ap.add_argument("--prefill-ubatch", type=int, default=None,
                    help="Set DFLASH27B_PREFILL_UBATCH for target split prefill")
    ap.add_argument("--cuda-visible-devices", default=None,
                    help="Optional CUDA_VISIBLE_DEVICES override for test_dflash")
    ap.add_argument("--target-tokenizer",
                    default=os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B"),
                    help="HuggingFace tokenizer repo for the target. Defaults to "
                         "$DFLASH_TOKENIZER, then Qwen/Qwen3.5-27B. Override for "
                         "Qwen3.6 or other variants, e.g. "
                         "--target-tokenizer Qwen/Qwen3.6-27B")
    args = ap.parse_args()

    # Tokenized prompts are cached at TMPDIR/he_prompt_<slug>_NN.bin so
    # different --target-tokenizer values never collide. Without the slug,
    # `--skip-tokenize` after a prior run with a different tokenizer would
    # silently feed the wrong token IDs to the bench.
    tok_slug = _tokenizer_slug(args.target_tokenizer)

    print(f"[bench] target    = {TARGET}")
    print(f"[bench] draft     = {DRAFT}")
    print(f"[bench] bin       = {TEST_DFLASH}")
    print(f"[bench] tmp       = {TMPDIR}")
    print(f"[bench] tokenizer = {args.target_tokenizer}")

    if not args.skip_tokenize:
        print("[bench] tokenizing prompts via HF…")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.target_tokenizer, trust_remote_code=True)
        for i, (name, p) in enumerate(PROMPTS):
            path = _prompt_path(i, tok_slug)
            n = tokenize_prompt(p, path, tok)
            print(f"  [{i:02d}] {name:26s}  {n:4d} tokens")
    else:
        if not _prompt_path(0, tok_slug).exists():
            sys.exit(
                f"[error] --skip-tokenize requested but no cache for "
                f"tokenizer={args.target_tokenizer!r} (looked for "
                f"{_prompt_path(0, tok_slug)}). Drop --skip-tokenize to "
                f"tokenize fresh, or pass --target-tokenizer matching a "
                f"previous run.")
        print(f"[bench] skipping tokenize (reusing {_prompt_path(0, tok_slug).parent})")

    print(f"\n[bench] mode={args.mode}  n_gen={args.n_gen}")
    print(f"{'prompt':28s}  {'steps':>6s} {'AL':>6s} {'pct%':>6s} {'prefill':>8s} {'decode':>8s}")
    print("-" * 72)

    extra_args = TestDflashLaunchArgs(
        draft_feature_mirror=args.draft_feature_mirror,
        peer_access=args.peer_access,
        target_gpu=args.target_gpu,
        draft_gpu=args.draft_gpu,
        target_gpus=args.target_gpus,
        target_layer_split=args.target_layer_split,
        target_split_load_draft=args.target_split_load_draft,
        target_split_dflash=args.target_split_dflash,
        draft_ipc_bin=args.draft_ipc_bin,
        draft_ipc_gpu=args.draft_ipc_gpu,
        draft_ipc_work_dir=args.draft_ipc_work_dir,
        draft_ipc_ring_cap=args.draft_ipc_ring_cap,
        max_ctx=args.max_ctx,
    ).to_cli_args()

    extra_env = {}
    if args.cuda_visible_devices:
        extra_env = apply_backend_visible_devices(
            "cuda",
            visible_devices=args.cuda_visible_devices,
            base_env=extra_env,
        )
    if args.prefill_ubatch is not None:
        extra_env["DFLASH27B_PREFILL_UBATCH"] = str(args.prefill_ubatch)

    results = []
    for i, (name, _) in enumerate(PROMPTS):
        path = _prompt_path(i, tok_slug)
        try:
            r = run_test_dflash(
                path,
                args.n_gen,
                fast_rollback=(args.mode == "fast" and not args.target_split_dflash),
                ddtree_budget=args.ddtree_budget,
                ddtree_temp=args.ddtree_temp,
                ddtree_no_chain_seed=args.ddtree_no_chain_seed,
                extra_args=extra_args,
                extra_env=extra_env,
            )
        except Exception as e:
            print(f"  [{i:02d}] {name:26s}  FAILED: {e}")
            continue
        results.append((name, r))
        prefill_s = (
            f"{r['prefill_tok_s']:8.2f}"
            if r["prefill_tok_s"] is not None else f"{'n/a':>8s}"
        )
        print(
            f"  {name:26s}  {r['steps']:6d} {r['commit_per_step']:6.2f} "
            f"{r['pct']:6.1f} {prefill_s} {r['tok_s']:8.2f}"
        )

    if not results:
        print("no successful runs")
        sys.exit(1)

    n = len(results)
    mean_al = sum(r["commit_per_step"] for _, r in results) / n
    mean_tps = sum(r["tok_s"] for _, r in results) / n
    mean_pct = sum(r["pct"] for _, r in results) / n
    prefill_vals = [r["prefill_tok_s"] for _, r in results if r["prefill_tok_s"] is not None]
    mean_prefill = sum(prefill_vals) / len(prefill_vals) if prefill_vals else None

    print("-" * 72)
    prefill_s = f"{mean_prefill:8.2f}" if mean_prefill is not None else f"{'n/a':>8s}"
    print(f"{'MEAN':28s}  {'':6s} {mean_al:6.2f} {mean_pct:6.1f} {prefill_s} {mean_tps:8.2f}")
    print()
    print(f"commit/step range: {min(r['commit_per_step'] for _,r in results):.2f} - "
          f"{max(r['commit_per_step'] for _,r in results):.2f}")
    print(f"tok/s range:        {min(r['tok_s'] for _,r in results):.1f} - "
          f"{max(r['tok_s'] for _,r in results):.1f}")


if __name__ == "__main__":
    main()
