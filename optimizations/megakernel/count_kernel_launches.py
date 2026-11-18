#!/usr/bin/env python3
"""Measure actual CUDA kernel launches per decoded token for the megakernel.

Compares the fused Lucebox megakernel (Decoder.step) against HuggingFace
Transformers (use_cache=True) using torch.profiler Chrome traces so compute
kernel launches are measured directly rather than estimated.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = "The weather today is"


@dataclass(frozen=True)
class LaunchCounts:
    """Measured CUDA dispatch counts collected across `steps` decode calls."""

    steps: int
    kernels: int
    gpu_memset: int
    gpu_memcpy: int

    @property
    def kernels_per_token(self) -> float:
        return self.kernels / self.steps

    @property
    def gpu_memset_per_token(self) -> float:
        return self.gpu_memset / self.steps

    @property
    def gpu_memcpy_per_token(self) -> float:
        return self.gpu_memcpy / self.steps


def count_trace_events(trace: dict[str, Any], *, steps: int = 1) -> LaunchCounts:
    """Count GPU compute kernels, memsets, and memcpys in a Chrome trace dict."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Chrome trace is missing a 'traceEvents' list")

    kernels = 0
    gpu_memset = 0
    gpu_memcpy = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        cat = event.get("cat")
        if cat == "kernel":
            kernels += 1
        elif cat == "gpu_memset":
            gpu_memset += 1
        elif cat == "gpu_memcpy":
            gpu_memcpy += 1

    return LaunchCounts(
        steps=steps,
        kernels=kernels,
        gpu_memset=gpu_memset,
        gpu_memcpy=gpu_memcpy,
    )


def parse_chrome_trace_file(path: Path, *, steps: int = 1) -> LaunchCounts:
    """Load a Chrome trace JSON file and count GPU kernel/memory dispatches."""
    with path.open(encoding="utf-8") as fh:
        trace = json.load(fh)
    if not isinstance(trace, dict):
        raise ValueError(f"Expected a JSON object in Chrome trace {path}")
    return count_trace_events(trace, steps=steps)


def count_kernel_launches(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    steps: int = 1,
) -> LaunchCounts:
    """Warm up `fn`, profile `steps` calls with torch.profiler, and return counts."""
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; a CUDA-capable GPU is required.")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()

    with tempfile.TemporaryDirectory(prefix="luce_kernel_launches_") as tmpdir:
        trace_path = Path(tmpdir) / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        return parse_chrome_trace_file(trace_path, steps=steps)


def detect_hf_fast_kernels() -> dict[str, bool]:
    """Check whether optional fast DeltaNet/conv packages are installed for HF."""
    return {
        "flash_linear_attention": importlib.util.find_spec("fla") is not None,
        "causal_conv1d": importlib.util.find_spec("causal_conv1d") is not None,
    }


def _ensure_megakernel_on_path() -> None:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def _select_half_dtype() -> Any:
    """Match model.py precision selection: bf16 on SM>=80, fp16 on SM<80."""
    import torch

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return torch.float16
    return torch.bfloat16


def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "bf16", "nvfp4"):
        raise ValueError(f"Unsupported backend: {backend!r}")
    if backend != "auto":
        return backend
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
        return "nvfp4"
    return "bf16"


def _release_cuda_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def profile_megakernel(
    model_name: str,
    prompt_ids: Sequence[int],
    *,
    backend: str = "auto",
    warmup: int = 3,
    steps: int = 1,
) -> tuple[str, LaunchCounts]:
    """Profile per-token CUDA launches for the fused megakernel Decoder.step."""
    _ensure_megakernel_on_path()
    resolved = _resolve_backend(backend)

    dec = None
    try:
        try:
            if resolved == "nvfp4":
                from model_nvfp4 import Decoder as Nvfp4Decoder

                dec = Nvfp4Decoder(model_name=model_name, backend="nvfp4", verbose=False)
            else:
                from model import Decoder as Bf16Decoder

                dec = Bf16Decoder(model_name=model_name, verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize megakernel ({resolved}) for {model_name!r}. "
                "Ensure the megakernel CUDA extension is built "
                "(e.g. `uv sync --extra megakernel`). "
                f"Cause: {exc}"
            ) from exc

        total_tokens = len(prompt_ids) + warmup + steps
        max_seq_len = getattr(dec, "max_seq_len", None)
        if max_seq_len is not None and total_tokens > max_seq_len:
            raise ValueError(
                f"Prompt ({len(prompt_ids)}) + warmup ({warmup}) + steps ({steps}) = "
                f"{total_tokens} exceeds megakernel max_seq_len ({max_seq_len})."
            )

        for tid in prompt_ids[:-1]:
            dec.step(tid)
        next_id = prompt_ids[-1]

        def megakernel_step() -> None:
            nonlocal next_id
            next_id = dec.step(next_id)

        counts = count_kernel_launches(megakernel_step, warmup=warmup, steps=steps)
        return resolved, counts
    finally:
        dec = None
        _release_cuda_memory()


def profile_hf(
    model_name: str,
    prompt_ids: Sequence[int],
    *,
    warmup: int = 3,
    steps: int = 1,
) -> LaunchCounts:
    """Profile per-token CUDA launches for HuggingFace Transformers with KV cache."""
    import torch
    from transformers import AutoModelForCausalLM

    model = None
    past = None
    try:
        dtype = _select_half_dtype()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map="cuda",
            )
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HuggingFace model {model_name!r} on CUDA: {exc}"
            ) from exc

        input_ids = torch.tensor([list(prompt_ids)], device="cuda")
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1:].argmax(-1)

        def hf_step() -> None:
            nonlocal past, next_tok
            with torch.no_grad():
                step_out = model(next_tok, past_key_values=past, use_cache=True)
            past = step_out.past_key_values
            next_tok = step_out.logits[:, -1:].argmax(-1)

        return count_kernel_launches(hf_step, warmup=warmup, steps=steps)
    finally:
        past = None
        model = None
        _release_cuda_memory()


def format_rate(total: int, steps: int) -> str:
    """Format a per-token rate as an integer when exact, else two decimals."""
    if total % steps == 0:
        return str(total // steps)
    return f"{total / steps:.2f}"


def format_path_report(label: str, counts: LaunchCounts) -> str:
    """Format measured compute-kernel and memory-op counts for one decode path."""
    step_word = "step" if counts.steps == 1 else "steps"
    lines = [
        f"=== {label} ===",
        (
            f"Compute kernel launches / token: "
            f"{format_rate(counts.kernels, counts.steps)} "
            f"(total: {counts.kernels} over {counts.steps} {step_word})"
        ),
        (
            f"Other GPU ops / token:           "
            f"{format_rate(counts.gpu_memset, counts.steps)} gpu_memset, "
            f"{format_rate(counts.gpu_memcpy, counts.steps)} gpu_memcpy"
        ),
    ]
    return "\n".join(lines)


def format_comparison(
    mega: LaunchCounts,
    hf: LaunchCounts,
    *,
    hf_fast_kernels: dict[str, bool] | None = None,
) -> str:
    """Format the megakernel vs. HuggingFace launch-count comparison."""
    mega_rate = format_rate(mega.kernels, mega.steps)
    hf_rate = format_rate(hf.kernels, hf.steps)

    if mega.kernels > 0:
        ratio = hf.kernels_per_token / mega.kernels_per_token
        reduction = f"{ratio:.1f}x fewer compute kernel launches per token"
    else:
        reduction = "N/A (megakernel reported 0 compute kernels)"

    lines = [
        "=== Comparison ===",
        f"Megakernel:        {mega_rate} launches/token",
        f"HF Transformers:   {hf_rate} launches/token",
        f"Reduction:         {reduction}",
    ]

    if hf_fast_kernels is not None:
        missing = [
            name.replace("_", "-")
            for name, available in hf_fast_kernels.items()
            if not available
        ]
        if missing:
            lines.append(
                "Note: HF ran without "
                + ", ".join(missing)
                + "; its reference fallback launches more small kernels."
            )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count actual CUDA compute kernel launches per decoded token for the "
            "Lucebox megakernel and/or HuggingFace Transformers."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt text to prefill before decode (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "megakernel", "hf"),
        default="both",
        help="Which decode path(s) to profile (default: both)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "bf16", "nvfp4"),
        default="auto",
        help="Megakernel backend (default: auto; selects nvfp4 on sm_12+, else bf16)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed warmup decode steps before profiling (default: 3)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of decode steps to profile and average (default: 1)",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {args.warmup}")
    if args.steps < 1:
        raise ValueError(f"--steps must be >= 1, got {args.steps}")
    if not args.prompt:
        raise ValueError("--prompt must be non-empty")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        import torch
    except ImportError as exc:
        print(
            f"Error: PyTorch is required to profile CUDA kernel launches ({exc}).",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available():
        print(
            "Error: CUDA is not available. Run this script on a machine with a CUDA-capable GPU.",
            file=sys.stderr,
        )
        return 1

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(
            f"Error: transformers is required to load the tokenizer/model ({exc}).",
            file=sys.stderr,
        )
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as exc:
        print(f"Error: failed to load tokenizer for {args.model!r}: {exc}", file=sys.stderr)
        return 1

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    if not prompt_ids:
        print("Error: prompt tokenized to zero tokens.", file=sys.stderr)
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    dtype = _select_half_dtype()
    print(
        f"GPU: {gpu_name} (sm_{major}{minor}) | torch {torch.__version__} | "
        f"CUDA {torch.version.cuda} | dtype: {str(dtype).removeprefix('torch.')}"
    )
    print(
        f"Model: {args.model} | prompt_tokens: {len(prompt_ids)} | "
        f"warmup: {args.warmup} | steps: {args.steps}"
    )

    mega_counts: LaunchCounts | None = None
    hf_counts: LaunchCounts | None = None
    hf_fast: dict[str, bool] | None = None

    try:
        if args.mode in ("both", "megakernel"):
            resolved_backend, mega_counts = profile_megakernel(
                args.model,
                prompt_ids,
                backend=args.backend,
                warmup=args.warmup,
                steps=args.steps,
            )
            print()
            print(format_path_report(f"Megakernel ({resolved_backend})", mega_counts))

        if args.mode in ("both", "hf"):
            hf_fast = detect_hf_fast_kernels()
            hf_counts = profile_hf(
                args.model,
                prompt_ids,
                warmup=args.warmup,
                steps=args.steps,
            )
            print()
            print(format_path_report("HuggingFace Transformers (use_cache=True)", hf_counts))
            fla_status = "installed" if hf_fast["flash_linear_attention"] else "not installed"
            conv_status = "installed" if hf_fast["causal_conv1d"] else "not installed"
            print(
                f"Optional HF fast kernels:        "
                f"flash-linear-attention={fla_status}, causal_conv1d={conv_status}"
            )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if mega_counts is not None and hf_counts is not None:
        print()
        print(format_comparison(mega_counts, hf_counts, hf_fast_kernels=hf_fast))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
