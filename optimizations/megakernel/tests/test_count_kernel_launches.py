"""CPU-only unit tests for optimizations/megakernel/count_kernel_launches.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizations.megakernel import count_kernel_launches as ckl
from optimizations.megakernel.count_kernel_launches import (
    LaunchCounts,
    build_parser,
    count_trace_events,
    format_comparison,
    format_path_report,
    format_rate,
    parse_chrome_trace_file,
    profile_megakernel,
    resolve_megakernel_max_seq_len,
    validate_args,
    validate_megakernel_sequence_length,
)


def test_count_trace_events_separates_kernels_and_memory_ops() -> None:
    trace = {
        "traceEvents": [
            {"name": "decode_kernel", "cat": "kernel", "ph": "X", "dur": 1200},
            {"name": "lm_head_kernel", "cat": "kernel", "ph": "X", "dur": 300},
            {"name": "Memset (Device)", "cat": "gpu_memset", "ph": "X", "dur": 2},
            {"name": "Memset (Device)", "cat": "gpu_memset", "ph": "X", "dur": 2},
            {"name": "Memset (Device)", "cat": "gpu_memset", "ph": "X", "dur": 2},
            {"name": "Memcpy DtoH (Pinned)", "cat": "gpu_memcpy", "ph": "X", "dur": 5},
            {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "dur": 10},
            {"name": "aten::item", "cat": "cpu_op", "ph": "X", "dur": 15},
            {"name": "process_name", "ph": "M"},
            "ignored-non-dict-entry",
        ]
    }

    counts = count_trace_events(trace, steps=1)

    assert counts == LaunchCounts(steps=1, kernels=2, gpu_memset=3, gpu_memcpy=1)
    assert counts.kernels_per_token == 2.0
    assert counts.gpu_memset_per_token == 3.0
    assert counts.gpu_memcpy_per_token == 1.0


def test_count_trace_events_averages_across_multiple_steps() -> None:
    trace = {
        "traceEvents": [
            {"name": "decode_kernel", "cat": "kernel"},
            {"name": "lm_head_kernel", "cat": "kernel"},
            {"name": "decode_kernel", "cat": "kernel"},
            {"name": "lm_head_kernel", "cat": "kernel"},
            {"name": "Memset (Device)", "cat": "gpu_memset"},
            {"name": "Memcpy DtoH", "cat": "gpu_memcpy"},
            {"name": "Memcpy DtoH", "cat": "gpu_memcpy"},
        ]
    }

    counts = count_trace_events(trace, steps=2)

    assert counts.kernels == 4
    assert counts.kernels_per_token == 2.0
    assert counts.gpu_memset_per_token == 0.5
    assert counts.gpu_memcpy_per_token == 1.0


def test_count_trace_events_validates_inputs() -> None:
    with pytest.raises(ValueError, match="steps must be >= 1"):
        count_trace_events({"traceEvents": []}, steps=0)

    with pytest.raises(ValueError, match="traceEvents"):
        count_trace_events({}, steps=1)

    with pytest.raises(ValueError, match="traceEvents"):
        count_trace_events({"traceEvents": "not-a-list"}, steps=1)


def test_parse_chrome_trace_file(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "k1", "cat": "kernel"},
                    {"name": "k2", "cat": "kernel"},
                    {"name": "m1", "cat": "gpu_memset"},
                ]
            }
        ),
        encoding="utf-8",
    )

    counts = parse_chrome_trace_file(trace_file, steps=1)
    assert counts == LaunchCounts(steps=1, kernels=2, gpu_memset=1, gpu_memcpy=0)


def test_parse_chrome_trace_file_rejects_non_object(tmp_path: Path) -> None:
    trace_file = tmp_path / "bad.json"
    trace_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        parse_chrome_trace_file(trace_file, steps=1)


def test_format_rate_and_reports() -> None:
    assert format_rate(4, 2) == "2"
    assert format_rate(5, 2) == "2.50"

    mega = LaunchCounts(steps=1, kernels=2, gpu_memset=3, gpu_memcpy=1)
    hf = LaunchCounts(steps=1, kernels=2075, gpu_memset=0, gpu_memcpy=1)

    path_report = format_path_report("Megakernel (bf16)", mega)
    assert "Compute kernel launches / token: 2 (total: 2 over 1 step)" in path_report
    assert "3 gpu_memset, 1 gpu_memcpy" in path_report

    comparison = format_comparison(
        mega,
        hf,
        hf_fast_kernels={"flash_linear_attention": False, "causal_conv1d": False},
    )
    assert "Megakernel:        2 launches/token" in comparison
    assert "HF Transformers:   2075 launches/token" in comparison
    assert "1037.5x fewer compute kernel launches per token" in comparison
    assert "flash-linear-attention, causal-conv1d" in comparison


def test_format_comparison_hf_fewer_and_equal_launches() -> None:
    mega = LaunchCounts(steps=1, kernels=10, gpu_memset=0, gpu_memcpy=0)
    hf = LaunchCounts(steps=1, kernels=2, gpu_memset=0, gpu_memcpy=0)

    more_cmp = format_comparison(mega, hf)
    assert "Megakernel vs HF:  5.0x more compute kernel launches per token" in more_cmp
    assert "fewer" not in more_cmp

    equal_cmp = format_comparison(
        LaunchCounts(steps=2, kernels=6, gpu_memset=0, gpu_memcpy=0),
        LaunchCounts(steps=1, kernels=3, gpu_memset=0, gpu_memcpy=0),
    )
    assert "Megakernel vs HF:  equal compute kernel launches per token (1.0x)" in equal_cmp


def test_format_comparison_handles_zero_counts() -> None:
    zero = LaunchCounts(steps=1, kernels=0, gpu_memset=0, gpu_memcpy=0)
    nonzero = LaunchCounts(steps=1, kernels=100, gpu_memset=0, gpu_memcpy=0)

    zero_mega = format_comparison(
        zero,
        nonzero,
        hf_fast_kernels={"flash_linear_attention": True, "causal_conv1d": True},
    )
    assert "N/A (megakernel reported 0 compute kernels)" in zero_mega
    assert "Note:" not in zero_mega

    zero_hf = format_comparison(nonzero, zero)
    assert "N/A (HF reported 0 compute kernels)" in zero_hf
    assert "fewer" not in zero_hf
    assert "more" not in zero_hf

    both_zero = format_comparison(zero, zero)
    assert "N/A (both megakernel and HF reported 0 compute kernels)" in both_zero


def test_resolve_megakernel_max_seq_len_is_backend_aware() -> None:
    nvfp4_module = SimpleNamespace(MAX_SEQ_LEN=2048)
    nvfp4_dec = SimpleNamespace()  # model_nvfp4.Decoder does not set dec.max_seq_len
    assert (
        resolve_megakernel_max_seq_len(
            nvfp4_dec, backend="nvfp4", decoder_module=nvfp4_module
        )
        == 2048
    )

    bf16_module = SimpleNamespace(MAX_SEQ_LEN=2048)
    bf16_dec = SimpleNamespace(max_seq_len=1024)
    assert (
        resolve_megakernel_max_seq_len(
            bf16_dec, backend="bf16", decoder_module=bf16_module
        )
        == 1024
    )
    assert (
        resolve_megakernel_max_seq_len(
            SimpleNamespace(), backend="bf16", decoder_module=bf16_module
        )
        is None
    )


def test_validate_megakernel_sequence_length_boundary() -> None:
    # Total tokens = prompt (5) + warmup (3) + steps (1) = 9 = max_seq_len + 1.
    # Consumed KV-cache positions = (5 - 1) + 3 + 1 = 8 == max_seq_len -> allowed.
    validate_megakernel_sequence_length(5, warmup=3, steps=1, max_seq_len=8)

    # One more step consumes 9 positions (> max_seq_len=8) -> rejected.
    with pytest.raises(ValueError, match="9 KV-cache positions.*max_seq_len \\(8\\)"):
        validate_megakernel_sequence_length(5, warmup=3, steps=2, max_seq_len=8)


def test_profile_megakernel_nvfp4_enforces_module_max_seq_len(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNvfp4Decoder:
        def __init__(self, *, model_name: str, backend: str, verbose: bool) -> None:
            self.steps = 0

        def step(self, token_id: int) -> int:
            self.steps += 1
            return token_id + 1

    fake_module = SimpleNamespace(MAX_SEQ_LEN=8, Decoder=FakeNvfp4Decoder)
    monkeypatch.setitem(sys.modules, "model_nvfp4", fake_module)
    monkeypatch.setattr(ckl, "_release_cuda_memory", lambda: None)

    # Boundary: prompt=5, warmup=3, steps=1 -> total=9 == MAX_SEQ_LEN+1 (8 consumed) -> allowed.
    def fake_count(fn, *, warmup: int, steps: int) -> LaunchCounts:
        for _ in range(warmup + steps):
            fn()
        return LaunchCounts(steps=steps, kernels=2 * steps, gpu_memset=0, gpu_memcpy=1)

    monkeypatch.setattr(ckl, "count_kernel_launches", fake_count)
    backend, counts = profile_megakernel(
        "Qwen/Qwen3.5-0.8B",
        [1, 2, 3, 4, 5],
        backend="nvfp4",
        warmup=3,
        steps=1,
    )
    assert backend == "nvfp4"
    assert counts.kernels_per_token == 2.0

    # Past boundary: prompt=5, warmup=3, steps=2 -> 9 consumed > MAX_SEQ_LEN (8) -> rejected.
    with pytest.raises(ValueError, match="max_seq_len \\(8\\)"):
        profile_megakernel(
            "Qwen/Qwen3.5-0.8B",
            [1, 2, 3, 4, 5],
            backend="nvfp4",
            warmup=3,
            steps=2,
        )


def test_cli_parser_defaults_and_validation() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    validate_args(args)

    assert args.model == "Qwen/Qwen3.5-0.8B"
    assert args.mode == "both"
    assert args.backend == "auto"
    assert args.warmup == 3
    assert args.steps == 1

    with pytest.raises(ValueError, match="--warmup"):
        validate_args(parser.parse_args(["--warmup", "-1"]))

    with pytest.raises(ValueError, match="--steps"):
        validate_args(parser.parse_args(["--steps", "0"]))

    with pytest.raises(ValueError, match="--prompt"):
        validate_args(parser.parse_args(["--prompt", ""]))
