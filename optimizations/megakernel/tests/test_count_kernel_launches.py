"""CPU-only unit tests for optimizations/megakernel/count_kernel_launches.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimizations.megakernel.count_kernel_launches import (
    LaunchCounts,
    build_parser,
    count_trace_events,
    format_comparison,
    format_path_report,
    format_rate,
    parse_chrome_trace_file,
    validate_args,
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


def test_format_comparison_handles_zero_megakernel_count() -> None:
    mega = LaunchCounts(steps=1, kernels=0, gpu_memset=0, gpu_memcpy=0)
    hf = LaunchCounts(steps=1, kernels=100, gpu_memset=0, gpu_memcpy=0)

    comparison = format_comparison(
        mega,
        hf,
        hf_fast_kernels={"flash_linear_attention": True, "causal_conv1d": True},
    )
    assert "N/A (megakernel reported 0 compute kernels)" in comparison
    assert "Note:" not in comparison


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

