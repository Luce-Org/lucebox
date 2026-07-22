import json

import pytest

from harness.validation_v2.http_runner import (
    _assistant_output,
    _response_timings,
    parse_sse_lines,
    validate_llama_cpp_evidence,
    validate_speculative_evidence,
)


def _config() -> dict[str, object]:
    return {
        "expected_configured_contract": "exact",
        "expected_effective_decode_mode": "autoregressive",
        "expected_fallback_reason": "exact_contract_requires_ar",
        "expected_draft_attached": True,
        "expected_draft_loaded": False,
    }


def _props() -> dict[str, object]:
    return {
        "speculative": {
            "configured_contract": "exact",
            "draft_attached": True,
            "draft_loaded": False,
        }
    }


def _usage() -> dict[str, object]:
    return {
        "completion_tokens": 2,
        "speculative": {
            "configured_contract": "exact",
            "effective_decode_mode": "autoregressive",
            "fallback_reason": "exact_contract_requires_ar",
        },
    }


def test_sse_parser_reconstructs_content_and_usage() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello "}}]}\n',
        b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}\n',
        b"data: [DONE]\n",
    ]
    text, usage, parsed = parse_sse_lines(chunks)
    assert text == "hello world"
    assert usage["completion_tokens"] == 2
    assert len(parsed) == 2


def test_sse_parser_preserves_nested_speculative_evidence() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
        (
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": _usage(),
                }
            )
            + "\n"
        ).encode(),
        b"data: [DONE]\n",
    ]
    _, usage, _ = parse_sse_lines(chunks)
    assert validate_speculative_evidence(
        usage=usage, props=_props(), config=_config()
    ) == {
        "configured_contract": "exact",
        "effective_decode_mode": "autoregressive",
        "fallback_reason": "exact_contract_requires_ar",
    }


def test_nonstream_usage_requires_nested_speculative_evidence() -> None:
    assert validate_speculative_evidence(
        usage=_usage(), props=_props(), config=_config()
    )["configured_contract"] == "exact"
    flattened = {
        "completion_tokens": 2,
        "configured_contract": "exact",
        "effective_decode_mode": "autoregressive",
        "fallback_reason": "exact_contract_requires_ar",
    }
    with pytest.raises(ValueError, match="nested speculative"):
        validate_speculative_evidence(
            usage=flattened, props=_props(), config=_config()
        )


def test_speculative_evidence_must_match_props_and_matrix() -> None:
    wrong_props = _props()
    wrong_props["speculative"]["draft_loaded"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="draft_loaded"):
        validate_speculative_evidence(
            usage=_usage(), props=wrong_props, config=_config()
        )

    missing_expectation = _config()
    del missing_expectation["expected_fallback_reason"]
    with pytest.raises(ValueError, match="lacks speculative evidence expectations"):
        validate_speculative_evidence(
            usage=_usage(), props=_props(), config=missing_expectation
        )


def test_sse_parser_rejects_non_data_lines() -> None:
    with pytest.raises(ValueError, match="non-SSE"):
        parse_sse_lines([b"not data\n"])


def test_sse_parser_rejects_non_object_chunks() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_sse_lines([f"data: {json.dumps([1, 2])}\n".encode()])


def test_sse_parser_rejects_truncated_or_unfinished_streams() -> None:
    with pytest.raises(ValueError, match=r"without \[DONE\]"):
        parse_sse_lines(
            [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n']
        )
    with pytest.raises(ValueError, match="without a finish reason"):
        parse_sse_lines(
            [
                b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
                b"data: [DONE]\n",
            ]
        )


def _llama_cpp_config(speculative_type: str) -> dict[str, object]:
    speculative = speculative_type != "none"
    return {
        "expected_build_info": "b10068-571d0d54",
        "expected_model_alias": "Qwen3.6-27B-Q4_K_M.gguf",
        "expected_speculative_type": speculative_type,
        "expected_configured_contract": "external-unverified"
        if speculative
        else "autoregressive",
        "expected_effective_decode_mode": "llama_cpp_"
        + speculative_type.removeprefix("draft-")
        if speculative
        else "autoregressive",
        "expected_fallback_reason": "none",
    }


def test_llama_cpp_evidence_requires_matching_identity_and_draft_activity() -> None:
    props = {
        "build_info": "b10068-571d0d54",
        "model_alias": "Qwen3.6-27B-Q4_K_M.gguf",
    }
    response = {
        "timings": {
            "predicted_ms": 1106.0,
            "predicted_per_second": 57.8,
            "draft_n": 55,
            "draft_n_accepted": 44,
        }
    }
    evidence = validate_llama_cpp_evidence(
        props=props,
        raw_response=response,
        config=_llama_cpp_config("draft-mtp"),
    )
    assert evidence["effective_decode_mode"] == "llama_cpp_mtp"
    assert evidence["draft_n_accepted"] == 44

    no_drafts = {"timings": {"predicted_ms": 1.0, "predicted_per_second": 1.0}}
    with pytest.raises(ValueError, match="no draft evidence"):
        validate_llama_cpp_evidence(
            props=props,
            raw_response=no_drafts,
            config=_llama_cpp_config("draft-dflash"),
        )


def test_llama_cpp_target_only_rejects_unexpected_drafts_and_wrong_build() -> None:
    props = {
        "build_info": "b10068-571d0d54",
        "model_alias": "Qwen3.6-27B-Q4_K_M.gguf",
    }
    response = {"timings": {"predicted_ms": 1.0, "predicted_per_second": 1.0}}
    evidence = validate_llama_cpp_evidence(
        props=props, raw_response=response, config=_llama_cpp_config("none")
    )
    assert evidence["configured_contract"] == "autoregressive"

    with pytest.raises(ValueError, match="unexpectedly drafted"):
        validate_llama_cpp_evidence(
            props=props,
            raw_response={"timings": {"draft_n": 1, "draft_n_accepted": 1}},
            config=_llama_cpp_config("none"),
        )
    with pytest.raises(ValueError, match="build_info"):
        validate_llama_cpp_evidence(
            props={**props, "build_info": "wrong"},
            raw_response=response,
            config=_llama_cpp_config("none"),
        )


def test_response_timings_reads_final_sse_chunk() -> None:
    assert _response_timings(
        [{"choices": []}, {"timings": {"predicted_per_second": 12.5}}]
    )["predicted_per_second"] == 12.5


def test_assistant_output_hash_material_includes_reasoning_and_final_content() -> None:
    nonstream = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "private trace",
                    "content": "VOLTA",
                }
            }
        ]
    }
    assert _assistant_output(nonstream) == {
        "reasoning_content": "private trace",
        "content": "VOLTA",
    }
    stream = [
        {"choices": [{"delta": {"reasoning_content": "private "}}]},
        {"choices": [{"delta": {"reasoning_content": "trace"}}]},
        {"choices": [{"delta": {"content": "VOL"}}]},
        {"choices": [{"delta": {"content": "TA"}, "finish_reason": "stop"}]},
    ]
    assert _assistant_output(stream) == _assistant_output(nonstream)
