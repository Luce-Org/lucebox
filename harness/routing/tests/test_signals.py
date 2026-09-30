import math

import pytest
from mocks import make_config, run

from lucerouter import signals
from lucerouter.config import ModelBackend

EXPECTED = {"easy": 0.6, "medium": 0.3, "hard": 0.1}

# Same distribution (plus distractor tokens) in every response shape we accept.
BRICK_FIXTURES = {
    "openai_chat_logprobs": {"choices": [{"logprobs": {"content": [{"token": "easy", "logprob": math.log(0.6),
        "top_logprobs": [{"token": "easy", "logprob": math.log(0.6)},
                         {"token": " medium", "logprob": math.log(0.3)},
                         {"token": "Hard", "logprob": math.log(0.1)},
                         {"token": "I", "logprob": math.log(0.05)}]}]}}]},
    "openai_completions_dict": {"choices": [{"logprobs": {"top_logprobs": [
        {" easy": math.log(0.6), "medium": math.log(0.3), " hard": math.log(0.1), "\n": math.log(0.02)}]}}]},
    "llama_new_top_probs": {"completion_probabilities": [{"token": "easy", "prob": 0.6, "top_probs": [
        {"id": 1, "token": " easy", "prob": 0.48}, {"id": 2, "token": "e", "prob": 0.12},
        {"id": 3, "token": "med", "prob": 0.3}, {"id": 4, "token": " HARD", "prob": 0.1},
        {"id": 5, "token": "The", "prob": 0.01}]}]},
    "llama_new_top_logprobs": {"completion_probabilities": [{"token": "easy", "logprob": -0.5, "top_logprobs": [
        {"token": "easy", "logprob": math.log(0.6)}, {"token": "medium", "logprob": math.log(0.3)},
        {"token": "hard", "logprob": math.log(0.1)}]}]},
    "llama_old_probs": {"completion_probabilities": [{"content": "easy", "probs": [
        {"tok_str": "easy", "prob": 0.6}, {"tok_str": " medium", "prob": 0.3}, {"tok_str": " hard.", "prob": 0.1}]}]},
    "think_token_first": {"completion_probabilities": [
        {"top_probs": [{"token": "<think>", "prob": 0.99}, {"token": "\n", "prob": 0.01}]},
        {"top_probs": [{"token": "easy", "prob": 0.6}, {"token": "medium", "prob": 0.3}, {"token": "hard", "prob": 0.1}]}]},
}


@pytest.mark.parametrize("name", sorted(BRICK_FIXTURES))
def test_parse_brick_probs_shapes(name):
    probs = signals.parse_brick_probs(BRICK_FIXTURES[name])
    for label, p in EXPECTED.items():
        assert probs[label] == pytest.approx(p, abs=1e-6)


def test_parse_brick_probs_without_labels_raises():
    with pytest.raises(ValueError):
        signals.parse_brick_probs({"completion_probabilities": [{"top_probs": [{"token": "The", "prob": 1.0}]}]})


def test_truncate_query_keeps_head_and_tail():
    text = "HEAD" + "x" * 10000 + "TAIL"
    out = signals.truncate_query(text, 3000)
    assert len(out) <= 3000 and out.startswith("HEAD") and out.endswith("TAIL")
    assert signals.truncate_query("short") == "short"


def test_brick_probs_luce_request_contract(mock):
    """luce_server path: system prompt + 'Classify:' user message, logprobs, thinking off."""
    probs = run(signals.brick_probs(f"{mock.base}/brick", "EASYQ what is 2+2", model="brick"))
    assert probs["easy"] == pytest.approx(0.9, abs=1e-3)
    body = mock.brick_requests[-1]
    assert body["messages"][0] == {"role": "system", "content": signals.BRICK_SYSTEM}
    assert body["messages"][1]["content"] == "Classify: EASYQ what is 2+2"
    assert body["logprobs"] is True and body["top_logprobs"] == 20 and body["max_tokens"] == 1
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_brick_llama_prompt_bytes():
    prompt = signals.brick_chatml_prompt("hi", signals.THINK_SUFFIX)
    assert prompt == ("<|im_start|>system\n" + signals.BRICK_SYSTEM + "<|im_end|>\n"
                      "<|im_start|>user\nClassify: hi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


LUCE = ModelBackend("qwen38-27b", "http://x/v1", "luce_server", 2, served_name="qwen3.6-27b", capacity=4)


@pytest.mark.parametrize("status,expected", [
    ({"models": [{"id": "other", "in_flight": 9, "capacity": 9},
                 {"id": "qwen3.6-27b", "in_flight": 1, "capacity": 2}]}, {"in_flight": 1, "capacity": 2}),
    ({"phase": "idle", "active_requests": 0, "parked_requests": 0}, {"in_flight": 0, "capacity": 4}),
    ({"phase": "decode", "active_requests": 2, "parked_requests": 1}, {"in_flight": 3, "capacity": 4}),
    ({"phase": "prefill", "active_requests": 0, "parked_requests": 0}, {"in_flight": 1, "capacity": 4}),
    ({"unrelated": True}, None),
])
def test_parse_luce_status(status, expected):
    assert signals.parse_luce_status(status, LUCE) == expected


def test_parse_llama_slots():
    llama = ModelBackend("s", "http://x/v1", "llama-server", 0)
    slots = [{"id": 0, "is_processing": True}, {"id": 1, "is_processing": False}]
    assert signals.parse_llama_slots(slots, llama) == {"in_flight": 1, "capacity": 2}


def test_load_state_live_and_down(mock):
    cfg = make_config(mock.base)
    mock.load["qwen35-2b"] = {"in_flight": 1, "capacity": 1}
    state = run(signals.load_state(cfg.models))
    assert state["qwen35-2b"] == {"in_flight": 1, "capacity": 1}
    assert state["qwen35-0.8b"] is None  # mock returns 500 -> unknown, not a crash


def test_brick_probs_fall_back_to_generated_word():
    resp = {"choices": [{"message": {"role": "assistant", "content": "medium"}}]}
    assert signals.parse_brick_probs(resp) == {"easy": 0.0, "medium": 1.0, "hard": 0.0}
