"""Card-driven sampling: precedence, provenance, and wire-body flow.

Covers the fix where luce-bench forces the resolved model card's
``sampling`` block by default (so card-less servers — OpenRouter / MLX —
run with the model's recommended decode params instead of the provider's
defaults), with explicit --temperature/--top-p/--top-k overriding per
field and --no-card-sampling opting out entirely.

  * ``card_sampling`` — returns the card's sampling block (or {}).
  * ``resolve_sampling`` — the per-field precedence + source classifier.
  * ``run_case`` — the resolved sampling reaches the wire body.

No live server. Uses mocked urlopen to capture the request body.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from lucebench.cli import resolve_sampling
from lucebench.model_cards import card_sampling
from lucebench.runner import run_case

# A card whose sampling block mirrors the bundled qwen3.6-27b card.
_QWEN_CARD = {
    "name": "Qwen3.6 27B",
    "sampling": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
}


def _mock_urlopen(response_body: dict, status: int = 200):
    resp = MagicMock()
    resp.read.return_value = json.dumps(response_body).encode()
    resp.status = status
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


def _chat_response() -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _capture_body(**kwargs) -> dict:
    case = {"id": "x", "kind": "integer", "question": "1+1?"}
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data)
        return _mock_urlopen(_chat_response())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        run_case(url="http://localhost:8080", case=case, stream=False, **kwargs)
    return sent["body"]


# ────────────────────────────────────────────────────────────────────
# card_sampling helper
# ────────────────────────────────────────────────────────────────────


def test_card_sampling_returns_block():
    assert card_sampling(_QWEN_CARD)["temperature"] == 1.0
    assert card_sampling(_QWEN_CARD)["top_k"] == 20


def test_card_sampling_empty_for_no_card_or_no_block():
    assert card_sampling(None) == {}
    assert card_sampling({"name": "no-sampling"}) == {}
    assert card_sampling({"sampling": "not-a-dict"}) == {}


def test_card_sampling_returns_fresh_dict():
    out = card_sampling(_QWEN_CARD)
    out["temperature"] = 99.0
    assert _QWEN_CARD["sampling"]["temperature"] == 1.0


# ────────────────────────────────────────────────────────────────────
# resolve_sampling — precedence + source classifier
# ────────────────────────────────────────────────────────────────────


def test_resolve_sampling_card_default():
    """Card resolved, no CLI flags, default → full card block, source=card."""
    sampling, source = resolve_sampling(
        card=_QWEN_CARD,
        no_card_sampling=False,
        cli_temperature=None,
        cli_top_p=None,
        cli_top_k=None,
    )
    assert source == "card"
    assert sampling == _QWEN_CARD["sampling"]


def test_resolve_sampling_cli_override_is_mixed():
    """--temperature overrides that field; the rest still come from the card."""
    sampling, source = resolve_sampling(
        card=_QWEN_CARD,
        no_card_sampling=False,
        cli_temperature=0.5,
        cli_top_p=None,
        cli_top_k=None,
    )
    assert source == "mixed"
    assert sampling["temperature"] == 0.5  # CLI wins
    assert sampling["top_p"] == 0.95  # still from card
    assert sampling["top_k"] == 20


def test_resolve_sampling_no_card_sampling_opts_out():
    """--no-card-sampling + no CLI → nothing resolves, source=none."""
    sampling, source = resolve_sampling(
        card=_QWEN_CARD,
        no_card_sampling=True,
        cli_temperature=None,
        cli_top_p=None,
        cli_top_k=None,
    )
    assert source == "none"
    assert sampling == {}


def test_resolve_sampling_no_card_sampling_with_cli_is_cli():
    """--no-card-sampling but CLI flags present → source=cli, card ignored."""
    sampling, source = resolve_sampling(
        card=_QWEN_CARD,
        no_card_sampling=True,
        cli_temperature=0.3,
        cli_top_p=None,
        cli_top_k=None,
    )
    assert source == "cli"
    assert sampling == {"temperature": 0.3}


def test_resolve_sampling_unknown_model_no_card():
    """No card resolved + no CLI → nothing, source=none."""
    sampling, source = resolve_sampling(
        card=None,
        no_card_sampling=False,
        cli_temperature=None,
        cli_top_p=None,
        cli_top_k=None,
    )
    assert source == "none"
    assert sampling == {}


def test_resolve_sampling_cli_only_no_card():
    sampling, source = resolve_sampling(
        card=None,
        no_card_sampling=False,
        cli_temperature=0.7,
        cli_top_p=0.9,
        cli_top_k=40,
    )
    assert source == "cli"
    assert sampling == {"temperature": 0.7, "top_p": 0.9, "top_k": 40}


# ────────────────────────────────────────────────────────────────────
# End-to-end: resolved sampling reaches the wire body
# ────────────────────────────────────────────────────────────────────


def test_card_sampling_flows_into_body():
    """Card resolved, no CLI: body carries temp/top_p/top_k/min_p from card."""
    sampling, source = resolve_sampling(
        card=_QWEN_CARD,
        no_card_sampling=False,
        cli_temperature=None,
        cli_top_p=None,
        cli_top_k=None,
    )
    body = _capture_body(
        temperature=sampling.get("temperature"),
        top_p=sampling.get("top_p"),
        top_k=sampling.get("top_k"),
        min_p=sampling.get("min_p"),
        presence_penalty=sampling.get("presence_penalty"),
        repetition_penalty=sampling.get("repetition_penalty"),
        sampling_source=source,
    )
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.95
    assert body["top_k"] == 20
    assert body["min_p"] == 0.0
    assert body["presence_penalty"] == 0.0
    assert body["repetition_penalty"] == 1.0


def test_cli_temperature_wins_but_card_fills_rest_in_body():
    """--temperature 0.5 + card → body temp=0.5, top_p/top_k from card; mixed."""
    sampling, source = resolve_sampling(
        card=_QWEN_CARD,
        no_card_sampling=False,
        cli_temperature=0.5,
        cli_top_p=None,
        cli_top_k=None,
    )
    assert source == "mixed"
    body = _capture_body(
        temperature=sampling.get("temperature"),
        top_p=sampling.get("top_p"),
        top_k=sampling.get("top_k"),
        min_p=sampling.get("min_p"),
        sampling_source=source,
    )
    assert body["temperature"] == 0.5
    assert body["top_p"] == 0.95
    assert body["top_k"] == 20


def test_no_card_sampling_omits_all_keys_in_body():
    """--no-card-sampling + no CLI → no sampling keys in body; source=none."""
    sampling, source = resolve_sampling(
        card=_QWEN_CARD,
        no_card_sampling=True,
        cli_temperature=None,
        cli_top_p=None,
        cli_top_k=None,
    )
    assert source == "none"
    body = _capture_body(
        temperature=sampling.get("temperature"),
        top_p=sampling.get("top_p"),
        top_k=sampling.get("top_k"),
        min_p=sampling.get("min_p"),
        sampling_source=source,
    )
    for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
        assert key not in body


def test_unknown_model_no_card_omits_in_body():
    """No card resolved + no CLI → sampling keys omitted from body."""
    sampling, source = resolve_sampling(
        card=None,
        no_card_sampling=False,
        cli_temperature=None,
        cli_top_p=None,
        cli_top_k=None,
    )
    assert source == "none"
    body = _capture_body(
        temperature=sampling.get("temperature"),
        top_p=sampling.get("top_p"),
        top_k=sampling.get("top_k"),
        sampling_source=source,
    )
    assert "temperature" not in body
    assert "top_p" not in body
    assert "top_k" not in body
