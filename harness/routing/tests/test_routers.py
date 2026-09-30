import numpy as np
import pytest
from mocks import make_config, run

from lucerouter.config import config_from_dict
from lucerouter.routers import build, parse_spec
from lucerouter.types import RouteRequest

S, M, L = "qwen35-0.8b", "qwen35-2b", "qwen38-27b"


def user(text, **kw):
    return RouteRequest(messages=[{"role": "user", "content": text}], **kw)


async def route_all(spec, cfg, reqs, **kw):
    router = build(spec, cfg, **kw)
    try:
        return [await router.route(r) for r in reqs]
    finally:
        await router.aclose()


# ─── spec grammar ──────────────────────────────────────────────────────

def test_string_and_json_specs_are_equivalent():
    s = parse_spec("load_aware:margin_down=0.2>brick:service=brick-max,t_small=0.8,small=none")
    j = parse_spec('{"router": "load_aware", "params": {"margin_down": "0.2"}, "inner": '
                   '{"router": "brick", "params": {"service": "brick-max", "t_small": "0.8", "small": "none"}}}')
    assert s == j == [("load_aware", {"margin_down": "0.2"}),
                      ("brick", {"service": "brick-max", "t_small": "0.8", "small": "none"})]


@pytest.mark.parametrize("spec", ["load_aware", "fixed:model=qwen35-2b>fixed:model=qwen35-2b", "nope:x=1",
                                  "brick:service=missing", "fixed"])
def test_bad_specs_rejected(spec):
    cfg = make_config("http://127.0.0.1:1")
    with pytest.raises((ValueError, KeyError)):
        build(spec, cfg)


# ─── brick ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("params,probs,expected", [
    ("", (0.85, 0.1, 0.05), S),
    ("", (0.6, 0.3, 0.1), M),                  # easy+medium 0.9 >= 0.5
    ("", (0.2, 0.2, 0.6), L),
    (",h_max=0.3", (0.2, 0.5, 0.3), L),        # p_hard not below h_max
    (",small=none", (0.95, 0.05, 0.0), M),     # no 0.8b tier: easy goes to 2b
    (",mid=none", (0.6, 0.3, 0.1), L),
    (",t_small=0.9", (0.85, 0.1, 0.05), M),
])
def test_brick_thresholds(params, probs, expected):
    router = build("brick:service=brick-max" + params, make_config("http://127.0.0.1:1"))
    model, _, margin = router.choose(dict(zip(("easy", "medium", "hard"), probs)))
    assert model == expected
    assert margin >= 0


def test_brick_routes_via_classifier(mock):
    cfg = make_config(mock.base)
    ds = run(route_all("brick:service=brick-max", cfg, [user("EASYQ"), user("MIDQ"), user("HARDQ")]))
    assert [d.model for d in ds] == [S, M, L]
    assert ds[0].scores["easy"] == pytest.approx(0.9, abs=1e-3)
    assert ds[0].router == "brick:service=brick-max" and ds[0].latency_ms > 0


def test_brick_classifier_down_falls_back_to_large():
    cfg = make_config("http://127.0.0.1:1")  # nothing listening
    [d] = run(route_all("brick:service=brick-max", cfg, [user("EASYQ")]))
    assert d.model == L and "brick error" in d.reason


# ─── load_aware ────────────────────────────────────────────────────────

def test_load_aware_shifts(mock):
    cfg = make_config(mock.base)
    free, full = {"in_flight": 0, "capacity": 1}, {"in_flight": 1, "capacity": 1}

    def go(spec, text):
        [d] = run(route_all(spec, cfg, [user(text)], load_ttl_s=0))
        return d

    mock.load.update({S: free, M: full, L: free})
    d = go("load_aware>brick:service=brick-max", "MIDQ")
    assert d.model == L and d.signals["inner_model"] == M and "shifted up" in d.reason

    mock.load.update({S: free, M: full, L: full})
    # MIDQ margin = min(0.8-0.5, 0.5-0.2, 0.8-0.3) = 0.3 > margin_down -> stays
    assert go("load_aware>brick:service=brick-max", "MIDQ").model == M
    assert go("load_aware:margin_down=0.5>brick:service=brick-max", "MIDQ").model == S

    mock.load.update({S: free, M: free, L: free})
    assert go("load_aware:promote=0.5>brick:service=brick-max", "MIDQ").model == L
    assert go("load_aware:promote=0.1>brick:service=brick-max", "MIDQ").model == M

    mock.load.clear()  # load unknown -> inner decision unchanged
    assert go("load_aware>brick:service=brick-max", "MIDQ").model == M


def test_offline_api_with_two_model_config():
    cfg = config_from_dict({"models": {M: {"base_url": "http://x/v1", "kind": "luce_server", "rank": 1},
                                       L: {"base_url": "http://y/v1", "kind": "luce_server", "rank": 2}},
                            "services": {"brick-max": {"base_url": "http://z"}}})
    router = build("brick:service=brick-max", cfg)
    assert router.small is None and router.mid == M and router.large == L
    assert router.choose({"easy": 0.95, "medium": 0.05, "hard": 0.0})[0] == M
