"""brick_skill: Brick's J = D + beta * latency rule, its latency model, and skill/latency fitting.

Arms (production layout): S=2b, M=27b thinking off, T=27b thinking on (same backend as M).
Toy skills: coding S .5 / M .8 / T .95; world_knowledge S .3 / M .5 / T .9; other dims .5.
Toy latency profile: S 1000 ms, M 5000 ms, T 20000 ms per request (no prompt-length term).
"""

import json
import math

import pytest
from eval.fit_latency import fit_profile
from eval.fit_skills import fit_skill_table
from mocks import ARMS, make_config, run

from lucerouter import brick_math as bm
from lucerouter.latency import LatencyProfile, expected_latency_ms, queue_wait_ms
from lucerouter.routers import build
from lucerouter.types import RouteRequest

S, M, T = "qwen35-2b", "qwen38-27b", "qwen38-27b-think"
BACKEND = "qwen38-27b"
CAPS = bm.CAPABILITIES


def onehot(cap):
    return {c: float(c == cap) for c in CAPS}


def skills_file(tmp_path):
    table = {"capabilities": list(CAPS), "arms": {
        S: {**dict.fromkeys(CAPS, 0.5), "coding": 0.5, "world_knowledge": 0.3},
        M: {**dict.fromkeys(CAPS, 0.5), "coding": 0.8, "world_knowledge": 0.5},
        T: {**dict.fromkeys(CAPS, 0.5), "coding": 0.95, "world_knowledge": 0.9}}}
    p = tmp_path / "skills.json"
    p.write_text(json.dumps(table))
    return p


PROFILE = {"chars_per_token": 4.0,
           "arms": {S: {"ttft_ms": 100, "tpot_ms": 10, "out_tokens": {"default": 90}},
                    M: {"ttft_ms": 200, "tpot_ms": 40, "out_tokens": {"default": 120}},
                    T: {"ttft_ms": 200, "tpot_ms": 40, "out_tokens": {"default": 495}}},
           "backends": {BACKEND: {"service_ms": 100000}}}


def profile_file(tmp_path):
    p = tmp_path / "latency.json"
    p.write_text(json.dumps(PROFILE))
    return p


def fake_capability(text):
    return onehot("coding" if "code" in text else "world_knowledge")


def user(text, **kw):
    return RouteRequest(messages=[{"role": "user", "content": text}], **kw)


async def route_one(spec, cfg, req, **kw):
    router = build(spec, cfg, **kw)
    try:
        return await router.route(req)
    finally:
        await router.aclose()


# ─── Brick maths ───────────────────────────────────────────────────────

def test_distance_matches_hand_computation_and_is_asymmetric():
    p = {**dict.fromkeys(CAPS, 0.0), "coding": 0.5, "math_reasoning": 0.5}
    z = 1.0
    s = {**dict.fromkeys(CAPS, 0.5), "coding": 1 / (1 + math.exp(-2.0)), "math_reasoning": 0.5}
    # coding: capacity .5*2 = 1 over requirement .5 -> over .5; math: capacity 0, under .5
    assert bm.distance(p, s, z, lam=0.04) == pytest.approx(math.sqrt(0.25 + 0.04 * 0.25))
    under_only = {**s, "coding": 0.5}                    # same size gap, but both under
    assert bm.distance(p, under_only, z, lam=0.04) == pytest.approx(math.sqrt(0.5))
    matched = {**s, "coding": 1 / (1 + math.exp(-1.0)), "math_reasoning": 1 / (1 + math.exp(-1.0))}
    assert bm.distance(p, matched, z, lam=0.04) == pytest.approx(0.0, abs=1e-9)


def test_tau_blends_label_toward_medium_by_confidence():
    assert bm.tau_query("hard", 0.5) == pytest.approx(0.5 * 0.88 + 0.5 * 0.72)
    assert bm.tau_query("bogus", 0.9) == bm.TAU["medium"]
    assert bm.complexity_label(None) == ("medium", 1.0)          # complexity service down


def test_r_knob_trades_quality_for_speed():
    lo, mid, hi = bm.effective_params(-1), bm.effective_params(0), bm.effective_params(1)
    assert (mid.mu, mid.b, mid.beta, mid.lam) == (bm.MU0, bm.B0, bm.BETA0, bm.LAM0)
    tau = bm.tau_query("hard", 1.0)
    assert bm.required_level(tau, lo) < bm.required_level(tau, mid) < bm.required_level(tau, hi)
    assert lo.beta > mid.beta > hi.beta
    assert bm.effective_params(5) == hi                          # clamped to [-1, 1]


def test_select_breaks_near_ties_on_expected_success_then_latency():
    p = onehot("coding")
    skills = {"a": {**onehot("coding"), "coding": 0.7}, "b": {**onehot("coding"), "coding": 0.72}}
    z = bm.logit(0.71, 0.02, 0.98)                               # a and b equally far from 0.71
    params = bm.BrickParams(mu=1, b=0, beta=0.01, lam=1.0)
    arm, scores, margin = bm.select(p, skills, {"a": 100.0, "b": 200.0}, z, params)
    assert scores[0].arm == "a" and arm == "b" and margin == 0.0  # a has the lower J, b wins the band
    arm, _, margin = bm.select(p, skills, {"a": 100.0, "b": 200.0}, z, params, tie_eps=0.0)
    assert arm == "a" and margin > 0


# ─── latency model ─────────────────────────────────────────────────────

def test_expected_latency_uses_class_mix_max_tokens_and_queue():
    prof = LatencyProfile.from_dict({"arms": {T: {"ttft_ms": 300, "ttft_ms_per_token": 0.5, "tpot_ms": 40,
                                                  "out_tokens": {"default": 1000, "easy": 400,
                                                                 "medium": 1200, "hard": 3000}}}})
    think = prof.arms[T]
    mix = {"easy": 0.5, "medium": 0.0, "hard": 0.5}
    lat = expected_latency_ms(think, prompt_tokens=100, complexity=mix)
    assert lat["out_tokens"] == 1700 and lat["total_ms"] == 300 + 50 + 40 * 1700
    assert expected_latency_ms(think, 100, mix, max_tokens=500)["out_tokens"] == 500
    assert expected_latency_ms(think, 100, None)["out_tokens"] == 1000          # no probs: default
    assert queue_wait_ms({"in_flight": 3, "capacity": 2}, 1000) == 1500
    assert queue_wait_ms(None, 1000) == 0.0


def test_config_latency_defaults_fill_arms_missing_from_the_profile_file(tmp_path):
    only_s = tmp_path / "p.json"
    only_s.write_text(json.dumps({"arms": {S: PROFILE["arms"][S]}}))
    prof = LatencyProfile.load(only_s, fallback=PROFILE)
    assert set(prof.arms) == {S, M, T} and prof.service_ms == {BACKEND: 100000}
    assert LatencyProfile.load(tmp_path / "absent.json", fallback=PROFILE).arms[T].out_tokens == {"default": 495}


# ─── router ────────────────────────────────────────────────────────────

def test_brick_skill_routes_by_capability_and_beta(mock, tmp_path):
    cfg = make_config(mock.base, arms=True)
    spec = f"brick_skill:service=brick-max,skills={skills_file(tmp_path)},latency={profile_file(tmp_path)}"

    def go(extra, text):
        return run(route_one(spec + extra, cfg, user(text), capability_fn=fake_capability))

    d = go("", "HARDQ explain history")        # world_knowledge: only the think arm has the skill
    assert d.model == T
    assert set(d.scores) == {S, M, T} and d.scores[T]["lat_ms"] == 20000 and d.scores[T]["a"] == 1.0
    assert d.signals["margin"] > 0 and d.signals["capability_ms"] >= 0 and d.signals["brick_ms"] > 0
    assert d.signals["capability"]["world_knowledge"] == 1.0
    assert go("", "HARDQ write code").model == M       # coding: 27B is enough, over-capacity costs
    assert go(",beta=100", "HARDQ explain history").model == S   # latency dominates
    assert go(",arms=qwen38-27b|qwen38-27b-think", "HARDQ write code").model == M


def test_brick_skill_live_queue_is_per_backend(mock, tmp_path):
    cfg = make_config(mock.base, arms=True)
    spec = (f"brick_skill:service=brick-max,beta=4,skills={skills_file(tmp_path)},"
            f"latency={profile_file(tmp_path)}")
    mock.load.update({S: {"in_flight": 0, "capacity": 4}, BACKEND: {"in_flight": 0, "capacity": 1}})
    idle = run(route_one(spec, cfg, user("HARDQ write code"), capability_fn=fake_capability, load_ttl_s=0))
    assert idle.model == M
    mock.load[BACKEND] = {"in_flight": 2, "capacity": 1}
    busy = run(route_one(spec, cfg, user("HARDQ write code"), capability_fn=fake_capability, load_ttl_s=0))
    assert busy.model == S
    assert busy.scores[M]["queue_ms"] == busy.scores[T]["queue_ms"] == 200000 and busy.scores[S]["queue_ms"] == 0
    offline = run(route_one(spec + ",live_load=off", cfg, user("HARDQ write code"),
                            capability_fn=fake_capability, load_ttl_s=0))
    assert offline.model == M


def test_brick_skill_capability_failure_falls_back_to_complexity_label(mock, tmp_path):
    cfg = make_config(mock.base, arms=True)
    spec = f"brick_skill:service=brick-max,skills={skills_file(tmp_path)},latency={profile_file(tmp_path)}"

    def broken(text):
        raise RuntimeError("no model")

    for text, arm in (("EASYQ", S), ("MIDQ", M), ("HARDQ", T)):
        d = run(route_one(spec, cfg, user(text), capability_fn=broken))
        assert d.model == arm and "capability error" in d.reason


def test_brick_skill_rejects_arms_without_skills_or_latency(tmp_path):
    cfg = make_config("http://127.0.0.1:1", arms=True)
    with pytest.raises(ValueError, match="skills="):
        build("brick_skill:service=brick-max", cfg, capability_fn=fake_capability)
    partial = tmp_path / "p.json"
    partial.write_text(json.dumps({"arms": {S: PROFILE["arms"][S]}}))
    with pytest.raises(ValueError, match="no latency profile"):
        build(f"brick_skill:service=brick-max,skills={skills_file(tmp_path)},latency={partial}", cfg,
              capability_fn=fake_capability)


# ─── fitting ───────────────────────────────────────────────────────────

def test_fit_skills_is_probability_weighted_success_with_clip():
    labels = [{"id": "q1", "correct": {S: True, M: True}},
              {"id": "q2", "correct": {S: False, M: True}},
              {"id": "q3", "correct": {S: False, M: True}}]
    cap = {"q1": {**onehot("coding"), "coding": 0.75, "math_reasoning": 0.25},
           "q2": {**onehot("coding"), "coding": 0.25, "math_reasoning": 0.75},
           "q3": onehot("world_knowledge")}
    table = fit_skill_table(labels, cap, [S, M])
    assert table["arms"][S]["coding"] == pytest.approx(0.75 / 1.0)
    assert table["arms"][S]["math_reasoning"] == pytest.approx(0.25 / 1.0)
    assert table["arms"][S]["world_knowledge"] == 0.02            # clipped from 0
    assert table["arms"][M]["coding"] == 0.98                     # clipped from 1
    assert table["arms"][S]["planning_agentic"] == pytest.approx(1 / 3)   # no mass: overall rate
    assert table["mass"][S]["planning_agentic"] == 0


def test_fitted_skill_table_loads_into_the_router(tmp_path):
    labels = [{"id": "q", "correct": {S: False, M: True, T: True}}]
    table = fit_skill_table(labels, {"q": onehot("coding")}, [S, M, T])
    path = tmp_path / "skills.json"
    path.write_text(json.dumps(table))
    cfg = make_config("http://127.0.0.1:1", arms=True, extra={"latency_defaults": PROFILE})
    router = build(f"brick_skill:service=brick-max,skills={path}", cfg, capability_fn=fake_capability)
    assert router.skills[S]["coding"] == 0.02 and router.skills[T]["coding"] == 0.98


def test_fit_latency_recovers_ttft_line_tpot_and_think_output_tokens():
    prompts = {f"p{i}": {"messages": [{"role": "user", "content": "x" * (40 * (i + 1))}]} for i in range(4)}
    answers = []
    for i in range(4):
        pt = 10 * (i + 1)
        answers.append({"id": f"p{i}", "model": BACKEND, "thinking": False, "error": None,
                        "prompt_tokens": pt, "completion_tokens": 101, "ttft_ms": 100 + 2 * pt,
                        "total_ms": 100 + 2 * pt + 100 * 30})
        answers.append({"id": f"p{i}", "model": BACKEND, "thinking": True, "error": None,
                        "prompt_tokens": pt, "completion_tokens": 1001 if i < 2 else 3001,
                        "ttft_ms": 100 + 2 * pt, "total_ms": 100 + 2 * pt + (1000 if i < 2 else 3000) * 30})
    complexity = {f"p{i}": ({"easy": 1.0, "medium": 0.0, "hard": 0.0} if i < 2 else
                            {"easy": 0.0, "medium": 0.0, "hard": 1.0}) for i in range(4)}
    prof = fit_profile(answers, prompts, {k: ARMS[k] for k in (M, T)}, complexity)
    m, t = prof["arms"][M], prof["arms"][T]
    assert (m["ttft_ms"], m["ttft_ms_per_token"], m["tpot_ms"]) == pytest.approx((100, 2, 30))
    assert t["out_tokens"]["easy"] == 1001 and t["out_tokens"]["hard"] == 3001
    assert t["out_tokens"]["default"] == 2001
    assert prof["chars_per_token"] == 4.0
    # one backend queue for both 27B arms
    assert set(prof["backends"]) == {BACKEND} and prof["backends"][BACKEND]["n"] == 8
