"""label.py headroom maths and evaluate_router.py scoring, Brick sweep, calibration and cache.

Four prompts with hand-computed outcomes over the default arms (S=2b, M=27b think off,
L=27b think on; M and L share one backend model and differ only in the thinking flag):
    a: everyone ok         -> cheapest S
    b: S fails, M/L ok     -> cheapest M
    c: only L ok           -> cheapest L
    d: nobody ok           -> cheapest None
Latency per answer: S 100 ms, M 200 ms, L 1000 ms.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from eval import evaluate_router as ev
from eval.common import DEFAULT_ARMS as ARMS
from eval.label import build_labels, headroom
from mocks import make_config

S, M, L = "qwen35-2b", "qwen38-27b", "qwen38-27b-think"
MODELS = [S, M, L]
OK = {"a": {S, M, L}, "b": {M, L}, "c": {L}, "d": set()}
TEXT = {"a": "EASYQ say hi", "b": "MIDQ summarize", "c": "HARDQ design", "d": "HARDQ prove"}
LAT = {S: 100.0, M: 200.0, L: 1000.0}


def prompts(split="test"):
    return [{"id": i, "split": split, "source": "t", "category": "easy" if i == "a" else "hard",
             "messages": [{"role": "user", "content": TEXT[i]}]} for i in OK]


def grades_and_answers(ids=OK):
    grades, answers = [], []
    for i in ids:
        for m in MODELS:
            key = {"id": i, "model": ARMS[m]["model"], "thinking": ARMS[m]["thinking"]}
            grades.append({**key, "correct": m in OK[i], "score": float(m in OK[i]), "grader": "x", "detail": {}})
            answers.append({**key, "total_ms": LAT[m], "ttft_ms": 5.0, "prompt_tokens": 10,
                            "completion_tokens": int(LAT[m] / 10)})
    return grades, answers


@pytest.fixture
def labels():
    return build_labels(prompts(), *grades_and_answers(), ARMS)


def test_labels_cheapest_ok_and_completeness(labels):
    assert {lab["id"]: lab["cheapest_ok"] for lab in labels} == {"a": S, "b": M, "c": L, "d": None}
    assert all(lab["complete"] for lab in labels)
    grades, answers = grades_and_answers()
    partial = build_labels(prompts(), [g for g in grades if not g["thinking"]], answers, ARMS)
    assert not any(lab["complete"] for lab in partial)
    assert {lab["id"]: sorted(lab["quality"]) for lab in partial}["a"] == [S, M]   # think arm missing


def test_headroom_shares_and_policies(labels):
    h = headroom(labels, MODELS)
    o = h["overall"]
    assert (o[f"{S}_ok"], o[f"{M}_ok"], o[f"{L}_ok"]) == (0.25, 0.5, 0.75)
    assert (o[f"only_{L}_ok"], o["none_ok"]) == (0.25, 0.25)
    assert h["by_category"]["easy"][f"{S}_ok"] == 1.0
    # oracle: a->S, b->M, c->L, d (nobody ok) -> S
    assert h["policies"]["oracle"]["quality"] == 0.75
    assert h["policies"]["oracle"]["mean_latency_ms"] == (100 + 200 + 1000 + 100) / 4
    assert h["policies"][f"always_{L}"]["quality"] == 0.75
    assert h["policies"][f"always_{L}"]["mean_latency_ms"] == 1000
    assert h["policies"][f"always_{S}"]["quality"] == 0.25


def test_score_under_over_latency_within_pool(labels):
    decisions = {"a": {"model": L, "latency_ms": 10.0}, "b": {"model": S, "latency_ms": 10.0},
                 "c": {"model": M, "latency_ms": 10.0}, "d": {"model": M, "latency_ms": 10.0}}
    r = ev.score_decisions(labels, decisions, MODELS)
    assert r["quality"] == 0.25                 # only a (L) is correct
    assert r["under_route_rate"] == 0.5         # b (S failed, M ok), c (M failed, L ok)
    assert r["over_route_rate"] == 0.25         # a (S was enough)
    assert r["mean_latency_ms"] == (1000 + 100 + 200 + 200 + 4 * 10) / 4
    assert r["distribution"] == {S: 0.25, M: 0.5, L: 0.25}
    # in the {M, L} pool, M is the cheapest option, so a->M is not over-routing
    r = ev.score_decisions(labels, {"a": {"model": M}, "b": {"model": L}}, [M, L])
    assert (r["over_route_rate"], r["under_route_rate"]) == (0.5, 0.0)   # b: M was enough


def test_baselines_oracle_has_no_under_or_over_routing(labels):
    base = ev.baselines(labels, MODELS)
    assert (base["oracle"]["under_route_rate"], base["oracle"]["over_route_rate"]) == (0.0, 0.0)
    assert base["oracle"]["quality"] == 0.75
    assert base[f"always_{S}"]["under_route_rate"] == 0.5   # b and c
    assert base[f"always_{L}"]["over_route_rate"] == 0.5    # a and b


BRICK = {  # same distributions the mock Brick service returns for the marker words
    "a": {"easy": 0.9, "medium": 0.08, "hard": 0.02},
    "b": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    "c": {"easy": 0.05, "medium": 0.15, "hard": 0.8},
    "d": {"easy": 0.05, "medium": 0.15, "hard": 0.8},
}


def test_calibration_reliability_against_cheapest_ok(labels):
    probs = {i: {"probs": p, "latency_ms": 1.0} for i, p in BRICK.items()}
    cal = ev.calibration(labels, probs, MODELS, n_bins=5)
    easy = cal["classes"]["easy"]
    # a: p=.9 and S ok (hit); b: p=.3, miss; c, d: p=.05, miss
    assert [(b["n"], b["observed"]) for b in easy["bins"]] == [(2, 0.0), (1, 0.0), (0, None), (0, None), (1, 1.0)]
    assert easy["ece"] == pytest.approx(0.25 * 0.1 + 0.25 * 0.3 + 0.5 * 0.05)
    assert cal["classes"]["hard"]["base_rate"] == 0.5          # c (only L) and d (none)
    assert cal["confusion"] == {"easy->easy": 1, "hard->hard": 2, "medium->medium": 1}


def test_pool_grids_only_vary_relevant_thresholds():
    grid = {"t_small": [0.5, 0.9], "t_mid": [0.4], "h_max": [0.3, 0.6]}
    pools = ev.model_pools(MODELS)
    assert {n: len(ev.pool_grid(t, grid)) for n, t in pools.items()} == {
        f"{S}+{M}+{L}": 4, f"{M}+{L}": 2, f"{S}+{L}": 2}


class CountingRouter:
    def __init__(self, fail_ids=()):
        self.calls = []
        self.fail_ids = set(fail_ids)

    async def route(self, req):
        self.calls.append(req.session_id)
        if req.session_id in self.fail_ids:
            raise RuntimeError("brick down")
        return SimpleNamespace(model=L, latency_ms=3.0, reason="", scores=BRICK[req.session_id], signals={})


def test_brick_probs_cache_is_one_pass_and_retries_failures(tmp_path):
    cache = tmp_path / "brick_probs.jsonl"
    ps = {p["id"]: p for p in prompts()}
    first = CountingRouter(fail_ids={"d"})
    asyncio.run(ev.classify_missing(first, "brick-max", ps, list(OK), cache))
    assert sorted(first.calls) == ["a", "b", "c", "d"]
    assert set(ev.load_probs_cache(cache, "brick-max")) == {"a", "b", "c"}
    second = CountingRouter()
    asyncio.run(ev.classify_missing(second, "brick-max", ps, list(OK), cache))
    assert second.calls == ["d"]                              # only the failure is retried
    assert ev.load_probs_cache(cache, "brick-eco") == {}      # cache is per service


def write_cli_inputs(mock, tmp_path, labels):
    cfg = make_config(mock.base)
    config = {"models": {m.name: {"base_url": m.base_url, "rank": m.rank} for m in cfg.models.values()},
              "services": {"brick-max": {"base_url": f"{mock.base}/brick"}}, "arms": ARMS}
    (tmp_path / "backends.json").write_text(json.dumps(config))
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts()))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "labels.jsonl").write_text("".join(json.dumps(lab) + "\n" for lab in labels))
    argv = ["--run-dir", str(run_dir), "--prompts", str(tmp_path / "prompts.jsonl"),
            "--config", str(tmp_path / "backends.json"), "--service", "brick-max",
            "--t-small-grid", "0.8,1.01", "--t-mid-grid", "0.5,1.01", "--h-max-grid", "0.5"]
    return run_dir, argv


def test_cli_brick_sweep_through_real_router(mock, tmp_path, labels):
    """End to end through lucerouter.build against the mock Brick service."""
    run_dir, argv = write_cli_inputs(mock, tmp_path, labels)
    ev.main(argv)
    assert len(mock.brick_requests) == 4

    report = json.loads((run_dir / "router_eval" / "results.json").read_text())
    pools = report["services"]["brick-max"]["pools"]
    full = {(p["t_small"], p["t_mid"]): p for p in pools[f"{S}+{M}+{L}"]["points"]}
    # EASYQ -> S, MIDQ -> M, HARDQ -> L: each prompt goes to its cheapest ok model
    best = full[(0.8, 0.5)]
    assert best["distribution"] == {S: 0.25, M: 0.25, L: 0.5}
    assert (best["quality"], best["under_route_rate"], best["over_route_rate"]) == (0.75, 0.0, 0.0)
    assert best["pareto"]
    assert full[(1.01, 1.01)]["distribution"] == {L: 1.0} and not full[(1.01, 1.01)]["pareto"]
    no_mid = {p["t_small"]: p for p in pools[f"{S}+{L}"]["points"]}
    assert no_mid[0.8]["distribution"] == {S: 0.25, L: 0.75}   # b (MIDQ) has nowhere but L
    assert report["baselines"]["oracle"]["quality"] == 0.75
    assert "p_easy" in (run_dir / "router_eval" / "results.md").read_text()

    ev.main(argv)                                              # rerun: probabilities come from cache
    assert len(mock.brick_requests) == 4


def test_cli_brick_skill_beta_sweep_from_cached_capabilities(mock, tmp_path, labels):
    """brick_skill replays over cached capability probs; beta trades the 27B arms for the 2b."""
    run_dir, argv = write_cli_inputs(mock, tmp_path, labels)
    caps = ("coding", "creative_synthesis", "instruction_following", "math_reasoning",
            "planning_agentic", "world_knowledge")
    kind = {"a": "creative_synthesis", "b": "coding", "c": "math_reasoning", "d": "math_reasoning"}
    (run_dir / "capability_probs.jsonl").write_text("".join(
        json.dumps({"id": i, "probs": {c: float(c == k) for c in caps}, "latency_ms": 50.0, "error": None}) + "\n"
        for i, k in kind.items()))
    skill = {S: 0.3, M: 0.8, L: 0.95}                      # same on every dimension
    (run_dir / "skills.json").write_text(json.dumps({"arms": {a: dict.fromkeys(caps, v) for a, v in skill.items()}}))
    (run_dir / "latency_profile.json").write_text(json.dumps({"arms": {
        a: {"ttft_ms": 0, "tpot_ms": 1, "out_tokens": {"default": LAT[a]}} for a in MODELS}}))
    ev.main(argv + ["--no-classify", "--beta-grid", "0,100"])

    report = json.loads((run_dir / "router_eval" / "results.json").read_text())
    pts = {p["beta"]: p for p in report["services"]["brick-max"]["pools"][f"{S}+{M}+{L}"]["brick_skill"]}
    assert pts[0.0]["distribution"] == {M: 1.0}           # 27B off meets every requirement; think over-capacity
    assert pts[100.0]["distribution"] == {S: 1.0}         # latency dominates
    brick_ms = [r["latency_ms"] for r in ev.load_probs_cache(run_dir / "brick_probs.jsonl", "brick-max").values()]
    assert pts[100.0]["mean_router_ms"] == pytest.approx(50.0 + sum(brick_ms) / 4)   # capability + Brick
    assert all("pareto_combined" in p for p in pts.values())
    assert "brick_skill / brick-max" in (run_dir / "router_eval" / "results.md").read_text()
