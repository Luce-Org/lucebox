"""Score the Brick routers offline against measured labels (runs/<name>/labels.jsonl).

Everything is per arm (config "arms": by default qwen35-2b, qwen38-27b and
qwen38-27b-think, the 27B with thinking on). Four reports, written to runs/<name>/router_eval/:

1. Baselines: always-<arm> for each arm, and the oracle (cheapest arm whose
   answer was correct; if none, the best-scoring one).
2. Brick threshold sweep, per Brick service (default brick-max and brick-eco) and per
   arm pool ({small, mid, large}, {mid, large}, {small, large}): a grid over the
   pool's relevant thresholds (t_small, t_mid, h_max) replayed through the router's
   own ``BrickRouter.choose``. Each point reports quality (mean score of the chosen
   model's answer), expected latency (chosen model's measured total_ms + the Brick
   call), the under-route rate (sent to a model that failed while a bigger model in
   the pool passed) and the over-route rate; Pareto-optimal points are marked.
3. brick_skill beta sweep (Brick's J = D + beta * latency rule), per service and pool, when a
   skill table exists (runs/<name>/skills.json from eval.fit_skills). The latency term uses
   the static profile (runs/<name>/latency_profile.json from eval.fit_latency, else the
   config's latency_defaults) with no live queue. The measured latency of a point is the chosen
   arm's total_ms plus capability_ms + brick_ms (an upper bound: live, the two classifiers run
   concurrently). Points are Pareto-marked within the sweep and together with the pool's
   threshold points (pareto_combined).
4. Brick calibration: reliability tables of each class probability against the
   cheapest arm that was actually OK (easy <-> small ok, medium <-> mid is the
   cheapest ok, hard <-> only large ok or none), with ECE, plus an argmax confusion
   table.

Brick is called once per prompt through the offline router API
(``lucerouter.routers.build("brick:service=S", config)`` + ``await router.route(...)``)
and the probabilities are cached in runs/<name>/brick_probs.jsonl, so a sweep is one
classifier pass however large the grid. Capability probabilities are cached the same way in
runs/<name>/capability_probs.jsonl (see eval.fit_skills).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .common import BACKENDS_PATH, PROMPTS_PATH, RUNS_DIR, append_jsonl, read_jsonl
from .fit_skills import classify_capabilities, load_capability_cache, local_capability_fn
from .label import oracle_choice, pool_cheapest_ok

DEFAULT_SERVICES = ["brick-max", "brick-eco"]
DEFAULT_GRID = {
    "t_small": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01],
    "t_mid": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
    "h_max": [0.2, 0.35, 0.5, 0.7, 1.01],
}
CLASSES = ("easy", "medium", "hard")
DEFAULT_BETAS = [0.0, 0.05, 0.1, 0.23, 0.5, 1.0, 2.0, 4.0, 8.0]


# --------------------------------------------------------------------------- scoring (pure)


def score_decisions(
    labels: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    pool: list[str],
) -> dict[str, Any]:
    """Score ``decisions`` ({id: {"model", "latency_ms"?, "error"?}}) within ``pool``
    (ordered smallest -> largest). Targets outside the pool count as the largest."""
    rank = {m: i for i, m in enumerate(pool)}
    dist: Counter[str] = Counter()
    q, lat, router_ms = [], [], []
    under = over = unmapped = errors = ok = n = 0
    for lab in labels:
        d = decisions.get(lab["id"])
        if d is None:
            continue
        n += 1
        m = d["model"]
        if m not in rank:
            m, unmapped = pool[-1], unmapped + 1
        errors += bool(d.get("error"))
        dist[m] += 1
        q.append(lab["quality"][m])
        ok += bool(lab["correct"].get(m))
        rms = float(d.get("latency_ms") or 0.0)
        router_ms.append(rms)
        lat.append(lab["latency"].get(m, 0.0) + rms)
        if not lab["correct"].get(m) and any(lab["correct"].get(b) for b in pool[rank[m] + 1:]):
            under += 1
        cheapest = pool_cheapest_ok(lab, pool)
        if cheapest is not None and rank[m] > rank[cheapest]:
            over += 1
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "distribution": {m: dist[m] / n for m in pool if dist[m]},
        "quality": statistics.fmean(q),
        "ok_rate": ok / n,
        "under_route_rate": under / n,
        "over_route_rate": over / n,
        "mean_latency_ms": statistics.fmean(lat),
        "median_latency_ms": statistics.median(lat),
        "mean_router_ms": statistics.fmean(router_ms),
        "unmapped_rate": unmapped / n,
        "error_rate": errors / n,
    }


def baselines(labels: list[dict[str, Any]], models: list[str]) -> dict[str, dict[str, Any]]:
    out = {f"always_{m}": score_decisions(labels, {lab["id"]: {"model": m} for lab in labels}, models)
           for m in models}
    out["oracle"] = score_decisions(
        labels, {lab["id"]: {"model": oracle_choice(lab, models)} for lab in labels}, models)
    return out


def pareto_front(points: list[dict[str, Any]], q_key: str = "quality", l_key: str = "mean_latency_ms") -> None:
    """Mark ``pareto`` on each point: no other point has >= quality and <= latency (one strict)."""
    for p in points:
        p["pareto"] = not any(
            o is not p and o[q_key] >= p[q_key] and o[l_key] <= p[l_key]
            and (o[q_key] > p[q_key] or o[l_key] < p[l_key])
            for o in points
        )


# --------------------------------------------------------------------------- pools


def model_pools(models: list[str]) -> dict[str, dict[str, str | None]]:
    """Named pools as Brick tiers. ``models`` ordered smallest -> largest (3 expected)."""
    if len(models) < 3:
        return {"+".join(models): {"small": None, "mid": models[0] if len(models) == 2 else None,
                                   "large": models[-1]}}
    s, m, lg = models[0], models[1], models[-1]
    return {
        f"{s}+{m}+{lg}": {"small": s, "mid": m, "large": lg},
        f"{m}+{lg}": {"small": None, "mid": m, "large": lg},
        f"{s}+{lg}": {"small": s, "mid": None, "large": lg},
    }


def pool_models(tiers: dict[str, str | None]) -> list[str]:
    return [tiers[k] for k in ("small", "mid", "large") if tiers[k]]


def pool_grid(tiers: dict[str, str | None], grid: dict[str, list[float]]) -> list[dict[str, float]]:
    """Only the thresholds that matter for this pool (t_small needs a small tier, etc.)."""
    keys = (["t_small"] if tiers["small"] else []) + (["t_mid", "h_max"] if tiers["mid"] else [])
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]


def sweep(
    labels: list[dict[str, Any]],
    probs: dict[str, dict[str, Any]],
    pool: list[str],
    grid: list[dict[str, float]],
    chooser_for: Callable[[dict[str, float]], Callable[[dict[str, float]], str]],
) -> list[dict[str, Any]]:
    """Replay each grid point over cached probs. ``chooser_for(params)(probs) -> model``."""
    usable = [lab for lab in labels if lab["id"] in probs]
    points = []
    for params in grid:
        choose = chooser_for(params)
        decisions = {lab["id"]: {"model": choose(probs[lab["id"]]["probs"]),
                                 "latency_ms": probs[lab["id"]]["latency_ms"]} for lab in usable}
        points.append({**params, **score_decisions(usable, decisions, pool)})
    pareto_front(points)
    return points


def skill_sweep(
    labels: list[dict[str, Any]],
    probs: dict[str, dict[str, Any]],
    cap: dict[str, dict[str, Any]],
    prompt_chars: dict[str, int],
    pool: list[str],
    betas: list[float],
    router_for: Callable[[float], Any],
) -> list[dict[str, Any]]:
    """Replay brick_skill's ``choose`` per beta. Router time = brick_ms + capability_ms."""
    usable = [lab for lab in labels if lab["id"] in probs and lab["id"] in cap]
    points = []
    for beta in betas:
        router = router_for(beta)
        decisions = {}
        for lab in usable:
            i = lab["id"]
            arm, _, _ = router.choose(probs[i]["probs"], cap[i]["probs"], prompt_chars[i])
            decisions[i] = {"model": arm, "latency_ms": probs[i]["latency_ms"] + cap[i]["latency_ms"]}
        points.append({"beta": beta, **score_decisions(usable, decisions, pool)})
    pareto_front(points)
    return points


def mark_combined_pareto(*sweeps: list[dict[str, Any]]) -> None:
    """Set ``pareto_combined`` on every point, judged against all the sweeps' points together."""
    everything = [p for s in sweeps for p in s if p.get("n")]
    flags = [dict(p) for p in everything]
    pareto_front(flags)
    for p, f in zip(everything, flags):
        p["pareto_combined"] = f["pareto"]


# --------------------------------------------------------------------------- calibration


def outcome_class(label: dict[str, Any], models: list[str]) -> str:
    """What Brick should have said: easy (small ok), medium (mid is cheapest ok), hard."""
    cheapest = pool_cheapest_ok(label, models)
    if cheapest == models[0]:
        return "easy"
    if len(models) >= 3 and cheapest == models[1]:
        return "medium"
    return "hard"


def calibration(
    labels: list[dict[str, Any]],
    probs: dict[str, dict[str, Any]],
    models: list[str],
    n_bins: int = 5,
) -> dict[str, Any]:
    usable = [lab for lab in labels if lab["id"] in probs]
    truth = {lab["id"]: outcome_class(lab, models) for lab in usable}
    out: dict[str, Any] = {"n": len(usable), "classes": {}}
    for cls in CLASSES:
        bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "p_sum": 0.0, "hits": 0}
                for i in range(n_bins)]
        for lab in usable:
            p = float(probs[lab["id"]]["probs"][cls])
            b = bins[min(int(p * n_bins), n_bins - 1)]
            b["n"] += 1
            b["p_sum"] += p
            b["hits"] += truth[lab["id"]] == cls
        ece = 0.0
        for b in bins:
            b["mean_p"] = b.pop("p_sum") / b["n"] if b["n"] else None
            b["observed"] = b["hits"] / b["n"] if b["n"] else None
            if b["n"]:
                ece += b["n"] / len(usable) * abs(b["observed"] - b["mean_p"])
        base = sum(t == cls for t in truth.values()) / len(usable) if usable else None
        out["classes"][cls] = {"bins": bins, "ece": ece, "base_rate": base}
    confusion: Counter[tuple[str, str]] = Counter()
    for lab in usable:
        pr = probs[lab["id"]]["probs"]
        confusion[(max(CLASSES, key=lambda c: pr[c]), truth[lab["id"]])] += 1
    out["confusion"] = {f"{p}->{t}": n for (p, t), n in sorted(confusion.items())}
    return out


# --------------------------------------------------------------------------- Brick pass (cached)


def _route_request(prompt: dict[str, Any], rid: str):
    from lucerouter.types import RouteRequest

    return RouteRequest(messages=prompt["messages"], tools=None, session_id=rid,
                        max_tokens=None, raw={"messages": prompt["messages"]})


def load_probs_cache(path: Path, service: str) -> dict[str, dict[str, Any]]:
    """Latest successful cached Brick result per prompt id for ``service``."""
    out = {}
    for row in read_jsonl(path):
        if row["service"] == service and not row.get("error") and row.get("probs"):
            out[row["id"]] = row
    return out


async def classify_missing(
    router: Any,
    service: str,
    prompts: dict[str, dict[str, Any]],
    ids: Iterable[str],
    cache_path: Path,
    make_request: Callable[[dict[str, Any], str], Any] = _route_request,
    concurrency: int = 1,
) -> int:
    """Route every uncached prompt once; append {id, service, probs, latency_ms, error}."""
    cached = load_probs_cache(cache_path, service)
    todo = [i for i in ids if i not in cached and i in prompts]
    sem = asyncio.Semaphore(concurrency)

    async def one(pid: str) -> None:
        async with sem:
            try:
                d = await router.route(make_request(prompts[pid], pid))
                err = (d.signals or {}).get("error")
                probs = None if err else {c: float(d.scores[c]) for c in CLASSES}
                row = {"id": pid, "service": service, "probs": probs,
                       "latency_ms": d.latency_ms, "error": err}
            except Exception as e:  # noqa: BLE001 - recorded, retried next run
                row = {"id": pid, "service": service, "probs": None, "latency_ms": None,
                       "error": f"{type(e).__name__}: {e}"}
            append_jsonl(cache_path, row)

    await asyncio.gather(*(one(i) for i in todo))
    return len(todo)


# --------------------------------------------------------------------------- report


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _dist(d: dict[str, float]) -> str:
    return ", ".join(f"{m} {100 * v:.0f}%" for m, v in d.items())


def baselines_markdown(base: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["| policy | quality | under | over | mean ms | median ms |", "|---|---|---|---|---|---|"]
    for name, r in base.items():
        lines.append(f"| {name} | {_pct(r['quality'])} | {_pct(r['under_route_rate'])} | "
                     f"{_pct(r['over_route_rate'])} | {r['mean_latency_ms']:.0f} | {r['median_latency_ms']:.0f} |")
    return lines


def sweep_markdown(service: str, pool_name: str, points: list[dict[str, Any]],
                   pool_base: dict[str, dict[str, Any]], target_q: float | None) -> list[str]:
    keys = [k for k in ("t_small", "t_mid", "h_max") if points and k in points[0]]
    lines = [f"### {service} / pool {pool_name}\n",
             "Pool baselines: " + "; ".join(
                 f"{n} {_pct(r['quality'])} @ {r['mean_latency_ms']:.0f} ms" for n, r in pool_base.items()),
             ""]
    if target_q is not None:
        ok = [p for p in points if p["quality"] >= target_q]
        if ok:
            best = min(ok, key=lambda p: p["mean_latency_ms"])
            params = ", ".join(f"{k}={best[k]}" for k in keys)
            lines += [f"Fastest point with quality >= {_pct(target_q)}: {params} -> "
                      f"{_pct(best['quality'])} @ {best['mean_latency_ms']:.0f} ms, "
                      f"under-route {_pct(best['under_route_rate'])}, {_dist(best['distribution'])}", ""]
    lines += ["| " + " | ".join(keys) + " | quality | mean ms | under (too small) | over | distribution |",
              "|---" * (len(keys) + 5) + "|"]
    for p in sorted((p for p in points if p["pareto"]), key=lambda p: p["mean_latency_ms"]):
        lines.append("| " + " | ".join(str(p[k]) for k in keys) + f" | {_pct(p['quality'])} | "
                     f"{p['mean_latency_ms']:.0f} | {_pct(p['under_route_rate'])} | "
                     f"{_pct(p['over_route_rate'])} | {_dist(p['distribution'])} |")
    return lines + [""]


def skill_markdown(service: str, pool_name: str, points: list[dict[str, Any]]) -> list[str]:
    lines = [f"### brick_skill / {service} / pool {pool_name}\n",
             "Every beta point; P = Pareto within the sweep, C = Pareto together with the threshold sweep.\n",
             "| beta | quality | mean ms | router ms | under (too small) | over | distribution | P | C |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in points:
        if not p.get("n"):
            continue
        lines.append(f"| {p['beta']} | {_pct(p['quality'])} | {p['mean_latency_ms']:.0f} | "
                     f"{p['mean_router_ms']:.0f} | {_pct(p['under_route_rate'])} | {_pct(p['over_route_rate'])} | "
                     f"{_dist(p['distribution'])} | {'P' if p['pareto'] else ''} | "
                     f"{'C' if p.get('pareto_combined') else ''} |")
    return lines + [""]


def calibration_markdown(service: str, cal: dict[str, Any]) -> list[str]:
    lines = [f"### {service} calibration ({cal['n']} prompts)\n",
             "Observed = share of prompts whose cheapest OK arm matches the class "
             "(easy: small ok; medium: mid is the cheapest ok; hard: only large ok or none).\n"]
    for cls, c in cal["classes"].items():
        lines += [f"**p_{cls}** (base rate {_pct(c['base_rate'])}, ECE {c['ece']:.3f})\n",
                  "| bin | n | mean p | observed |", "|---|---|---|---|"]
        for b in c["bins"]:
            lines.append(f"| {b['lo']:.1f}-{b['hi']:.1f} | {b['n']} | "
                         f"{'-' if b['mean_p'] is None else f'{b['mean_p']:.2f}'} | {_pct(b['observed'])} |")
        lines.append("")
    lines += ["Argmax class -> actual class: " + ", ".join(f"{k}: {v}" for k, v in cal["confusion"].items()), ""]
    return lines


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "_", s)[:120]


# --------------------------------------------------------------------------- CLI


async def run(args: argparse.Namespace) -> None:
    import httpx
    from lucerouter.config import load_config
    from lucerouter.routers import build

    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / args.run
    out_dir = run_dir / "router_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    models = [a.name for a in config.arms_by_rank()]
    all_labels = [lab for lab in read_jsonl(run_dir / "labels.jsonl") if lab["complete"]]
    missing = [m for m in models if all_labels and m not in all_labels[0]["quality"]]
    if missing:
        models = [m for m in models if m not in missing]
        print(f"note: no labels for arms {missing}; evaluating over {models}")
    labels = all_labels if args.split == "all" else [lab for lab in all_labels if lab["split"] == args.split]
    prompts = {p["id"]: p for p in read_jsonl(args.prompts)}
    grid = {k: _floats(getattr(args, f"{k}_grid")) for k in DEFAULT_GRID}
    cache = run_dir / "brick_probs.jsonl"
    skills_path = Path(args.skills) if args.skills else run_dir / "skills.json"
    latency_path = Path(args.latency_profile) if args.latency_profile else run_dir / "latency_profile.json"
    cap = load_skill_inputs(args, run_dir, prompts, [lab["id"] for lab in labels], skills_path)
    chars = prompt_chars(prompts)

    base = baselines(labels, models)
    report: dict[str, Any] = {"split": args.split, "n": len(labels), "models": models,
                              "baselines": base, "services": {}}
    md = [f"# Brick router evaluation ({args.split} split, {len(labels)} prompts)\n",
          "quality = mean score of the chosen arm's answer; latency = chosen arm's measured "
          "total_ms + router time (Brick call, plus the capability classifier for brick_skill); "
          "under = sent to an arm that failed while a higher-rank arm in the pool passed; "
          "over = a lower-rank arm in the pool would have passed.\n",
          "## Baselines\n", *baselines_markdown(base), ""]

    async with httpx.AsyncClient() as client:
        for service in args.service:
            router = build(f"brick:service={service}", config, client=client)
            n_new = await classify_missing(router, service, prompts, [lab["id"] for lab in all_labels],
                                           cache, concurrency=args.concurrency)
            probs = load_probs_cache(cache, service)
            n_ok = sum(lab["id"] in probs for lab in labels)
            print(f"{service}: classified {n_new} new prompts; {n_ok}/{len(labels)} have probabilities")
            svc: dict[str, Any] = {"n_with_probs": n_ok, "pools": {}}
            md += [f"## {service}\n", f"{n_ok}/{len(labels)} prompts have Brick probabilities; "
                   f"mean Brick latency "
                   f"{statistics.fmean([probs[lab['id']]['latency_ms'] for lab in labels if lab['id'] in probs] or [0]):.1f} ms.\n"]
            for pool_name, tiers in model_pools(models).items():
                pool = pool_models(tiers)

                def chooser_for(params, tiers=tiers, service=service):
                    spec_params = {k: ("none" if v is None else v) for k, v in tiers.items()}
                    spec_params.update(params)
                    spec = "brick:" + ",".join(f"{k}={v}" for k, v in {"service": service, **spec_params}.items())
                    r = build(spec, config, client=client)
                    return lambda p: r.choose(p)[0]

                points = sweep(labels, probs, pool, pool_grid(tiers, grid), chooser_for)
                pool_base = baselines([lab for lab in labels if lab["id"] in probs], pool)
                target = base[f"always_{models[-1]}"]["quality"] - args.quality_tolerance
                svc["pools"][pool_name] = {"baselines": pool_base, "points": points}
                if cap:
                    spec = (f"brick_skill:service={service},skills={skills_path},live_load=off,"
                            f"r={args.skill_r},arms={'|'.join(pool)}"
                            + (f",latency={latency_path}" if latency_path.exists() else ""))

                    def router_for(beta, spec=spec):
                        return build(f"{spec},beta={beta}", config, client=client, capability_fn=_no_capability)

                    skill_points = skill_sweep(labels, probs, cap, chars, pool, _floats(args.beta_grid), router_for)
                    mark_combined_pareto(points, skill_points)
                    svc["pools"][pool_name]["brick_skill"] = skill_points
                md += sweep_markdown(service, pool_name, points, pool_base, target)
                if cap:
                    md += skill_markdown(service, pool_name, svc["pools"][pool_name]["brick_skill"])
            cal = calibration(labels, probs, models, n_bins=args.bins)
            svc["calibration"] = cal
            md += calibration_markdown(service, cal)
            report["services"][service] = svc

    (out_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    text = "\n".join(md) + "\n"
    (out_dir / "results.md").write_text(text)
    print(text)
    print(f"wrote {out_dir / 'results.md'} and results.json (probabilities cached in {cache})")


def _no_capability(text: str) -> dict[str, float]:
    raise RuntimeError("offline replay uses cached capability probs")


def prompt_chars(prompts: dict[str, dict[str, Any]]) -> dict[str, int]:
    from lucerouter.types import RouteRequest

    return {i: RouteRequest(messages=p["messages"]).prompt_chars() for i, p in prompts.items()}


def load_skill_inputs(args: argparse.Namespace, run_dir: Path, prompts: dict[str, dict[str, Any]],
                      ids: list[str], skills_path: Path) -> dict[str, dict[str, Any]]:
    """Cached capability probs for ``ids`` (classifying missing ones), or {} to skip brick_skill."""
    if not skills_path.exists():
        print(f"note: no skill table at {skills_path}; skipping brick_skill (run eval.fit_skills)")
        return {}
    cache = run_dir / "capability_probs.jsonl"
    missing = [i for i in ids if i not in load_capability_cache(cache)]
    if missing and not args.no_classify:
        try:
            fn = local_capability_fn(args.capability_model)
            fn("warm up")
        except Exception as e:  # noqa: BLE001 - optional extra or model missing
            print(f"note: capability classifier unavailable ({e}); using cached probs only")
        else:
            n = classify_capabilities(fn, prompts, missing, cache)
            print(f"capability: classified {n} new prompts")
    return load_capability_cache(cache)


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--config", type=Path, default=BACKENDS_PATH)
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--service", action="append", default=None, metavar="BRICK_SERVICE",
                    help=f"Brick services to evaluate (default {DEFAULT_SERVICES})")
    for k, v in DEFAULT_GRID.items():
        ap.add_argument(f"--{k.replace('_', '-')}-grid", dest=f"{k}_grid", default=",".join(map(str, v)))
    ap.add_argument("--quality-tolerance", type=float, default=0.02,
                    help="headline: fastest point within this of always-largest quality")
    ap.add_argument("--bins", type=int, default=5, help="calibration bins")
    ap.add_argument("--skills", default=None, help="brick_skill skill table (default runs/<name>/skills.json)")
    ap.add_argument("--latency-profile", default=None,
                    help="brick_skill latency profile (default runs/<name>/latency_profile.json, "
                         "else the config's latency_defaults)")
    ap.add_argument("--beta-grid", default=",".join(map(str, DEFAULT_BETAS)))
    ap.add_argument("--skill-r", type=float, default=0.0, help="Brick knob r for mu/b/lambda in the beta sweep")
    ap.add_argument("--capability-model", default=None, help="ModernBERT dir for missing capability probs")
    ap.add_argument("--no-classify", action="store_true", help="use cached capability probs only")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel Brick calls")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not (args.run or args.run_dir):
        raise SystemExit("--run or --run-dir required")
    args.service = args.service or DEFAULT_SERVICES
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
