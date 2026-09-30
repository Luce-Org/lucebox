"""Fit Brick skill vectors s_m for our arms from measured success on the train split.

Brick's estimator (arXiv 2606.13241 section 7.3; brick-SR1
`packages/evals/baselines/sweep_brick_v2_wandb.py` `calibrate_skills`) is a soft,
probability-weighted success rate per capability dimension:

    s_mc = clip( sum_q p_c(q) * 1[m correct on q] / sum_q p_c(q), 0.02, 0.98 )

with p(q) the capability classifier's distribution for prompt q. Here m is an arm (a model
plus thinking mode), and "correct" is the arm's graded answer in runs/<name>/labels.jsonl.
There is no smoothing, as in the paper. A dimension with no probability mass on the train
split gets the arm's overall success rate instead of 0/0; the output records the mass per
dimension so thin dimensions are visible.

Capability probabilities are computed once per prompt with the local ModernBERT classifier
and cached in runs/<name>/capability_probs.jsonl ({id, probs, latency_ms, error}), which
evaluate_router also reads.

    uv run --extra brick python -m eval.fit_skills --run pilot     # -> runs/pilot/skills.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from lucerouter.brick_math import CAPABILITIES, SKILL_CLIP, normalise_probs

from .common import BACKENDS_PATH, PROMPTS_PATH, RUNS_DIR, append_jsonl, load_arms, read_jsonl


def fit_skill_table(labels: list[dict[str, Any]], cap_probs: dict[str, dict[str, float]],
                    arms: list[str], clip: tuple[float, float] = SKILL_CLIP) -> dict[str, Any]:
    """Skill table JSON (the format lucerouter.routers.load_skill_table reads)."""
    usable = [lab for lab in labels if lab["id"] in cap_probs]
    out: dict[str, Any] = {"capabilities": list(CAPABILITIES), "method": "brick soft success rate",
                           "clip": list(clip), "arms": {}, "mass": {}, "n": {}}
    for arm in arms:
        rows = [lab for lab in usable if arm in lab["correct"]]
        overall = sum(bool(lab["correct"][arm]) for lab in rows) / len(rows) if rows else 0.5
        skills, mass = {}, {}
        for c in CAPABILITIES:
            w = sum(cap_probs[lab["id"]][c] for lab in rows)
            hits = sum(cap_probs[lab["id"]][c] * bool(lab["correct"][arm]) for lab in rows)
            s = hits / w if w > 1e-9 else overall
            skills[c] = round(min(max(s, clip[0]), clip[1]), 6)
            mass[c] = round(w, 3)
        out["arms"][arm] = skills
        out["mass"][arm] = mass
        out["n"][arm] = len(rows)
    return out


def load_capability_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Latest successful row per prompt id."""
    return {row["id"]: row for row in read_jsonl(path) if not row.get("error") and row.get("probs")}


def classify_capabilities(fn: Callable[[str], dict[str, float]], prompts: dict[str, dict[str, Any]],
                          ids: Iterable[str], cache_path: Path) -> int:
    """Run ``fn`` on every uncached prompt; append {id, probs, latency_ms, error}. Returns the count."""
    from lucerouter.capability import routing_text

    cached = load_capability_cache(cache_path)
    todo = [i for i in dict.fromkeys(ids) if i not in cached and i in prompts]
    for pid in todo:
        t0 = time.perf_counter()
        try:
            probs, err = normalise_probs(fn(routing_text(prompts[pid]["messages"]))), None
        except Exception as e:  # noqa: BLE001 - recorded, retried next run
            probs, err = None, f"{type(e).__name__}: {e}"
        append_jsonl(cache_path, {"id": pid, "probs": probs, "error": err,
                                  "latency_ms": round((time.perf_counter() - t0) * 1000, 2)})
    return len(todo)


def local_capability_fn(model_dir: str | None = None) -> Callable[[str], dict[str, float]]:
    from lucerouter.capability import DEFAULT_MODEL_DIR, get_classifier

    return get_classifier(model_dir or DEFAULT_MODEL_DIR).predict


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--config", type=Path, default=BACKENDS_PATH)
    ap.add_argument("--split", choices=["train", "test", "all"], default="train")
    ap.add_argument("--capability-model", default=None, help="ModernBERT dir (default lucerouter's)")
    ap.add_argument("--out", type=Path, default=None, help="default runs/<name>/skills.json")
    args = ap.parse_args(argv)
    if not (args.run or args.run_dir):
        ap.error("--run or --run-dir required")
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / args.run
    arms = list(load_arms(args.config))
    labels = [lab for lab in read_jsonl(run_dir / "labels.jsonl")
              if args.split == "all" or lab["split"] == args.split]
    prompts = {p["id"]: p for p in read_jsonl(args.prompts)}
    cache = run_dir / "capability_probs.jsonl"
    n_new = classify_capabilities(local_capability_fn(args.capability_model), prompts,
                                  [lab["id"] for lab in labels], cache)
    probs = {i: r["probs"] for i, r in load_capability_cache(cache).items()}
    table = fit_skill_table(labels, probs, arms)
    table.update(run=str(run_dir), split=args.split)
    out = args.out or run_dir / "skills.json"
    out.write_text(json.dumps(table, indent=2) + "\n")
    print(f"classified {n_new} new prompts; fitted {len(arms)} arms on {len(labels)} {args.split} labels")
    for arm in arms:
        print(f"  {arm:<18} " + " ".join(f"{c[:6]}={table['arms'][arm][c]:.2f}" for c in CAPABILITIES))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
