#!/usr/bin/env python3
"""Find each model's baseline: the newest baseline artifact from a main run.

Baseline runs on main (every merge, or dispatched with update_baseline) upload
their result as the artifact `model-e2e-baseline-<model>-<device>`. This reads
the select job's matrix on stdin and adds to each entry the `baseline_run` that
holds its baseline, or "" when there is none yet.

With --only-changed HEAD, it keeps only the models whose code changed between
their baseline's commit and HEAD (or that have no baseline): a merge that
cannot change a model's output does not need a new baseline. Measuring from the
baseline rather than the previous commit means a merge whose run was skipped
or failed is picked up by the next one.

Only artifacts from pushes to or dispatched runs on this repository's main
count: pull request code runs in the same workflow and could upload an artifact
with the same name.

    python3 find_baseline.py --repo OWNER/REPO [--only-changed SHA] < matrix.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable

from select_models import models_for

WORKFLOW = ".github/workflows/model-e2e.yml"
TRUSTED_EVENTS = ("push", "workflow_dispatch")
# The compare API lists at most this many files; a longer diff counts as all changed.
COMPARE_FILE_LIMIT = 300

Api = Callable[[str], dict]


def artifact_name(entry: dict) -> str:
    return f"model-e2e-baseline-{entry['model']}-{entry['device']}"


def github_api(token: str) -> Api:
    def get(path: str) -> dict:
        request = urllib.request.Request(
            f"https://api.github.com/{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    return get


def trusted(run: dict, repo: str) -> bool:
    return (
        run.get("event") in TRUSTED_EVENTS
        and run.get("head_branch") == "main"
        and (run.get("head_repository") or {}).get("full_name") == repo
        and (run.get("path") or "").split("@")[0] == WORKFLOW
    )


def find_run(api: Api, repo: str, name: str) -> dict:
    """The newest trusted run that uploaded `name`, or {}."""
    query = urllib.parse.urlencode({"name": name, "per_page": 50})
    artifacts = api(f"repos/{repo}/actions/artifacts?{query}").get("artifacts", [])
    artifacts = [a for a in artifacts if a.get("name") == name and not a.get("expired")]
    artifacts.sort(key=lambda a: a.get("created_at") or "", reverse=True)
    for artifact in artifacts:
        run_id = (artifact.get("workflow_run") or {}).get("id")
        if run_id:
            run = api(f"repos/{repo}/actions/runs/{run_id}")
            if trusted(run, repo):
                return run
    return {}


def add_baselines(matrix: dict, api: Api, repo: str) -> dict:
    found: dict[str, dict] = {}
    for entry in matrix["include"]:
        name = artifact_name(entry)
        if name not in found:
            found[name] = find_run(api, repo, name)
        entry["baseline_run"] = str(found[name].get("id") or "")
        entry["baseline_commit"] = found[name].get("head_sha") or ""
    return matrix


def keep_changed(matrix: dict, api: Api, repo: str, head: str) -> dict:
    """Drop the entries whose model code is unchanged since their baseline."""
    kept = []
    for entry in matrix["include"]:
        base = entry["baseline_commit"]
        if base:
            files = api(f"repos/{repo}/compare/{base}...{head}").get("files", [])
            paths = [f["filename"] for f in files]
            if len(files) < COMPARE_FILE_LIMIT and entry["model"] not in models_for(paths):
                continue
        kept.append(entry)
    return {"include": kept}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", required=True, help="OWNER/REPO")
    parser.add_argument(
        "--only-changed",
        metavar="SHA",
        help="keep only the models whose code changed between their baseline and SHA",
    )
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    api = github_api(token)
    matrix = add_baselines(json.load(sys.stdin), api, args.repo)
    for entry in matrix["include"]:
        where = f"run {entry['baseline_run']}" if entry["baseline_run"] else "none yet"
        print(f"Baseline for {entry['model']}: {where}", file=sys.stderr)
    if args.only_changed:
        matrix = keep_changed(matrix, api, args.repo, args.only_changed)
        kept = [e["model"] for e in matrix["include"]]
        print(f"Changed since their baseline: {', '.join(kept) or 'none'}", file=sys.stderr)
    print(json.dumps(matrix, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
