"""Run a luce-bench area (or full sweep) against a Lucebox server.

The function form of ``harness/clients/run_lucebench.sh``. Same contract:
build a luce-bench argv with the per-area knobs, exec it against a running
server, parse the JSON snapshot back. Used by ``lucebox profile`` so the
StepDefinition framework doesn't have to re-derive argv.

The shell wrapper still exists for operator use (``harness/clients/run_lucebench.sh``).
Both ultimately do the same thing — single source of truth for what
"run luce-bench against this server" means.

Stdlib-only at runtime. luce-bench is invoked as a subprocess so we don't
have to import it (its CLI module owns argv parsing + dispatch).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

Area = Literal["ds4-eval", "code", "longctx", "agent", "forge"]


def run_bench(
    *,
    base_url: str,
    area: Area | None = None,
    model: str = "default",
    think: bool | None = None,
    max_tokens: int | None = None,
    timeout: int = 300,
    parallel: int = 1,
    auth_env: str | None = None,
    out_dir: Path | None = None,
    name: str | None = None,
    json_out: Path | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a luce-bench area (or the full sweep) and return the parsed result.

    Args:
        base_url: Lucebox server's HTTP base, e.g. ``http://localhost:8080``.
        area: Single area name, or ``None`` for ``--sweep`` (all stdlib areas).
        model: Model ID. ``"default"`` triggers luce-bench's ``/v1/models``
            auto-resolve (uses the single exposed model if there's exactly one).
        think: ``True`` → ``--think``, ``False`` → ``--no-think``, ``None`` →
            omit the flag and let the server's card defaults govern.
        max_tokens: Per-request decode cap. ``None`` → use luce-bench area default.
        timeout: Per-case wall timeout (seconds).
        parallel: In-flight concurrency.
        auth_env: Env var name to read Authorization bearer from (e.g.
            ``OPENROUTER_API_KEY``).
        out_dir: Directory for sweep output. Required when ``area`` is None.
        name: Name for the sweep dir. Required when ``area`` is None.
        json_out: Single-area mode only — override the output JSON path.
            Used by ``lucebox profile`` to land snapshots where its
            framework expects them (``dest/bench-<area>.json``). Ignored
            in sweep mode (sweep always writes per-area files + summary
            under ``out_dir/name/``).
        extra_body: Additional fields to merge into every chat-completion
            request body. Use for provider-specific knobs.

    Returns:
        For single-area: the parsed area JSON (rows, pass count, timings).
        For sweep: the parsed ``_summary.json`` (cross-area aggregate).
    """
    if area is None and (out_dir is None or name is None):
        raise ValueError("sweep mode (area=None) requires out_dir and name")

    argv: list[str] = [
        sys.executable,
        "-m",
        "lucebench.cli",
        "--base-url",
        base_url,
        "--model",
        model,
        "--timeout",
        str(timeout),
        "--parallel",
        str(parallel),
    ]

    resolved_json_out: Path
    if area is not None:
        if json_out is not None:
            resolved_json_out = json_out
        else:
            resolved_json_out = (out_dir or Path.cwd()) / f"lucebench-{area}.json"
        resolved_json_out.parent.mkdir(parents=True, exist_ok=True)
        argv += ["--area", area, "--json-out", str(resolved_json_out)]
    else:
        assert out_dir is not None and name is not None  # narrowed by check above
        out_dir.mkdir(parents=True, exist_ok=True)
        argv += ["--sweep", "--out-dir", str(out_dir), "--name", name]
        resolved_json_out = out_dir / name / "_summary.json"

    if think is True:
        argv += ["--think"]
    elif think is False:
        argv += ["--no-think"]
    if max_tokens is not None:
        argv += ["--max-tokens", str(max_tokens)]
    if auth_env is not None:
        argv += ["--auth-env", auth_env]
    if extra_body is not None:
        argv += ["--extra-body", json.dumps(extra_body)]

    subprocess.run(argv, check=True)
    return json.loads(resolved_json_out.read_text())


def main() -> int:
    """Thin CLI wrapping ``run_bench`` for the ``harness-run-bench`` console script.

    Most operator invocations go through ``harness/clients/run_lucebench.sh``
    (which handles the server lifecycle too). This entry exists so the
    function form has a working CLI surface for ad-hoc use.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="harness-run-bench")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--area", default=None,
                        choices=["ds4-eval", "code", "longctx", "agent", "forge"])
    parser.add_argument("--model", default="default")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--think", action="store_true")
    grp.add_argument("--no-think", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--auth-env", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path.cwd())
    parser.add_argument("--name", default="harness-run")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="single-area only — explicit output JSON path")
    args = parser.parse_args()

    think: bool | None = None
    if args.think:
        think = True
    elif args.no_think:
        think = False

    # Caller can be on a fresh test box; check luce-bench is reachable.
    if shutil.which(sys.executable) is None:
        print(f"[harness] missing python: {sys.executable}", file=sys.stderr)
        return 2

    result = run_bench(
        base_url=args.base_url,
        area=args.area,
        model=args.model,
        think=think,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        parallel=args.parallel,
        auth_env=args.auth_env,
        out_dir=args.out_dir,
        name=args.name,
        json_out=args.json_out,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
