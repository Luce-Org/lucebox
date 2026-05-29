"""Launch Codex pointed at a Lucebox server.

Mirrors ``harness/clients/run_codex.sh`` — writes a per-run CODEX_HOME
config.toml that registers Lucebox as a custom model provider, then exec's
the codex binary with the right env. The Responses API is the default
wire format (matches what current Codex versions speak).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from harness.clients._common import (
    DEFAULT_API_KEY,
    DEFAULT_MODEL_ID,
    exec_client,
    find_bin,
    mktempdir,
)


def write_config(home: Path, *, base_url: str, model: str, sandbox: str,
                 wire_api: str) -> None:
    config_path = home / "config.toml"
    config_path.write_text(
        f"""model = "{model}"
model_provider = "luce"
approval_policy = "never"
sandbox_mode = "{sandbox}"

[model_providers.luce]
name = "Lucebox"
base_url = "{base_url.rstrip('/')}/v1"
env_key = "OPENAI_API_KEY"
wire_api = "{wire_api}"
"""
    )


def launch(
    *,
    base_url: str,
    model: str = DEFAULT_MODEL_ID,
    api_key: str = DEFAULT_API_KEY,
    prompt: str | None = None,
    timeout: int | None = None,
    interactive: bool = True,
    sandbox: str = "danger-full-access",
    wire_api: str = "responses",
    work_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Run Codex against the given Lucebox server.

    Codex isolates its config + session state under $CODEX_HOME (and falls
    back to $HOME). We point both at a per-run tempdir so the user's actual
    codex config isn't disturbed by a lucebox-pointed run.
    """
    codex_bin = find_bin("codex", env_var="CODEX_BIN",
                         work_dir_hint="clients/codex/npm/bin/codex")
    home = work_dir or mktempdir("codex")
    write_config(home, base_url=base_url, model=model,
                 sandbox=sandbox, wire_api=wire_api)

    env = {
        **os.environ,
        "HOME": str(home),
        "CODEX_HOME": str(home),
        "OPENAI_API_KEY": api_key,
    }

    argv: list[str] = [codex_bin]
    if interactive:
        # Bare interactive — codex picks up config.toml from $CODEX_HOME.
        if extra_args:
            argv += extra_args
    else:
        if prompt is None:
            raise ValueError("non-interactive mode requires prompt=...")
        argv += [
            "exec",
            "--skip-git-repo-check",
            "--sandbox", sandbox,
            "--model", model,
            "--json",
        ]
        if extra_args:
            argv += extra_args
        argv += [prompt]

    return exec_client(argv, env, interactive=interactive, timeout=timeout)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="harness-codex")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--sandbox", default="danger-full-access")
    parser.add_argument("--wire-api", default="responses",
                        choices=["responses", "chat"])
    args, extra = parser.parse_known_args()

    try:
        return launch(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            prompt=args.prompt,
            timeout=args.timeout,
            interactive=args.prompt is None,
            sandbox=args.sandbox,
            wire_api=args.wire_api,
            extra_args=extra or None,
        )
    except FileNotFoundError as e:
        print(f"[harness-codex] {e}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
