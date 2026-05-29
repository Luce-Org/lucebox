"""Launch OpenCode pointed at a Lucebox server.

Mirrors ``harness/clients/run_opencode.sh``. OpenCode uses a per-project
opencode.json that registers Lucebox via the OpenAI-compatible AI SDK
provider. We write it to the project dir (cwd by default), set env, exec.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from harness.clients._common import (
    DEFAULT_API_KEY,
    DEFAULT_MODEL_ID,
    exec_client,
    find_bin,
    mktempdir,
)


def write_config(
    project_dir: Path,
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_ctx: int = 32768,
    max_tokens: int = 4096,
) -> None:
    config_path = project_dir / "opencode.json"
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"lucebox/{model}",
        "small_model": f"lucebox/{model}",
        "provider": {
            "lucebox": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Lucebox",
                "options": {
                    "baseURL": f"{base_url.rstrip('/')}/v1",
                    "apiKey": api_key,
                    "timeout": 600000,
                    "chunkTimeout": 60000,
                },
                "models": {
                    model: {
                        "name": "Lucebox DFlash",
                        "limit": {"context": max_ctx, "output": max_tokens},
                    }
                },
            }
        },
    }
    config_path.write_text(json.dumps(config, indent=2))


def launch(
    *,
    base_url: str,
    model: str = DEFAULT_MODEL_ID,
    api_key: str = DEFAULT_API_KEY,
    prompt: str | None = None,
    timeout: int | None = None,
    interactive: bool = True,
    project_dir: Path | None = None,
    max_ctx: int = 32768,
    max_tokens: int = 4096,
    extra_args: list[str] | None = None,
) -> int:
    """Run OpenCode against the given Lucebox server.

    OpenCode reads opencode.json from the cwd. For interactive mode we
    use the current cwd (the user's project). For non-interactive runs
    we make a fresh tempdir so the test config doesn't pollute the user's
    project tree.
    """
    bin_path = find_bin("opencode", env_var="OPENCODE_BIN",
                        work_dir_hint="clients/opencode/npm/bin/opencode")
    cwd = project_dir if project_dir else (Path.cwd() if interactive else mktempdir("opencode"))
    cwd.mkdir(parents=True, exist_ok=True)
    write_config(cwd, base_url=base_url, model=model, api_key=api_key,
                 max_ctx=max_ctx, max_tokens=max_tokens)

    # OpenCode resolves XDG_* for state; sandbox these too in test mode
    # so the user's real opencode state isn't touched.
    home = cwd / ".lucebox-opencode-home"
    home.mkdir(exist_ok=True)
    (home / ".config").mkdir(exist_ok=True)
    (home / ".local" / "share").mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "OPENAI_API_KEY": api_key,
    }

    argv: list[str] = [bin_path]
    # Run from the project dir so opencode.json is picked up.
    env["OPENCODE_CWD"] = str(cwd)
    if interactive:
        if extra_args:
            argv += extra_args
    else:
        if prompt is None:
            raise ValueError("non-interactive mode requires prompt=...")
        argv += [
            "run",
            "--pure",
            "--model", f"lucebox/{model}",
            "--format", "json",
        ]
        if extra_args:
            argv += extra_args
        argv += [prompt]

    # chdir into the project so opencode resolves the right config.
    old_cwd = os.getcwd()
    try:
        os.chdir(cwd)
        return exec_client(argv, env, interactive=interactive, timeout=timeout)
    finally:
        os.chdir(old_cwd)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="harness-opencode")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-ctx", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args, extra = parser.parse_known_args()

    try:
        return launch(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            prompt=args.prompt,
            timeout=args.timeout,
            interactive=args.prompt is None,
            max_ctx=args.max_ctx,
            max_tokens=args.max_tokens,
            extra_args=extra or None,
        )
    except FileNotFoundError as e:
        print(f"[harness-opencode] {e}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
