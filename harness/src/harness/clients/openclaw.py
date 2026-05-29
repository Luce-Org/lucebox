"""Launch OpenClaw pointed at a Lucebox server.

Mirrors ``harness/clients/run_openclaw.sh``. OpenClaw takes a JSON config
patch via $HOME/openclaw.patch.json that's merged into its baked-in
provider registry on startup.
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


def write_config(home: Path, *, base_url: str, model: str, api_key: str,
                 api: str = "openai-completions",
                 max_ctx: int = 204800, max_tokens: int = 4096) -> Path:
    patch_path = home / "openclaw.patch.json"
    patch_path.write_text(json.dumps({
        "models": {
            "mode": "merge",
            "providers": {
                "lucebox": {
                    "baseUrl": f"{base_url.rstrip('/')}/v1",
                    "apiKey": api_key,
                    "auth": "api-key",
                    "api": api,
                    "contextWindow": max_ctx,
                    "maxTokens": max_tokens,
                    "models": [
                        {
                            "id": model,
                            "name": "Lucebox DFlash",
                            "api": api,
                            "contextWindow": max_ctx,
                            "maxTokens": max_tokens,
                            "input": ["text"],
                            "output": ["text"],
                            "supportsTools": True,
                        }
                    ],
                }
            },
            "defaultProvider": "lucebox",
            "defaultModel": model,
        }
    }, indent=2))
    return patch_path


def launch(
    *,
    base_url: str,
    model: str = DEFAULT_MODEL_ID,
    api_key: str = DEFAULT_API_KEY,
    prompt: str | None = None,
    timeout: int | None = None,
    interactive: bool = True,
    work_dir: Path | None = None,
    max_ctx: int = 204800,
    max_tokens: int = 4096,
    extra_args: list[str] | None = None,
) -> int:
    bin_path = find_bin("openclaw", env_var="OPENCLAW_BIN",
                        work_dir_hint="clients/openclaw/npm/bin/openclaw")
    home = work_dir or mktempdir("openclaw")
    patch_path = write_config(home, base_url=base_url, model=model,
                              api_key=api_key, max_ctx=max_ctx,
                              max_tokens=max_tokens)

    env = {
        **os.environ,
        "HOME": str(home),
        "OPENCLAW_CONFIG_PATCH": str(patch_path),
    }
    argv: list[str] = [bin_path]
    if interactive:
        if extra_args:
            argv += extra_args
    else:
        if prompt is None:
            raise ValueError("non-interactive mode requires prompt=...")
        argv += ["agent"]
        if extra_args:
            argv += extra_args
        argv += [prompt]
    return exec_client(argv, env, interactive=interactive, timeout=timeout)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="harness-openclaw")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-ctx", type=int, default=204800)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args, extra = parser.parse_known_args()
    try:
        return launch(
            base_url=args.base_url, model=args.model, api_key=args.api_key,
            prompt=args.prompt, timeout=args.timeout,
            interactive=args.prompt is None,
            max_ctx=args.max_ctx, max_tokens=args.max_tokens,
            extra_args=extra or None,
        )
    except FileNotFoundError as e:
        print(f"[harness-openclaw] {e}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
