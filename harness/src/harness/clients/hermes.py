"""Launch Hermes Agent pointed at a Lucebox server.

Mirrors ``harness/clients/run_hermes.sh``. Hermes reads YAML config from
$HOME/config.yaml plus a $HOME/.env file. Both get written to a per-run
working dir to keep the user's real Hermes state untouched.
"""

from __future__ import annotations

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
                 max_ctx: int, max_tokens: int, repo_dir: str) -> None:
    base = f"{base_url.rstrip('/')}/v1"
    (home / "config.yaml").write_text(
        f"""model:
  default: "{model}"
  provider: "lucebox"
  base_url: "{base}"
  api_key: "{api_key}"
  api_mode: "chat_completions"
  context_length: {max_ctx}
  max_tokens: {max_tokens}

custom_providers:
  - name: "lucebox"
    base_url: "{base}"
    api_key: "{api_key}"
    api_mode: "chat_completions"
    models:
      "{model}":
        context_length: {max_ctx}
        max_tokens: {max_tokens}

terminal:
  backend: "local"
  cwd: "{repo_dir}"
  timeout: 180
  lifetime_seconds: 300
"""
    )
    (home / ".env").write_text(
        f"""OPENAI_API_KEY={api_key}
OPENAI_BASE_URL={base}
HERMES_INFERENCE_PROVIDER=lucebox
HERMES_INFERENCE_MODEL={model}
HERMES_ACCEPT_HOOKS=1
HERMES_API_TIMEOUT=600
HERMES_API_CALL_STALE_TIMEOUT=600
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
    work_dir: Path | None = None,
    max_ctx: int = 98304,
    max_tokens: int = 4096,
    max_turns: int = 40,
    extra_args: list[str] | None = None,
) -> int:
    bin_path = find_bin("hermes", env_var="HERMES_BIN",
                        work_dir_hint="clients/hermes/home/.local/bin/hermes")
    home = work_dir or mktempdir("hermes")
    repo_dir = os.environ.get("REPO_DIR", str(Path.cwd()))
    write_config(home, base_url=base_url, model=model, api_key=api_key,
                 max_ctx=max_ctx, max_tokens=max_tokens, repo_dir=repo_dir)

    base = f"{base_url.rstrip('/')}/v1"
    # Mirror harness/clients/run_hermes.sh: HERMES_HOME tells the binary
    # which config dir to read (Hermes does not always honor HOME alone);
    # the OPENAI_/HERMES_INFERENCE_* env vars are the canonical wiring;
    # NO_COLOR keeps the batch log diffable.
    env = {
        **os.environ,
        "HOME": str(home),
        "HERMES_HOME": str(home),
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": base,
        "HERMES_INFERENCE_PROVIDER": "lucebox",
        "HERMES_INFERENCE_MODEL": model,
        "HERMES_ACCEPT_HOOKS": "1",
        "NO_COLOR": "1",
    }
    argv: list[str] = [bin_path]
    if interactive:
        if extra_args:
            argv += extra_args
    else:
        if prompt is None:
            raise ValueError("non-interactive mode requires prompt=...")
        # Mirror run_hermes.sh's validated batch invocation: `chat` subcommand
        # with the lucebox provider, accept-hooks/yolo so it doesn't stop on
        # interactive prompts, `--query` for the user prompt (not positional).
        argv += [
            "chat",
            "--quiet",
            "--provider", "lucebox",
            "--model", model,
            "--accept-hooks",
            "--yolo",
            "--max-turns", str(max_turns),
            "--source", "lucebox-harness",
        ]
        if extra_args:
            argv += extra_args
        argv += ["--query", prompt]

    old_cwd = os.getcwd()
    try:
        os.chdir(repo_dir)
        return exec_client(argv, env, interactive=interactive, timeout=timeout)
    finally:
        os.chdir(old_cwd)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="harness-hermes")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-ctx", type=int, default=98304)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=40,
                        help="Max agent turns for `hermes chat --max-turns` "
                        "(mirrors HERMES_MAX_TURNS in run_hermes.sh).")
    args, extra = parser.parse_known_args()
    try:
        return launch(
            base_url=args.base_url, model=args.model, api_key=args.api_key,
            prompt=args.prompt, timeout=args.timeout,
            interactive=args.prompt is None,
            max_ctx=args.max_ctx, max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            extra_args=extra or None,
        )
    except FileNotFoundError as e:
        print(f"[harness-hermes] {e}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
