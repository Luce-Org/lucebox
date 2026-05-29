"""Launch Pi pointed at a Lucebox server.

Mirrors ``harness/clients/run_pi.sh``. Pi reads $HOME/agent/{settings,models}.json.
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
                 api: str = "openai-responses",
                 tools: str = "read,grep,find,ls") -> None:
    agent = home / "agent"
    agent.mkdir(parents=True, exist_ok=True)
    (home / "sessions").mkdir(parents=True, exist_ok=True)
    (agent / "settings.json").write_text(json.dumps({"compaction": {"enabled": False}}))
    (agent / "models.json").write_text(json.dumps({
        "providers": {
            "lucebox": {
                "baseUrl": f"{base_url.rstrip('/')}/v1",
                "api": api,
                "apiKey": api_key,
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsUsageInStreaming": True,
                    "maxTokensField": "max_tokens",
                },
                "models": [
                    {"id": model, "name": "Lucebox DFlash"},
                ],
            }
        },
        "defaultModel": {"provider": "lucebox", "id": model},
    }))


def launch(
    *,
    base_url: str,
    model: str = DEFAULT_MODEL_ID,
    api_key: str = DEFAULT_API_KEY,
    prompt: str | None = None,
    timeout: int | None = None,
    interactive: bool = True,
    work_dir: Path | None = None,
    tools: str = "read,grep,find,ls",
    extra_args: list[str] | None = None,
) -> int:
    bin_path = find_bin("pi", env_var="PI_BIN",
                        work_dir_hint="clients/pi/npm/bin/pi")
    home = work_dir or mktempdir("pi")
    write_config(home, base_url=base_url, model=model, api_key=api_key)

    agent_dir = home / "agent"
    sessions_dir = home / "sessions"
    # Mirror the env exports in harness/clients/run_pi.sh (PI_CODING_AGENT_*
    # tell Pi where its config + session state live; PI_OFFLINE keeps it
    # from reaching out to the public Pi API).
    env = {
        **os.environ,
        "HOME": str(home),
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "PI_CODING_AGENT_SESSION_DIR": str(sessions_dir),
        "PI_OFFLINE": "1",
    }
    argv: list[str] = [bin_path]
    if interactive:
        if extra_args:
            argv += extra_args
    else:
        if prompt is None:
            raise ValueError("non-interactive mode requires prompt=...")
        # Mirror harness/clients/run_pi.sh's validated invocation: route via
        # the lucebox provider with the json print mode + a fixed tool
        # allowlist, no session persistence, offline (no cloud Pi calls).
        argv += [
            "--provider", "lucebox",
            "--model", model,
            "--print",
            "--mode", "json",
            "--tools", tools,
            "--no-session",
            "--offline",
        ]
        if extra_args:
            argv += extra_args
        argv += [prompt]
    return exec_client(argv, env, interactive=interactive, timeout=timeout)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="harness-pi")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--tools", default="read,grep,find,ls",
                        help="Comma-separated tool allowlist passed to "
                        "`pi --tools` (matches PI_TOOLS in run_pi.sh).")
    args, extra = parser.parse_known_args()
    try:
        return launch(
            base_url=args.base_url, model=args.model, api_key=args.api_key,
            prompt=args.prompt, timeout=args.timeout,
            interactive=args.prompt is None, tools=args.tools,
            extra_args=extra or None,
        )
    except FileNotFoundError as e:
        print(f"[harness-pi] {e}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
