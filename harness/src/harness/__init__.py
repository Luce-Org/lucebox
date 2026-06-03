"""Lucebox harness — client launchers, bench orchestration, profile sweeps.

The harness is the "run X against a Lucebox server" abstraction. It owns the
server-lifecycle + client-config patterns that the shell launchers under
``harness/clients/`` use, exposed here as importable Python so callers like
``lucebox profile`` can build on it without re-implementing argv.

Modules:
  - `harness.bench` — run a luce-bench area (or full sweep) against a server,
    return the parsed JSON. The Python entry point for
    ``harness/clients/run_lucebench.sh``.
  - `harness.clients.claude_code` — launch Claude Code against a Lucebox
    server with the right env (ANTHROPIC_BASE_URL, telemetry-off knobs,
    etc.). The Python entry point for ``harness/clients/run_claude_code.sh``
    and for the host-side ``lucebox claude`` subcommand.

All entry points keep the stdlib-only invariant — fresh test boxes can run
the harness before any project Python deps are installed.
"""

__version__ = "0.1.0"
