"""Minimal OpenAI-function-calling agent for harbor / terminal-bench.

Talks to a dflash_server (OpenAI-compatible) and executes tools inside the
task container with `docker exec`. When the server returns a speculative
tool result (`dflash_tool_speculation.status == "hit"`) for the tool call the
model actually made, the agent consumes it instead of executing the tool.

Two arms, selected with --ak arm=control|spec, differ ONLY in the request
field `automatic_tool_speculation` (false/true). Everything else is identical.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tb_tools  # noqa: E402
from harbor.agents.base import BaseAgent  # noqa: E402
from harbor.environments.base import BaseEnvironment  # noqa: E402
from harbor.environments.docker.docker import (  # noqa: E402
    _sanitize_docker_compose_project_name,
)
from harbor.models.agent.context import AgentContext  # noqa: E402

SYSTEM_PROMPT = """You are an autonomous software engineer working inside a Linux container to complete a task.
You interact ONLY through the provided tools. Rules:
- Inspect first: use list_dir, read_file and search_files (read-only tools) to look at files and directories.
- Use bash for commands that change state or run programs (install, compile, run tests, git, python ...).
- Use write_file to create or overwrite whole files.
- Make exactly one tool call per response, then wait for its result. Keep any prose very short.
- Do not ask the user questions; there is no user. Work autonomously until the task is fully done.
- Verify your work (run the program/tests) before finishing.
- When the task is complete, call task_complete with a one-sentence summary."""

NUDGE = (
    "There is no user to talk to. If the task is fully complete, call the task_complete tool now. "
    "Otherwise continue working by calling exactly one tool."
)


class ToolSpecAgent(BaseAgent):
    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger=None,
        *args,
        base_url: str = "http://127.0.0.1:18145",
        arm: str = "control",
        model: str = "deepseek-v4-flash",
        max_turns: int = 30,
        max_prompt_tokens: int = 13500,
        max_tokens: int = 2048,
        results_dir: str = "/home/lucebox5/tbspec/results",
        run_tag: str = "run",
        **kwargs,
    ):
        super().__init__(logs_dir, model_name, logger, *args, **kwargs)
        self.base_url = base_url.rstrip("/")
        if arm not in ("control", "spec"):
            raise ValueError("arm must be control or spec")
        self.arm = arm
        self.model = model
        self.max_turns = int(max_turns)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_tokens = int(max_tokens)
        self.results_dir = Path(results_dir)
        self.run_tag = run_tag

    @staticmethod
    def name() -> str:
        return "tbspec-tool-agent"

    def version(self) -> str | None:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    # ------------------------------------------------------------------ helpers
    def _find_container(self, environment: BaseEnvironment) -> str:
        proj = _sanitize_docker_compose_project_name(environment.session_id)
        out = subprocess.run(
            [
                tb_tools.DOCKER, "ps", "-q", "--no-trunc",
                "--filter", f"label=com.docker.compose.project={proj}",
                "--filter", "label=com.docker.compose.service=main",
            ],
            capture_output=True, text=True, timeout=30,
        ).stdout.split()
        if len(out) == 1:
            return out[0]
        raise RuntimeError(f"could not resolve main container for project {proj!r}: {out}")

    async def _chat(self, client: httpx.AsyncClient, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tb_tools.TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "automatic_tool_speculation": self.arm == "spec",
        }
        started = time.perf_counter()
        resp = await client.post(f"{self.base_url}/v1/chat/completions", json=body)
        wall_ms = (time.perf_counter() - started) * 1000.0
        resp.raise_for_status()
        return resp.json(), wall_ms

    async def _run_tool(self, cid: str, workdir: str | None, user: str | None, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(tb_tools.run_tool_sync, cid, workdir, user, name, args)

    # ---------------------------------------------------------------------- run
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        run_started = time.perf_counter()
        cid = self._find_container(environment)
        workdir = getattr(environment.task_env_config, "workdir", None) or None
        user = environment.default_user
        state = {"cid": cid, "workdir": workdir, "user": user, "arm": self.arm, "session_id": environment.session_id}
        os.makedirs(os.path.dirname(tb_tools.STATE_FILE), exist_ok=True)
        with open(tb_tools.STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(state, fh)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        turns: list[dict[str, Any]] = []
        totals = {
            "prompt_tokens": 0, "completion_tokens": 0, "model_wall_ms": 0.0, "tool_wall_ms": 0.0,
            "tool_calls": 0, "readonly_calls": 0, "spec_hits": 0, "spec_status": {},
            "predictor_wall_ms": 0.0, "hit_saved_ms": 0.0,
        }
        stop_reason = "max_turns"
        nudged = False
        tool_call_ids_seen = 0

        def flush(final: bool = False) -> None:
            summary = {
                "arm": self.arm, "run_tag": self.run_tag, "task_session": environment.session_id,
                "container": cid, "workdir": workdir, "user": user,
                "turns": turns, "totals": totals, "stop_reason": stop_reason if final else None,
                "wall_ms": (time.perf_counter() - run_started) * 1000.0,
                "instruction_chars": len(instruction),
            }
            try:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                (self.logs_dir / "tbspec_trace.json").write_text(json.dumps(summary, indent=1))
                self.results_dir.mkdir(parents=True, exist_ok=True)
                (self.results_dir / f"{self.run_tag}__{environment.session_id}.json").write_text(json.dumps(summary, indent=1))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("trace flush failed: %s", exc)
            context.n_input_tokens = totals["prompt_tokens"]
            context.n_output_tokens = totals["completion_tokens"]
            context.metadata = {k: v for k, v in totals.items()} | {"arm": self.arm, "stop_reason": stop_reason if final else None, "n_turns": len(turns)}

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=30.0)) as client:
                for turn_index in range(self.max_turns):
                    turn: dict[str, Any] = {"turn": turn_index}
                    try:
                        response, model_wall_ms = await self._chat(client, messages)
                    except Exception as exc:  # noqa: BLE001
                        turn["error"] = f"chat failed: {exc!r}"
                        turns.append(turn)
                        stop_reason = "chat_error"
                        break
                    usage = response.get("usage") or {}
                    spec = response.get("dflash_tool_speculation")
                    choice = (response.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    finish_reason = choice.get("finish_reason")
                    turn.update({
                        "model_wall_ms": model_wall_ms,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "finish_reason": finish_reason,
                        "spec": spec,
                        "timings": response.get("timings"),
                    })
                    totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                    totals["model_wall_ms"] += model_wall_ms
                    if isinstance(spec, dict):
                        st = str(spec.get("status"))
                        totals["spec_status"][st] = totals["spec_status"].get(st, 0) + 1
                        totals["predictor_wall_ms"] += float(spec.get("predictor_wall_ms") or 0.0)

                    tool_calls = message.get("tool_calls") or []
                    assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    messages.append(assistant_msg)
                    turn["assistant_content"] = (message.get("content") or "")[:2000]

                    if not tool_calls:
                        if nudged:
                            stop_reason = "no_tool_call_after_nudge"
                            turns.append(turn)
                            break
                        nudged = True
                        messages.append({"role": "user", "content": NUDGE})
                        turn["nudged"] = True
                        turns.append(turn)
                        continue
                    nudged = False

                    turn_tools = []
                    done = False
                    for call_index, call in enumerate(tool_calls):
                        fn = call.get("function") or {}
                        name = fn.get("name")
                        call_id = call.get("id") or f"call_{tool_call_ids_seen}"
                        tool_call_ids_seen += 1
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                            if not isinstance(args, dict):
                                raise ValueError("arguments must be an object")
                        except Exception as exc:  # noqa: BLE001
                            args = None
                            content = f"[tool error] invalid JSON arguments: {exc}"
                        rec: dict[str, Any] = {"name": name, "args": args, "call_id": call_id, "source": None, "tool_wall_ms": 0.0}
                        totals["tool_calls"] += 1
                        if call_index > 0:
                            content = "[not executed: only the FIRST tool call of a response is executed. Make exactly one tool call per response.]"
                            rec["source"] = "skipped_extra_call"
                        elif args is None:
                            rec["source"] = "invalid_args"
                        elif name == "task_complete":
                            content = "acknowledged"
                            rec["source"] = "task_complete"
                            done = True
                        elif name not in tb_tools.TOOL_NAMES:
                            content = f"[tool error] unknown tool {name!r}"
                            rec["source"] = "unknown_tool"
                        else:
                            if name in tb_tools.READ_ONLY_TOOLS:
                                totals["readonly_calls"] += 1
                            hit = (
                                isinstance(spec, dict)
                                and spec.get("status") == "hit"
                                and len(tool_calls) == 1
                                and call_index == 0
                                and name in tb_tools.READ_ONLY_TOOLS
                                and isinstance(spec.get("result"), dict)
                                and spec["result"].get("call_sha256") == tb_tools.call_sha256(name, args)
                                and spec["result"].get("container") == cid
                            )
                            if hit:
                                raw = spec["result"]
                                rec["source"] = "speculative_hit"
                                rec["executor_elapsed_ms"] = raw.get("elapsed_ms")
                                totals["spec_hits"] += 1
                                totals["hit_saved_ms"] += float(raw.get("elapsed_ms") or 0.0)
                            else:
                                started = time.perf_counter()
                                raw = await self._run_tool(cid, workdir, user, name, args)
                                rec["tool_wall_ms"] = (time.perf_counter() - started) * 1000.0
                                rec["source"] = "client_exec"
                                totals["tool_wall_ms"] += rec["tool_wall_ms"]
                            rec["rc"] = raw.get("rc")
                            content = tb_tools.format_result(raw, visible_chars=12000 if name == "read_file" else 6000)
                        rec["result_chars"] = len(content)
                        turn_tools.append(rec)
                        messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
                    turn["tools"] = turn_tools
                    turns.append(turn)
                    flush()
                    if done:
                        stop_reason = "task_complete"
                        break
                    if (usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0) > self.max_prompt_tokens:
                        stop_reason = "context_budget"
                        break
                    if finish_reason == "length" and not tool_calls:
                        stop_reason = "length"
                        break
        finally:
            flush(final=True)
            try:
                with open(tb_tools.STATE_FILE, "r", encoding="utf-8") as fh:
                    if json.load(fh).get("cid") == cid:
                        os.remove(tb_tools.STATE_FILE)
            except (OSError, ValueError):
                pass

