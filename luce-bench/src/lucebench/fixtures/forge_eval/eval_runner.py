"""Eval runner — run scenarios N times, collect per-run results.

Vendored from forge-guardrails 0.7.1's ``tests/eval/eval_runner.py`` so it
can be imported without the upstream tests/ tree on sys.path. See
``forge_eval/__init__.py`` for the rationale and the list of local
modifications.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from ._forge.clients.base import ChunkType, LLMClient, StreamChunk
from ._forge.context.manager import CompactEvent, ContextManager
from ._forge.context.strategies import CompactStrategy, NoCompact, SlidingWindowCompact, TieredCompact
from ._forge.core.messages import Message, MessageType
from ._forge.core.runner import WorkflowRunner
from ._forge.core.workflow import ToolCall, ToolDef, ToolSpec, Workflow
from ._forge.errors import ForgeError, StreamError
from ._forge.server import BudgetMode, ServerManager

# Vendored: tests.eval.* imports rewired to local relative imports.
from .ablation import ABLATION_PRESETS, AblationConfig
from .scenarios import ALL_SCENARIOS, EvalScenario


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Stub for the upstream ``tests.eval.batch_eval._compute_cost``.

    The real pricing table only covers Anthropic-API model IDs; for a
    self-hosted dflash bench the model name will be something like
    ``luce-dflash`` and the real function returns 0.0 anyway. Inlining
    the no-op here keeps the eval_runner self-contained.
    """
    return 0.0

# Scenarios that always use their own hardcoded budget (MANUAL override).
_COMPACTION_SCENARIOS = {
    "compaction_stress", "phase2_compaction",
    "compaction_stress_stateful", "phase2_compaction_stateful",
    "inventory_audit", "supplier_deep_dive",
    "compaction_chain_p1", "compaction_chain_p2", "compaction_chain_p3",
}


@dataclass
class RunResult:
    """Result of a single eval run."""

    scenario_name: str
    completeness: bool
    iterations_used: int
    terminal_args: dict[str, Any] | None = None
    accuracy: bool | None = None
    validate_error: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    compaction_events: list[CompactEvent] = field(default_factory=list)
    messages: list[Message] | None = None
    elapsed_seconds: float = 0.0
    stream_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class EvalConfig:
    """Configuration for an eval run."""

    runs_per_scenario: int = 10
    stream: bool = False
    compact_strategy: CompactStrategy | None = None
    strategy_overrides: dict[str, CompactStrategy] = field(default_factory=dict)
    keep_message_history: bool = True
    verbose: bool = False
    budget_override: int | None = None
    stream_retries: int = 2


class CountingClientWrapper:
    """Wraps an LLMClient to count send() calls and accumulate token usage."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _collect_usage(self) -> None:
        """Read last_usage from the wrapped client if available."""
        usage = getattr(self._client, "last_usage", None)
        if usage:
            # Slot-keyed {slot_id: TokenUsage} across all clients (llamaserver,
            # ollama, anthropic). Sum across slots (usually one).
            for tu in usage.values():
                self.total_input_tokens += tu.prompt_tokens
                self.total_output_tokens += tu.completion_tokens

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> Any:
        self.call_count += 1
        result = await self._client.send(
            messages, tools=tools, sampling=sampling, passthrough=passthrough,
            inbound_anthropic_body=inbound_anthropic_body,
        )
        self._collect_usage()
        return result

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.call_count += 1
        async for chunk in self._client.send_stream(
            messages, tools=tools, sampling=sampling, passthrough=passthrough,
            inbound_anthropic_body=inbound_anthropic_body,
        ):
            yield chunk
        self._collect_usage()

    async def get_context_length(self) -> int | None:
        return await self._client.get_context_length()


def _resolve_strategy(
    scenario: EvalScenario,
    config: EvalConfig,
) -> CompactStrategy:
    """Determine which compact strategy to use for a scenario."""
    # Check tag-based overrides
    for tag in scenario.tags:
        if tag in config.strategy_overrides:
            return config.strategy_overrides[tag]
    # Fall back to config-level strategy, then NoCompact
    return config.compact_strategy or NoCompact()


def _build_workflow_with_capture(
    scenario: EvalScenario,
    ablation: AblationConfig | None = None,
) -> tuple[Workflow, dict[str, Any], Callable[[], bool] | None]:
    """Build a per-run workflow copy with terminal arg capture.

    Returns (workflow, capture_dict, validate_state_fn) where capture_dict
    will be populated with {"args": {...}} when the terminal tool is called.

    If the scenario has a ``build_workflow`` factory (stateful scenarios),
    it is called to get a fresh workflow + validate_state closure per run.
    Otherwise the scenario's static workflow is copied and validate_state
    comes from the scenario dataclass field.

    If ablation disables step enforcement, required_steps is set to []
    so StepTracker becomes a no-op.
    """
    capture: dict[str, Any] = {}
    validate_state_fn: Callable[[], bool] | None = None

    if scenario.build_workflow is not None:
        base_workflow, validate_state_fn = scenario.build_workflow()
    else:
        base_workflow = scenario.workflow
        validate_state_fn = scenario.validate_state

    tools = dict(base_workflow.tools)
    for tt_name in base_workflow.terminal_tools:
        original_fn = base_workflow.get_callable(tt_name)
        terminal_spec = base_workflow.tools[tt_name].spec

        def capturing_terminal(_fn=original_fn, **kwargs: Any) -> Any:
            capture["args"] = kwargs
            return _fn(**kwargs)

        tools[tt_name] = ToolDef(
            spec=terminal_spec,
            callable=capturing_terminal,
        )

    # Ablation: disable step enforcement by clearing required_steps
    required_steps = base_workflow.required_steps
    if ablation is not None and not ablation.step_enforcement_enabled:
        required_steps = []

    workflow = Workflow(
        name=base_workflow.name,
        description=base_workflow.description,
        tools=tools,
        required_steps=required_steps,
        terminal_tool=base_workflow.terminal_tool,
        system_prompt_template=base_workflow.system_prompt_template,
    )
    return workflow, capture, validate_state_fn


def _verbose_printer(msg: Message) -> None:
    """Print a live trace line for a single message."""
    _MAX = 120
    match msg.metadata.type:
        case MessageType.TOOL_CALL:
            if msg.tool_calls:
                names = [tc.name for tc in msg.tool_calls]
                label = ", ".join(names)
                if len(names) > 1:
                    print(f"    [tool_call] *** PARALLEL {len(names)} *** {label}")
                else:
                    print(f"    [tool_call] {label}")
            else:
                print(f"    [tool_call] {msg.content}")
        case MessageType.TOOL_RESULT:
            text = msg.content[:_MAX] + "..." if len(msg.content) > _MAX else msg.content
            print(f"    [result]    {text}")
        case MessageType.REASONING:
            text = msg.content[:_MAX] + "..." if len(msg.content) > _MAX else msg.content
            print(f"    [thinking]  {text}")
        case MessageType.RETRY_NUDGE:
            print("    [nudge]     retry")
        case MessageType.STEP_NUDGE:
            print("    [nudge]     step enforcement")


async def run_scenario(
    client: LLMClient,
    scenario: EvalScenario,
    config: EvalConfig,
    ablation: AblationConfig | None = None,
) -> RunResult:
    """Run a single eval scenario once. Returns a RunResult."""
    # Set up per-run state
    counting_client = CountingClientWrapper(client)
    compaction_events: list[CompactEvent] = []
    collected_messages: list[Message] = []

    # Ablation: force NoCompact when compaction is disabled
    if ablation is not None and not ablation.compaction_enabled:
        strategy = NoCompact()
    else:
        strategy = _resolve_strategy(scenario, config)
    budget = config.budget_override if config.budget_override is not None else scenario.budget_tokens

    ctx = ContextManager(
        strategy=strategy,
        budget_tokens=budget,
        on_compact=compaction_events.append,
    )

    workflow, capture, validate_state_fn = _build_workflow_with_capture(scenario, ablation=ablation)

    # Build on_message callback: verbose print, history collection, or both
    callbacks: list[Any] = []
    if config.verbose:
        callbacks.append(_verbose_printer)
    if config.keep_message_history:
        callbacks.append(collected_messages.append)

    if not callbacks:
        on_message = None
    elif len(callbacks) == 1:
        on_message = callbacks[0]
    else:
        def on_message(msg: Message) -> None:
            for cb in callbacks:
                cb(msg)

    # Apply ablation overrides to runner params
    max_retries = scenario.max_retries_per_step
    max_tool_errors = scenario.max_tool_errors
    rescue_enabled = True
    if ablation is not None:
        max_retries = ablation.max_retries_per_step
        max_tool_errors = ablation.max_tool_errors
        rescue_enabled = ablation.rescue_enabled

    runner = WorkflowRunner(
        client=counting_client,
        context_manager=ctx,
        max_iterations=scenario.max_iterations,
        max_retries_per_step=max_retries,
        max_tool_errors=max_tool_errors,
        stream=config.stream,
        on_message=on_message,
        rescue_enabled=rescue_enabled,
    )

    start = time.monotonic()
    last_stream_error: StreamError | None = None
    for attempt in range(1 + config.stream_retries):
        if attempt > 0:
            # Reset state for retry — fresh run from scratch
            print(f"    [retry {attempt}/{config.stream_retries}] StreamError, retrying...", flush=True)
            counting_client.call_count = 0
            counting_client.total_input_tokens = 0
            counting_client.total_output_tokens = 0
            compaction_events.clear()
            collected_messages.clear()
            workflow, capture, validate_state_fn = _build_workflow_with_capture(scenario, ablation=ablation)
            start = time.monotonic()

        try:
            await runner.run(workflow, scenario.user_message)
            elapsed = time.monotonic() - start
            accuracy: bool | None = None
            validate_error: str | None = None
            if scenario.validate and capture.get("args") is not None:
                try:
                    accuracy = scenario.validate(capture["args"])
                except Exception as exc:
                    accuracy = None
                    validate_error = type(exc).__name__
            if validate_state_fn is not None:
                try:
                    state_ok = validate_state_fn()
                    if accuracy is None:
                        accuracy = state_ok
                    else:
                        accuracy = accuracy and state_ok
                except Exception as exc:
                    accuracy = False
                    validate_error = f"validate_state: {type(exc).__name__}"
            return RunResult(
                scenario_name=scenario.name,
                completeness=True,
                iterations_used=counting_client.call_count,
                terminal_args=capture.get("args"),
                accuracy=accuracy,
                validate_error=validate_error,
                compaction_events=compaction_events,
                messages=collected_messages if config.keep_message_history else None,
                elapsed_seconds=elapsed,
                stream_retries=attempt,
                input_tokens=counting_client.total_input_tokens,
                output_tokens=counting_client.total_output_tokens,
            )
        except StreamError as exc:
            last_stream_error = exc
            continue
        except ForgeError as exc:
            elapsed = time.monotonic() - start
            return RunResult(
                scenario_name=scenario.name,
                completeness=False,
                iterations_used=counting_client.call_count,
                error_type=type(exc).__name__,
                error_message=str(exc),
                compaction_events=compaction_events,
                messages=collected_messages if config.keep_message_history else None,
                elapsed_seconds=elapsed,
                stream_retries=attempt,
                input_tokens=counting_client.total_input_tokens,
                output_tokens=counting_client.total_output_tokens,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return RunResult(
                scenario_name=scenario.name,
                completeness=False,
                iterations_used=counting_client.call_count,
                error_type=type(exc).__name__,
                error_message=str(exc),
                compaction_events=compaction_events,
                messages=collected_messages if config.keep_message_history else None,
                elapsed_seconds=elapsed,
                stream_retries=attempt,
                input_tokens=counting_client.total_input_tokens,
                output_tokens=counting_client.total_output_tokens,
            )

    # All stream retries exhausted
    elapsed = time.monotonic() - start
    assert last_stream_error is not None
    return RunResult(
        scenario_name=scenario.name,
        completeness=False,
        iterations_used=counting_client.call_count,
        error_type=type(last_stream_error).__name__,
        error_message=str(last_stream_error),
        compaction_events=compaction_events,
        messages=collected_messages if config.keep_message_history else None,
        elapsed_seconds=elapsed,
        stream_retries=config.stream_retries,
        input_tokens=counting_client.total_input_tokens,
        output_tokens=counting_client.total_output_tokens,
    )


async def run_eval(
    client: LLMClient,
    scenarios: list[EvalScenario],
    config: EvalConfig,
    resolved_budget: int | None = None,
    tags: list[str] | None = None,
    names: list[str] | None = None,
    ablation: AblationConfig | None = None,
) -> dict[str, list[RunResult]]:
    """Run all scenarios, return results grouped by scenario name.

    Args:
        resolved_budget: The globally resolved budget from ServerManager.
            Compaction scenarios override this with their own hardcoded
            budget; all others use it.  When provided, also sets
            ``client.set_num_ctx()`` per-scenario for Ollama backends.
    """
    if tags:
        scenarios = [
            s for s in scenarios if any(t in s.tags for t in tags)
        ]
    if names:
        scenarios = [s for s in scenarios if s.name in names]

    results: dict[str, list[RunResult]] = {}

    for scenario in scenarios:
        # Skip compaction scenarios when ablation disables compaction
        if scenario.name in _COMPACTION_SCENARIOS and ablation is not None and not ablation.compaction_enabled:
            print(f"  Skipping {scenario.name} (compaction disabled by ablation={ablation.name})")
            continue

        # Wire per-scenario budget: compaction scenarios use their own
        # hardcoded value, everything else uses the resolved budget.
        if scenario.name in _COMPACTION_SCENARIOS:
            scenario_budget = scenario.budget_tokens
        else:
            scenario_budget = resolved_budget

        # Set num_ctx on Ollama client to match this scenario's budget
        if scenario_budget is not None and hasattr(client, "set_num_ctx"):
            client.set_num_ctx(scenario_budget)

        # Pass as budget_override so run_scenario uses it
        per_scenario_config = EvalConfig(
            runs_per_scenario=config.runs_per_scenario,
            stream=config.stream,
            compact_strategy=config.compact_strategy,
            strategy_overrides=config.strategy_overrides,
            keep_message_history=config.keep_message_history,
            verbose=config.verbose,
            budget_override=scenario_budget,
            stream_retries=config.stream_retries,
        )

        scenario_results: list[RunResult] = []
        for run_idx in range(config.runs_per_scenario):
            print(
                f"  Running {scenario.name} "
                f"[{run_idx + 1}/{config.runs_per_scenario}]...",
                flush=True,
            )
            result = await run_scenario(client, scenario, per_scenario_config, ablation=ablation)
            scenario_results.append(result)
            if not result.completeness:
                status = f"FAIL ({result.error_type})"
            elif result.accuracy is False:
                status = "OK (incorrect)"
            else:
                status = "OK"
            cost_str = ""
            if result.input_tokens:
                # Vendored: was ``from tests.eval.batch_eval import
                # _compute_cost`` — see module-level stub.
                cost = _compute_cost(
                    client.model if hasattr(client, "model") else "",
                    result.input_tokens,
                    result.output_tokens,
                )
                if cost > 0:
                    cost_str = f", ${cost:.4f}"
            print(
                f"    {status} — {result.iterations_used} iterations, "
                f"{result.elapsed_seconds:.1f}s{cost_str}",
                flush=True,
            )
        results[scenario.name] = scenario_results

    return results


# NOTE: Upstream's ``async def main()`` CLI was dropped during vendoring —
# ``bench_http_capability.py --area forge`` owns the entrypoint and only
# imports ``run_eval``, ``EvalConfig``, ``RunResult``, ``ALL_SCENARIOS``,
# and ``ABLATION_PRESETS``. If you need to resync, copy the upstream
# ``tests/eval/eval_runner.py`` and re-apply the patches noted in
# ``forge_eval/__init__.py``.
