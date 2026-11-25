"""Ablation configuration — selectively disable forge guardrails."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    """Selectively disable forge guardrails for ablation runs."""

    name: str
    rescue_enabled: bool = True
    max_retries_per_step: int = 5  # 0 = no retry/unknown-tool nudge
    step_enforcement_enabled: bool = True
    max_tool_errors: int = 2  # 0 = no error recovery
    compaction_enabled: bool = True


ABLATION_PRESETS: dict[str, AblationConfig] = {
    "reforged": AblationConfig(name="reforged"),
    "no_rescue": AblationConfig(name="no_rescue", rescue_enabled=False),
    "no_nudge": AblationConfig(
        name="no_nudge", rescue_enabled=False, max_retries_per_step=0,
    ),
    "no_steps": AblationConfig(name="no_steps", step_enforcement_enabled=False),
    "no_recovery": AblationConfig(name="no_recovery", max_tool_errors=0),
    "no_compact": AblationConfig(name="no_compact", compaction_enabled=False),
    "bare": AblationConfig(
        name="bare",
        rescue_enabled=False,
        max_retries_per_step=0,
        step_enforcement_enabled=False,
        max_tool_errors=0,
        compaction_enabled=False,
    ),
}
