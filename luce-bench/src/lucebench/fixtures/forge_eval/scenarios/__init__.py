"""Eval scenario definitions — re-exports from submodules."""

from ._base import EvalScenario, _check
from ._compaction import relevance_detection
from ._model_quality import (
    argument_fidelity,
    conditional_routing,
    data_gap_recovery,
    sequential_reasoning,
    tool_selection,
)
from ._model_reasoning import (
    argument_transformation,
    data_gap_recovery_extended,
    grounded_synthesis,
    inconsistent_api_recovery,
)
from ._plumbing import basic_2step, error_recovery, sequential_3step
from ._compaction_chain import (
    compaction_chain_baseline,
    compaction_chain_p1,
    compaction_chain_p2,
    compaction_chain_p3,
)
from ._stateful_model_quality import (
    argument_fidelity_stateful,
    conditional_routing_stateful,
    data_gap_recovery_stateful,
    sequential_reasoning_stateful,
    tool_selection_stateful,
)
from ._stateful_model_reasoning import (
    argument_transformation_stateful,
    data_gap_recovery_extended_stateful,
    grounded_synthesis_stateful,
    inconsistent_api_recovery_stateful,
)
from ._stateful_plumbing import (
    basic_2step_stateful,
    error_recovery_stateful,
    sequential_3step_stateful,
)
from ._stateful_relevance import relevance_detection_stateful

ALL_SCENARIOS: list[EvalScenario] = [
    # Lambda (non-stateful)
    basic_2step,
    sequential_3step,
    error_recovery,
    tool_selection,
    argument_fidelity,
    sequential_reasoning,
    conditional_routing,
    data_gap_recovery,
    data_gap_recovery_extended,
    argument_transformation,
    inconsistent_api_recovery,
    grounded_synthesis,
    relevance_detection,
    # Stateful
    basic_2step_stateful,
    sequential_3step_stateful,
    error_recovery_stateful,
    tool_selection_stateful,
    argument_fidelity_stateful,
    sequential_reasoning_stateful,
    conditional_routing_stateful,
    data_gap_recovery_stateful,
    data_gap_recovery_extended_stateful,
    argument_transformation_stateful,
    inconsistent_api_recovery_stateful,
    grounded_synthesis_stateful,
    relevance_detection_stateful,
    # Compaction chain
    compaction_chain_baseline,
    compaction_chain_p1,
    compaction_chain_p2,
    compaction_chain_p3,
]

__all__ = [
    "EvalScenario",
    "_check",
    "ALL_SCENARIOS",
    "basic_2step",
    "sequential_3step",
    "error_recovery",
    "tool_selection",
    "argument_fidelity",
    "sequential_reasoning",
    "conditional_routing",
    "data_gap_recovery",
    "data_gap_recovery_extended",
    "argument_transformation",
    "inconsistent_api_recovery",
    "grounded_synthesis",
    "relevance_detection",
    "basic_2step_stateful",
    "sequential_3step_stateful",
    "error_recovery_stateful",
    "tool_selection_stateful",
    "argument_fidelity_stateful",
    "sequential_reasoning_stateful",
    "conditional_routing_stateful",
    "data_gap_recovery_stateful",
    "data_gap_recovery_extended_stateful",
    "argument_transformation_stateful",
    "inconsistent_api_recovery_stateful",
    "grounded_synthesis_stateful",
    "relevance_detection_stateful",
    "compaction_chain_baseline",
    "compaction_chain_p1",
    "compaction_chain_p2",
    "compaction_chain_p3",
]
