# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Composable guardrail middleware for external agent loops.

Use these components inside your own orchestration loop to get forge's
reliability stack (retry nudges, rescue parsing, step enforcement, error
tracking) without adopting WorkflowRunner.

Most integrators should use ``Guardrails`` (two-method API). For granular
control, use ResponseValidator, StepEnforcer, and ErrorTracker directly.

See ADR-011 (docs/decisions/011-guardrail-middleware.md) for design rationale.
"""

from .nudge import Nudge
from .response_validator import ResponseValidator, ValidationResult
from .step_enforcer import StepEnforcer, StepCheck
from .error_tracker import ErrorTracker
from .guardrails import CheckResult, Guardrails

__all__ = [
    # Bundled API
    "CheckResult",
    "Guardrails",
    # Granular components
    "ErrorTracker",
    "Nudge",
    "ResponseValidator",
    "StepCheck",
    "StepEnforcer",
    "ValidationResult",
]
