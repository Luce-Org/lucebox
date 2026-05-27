# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Exception hierarchy for the forge library."""


class ForgeError(Exception):
    """Base exception for the forge library."""

    pass


class UnsupportedModelError(ForgeError):
    """Caller opted into recommended sampling for a model not in the map.

    Raised by ``apply_sampling_defaults(model, strict=True)`` when ``model``
    has no entry in ``MODEL_SAMPLING_DEFAULTS``. Failing loud is intentional:
    ``recommended_sampling=True`` declares "I want the per-card sampling
    profile for this model"; falling through to backend defaults silently
    would defeat that intent.
    """

    def __init__(self, model: str):
        super().__init__(
            f"No recommended sampling defaults registered for model {model!r}. "
            f"Either add an entry to MODEL_SAMPLING_DEFAULTS (with HF card URL) "
            f"or drop recommended_sampling=True."
        )
        self.model = model


class ToolCallError(ForgeError):
    """LLM failed to produce a valid tool call after retries."""

    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.raw_response = raw_response
        self.cause = cause


class ToolExecutionError(ForgeError):
    """A tool callable raised during execution."""

    def __init__(self, tool_name: str, cause: Exception):
        super().__init__(f"Tool '{tool_name}' raised: {cause}")
        self.tool_name = tool_name
        self.cause = cause


class ToolResolutionError(Exception):
    """Tool arguments were valid but the data didn't resolve.

    The tool equivalent of HTTP 4xx — the call was well-formed and the
    schema was satisfied, but the arguments couldn't be resolved against
    the underlying data (wrong key, empty result set, unrecognized ID,
    etc.).

    Raise this from a tool callable to signal "try again with different
    arguments" without counting toward consecutive_tool_errors.  The
    runner feeds the message back to the model and does NOT mark the
    step as completed.

    Not a ForgeError — this is a tool-author exception, not a framework
    error.  The runner catches it explicitly.
    """

    def __init__(self, message: str, tool_name: str | None = None):
        super().__init__(message)
        self.tool_name = tool_name


class WorkflowCancelledError(ForgeError):
    """Workflow was cancelled via cancel_event before completion."""

    def __init__(
        self,
        messages: list,
        completed_steps: dict[str, None],
        iteration: int,
    ):
        super().__init__(
            f"Workflow cancelled at iteration {iteration}. "
            f"Completed steps: {completed_steps}"
        )
        self.messages = messages
        self.completed_steps = completed_steps
        self.iteration = iteration


class MaxIterationsError(ForgeError):
    """Workflow exceeded max_iterations without calling the terminal tool."""

    def __init__(
        self,
        iterations: int,
        completed_steps: dict[str, None],
        pending_steps: list[str],
    ):
        super().__init__(
            f"Max iterations ({iterations}) exceeded. "
            f"Completed: {completed_steps}, Pending: {pending_steps}"
        )
        self.iterations = iterations
        self.completed_steps = completed_steps
        self.pending_steps = pending_steps


class StepEnforcementError(ForgeError):
    """Model repeatedly tried to call the terminal tool before completing required steps."""

    def __init__(
        self,
        terminal_tool: str,
        attempts: int,
        pending_steps: list[str],
    ):
        super().__init__(
            f"Model called '{terminal_tool}' prematurely {attempts} times "
            f"without completing required steps: {pending_steps}"
        )
        self.terminal_tool = terminal_tool
        self.attempts = attempts
        self.pending_steps = pending_steps


class PrerequisiteError(ForgeError):
    """Model repeatedly called a tool without satisfying its prerequisites."""

    def __init__(
        self,
        tool_name: str,
        violations: int,
        missing_prereqs: list[str],
    ):
        super().__init__(
            f"Tool '{tool_name}' called {violations} times "
            f"without satisfying prerequisites: {missing_prereqs}"
        )
        self.tool_name = tool_name
        self.violations = violations
        self.missing_prereqs = missing_prereqs


class ContextBudgetExceeded(ForgeError):
    """Context exceeded budget even after compaction. Unrecoverable."""

    def __init__(self, estimated_tokens: int, budget_tokens: int):
        super().__init__(
            f"Context budget exceeded: {estimated_tokens} tokens "
            f"estimated, budget is {budget_tokens}"
        )
        self.estimated_tokens = estimated_tokens
        self.budget_tokens = budget_tokens


class HardwareDetectionError(ForgeError):
    """nvidia-smi responded but output couldn't be parsed."""

    def __init__(self, cause: Exception):
        super().__init__(f"Hardware detection failed: {cause}")
        self.cause = cause


class ContextDiscoveryError(ForgeError):
    """Backend context length response couldn't be parsed."""

    def __init__(self, cause: Exception):
        super().__init__(f"Context discovery failed: {cause}")
        self.cause = cause


class BudgetResolutionError(ForgeError):
    """No context budget could be determined from any source."""

    def __init__(self, cause: Exception | None = None) -> None:
        if cause is not None:
            super().__init__(f"Budget resolution failed: {cause}")
            self.__cause__ = cause
        else:
            super().__init__(
                "No context budget could be determined: "
                "no GPU detected and no explicit budget_tokens provided"
            )


class BackendError(ForgeError):
    """Unexpected HTTP error from the LLM backend."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"Backend returned {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class ThinkingNotSupportedError(BackendError):
    """Model does not support thinking mode, but think=True was explicitly requested."""

    def __init__(self, model: str, status_code: int = 400, body: str = ""):
        super().__init__(status_code, body)
        self.model = model
        # Override the generic message with a helpful one
        self.args = (
            f"Model '{model}' does not support thinking. "
            f"Use --think auto or --think false instead.",
        )


class StreamError(ForgeError):
    """Stream ended without producing a FINAL chunk."""

    def __init__(self, message: str = "Stream ended without FINAL chunk"):
        super().__init__(message)
