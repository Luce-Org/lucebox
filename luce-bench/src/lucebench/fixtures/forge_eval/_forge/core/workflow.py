# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Tool and workflow definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


def _to_pascal(name: str) -> str:
    """Convert snake_case tool name to PascalCaseParams."""
    return "".join(part.capitalize() for part in name.split("_")) + "Params"


def _json_schema_to_type(
    prop: dict[str, Any],
    field_name: str,
    model_name_prefix: str,
) -> type:
    """Convert a single JSON Schema property dict to a Python type.

    Handles primitives, enums, nested objects, and arrays recursively.
    """
    # Enum takes priority — Literal type
    if "enum" in prop:
        values = tuple(prop["enum"])
        return Literal[values]  # type: ignore[valid-type]

    json_type = prop.get("type", "string")

    type_map: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    if json_type in type_map:
        return type_map[json_type]

    if json_type == "object":
        # Nested object with its own properties → recursive sub-model
        sub_props = prop.get("properties", {})
        if sub_props:
            sub_required = set(prop.get("required", []))
            return _build_model(
                sub_props,
                sub_required,
                f"{model_name_prefix}_{field_name.capitalize()}",
            )
        return dict

    if json_type == "array":
        items = prop.get("items", {})
        if items:
            item_type = _json_schema_to_type(
                items, field_name + "Item", model_name_prefix
            )
            return list[item_type]  # type: ignore[valid-type]
        return list

    # Unknown type → Any
    return Any  # type: ignore[return-value]


def _build_model(
    properties: dict[str, Any],
    required: set[str],
    model_name: str,
) -> type[BaseModel]:
    """Build a dynamic Pydantic model from JSON Schema properties."""
    fields: dict[str, Any] = {}

    for fname, fprop in properties.items():
        python_type = _json_schema_to_type(fprop, fname, model_name)
        description = fprop.get("description")
        default = fprop.get("default")

        if fname in required:
            if description is not None:
                fields[fname] = (python_type, Field(description=description))
            else:
                fields[fname] = (python_type, ...)
        else:
            # Optional field
            if default is not None:
                if description is not None:
                    fields[fname] = (
                        python_type | None,
                        Field(default=default, description=description),
                    )
                else:
                    fields[fname] = (python_type | None, Field(default=default))
            else:
                if description is not None:
                    fields[fname] = (
                        python_type | None,
                        Field(default=None, description=description),
                    )
                else:
                    fields[fname] = (python_type | None, None)

    return create_model(model_name, **fields)  # type: ignore[call-overload]


class ToolSpec(BaseModel):
    """Declarative tool schema — what the LLM sees."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: type[BaseModel]

    @classmethod
    def from_json_schema(
        cls,
        name: str,
        description: str,
        schema: dict[str, Any],
    ) -> ToolSpec:
        """Create a ToolSpec from a raw JSON Schema dict.

        The *schema* argument is the ``parameters`` object from an OpenAI-style
        tool definition (i.e. a JSON Schema with ``properties``, ``required``,
        etc.).
        """
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        model_name = _to_pascal(name)
        params_cls = _build_model(properties, required, model_name)
        return cls(name=name, description=description, parameters=params_cls)

    def get_json_schema(self) -> dict[str, Any]:
        """Return JSON Schema dict for this tool's parameters."""
        return self.parameters.model_json_schema()


@dataclass
class ToolDef:
    """Binds a tool schema to its implementation.

    Downstream projects define tools as ToolDefs. The Workflow holds these
    in a dict keyed by name, deriving the spec list (for the LLM) and
    callable lookup (for execution) internally.

    Prerequisites express conditional dependencies: "if you call this tool,
    you must have called tool X first." Entries can be:
    - str: name-only ("read_file" — any prior call to read_file satisfies it)
    - dict: arg-matched ({"tool": "read_file", "match_arg": "path"} — a prior
      call to read_file with the same ``path`` value satisfies it)
    """

    spec: ToolSpec
    callable: Callable[..., Any]
    prerequisites: list[str | dict[str, str]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name


class ToolCall(BaseModel):
    """Validated tool invocation returned by an LLMClient."""

    tool: str
    args: dict[str, Any]
    reasoning: str | None = None


class TextResponse(BaseModel):
    """Non-tool-call response from the model (reasoning trace, refusal, etc.)."""

    content: str


type LLMResponse = list[ToolCall] | TextResponse


@dataclass
class Workflow:
    """Declarative workflow definition. Provided by downstream projects.

    The Workflow holds ToolDefs in an ordered dict keyed by tool name.
    Keys must match ToolDef.spec.name — validated at construction time.
    It does NOT contain execution logic — that's the WorkflowRunner's job.
    """

    name: str
    description: str
    tools: dict[str, ToolDef]
    required_steps: list[str]
    terminal_tool: str | list[str]
    system_prompt_template: str
    terminal_tools: frozenset[str] = field(default_factory=frozenset, init=False)

    def __post_init__(self) -> None:
        # Normalize terminal_tool to frozenset for O(1) membership checks.
        if isinstance(self.terminal_tool, str):
            self.terminal_tools = frozenset([self.terminal_tool])
        else:
            self.terminal_tools = frozenset(self.terminal_tool)

        for key, tool_def in self.tools.items():
            if key != tool_def.name:
                raise ValueError(
                    f"Tool key '{key}' does not match ToolDef name '{tool_def.name}'"
                )
        tool_names = set(self.tools.keys())
        for step in self.required_steps:
            if step not in tool_names:
                raise ValueError(
                    f"Required step '{step}' not in tools: {tool_names}"
                )
        for tt in self.terminal_tools:
            if tt not in tool_names:
                raise ValueError(
                    f"Terminal tool '{tt}' not in tools: {tool_names}"
                )
            if tt in self.required_steps:
                raise ValueError(
                    f"Terminal tool '{tt}' cannot also be a required step"
                )
        for key, tool_def in self.tools.items():
            for prereq in tool_def.prerequisites:
                prereq_name = prereq if isinstance(prereq, str) else prereq["tool"]
                if prereq_name not in tool_names:
                    raise ValueError(
                        f"Prerequisite '{prereq_name}' for tool '{key}' "
                        f"not in tools: {tool_names}"
                    )

    def build_system_prompt(self, **kwargs: str) -> str:
        """Render the system prompt with user-provided values."""
        return self.system_prompt_template.format(**kwargs)

    def get_tool_specs(self) -> list[ToolSpec]:
        """Return all tool specs for passing to the LLM client."""
        return [t.spec for t in self.tools.values()]

    def get_callable(self, tool_name: str) -> Callable[..., Any]:
        """Return the callable for a tool by name. Raises KeyError if not found."""
        return self.tools[tool_name].callable
