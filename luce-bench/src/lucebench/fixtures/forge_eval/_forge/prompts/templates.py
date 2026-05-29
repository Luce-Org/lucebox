# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Tool prompt builders for the prompt-injected tool calling path."""

from __future__ import annotations

import json
import re

from ..core.workflow import ToolCall, ToolSpec


def build_tool_prompt(tools: list[ToolSpec]) -> str:
    """Build tool description block for prompt injection.

    Args:
        tools: The list of tool specs to describe.
    """
    lines = ["You have access to the following tools:", ""]

    for tool in tools:
        schema = tool.get_json_schema()
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        lines.append(f"## {tool.name}")
        lines.append(f"Description: {tool.description}")
        if properties:
            lines.append("Parameters:")
            for name, prop in properties.items():
                req = " (required)" if name in required else " (optional)"
                ptype = prop.get("type", "any")
                desc = prop.get("description", "")
                lines.append(f"  - {name} ({ptype}{req}): {desc}")
                if "enum" in prop:
                    lines.append(f"    Allowed values: {', '.join(str(v) for v in prop['enum'])}")
        lines.append("")

    lines.append("To call a tool, respond with ONLY a JSON object in this exact format:")
    lines.append('{"tool": "<tool_name>", "args": {<arguments>}}')
    lines.append("")
    lines.append("Example:")
    if tools:
        example_tool = tools[0]
        example_schema = example_tool.get_json_schema()
        example_args = {
            name: f"<{name}>" for name in example_schema.get("properties", {})
        }
        lines.append(json.dumps({"tool": example_tool.name, "args": example_args}))
    lines.append("")
    lines.append("Respond with ONLY the JSON tool call. Do not include any other text.")

    return "\n".join(lines)


def extract_tool_call(text: str, available_tools: list[str]) -> list[ToolCall]:
    """Extract all ToolCalls from free-text model output.

    Handles JSON wrapped in code fences or embedded in surrounding text.
    Returns all valid tool calls found, or an empty list if none.

    Args:
        text: The raw model output text.
        available_tools: List of valid tool names to match against.
    """
    # Strip code fences if present
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", text)
    cleaned = re.sub(r"```", "", cleaned)

    found: list[ToolCall] = []
    # Try to find JSON objects by scanning for opening braces
    i = 0
    while i < len(cleaned):
        if cleaned[i] == "{":
            # Find matching closing brace
            depth = 0
            for j in range(i, len(cleaned)):
                if cleaned[j] == "{":
                    depth += 1
                elif cleaned[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[i : j + 1]
                        result = _try_parse_tool_call(candidate, available_tools)
                        if result is not None:
                            found.append(result)
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return found


def _try_parse_tool_call(json_str: str, available_tools: list[str]) -> ToolCall | None:
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    # Forge style: {"tool": "...", "args": {...}}
    # OpenAI style: {"name": "...", "arguments": {...}}
    # Granite 4.0 emits OpenAI-style inside <tool_call> tags.
    tool_name = data.get("tool") or data.get("name")
    if tool_name not in available_tools:
        return None

    args = data.get("args")
    if args is None:
        args = data.get("arguments", {})

    return ToolCall(tool=tool_name, args=args)


# Pattern for native FC rehearsal syntax: tool_name[ARGS]{...}
# Reasoning models rehearse tool calls in thinking tokens using this format.
# Captures: tool name and the JSON args blob.
_REHEARSAL_RE = re.compile(
    r"(\w+)\[ARGS\](\{.*\})", re.DOTALL
)

# Think tag patterns (same as llamafile._THINK_TAG_RE) — needed to strip
# thinking blocks before rescue parsing.
_THINK_TAG_RE = re.compile(
    r"\[THINK\].*?\[/THINK\]|<think>.*?</think>", re.DOTALL
)

# Qwen Coder XML tool call format.
# <function=name>
#   <parameter=key>value</parameter>
#   <parameter=other>value</parameter>
# </function>
# Pattern adapted from Qwen's reference parser:
# https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/qwen3coder_tool_parser.py
_QWEN_FUNCTION_RE = re.compile(
    r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL
)
_QWEN_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)

# Mistral native bracket-tag tool call format:
#   [TOOL_CALLS]<tool_name>{<json_args>}
# with optional whitespace/newline between the name and the opening brace.
# Emitted by Devstral-Small-2 and Mistral-Small-3.x family in prompt mode
# when the model falls back to its native serialization. Anchor matches
# only the [TOOL_CALLS]<name> prefix; the JSON args are extracted via
# brace-balance scan in _parse_mistral_bracket_tool_calls.
_MISTRAL_BRACKET_RE = re.compile(r"\[TOOL_CALLS\](\w+)\s*(?=\{)")


def _parse_qwen_xml_tool_calls(text: str, available_tools: list[str]) -> list[ToolCall]:
    """Parse Qwen Coder XML-format tool calls from model output.

    Handles the format emitted by Qwen3-Coder models (and occasionally other
    models trained on similar data), with or without the outer <tool_call>
    wrapper. Whitespace behavior matches Qwen's reference parser: one leading
    and one trailing newline are stripped from each parameter value.

    Type coercion is deferred to Pydantic — all parameter values are passed
    as strings, and the tool's parameter model coerces at ToolCall construction.
    """
    found: list[ToolCall] = []
    for fn_match in _QWEN_FUNCTION_RE.finditer(text):
        tool_name = fn_match.group(1).strip()
        if tool_name not in available_tools:
            continue

        body = fn_match.group(2)
        args: dict[str, str] = {}
        for param_match in _QWEN_PARAMETER_RE.finditer(body):
            key = param_match.group(1).strip()
            value = param_match.group(2)
            # Strip the first newline after the opening tag and the last
            # newline before the closing tag — matches Qwen's parser.
            if value.startswith("\n"):
                value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
            args[key] = value

        found.append(ToolCall(tool=tool_name, args=args))

    return found


def _parse_mistral_bracket_tool_calls(text: str, available_tools: list[str]) -> list[ToolCall]:
    """Parse Mistral native ``[TOOL_CALLS]<name>{<args>}`` tool-call format.

    Devstral-Small-2 and Mistral-Small-3.x emit this shape in prompt mode when
    they fall back to their training-data tool-call serialization. The args
    are JSON; extracted via brace-balance scan to handle nested objects /
    strings that contain literal braces.

    Optional whitespace (including newlines) is permitted between the tool
    name and the opening ``{``. Multiple bracket-tagged calls in one message
    are returned as a list.
    """
    found: list[ToolCall] = []
    for m in _MISTRAL_BRACKET_RE.finditer(text):
        tool_name = m.group(1)
        if tool_name not in available_tools:
            continue
        # Brace-balance scan starting at the opening brace (lookahead-anchored).
        i = m.end()
        if i >= len(text) or text[i] != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for j in range(i, len(text)):
            ch = text[j]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    try:
                        args = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(args, dict):
                        found.append(ToolCall(tool=tool_name, args=args))
                    break
    return found


def rescue_tool_call(text: str, available_tools: list[str]) -> list[ToolCall]:
    """Try to parse ToolCalls from a TextResponse that failed native FC.

    Used by the runner to rescue valid tool calls that the model emitted as
    free text instead of structured output. Returns an empty list if nothing
    parseable is found — caller falls through to the normal retry nudge.

    Parsing strategies (in order):
    1. Prompt-injected JSON: {"tool": "name", "args": {...}}
    2. Rehearsal syntax: tool_name[ARGS]{...}
    3. Qwen Coder XML: <function=name><parameter=key>value</parameter></function>
    4. Mistral bracket-tag: [TOOL_CALLS]<name>{<args>}
    """
    # Strip think tags — the tool call may be after or outside thinking blocks
    cleaned = _THINK_TAG_RE.sub("", text).strip()
    if not cleaned:
        return []

    # Strategy 1: existing JSON extraction (handles code fences, embedded JSON)
    found = extract_tool_call(cleaned, available_tools)

    # Strategy 2: rehearsal syntax — tool_name[ARGS]{...}
    # Only try if JSON extraction found nothing (avoid double-counting)
    if not found:
        for m in _REHEARSAL_RE.finditer(cleaned):
            tool_name, args_str = m.group(1), m.group(2)
            if tool_name in available_tools:
                try:
                    args = json.loads(args_str)
                    if isinstance(args, dict):
                        found.append(ToolCall(tool=tool_name, args=args))
                except json.JSONDecodeError:
                    pass

    # Strategy 3: Qwen Coder XML — <function=name><parameter=key>value</parameter></function>
    if not found:
        found = _parse_qwen_xml_tool_calls(cleaned, available_tools)

    # Strategy 4: Mistral bracket-tag — [TOOL_CALLS]<name>{<args>}
    if not found:
        found = _parse_mistral_bracket_tool_calls(cleaned, available_tools)

    return found
