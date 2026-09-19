import json


def same_json_type_and_value(actual, expected):
    """Compare decoded JSON without Python's bool/int or int/float coercions."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same_json_type_and_value(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_json_type_and_value(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def score_message(name, message, finish_reason, expected):
    """Apply the frozen JSON/tool rubric without repairing model output."""
    try:
        if name == "tool_call":
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                return False
            function = calls[0]["function"]
            if function["name"] != "lookup_weather" or finish_reason != "tool_calls":
                return False
            value = json.loads(function["arguments"])
        else:
            if finish_reason != "stop":
                return False
            # json.loads permits surrounding JSON whitespace, but fenced JSON,
            # prose and <think> wrappers remain format failures.
            value = json.loads(message.get("content") or "")
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        return False
    return same_json_type_and_value(value, expected)
