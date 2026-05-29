import json

import httpx

from lucebox import smoke


def test_tool_smoke_disables_thinking_for_deterministic_tool_calls():
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        seen_body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "report_status", "arguments": "{}"}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    ok, err = smoke._check_tool_call(client, "http://test", 1.0)

    assert ok is True
    assert err == ""
    assert seen_body["chat_template_kwargs"] == {"enable_thinking": False}
