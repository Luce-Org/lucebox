"""collect.py streaming/resume and grade.py judge routing, against in-process fakes."""

import asyncio
import json

import httpx
from eval.collect import collect_targets, completed_keys, plan_jobs, stream_chat
from eval.common import DEFAULT_ARMS
from eval.common import read_jsonl
from eval.grade import grade_rows, needs_regrade, reference_answers
from eval.graders import GradeResult
from eval.judge import Judge, JudgeConfig


def sse(*chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def test_stream_chat_records_text_reasoning_usage_and_request_shape():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return sse(
            {"choices": [{"delta": {"reasoning_content": "thinking..."}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4}},
        )

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await stream_chat(c, "http://x/v1", "m", [{"role": "user", "content": "hi"}],
                                     thinking=True, max_tokens=64, temperature=0.0)

    rec = asyncio.run(go())
    assert (rec["text"], rec["reasoning"], rec["finish_reason"]) == ("Hello world", "thinking...", "stop")
    assert (rec["prompt_tokens"], rec["completion_tokens"], rec["error"]) == (11, 4, None)
    assert 0 <= rec["ttft_ms"] <= rec["total_ms"]
    assert seen["chat_template_kwargs"] == {"enable_thinking": True}
    assert seen["stream"] is True and seen["max_tokens"] == 64


def test_stream_chat_http_error_is_recorded_not_raised():
    async def go():
        t = httpx.MockTransport(lambda r: httpx.Response(503, text="busy"))
        async with httpx.AsyncClient(transport=t) as c:
            return await stream_chat(c, "http://x/v1", "m", [], thinking=False,
                                     max_tokens=8, temperature=0.0)

    rec = asyncio.run(go())
    assert rec["error"].startswith("HTTP 503") and rec["text"] == ""


def test_resume_skips_successes_and_retries_errors(tmp_path):
    answers = tmp_path / "answers.jsonl"
    rows = [
        {"id": "a", "model": "s", "thinking": False, "error": None},
        {"id": "b", "model": "s", "thinking": False, "error": "timeout"},
        {"id": "c", "model": "s", "thinking": False, "error": "timeout"},
        {"id": "c", "model": "s", "thinking": False, "error": None},  # retried later, succeeded
    ]
    answers.write_text("".join(json.dumps(r) + "\n" for r in rows))
    prompts = [{"id": i} for i in "abcd"]
    jobs = plan_jobs(prompts, [("s", False), ("l", False)], completed_keys(answers))
    assert [(p["id"], m) for p, m, _ in jobs] == [
        ("b", "s"), ("d", "s"), ("a", "l"), ("b", "l"), ("c", "l"), ("d", "l")]


def test_default_targets_are_the_arms_with_shared_backend():
    assert collect_targets(DEFAULT_ARMS) == [("qwen35-2b", False), ("qwen38-27b", False), ("qwen38-27b", True)]
    assert collect_targets(DEFAULT_ARMS, models=["qwen35-0.8b"], thinking="both") == [
        ("qwen35-0.8b", False), ("qwen35-0.8b", True)]


class FakeJudge:
    def __init__(self, verdict=True):
        self.calls = []
        self.verdict = verdict

    async def grade(self, request_text, raw_answer, reference=None, rubric=None):
        self.calls.append({"answer": raw_answer, "reference": reference, "rubric": rubric})
        return GradeResult(self.verdict, float(self.verdict), "judge", {"reference": bool(reference)})


PROMPTS = {
    "open": {"id": "open", "messages": [{"role": "user", "content": "design X"}],
             "grader": {"type": "judge", "rubric": "mention Y"}},
    "calc": {"id": "calc", "messages": [{"role": "user", "content": "1/2?"}],
             "grader": {"type": "math", "answer": "\\frac{1}{2}", "judge_fallback": True}},
}


def ans(pid, model, thinking, text, error=None):
    return {"id": pid, "model": model, "thinking": thinking, "text": text, "error": error,
            "finish_reason": "stop"}


def test_grade_rows_routes_to_judge_with_27b_thinking_reference(tmp_path):
    answers = [
        ans("open", "qwen38-27b", True, "<think>...</think>REF ANSWER"),
        ans("open", "qwen35-0.8b", False, "small answer"),
        ans("open", "qwen35-2b", False, "", error="timeout"),
        ans("calc", "qwen35-0.8b", False, "\\boxed{0.5}"),          # deterministic pass
        ans("calc", "qwen35-2b", False, "it is one half, \\boxed{1/2 }"),  # deterministic pass
        ans("calc", "qwen38-27b", False, "\\boxed{\\sin(\\pi/6)}"),  # miss -> judge fallback
    ]
    keyed = {(a["id"], a["model"], a["thinking"]): a for a in answers}
    refs = reference_answers(keyed, "qwen38-27b")
    assert refs == {"open": "REF ANSWER"}
    judge = FakeJudge()
    out = tmp_path / "grades.jsonl"
    asyncio.run(grade_rows(PROMPTS, answers, judge, refs, out))
    grades = {(g["id"], g["model"]): g for g in read_jsonl(out)}

    # the reference row is judged without itself as reference; the small model gets it
    by_answer = {c["answer"]: c for c in judge.calls}
    assert by_answer["small answer"]["reference"] == "REF ANSWER"
    assert by_answer["small answer"]["rubric"] == "mention Y"
    assert by_answer["<think>...</think>REF ANSWER"]["reference"] is None
    # failed request: graded 0 without spending a judge call
    assert grades[("open", "qwen35-2b")]["grader"] == "no_answer"
    assert not grades[("open", "qwen35-2b")]["correct"]
    assert grades[("calc", "qwen35-0.8b")]["grader"] == "math"
    assert grades[("calc", "qwen38-27b")]["grader"] == "math+judge"
    assert by_answer["\\boxed{\\sin(\\pi/6)}"]["reference"] == "Final answer: \\frac{1}{2}"
    assert len(judge.calls) == 3


def test_needs_regrade_once_reference_exists():
    g = {"id": "open", "model": "qwen35-2b", "thinking": False, "grader": "judge",
         "detail": {"reference": False}}
    assert needs_regrade(g, {"open": "REF"})
    assert not needs_regrade(g, {})
    assert not needs_regrade({**g, "detail": {"reference": True}}, {"open": "REF"})
    assert needs_regrade({**g, "grader": "judge_error"}, {})


def test_judge_request_and_verdict_parsing():
    bodies = []
    replies = iter(['Sure.\n{"score": 1, "reason": "correct"}', "garbage"])

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    async def go():
        judge = Judge(JudgeConfig(base_url="http://j/v1", retries=0),
                      client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            ok = await judge.grade("Q?", "<think>x</think>A", reference="R", rubric="must say A")
            bad = await judge.grade("Q?", "A")
        finally:
            await judge.aclose()
        return ok, bad

    ok, bad = asyncio.run(go())
    assert (ok.correct, ok.grader, ok.detail["reason"]) == (True, "judge", "correct")
    assert (bad.correct, bad.grader) == (False, "judge_error")
    body = bodies[0]
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    user = body["messages"][-1]["content"]
    assert "REFERENCE" in user and "must say A" in user and "<think>" not in user
