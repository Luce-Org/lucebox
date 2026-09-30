#!/usr/bin/env python3
"""Live quality checks for the Jev-like /v1/systemone endpoint.

The cases are adapted from public Jev examples:

- TypeSafe State and Primitives documentation:
  https://docs.typesafe.ai/concepts/state
  https://docs.typesafe.ai/primitives
- TypeSafe API reference:
  https://docs.typesafe.ai/api
- Community ticket-triage example and published Jev observations:
  https://github.com/rajivkuriakose/typesafe-jev-examples
  https://github.com/ItBayMax/typesafe-ai-jev-example

Run against an existing server:

    python3 server/tests/test_systemone_quality.py \
        --base-url http://127.0.0.1:18080 --model-name luce-dflash

Or launch a local Qwen3.5/Qwen3.6 model:

    python3 server/tests/test_systemone_quality.py \
        --launch server/models/Qwen3.6-27B-Q4_K_M.gguf \
        --server-bin server/build/dflash_server
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRIAGE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "Charges, refunds, invoices, subscription changes, cancellations.",
            "technical": "Bugs, failed integrations, errors, or anything not working.",
            "sales": "Pricing, plan comparisons, or capabilities of plans not yet purchased.",
            "other": "None of the other options clearly applies.",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound?",
        "criteria": [
            "Neutral or friendly; stating facts or asking a question.",
            "Visibly annoyed but still civil.",
            "Angry; strong language, threats, or accusations.",
        ],
    },
    "business_impact": {
        "type": "score",
        "instructions": "How much is the customer's own business being harmed right now?",
        "criteria": [
            "No harm; a question or preference.",
            "Inconvenience; a workaround exists.",
            "Active revenue or operational loss while this continues.",
        ],
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "Does the message convey urgency or time-sensitivity?",
    },
    "is_repeat_contact": {
        "type": "noul",
        "instructions": "Does the customer say they contacted support about this before?",
    },
    "threatens_churn": {
        "type": "noul",
        "instructions": "Does the customer threaten to cancel, charge back, or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly ask for a refund or chargeback?",
    },
}


@dataclass(frozen=True)
class QualityCase:
    name: str
    state: Any
    questions: dict[str, Any]
    expectations: dict[str, dict[str, Any]]


CASES = [
    QualityCase(
        name="official_structured_duplicate_charge",
        state={
            "ticket": {
                "subject": "Duplicate charge",
                "messages": [
                    {
                        "from": "customer",
                        "text": (
                            "I was charged twice for order A-104. "
                            "Please refund the duplicate."
                        ),
                    },
                    {"from": "support", "text": "We are checking the charges."},
                ],
            },
            "order": {
                "id": "A-104",
                "charges": [
                    {"amount_usd": 49, "status": "captured"},
                    {"amount_usd": 49, "status": "captured"},
                ],
            },
            "refund_policy": "Duplicate charges are eligible for a refund.",
        },
        questions={
            "department": TRIAGE_QUESTIONS["department"],
            "refund_requested": {
                "type": "noul",
                "instructions": "Does `ticket.messages[0].text` request a refund?",
            },
            "policy_supports_refund": {
                "type": "noul",
                "instructions": (
                    "Does `refund_policy` support the refund requested in "
                    "`ticket.messages[0].text`, given `order.charges`?"
                ),
            },
        },
        expectations={
            "department": {"choice": {"billing"}},
            "refund_requested": {"noul_min": 0.55},
            "policy_supports_refund": {"noul_min": 0.55},
        },
    ),
    QualityCase(
        name="official_cancelled_flight_refund",
        state={
            "ticket_message": "My flight was cancelled. Can I get a refund?",
            "refund_policy": "Cancelled flights are eligible for a full refund.",
        },
        questions={
            "refund_requested": {
                "type": "noul",
                "instructions": "Does `ticket_message` request a refund?",
            },
            "request_type": {
                "type": "choice",
                "instructions": "What is the main request in `ticket_message`?",
                "criteria": {
                    "refund": "The customer wants money returned.",
                    "rebooking": "The customer wants a replacement flight.",
                    "information": "The customer is asking for information only.",
                },
            },
            "frustration": {
                "type": "score",
                "instructions": (
                    "How frustrated does the customer appear in `ticket_message`?"
                ),
                "criteria": [
                    "Calm and neutral.",
                    "Concerned but civil.",
                    "Very angry or using strong language.",
                ],
            },
        },
        expectations={
            "refund_requested": {"noul_min": 0.55},
            "request_type": {"choice": {"refund"}},
            "frustration": {"score_max": 1.5},
        },
    ),
    QualityCase(
        name="official_failed_payouts",
        state="Help! My payouts have been failing for 3 days.",
        questions={"department": TRIAGE_QUESTIONS["department"]},
        expectations={"department": {"choice": {"technical"}}},
    ),
    QualityCase(
        name="community_stripe_outage",
        state={
            "ticket": (
                "Hi, I've been trying to connect my Stripe account for 3 days "
                "and it keeps failing. I'm losing sales. Please help ASAP."
            )
        },
        questions=TRIAGE_QUESTIONS,
        expectations={
            "department": {"choice": {"technical"}},
            "business_impact": {"score_min": 1.25},
            "is_urgent": {"noul_min": 0.55},
            "threatens_churn": {"noul_max": 0.45},
            "refund_requested": {"noul_max": 0.45},
        },
    ),
    QualityCase(
        name="community_plan_question",
        state={
            "ticket": (
                "Quick question: does the Team plan include SSO, or is that "
                "Enterprise only? No rush, just planning next quarter."
            )
        },
        questions=TRIAGE_QUESTIONS,
        expectations={
            "department": {"choice": {"sales"}},
            "business_impact": {"score_max": 0.75},
            "is_urgent": {"noul_max": 0.45},
            "threatens_churn": {"noul_max": 0.45},
            "refund_requested": {"noul_max": 0.45},
        },
    ),
    QualityCase(
        name="community_repeat_churn_refund",
        state={
            "ticket": (
                "This is the fourth time I've written in. Nobody reads these. "
                "Cancel my account and refund the last two months or I'm "
                "filing a chargeback."
            )
        },
        questions=TRIAGE_QUESTIONS,
        expectations={
            "department": {"choice": {"billing"}},
            "frustration": {"score_min": 1.25},
            "is_repeat_contact": {"noul_min": 0.55},
            "threatens_churn": {"noul_min": 0.55},
            "refund_requested": {"noul_min": 0.55},
        },
    ),
    QualityCase(
        name="community_mixed_intent_fallback",
        state={
            "ticket": (
                "The page occasionally freezes for a moment. Also, I think "
                "an invoice may not have arrived, but I am not sure which one."
            )
        },
        questions={
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": {
                    "billing": "A clear billing, invoice, charge, or refund request.",
                    "technical": "A clear bug, outage, or integration problem.",
                    "account": "A clear login, permission, or profile problem.",
                    "unclear": (
                        "Information is insufficient or multiple unrelated "
                        "requests are mixed together; a person should inspect it."
                    ),
                },
            }
        },
        expectations={"department": {"choice": {"unclear"}}},
    ),
]


def post_json(base_url: str, path: str, body: dict[str, Any],
              timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def wait_ready(base_url: str, process: subprocess.Popen[Any] | None,
               timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(
                base_url.rstrip("/") + "/health", timeout=2
            ) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def validate_answer_shape(question: dict[str, Any], answer: dict[str, Any]) -> None:
    expected_type = question["type"]
    if answer.get("type") != expected_type:
        raise AssertionError(
            f"type mismatch: expected {expected_type}, got {answer.get('type')}"
        )
    if expected_type == "noul":
        value = answer.get("noul")
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise AssertionError(f"invalid noul probability: {value!r}")
        return

    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise AssertionError("missing probabilities")
    if any(
        not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0
        for value in probabilities.values()
    ):
        raise AssertionError(f"invalid probabilities: {probabilities!r}")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
        raise AssertionError(
            f"probabilities sum to {sum(probabilities.values()):.9f}"
        )
    confidence = answer.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise AssertionError(f"invalid confidence: {confidence!r}")

    if expected_type == "choice":
        criteria = question["criteria"]
        if set(probabilities) != set(criteria):
            raise AssertionError(
                f"choice probability keys differ: {set(probabilities)}"
            )
        if answer.get("choice") not in criteria:
            raise AssertionError(f"invalid selected choice: {answer.get('choice')!r}")
        return

    score = answer.get("score")
    criteria = question["criteria"]
    if not isinstance(score, (int, float)) or not 0.0 <= score <= len(criteria) - 1:
        raise AssertionError(f"invalid score: {score!r}")
    expected_levels = {str(index) for index in range(len(criteria))}
    if set(probabilities) != expected_levels:
        raise AssertionError(
            f"score probability keys differ: {set(probabilities)}"
        )
    if set(answer.get("legend", {})) != expected_levels:
        raise AssertionError(f"invalid score legend: {answer.get('legend')!r}")


def semantic_checks(answer: dict[str, Any],
                    expectation: dict[str, Any]) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    if "choice" in expectation:
        expected = expectation["choice"]
        actual = answer["choice"]
        checks.append((actual in expected, f"choice={actual}, expected={sorted(expected)}"))
    if "noul_min" in expectation:
        actual = answer["noul"]
        threshold = expectation["noul_min"]
        checks.append((actual >= threshold, f"noul={actual:.4f} >= {threshold:.2f}"))
    if "noul_max" in expectation:
        actual = answer["noul"]
        threshold = expectation["noul_max"]
        checks.append((actual <= threshold, f"noul={actual:.4f} <= {threshold:.2f}"))
    if "score_min" in expectation:
        actual = answer["score"]
        threshold = expectation["score_min"]
        checks.append((actual >= threshold, f"score={actual:.4f} >= {threshold:.2f}"))
    if "score_max" in expectation:
        actual = answer["score"]
        threshold = expectation["score_max"]
        checks.append((actual <= threshold, f"score={actual:.4f} <= {threshold:.2f}"))
    return checks


def run_suite(base_url: str, model_name: str, timeout: float,
              min_quality: float) -> int:
    quality_passed = 0
    quality_total = 0
    schema_failed = 0

    for case in CASES:
        print(f"\n[{case.name}]")
        body = {
            "model": model_name,
            "state": case.state,
            "questions": case.questions,
        }
        try:
            response = post_json(base_url, "/v1/systemone", body, timeout)
        except Exception as error:
            print(f"  FAIL request: {error}")
            schema_failed += 1
            continue

        if response.get("model") != model_name:
            print(
                f"  FAIL response model={response.get('model')!r}, "
                f"expected={model_name!r}"
            )
            schema_failed += 1
        if response.get("usage", {}).get("output_tokens") != 0:
            print(f"  FAIL expected zero output tokens: {response.get('usage')!r}")
            schema_failed += 1

        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(case.questions):
            print(f"  FAIL answer ids: {answers!r}")
            schema_failed += 1
            continue

        shape_ok = True
        for question_id, question in case.questions.items():
            try:
                validate_answer_shape(question, answers[question_id])
            except AssertionError as error:
                print(f"  FAIL schema {question_id}: {error}")
                schema_failed += 1
                shape_ok = False
        if not shape_ok:
            continue

        for question_id, expectation in case.expectations.items():
            for passed, detail in semantic_checks(
                answers[question_id], expectation
            ):
                quality_total += 1
                quality_passed += int(passed)
                marker = "PASS" if passed else "FAIL"
                print(f"  {marker} {question_id}: {detail}")

    quality = quality_passed / quality_total if quality_total else 0.0
    print(
        f"\nQuality: {quality_passed}/{quality_total} "
        f"({quality:.1%}), required {min_quality:.1%}"
    )
    print(f"Schema failures: {schema_failed}")
    return 0 if schema_failed == 0 and quality >= min_quality else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live quality checks for /v1/systemone"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model-name", default="luce-dflash")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--min-quality", type=float, default=0.75)
    parser.add_argument("--launch", metavar="MODEL_GGUF")
    parser.add_argument(
        "--server-bin", default="server/build/dflash_server"
    )
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--server-log", default="/tmp/systemone-quality-server.log")
    args = parser.parse_args()

    if not 0.0 <= args.min_quality <= 1.0:
        parser.error("--min-quality must be between 0 and 1")

    process: subprocess.Popen[Any] | None = None
    log_file = None
    base_url = args.base_url
    try:
        if args.launch:
            server_bin = Path(args.server_bin)
            model_path = Path(args.launch)
            if not server_bin.is_file():
                parser.error(f"server binary not found: {server_bin}")
            if not model_path.is_file():
                parser.error(f"model not found: {model_path}")
            base_url = f"http://127.0.0.1:{args.port}"
            command = [
                str(server_bin),
                str(model_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--max-ctx",
                "4096",
                "--default-max-tokens",
                "1",
                "--prefix-cache-slots",
                "0",
                "--prefill-cache-slots",
                "0",
                "--model-name",
                args.model_name,
            ]
            print("Launching:", " ".join(command))
            log_file = open(args.server_log, "w", encoding="utf-8")
            process = subprocess.Popen(
                command, stdout=log_file, stderr=subprocess.STDOUT
            )
            if not wait_ready(base_url, process, args.startup_timeout):
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                print(f"Server failed to start; see {args.server_log}")
                return 2

        return run_suite(
            base_url, args.model_name, args.timeout, args.min_quality
        )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
