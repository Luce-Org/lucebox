"""Deterministic held-out NIAH-v2 case generation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

DEPTHS = (0.05, 0.25, 0.50, 0.75, 0.95)
ANSWER_TYPES = ("seven-digit", "uuid", "word-pair", "short-phrase")

FILLER_FAMILIES = (
    (
        "field-notes",
        "Field note {index}: the archive records routine observations about weather, "
        "equipment checks, and ordinary travel arrangements. No instruction in this "
        "note changes the requested response format.\n",
    ),
    (
        "catalog",
        "Catalog record {index}: item descriptions, shelf locations, and unrelated numeric "
        "inventory references are listed for background context only.\n",
    ),
    (
        "meeting-minutes",
        "Meeting minute {index}: participants reviewed schedules, maintenance, documentation, "
        "and several unrelated identifiers before moving to the next agenda item.\n",
    ),
)

HELD_OUT_TEMPLATES = (
    (
        "archive-key",
        "The authoritative archive key is {answer}. Preserve it exactly.",
        "Return only the authoritative archive key.",
    ),
    (
        "verification-token",
        "For the final audit, the verification token is {answer}. This statement supersedes "
        "all unrelated identifiers.",
        "Return only the final audit verification token.",
    ),
    (
        "reference-label",
        "The reference label requested by the operator is {answer}. Keep its spelling and case.",
        "Return only the operator's requested reference label.",
    ),
)

WORD_BANK = (
    "amber",
    "birch",
    "cobalt",
    "delta",
    "ember",
    "fjord",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "kepler",
    "lumen",
)


@dataclasses.dataclass(frozen=True)
class NiahCase:
    case_id: str
    seed: int
    context_tokens: int
    depth_fraction: float
    filler_id: str
    template_id: str
    answer_type: str
    answer: str
    prompt: str
    prompt_sha256: str
    tokenizer_revision: str
    actual_context_tokens: int
    actual_depth_fraction: float
    control: str | None = None

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _answer(answer_type: str, rng: random.Random) -> str:
    if answer_type == "seven-digit":
        return f"{rng.randrange(1_000_000, 10_000_000)}"
    if answer_type == "uuid":
        return str(uuid.UUID(int=rng.getrandbits(128)))
    if answer_type == "word-pair":
        left, right = rng.sample(WORD_BANK, 2)
        return f"{left}-{right}"
    if answer_type == "short-phrase":
        words = rng.sample(WORD_BANK, 3)
        return " ".join(words)
    raise ValueError(f"unknown answer type: {answer_type}")


def _filler(target_characters: int, family: tuple[str, str], rng: random.Random) -> str:
    name, template = family
    records: list[str] = []
    characters = 0
    index = 0
    while characters < target_characters:
        unrelated = rng.randrange(1_000_000, 10_000_000)
        record = template.format(index=f"{index}-{unrelated}")
        records.append(record)
        characters += len(record)
        index += 1
    rng.shuffle(records)
    return "".join(records), name


def _validate_revision(revision: str) -> None:
    if len(revision) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in revision):
        raise ValueError("tokenizer_revision must be an immutable lowercase SHA")


def _sized_prompt(
    *,
    context_tokens: int,
    depth_fraction: float,
    body: str,
    question: str,
    family: tuple[str, str],
    rng: random.Random,
    token_counter: Callable[[str], int],
    chars_per_token: float,
    token_tolerance: int,
) -> tuple[str, str, int, float]:
    """Size and place a case using the campaign tokenizer, never a char proxy."""

    if context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if token_tolerance < 0:
        raise ValueError("token_tolerance must be non-negative")
    filler, filler_id = _filler(
        round(context_tokens * chars_per_token * 2.0), family, rng
    )

    def render(filler_characters: int) -> tuple[str, int]:
        selected = filler[:filler_characters]
        insertion = min(len(selected), round(len(selected) * depth_fraction))
        prompt = selected[:insertion] + "\n" + body + "\n" + selected[insertion:] + "\n" + question
        return prompt, insertion

    filler_characters = min(len(filler), round(context_tokens * chars_per_token))
    best: tuple[int, str, int] | None = None
    for _ in range(16):
        prompt, insertion = render(filler_characters)
        actual = token_counter(prompt)
        if not isinstance(actual, int) or isinstance(actual, bool) or actual <= 0:
            raise ValueError("token counter must return a positive integer")
        if actual <= context_tokens and (best is None or actual > best[0]):
            best = (actual, prompt, insertion)
        if actual <= context_tokens and context_tokens - actual <= token_tolerance:
            break
        observed_chars_per_token = len(prompt) / actual
        adjustment = round((context_tokens - actual) * observed_chars_per_token)
        if adjustment == 0:
            adjustment = 1 if actual < context_tokens else -1
        next_characters = max(0, min(len(filler), filler_characters + adjustment))
        if next_characters == filler_characters:
            break
        filler_characters = next_characters

    if best is None or context_tokens - best[0] > token_tolerance:
        observed = None if best is None else best[0]
        raise ValueError(
            f"tokenizer-sized prompt missed target {context_tokens} by more than "
            f"{token_tolerance} tokens (best={observed})"
        )
    actual, prompt, insertion = best
    prefix_tokens = token_counter(prompt[:insertion]) if insertion else 0
    actual_depth = prefix_tokens / actual
    if abs(actual_depth - depth_fraction) > 0.02:
        raise ValueError(
            f"tokenizer-sized needle depth {actual_depth:.4f} differs from "
            f"planned {depth_fraction:.4f}"
        )
    return prompt, filler_id, actual, actual_depth


def generate_primary_cases(
    *,
    context_tokens: int,
    tokenizer_revision: str,
    seed_base: int = 20260718,
    chars_per_token: float = 4.0,
    token_counter: Callable[[str], int] | None = None,
    token_tolerance: int = 8,
) -> list[NiahCase]:
    """Generate the 5 depths x 5 seed families x 4 answer types design."""

    _validate_revision(tokenizer_revision)
    if token_counter is None:
        raise ValueError("publication NIAH generation requires an actual tokenizer counter")
    cases: list[NiahCase] = []
    for depth_index, depth in enumerate(DEPTHS):
        for seed_family in range(5):
            for answer_index, answer_type in enumerate(ANSWER_TYPES):
                seed = seed_base + depth_index * 10_000 + seed_family * 100 + answer_index
                rng = random.Random(seed)
                family = FILLER_FAMILIES[(seed_family + answer_index) % len(FILLER_FAMILIES)]
                template_id, needle_template, question = HELD_OUT_TEMPLATES[
                    (depth_index + seed_family) % len(HELD_OUT_TEMPLATES)
                ]
                answer = _answer(answer_type, rng)
                prompt, filler_id, actual_tokens, actual_depth = _sized_prompt(
                    context_tokens=context_tokens,
                    depth_fraction=depth,
                    body=needle_template.format(answer=answer),
                    question=question,
                    family=family,
                    rng=rng,
                    token_counter=token_counter,
                    chars_per_token=chars_per_token,
                    token_tolerance=token_tolerance,
                )
                case_id = (
                    f"niah-{context_tokens}-d{depth_index}-s{seed_family}-a{answer_index}"
                )
                cases.append(
                    NiahCase(
                        case_id=case_id,
                        seed=seed,
                        context_tokens=context_tokens,
                        depth_fraction=depth,
                        filler_id=filler_id,
                        template_id=template_id,
                        answer_type=answer_type,
                        answer=answer,
                        prompt=prompt,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        tokenizer_revision=tokenizer_revision,
                        actual_context_tokens=actual_tokens,
                        actual_depth_fraction=actual_depth,
                    )
                )
    if len(cases) != 100:
        raise AssertionError(f"primary NIAH design produced {len(cases)} cases")
    return cases


def generate_control_cases(
    *,
    context_tokens: int,
    tokenizer_revision: str,
    seed_base: int = 20260718,
    chars_per_token: float = 4.0,
    token_counter: Callable[[str], int] | None = None,
    token_tolerance: int = 8,
) -> list[NiahCase]:
    _validate_revision(tokenizer_revision)
    if token_counter is None:
        raise ValueError("publication NIAH generation requires an actual tokenizer counter")
    controls = ("answer-absent", "misleading", "multiple", "prefix", "prompt-echo")
    cases: list[NiahCase] = []
    for control_index, control in enumerate(controls):
        for repetition in range(4):
            seed = seed_base + 900_000 + control_index * 100 + repetition
            rng = random.Random(seed)
            answer = _answer(ANSWER_TYPES[repetition], rng)
            distractor = _answer(ANSWER_TYPES[repetition], rng)
            if control == "answer-absent":
                body = "No authoritative value is present in this control context."
            elif control == "misleading":
                body = f"A superseded record contains {distractor}; it is not authoritative."
            elif control == "multiple":
                body = f"Two conflicting unverified records contain {answer} and {distractor}."
            elif control == "prefix":
                body = f"An invalid longer identifier is {answer}{rng.randrange(10, 99)}."
            else:
                body = f"The question text mentions a placeholder such as {distractor}."
            question = "Return the authoritative value only, or NONE if it is unavailable."
            family = FILLER_FAMILIES[(control_index + repetition) % len(FILLER_FAMILIES)]
            prompt, filler_id, actual_tokens, actual_depth = _sized_prompt(
                context_tokens=context_tokens,
                depth_fraction=0.5,
                body=body,
                question=question,
                family=family,
                rng=rng,
                token_counter=token_counter,
                chars_per_token=chars_per_token,
                token_tolerance=token_tolerance,
            )
            cases.append(
                NiahCase(
                    case_id=f"niah-control-{context_tokens}-{control}-{repetition}",
                    seed=seed,
                    context_tokens=context_tokens,
                    depth_fraction=0.5,
                    filler_id=filler_id,
                    template_id="control",
                    answer_type=ANSWER_TYPES[repetition],
                    answer="NONE",
                    prompt=prompt,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    tokenizer_revision=tokenizer_revision,
                    actual_context_tokens=actual_tokens,
                    actual_depth_fraction=actual_depth,
                    control=control,
                )
            )
    return cases


def write_cases(path: Path, cases: Iterable[NiahCase]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            line = json.dumps(case.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
    return digest.hexdigest()
