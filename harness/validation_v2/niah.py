"""Deterministic held-out NIAH-v2 case generation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import uuid
from collections.abc import Iterable
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
    index = 0
    while sum(map(len, records)) < target_characters:
        unrelated = rng.randrange(1_000_000, 10_000_000)
        records.append(template.format(index=f"{index}-{unrelated}"))
        index += 1
    rng.shuffle(records)
    return "".join(records), name


def generate_primary_cases(
    *,
    context_tokens: int,
    tokenizer_revision: str,
    seed_base: int = 20260718,
    chars_per_token: float = 4.0,
) -> list[NiahCase]:
    """Generate the 5 depths x 5 seed families x 4 answer types design."""

    cases: list[NiahCase] = []
    target_characters = round(context_tokens * chars_per_token)
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
                filler, filler_id = _filler(target_characters, family, rng)
                insertion = min(len(filler), round(len(filler) * depth))
                needle = "\n" + needle_template.format(answer=answer) + "\n"
                prompt = filler[:insertion] + needle + filler[insertion:] + "\n" + question
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
) -> list[NiahCase]:
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
            prompt = body + "\nReturn the authoritative value only, or NONE if it is unavailable."
            cases.append(
                NiahCase(
                    case_id=f"niah-control-{context_tokens}-{control}-{repetition}",
                    seed=seed,
                    context_tokens=context_tokens,
                    depth_fraction=0.5,
                    filler_id="control",
                    template_id="control",
                    answer_type=ANSWER_TYPES[repetition],
                    answer="NONE" if control in {"answer-absent", "misleading", "prompt-echo"} else answer,
                    prompt=prompt,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    tokenizer_revision=tokenizer_revision,
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
