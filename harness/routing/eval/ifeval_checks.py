"""Subset of the IFEval instruction checkers (Zhou et al. 2023), re-implemented.

Follows the reference semantics of google-research/instruction_following_eval for
the instruction ids in ``CHECKERS``. ``language:response_language`` is excluded
(needs a language-ID model); build_set.py only samples prompts whose every
instruction is supported here.

Grading uses the paper's *loose* mode: an instruction passes if it holds for the
response or for any variant with the first/last line removed and/or ``*`` stripped
(so a markdown preamble like "Sure, here it is:" does not fail it).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any


def _relation_ok(value: int, relation: str, target: int) -> bool:
    if relation == "less than":
        return value < target
    if relation == "at least":
        return value >= target
    raise ValueError(f"unknown relation {relation!r}")


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip())
    return [p for p in parts if p.strip()]


def no_comma(text: str, **_: Any) -> bool:
    return "," not in text


def number_highlighted_sections(text: str, num_highlights: int, **_: Any) -> bool:
    hits = [h for h in re.findall(r"\*[^\n\*]*\*", text) if h.strip("*").strip()]
    hits += [h for h in re.findall(r"\*\*[^\n\*]*\*\*", text) if h.strip("*").strip()]
    return len(hits) >= num_highlights


def number_words(text: str, num_words: int, relation: str, **_: Any) -> bool:
    return _relation_ok(len(_words(text)), relation, num_words)


def number_sentences(text: str, num_sentences: int, relation: str, **_: Any) -> bool:
    return _relation_ok(len(_sentences(text)), relation, num_sentences)


def number_paragraphs(text: str, num_paragraphs: int, **_: Any) -> bool:
    paras = re.split(r"\s?\*\*\*\s?", text)
    n = len(paras)
    for i, p in enumerate(paras):
        if not p.strip():
            if i in (0, len(paras) - 1):
                n -= 1
            else:
                return False
    return n == num_paragraphs


def nth_paragraph_first_word(
    text: str, num_paragraphs: int, nth_paragraph: int, first_word: str, **_: Any
) -> bool:
    paras = [p for p in re.split(r"\n\n", text) if p.strip()]
    if len(paras) != num_paragraphs or nth_paragraph > len(paras):
        return False
    para = paras[nth_paragraph - 1].strip()
    word = para.split()[0] if para.split() else ""
    word = re.sub(r"^[^\w]+|[^\w]+$", "", word).lower()
    return word == first_word.lower()


def number_bullet_lists(text: str, num_bullets: int, **_: Any) -> bool:
    bullets = re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE)
    bullets += re.findall(r"^\s*-.*$", text, flags=re.MULTILINE)
    return len(bullets) == num_bullets


def title(text: str, **_: Any) -> bool:
    return any(t.strip() for t in re.findall(r"<<([^\n]+)>>", text))


def json_format(text: str, **_: Any) -> bool:
    value = text.strip()
    value = re.sub(r"^```(?:json|Json|JSON)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value).strip()
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def multiple_sections(text: str, section_spliter: str, num_sections: int, **_: Any) -> bool:
    sections = re.split(r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?", text)
    return len(sections) - 1 >= num_sections


def constrained_response(text: str, **_: Any) -> bool:
    return any(opt in text for opt in ("My answer is yes.", "My answer is no.", "My answer is maybe."))


def english_lowercase(text: str, **_: Any) -> bool:
    return text == text.lower()


def english_capital(text: str, **_: Any) -> bool:
    return text == text.upper()


def capital_word_frequency(text: str, capital_frequency: int, capital_relation: str, **_: Any) -> bool:
    caps = [w for w in _words(text) if w.isupper()]
    return _relation_ok(len(caps), capital_relation, capital_frequency)


def keyword_existence(text: str, keywords: list[str], **_: Any) -> bool:
    return all(re.search(re.escape(k), text, flags=re.IGNORECASE) for k in keywords)


def forbidden_words(text: str, forbidden_words: list[str], **_: Any) -> bool:
    return not any(re.search(r"\b" + re.escape(w) + r"\b", text, flags=re.IGNORECASE)
                   for w in forbidden_words)


def keyword_frequency(text: str, keyword: str, frequency: int, relation: str, **_: Any) -> bool:
    n = len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))
    return _relation_ok(n, relation, frequency)


def letter_frequency(text: str, letter: str, let_frequency: int, let_relation: str, **_: Any) -> bool:
    return _relation_ok(text.lower().count(letter.lower()), let_relation, let_frequency)


def postscript(text: str, postscript_marker: str, **_: Any) -> bool:
    marker = postscript_marker.strip()
    if marker.lower() == "p.p.s":
        pat = r"\s*p\.\s?p\.\s?s.*$"
    elif marker.lower() == "p.s.":
        pat = r"\s*p\.\s?s\..*$"
    else:
        pat = r"\s*" + re.escape(marker.lower()) + r".*$"
    return bool(re.findall(pat, text.lower(), flags=re.MULTILINE))


def number_placeholders(text: str, num_placeholders: int, **_: Any) -> bool:
    return len(re.findall(r"\[.*?\]", text)) >= num_placeholders


def end_checker(text: str, end_phrase: str, **_: Any) -> bool:
    return text.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def quotation(text: str, **_: Any) -> bool:
    t = text.strip()
    return len(t) > 1 and t[0] == '"' and t[-1] == '"'


def repeat_prompt(text: str, prompt_to_repeat: str, **_: Any) -> bool:
    return text.strip().lower().startswith(prompt_to_repeat.strip().lower())


def two_responses(text: str, **_: Any) -> bool:
    parts = text.split("******")
    valid = []
    for i, part in enumerate(parts):
        if not part.strip():
            if i not in (0, len(parts) - 1):
                return False
        else:
            valid.append(part.strip())
    return len(valid) == 2 and valid[0] != valid[1]


CHECKERS: dict[str, Callable[..., bool]] = {
    "punctuation:no_comma": no_comma,
    "detectable_format:number_highlighted_sections": number_highlighted_sections,
    "length_constraints:number_words": number_words,
    "length_constraints:number_sentences": number_sentences,
    "length_constraints:number_paragraphs": number_paragraphs,
    "length_constraints:nth_paragraph_first_word": nth_paragraph_first_word,
    "detectable_format:number_bullet_lists": number_bullet_lists,
    "detectable_format:title": title,
    "detectable_format:json_format": json_format,
    "detectable_format:multiple_sections": multiple_sections,
    "detectable_format:constrained_response": constrained_response,
    "change_case:english_lowercase": english_lowercase,
    "change_case:english_capital": english_capital,
    "change_case:capital_word_frequency": capital_word_frequency,
    "keywords:existence": keyword_existence,
    "keywords:forbidden_words": forbidden_words,
    "keywords:frequency": keyword_frequency,
    "keywords:letter_frequency": letter_frequency,
    "detectable_content:postscript": postscript,
    "detectable_content:number_placeholders": number_placeholders,
    "startend:end_checker": end_checker,
    "startend:quotation": quotation,
    "combination:repeat_prompt": repeat_prompt,
    "combination:two_responses": two_responses,
}

SUPPORTED = frozenset(CHECKERS)


def _loose_variants(text: str) -> list[str]:
    lines = text.split("\n")
    base = [
        text,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]
    return base + [v.replace("*", "") for v in base]


def check_one(text: str, instruction_id: str, kwargs: dict[str, Any], prompt: str = "") -> bool:
    fn = CHECKERS[instruction_id]
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None}
    if instruction_id == "combination:repeat_prompt" and "prompt_to_repeat" not in kw:
        kw["prompt_to_repeat"] = prompt
    for variant in _loose_variants(text):
        if variant.strip() and fn(variant, **kw):
            return True
    return False


def check_all(
    text: str, instruction_ids: list[str], kwargs_list: list[dict[str, Any]], prompt: str = ""
) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for iid, kw in zip(instruction_ids, kwargs_list, strict=True):
        out[iid] = out.get(iid, True) and check_one(text, iid, kw, prompt)
    return out
