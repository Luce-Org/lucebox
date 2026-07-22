"""Build sealed tuning and held-out decode corpora from pinned public sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .records import canonical_json

TUNING_SOURCE = {
    "url": (
        "https://raw.githubusercontent.com/thc1006/qwen3.6-speculative-decoding-v100/"
        "e9894887828fc7b9c15d7af170b4da36ce0758fa/prompts/prompts40.json"
    ),
    "sha256": "61d04afa8243989ee7ec951973fd9143f7818bc890fc6fee1fe3d307467bb197",
    "revision": "e9894887828fc7b9c15d7af170b4da36ce0758fa",
    "license": "CC0-1.0",
    "source": "thc1006/qwen3.6-speculative-decoding-v100",
}

VALIDATION_SOURCE = {
    "url": (
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
        "761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json"
    ),
    "sha256": "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419",
    "revision": "761dc5bfbdeeffa89b8bff5d038781a4055f796a",
    "license": "CC-BY-NC-4.0",
    "source": "tatsu-lab/stanford_alpaca",
}

FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("code", r"\b(code|program|function|algorithm|python|javascript|java|sql)\b"),
    ("math", r"\b(calculate|solve|equation|mathematical|arithmetic|probability|percentage)\b"),
    ("summarization", r"\b(summarize|summary|summarise)\b"),
    ("classification", r"\b(classify|classification|categorize|category|label)\b"),
    ("extraction", r"\b(extract|identify|find)\b.*\b(from|in the)\b"),
    ("rewriting", r"\b(rewrite|rephrase|paraphrase|edit|correct the grammar)\b"),
    ("brainstorming", r"\b(brainstorm|ideas|suggest|recommend)\b"),
    ("question-answering", r"^(what|who|when|where|why|how)\b|answer the following question"),
    ("reasoning", r"\b(reason|logic|infer|analyze|analyse|compare|evaluate)\b"),
    ("creative", r"\b(story|poem|creative|dialogue|slogan|song|fiction)\b"),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch_pinned(source: Mapping[str, str], cache: Path) -> bytes:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{source['revision']}-{Path(source['url']).name}"
    if target.is_file():
        value = target.read_bytes()
    else:
        with urllib.request.urlopen(source["url"], timeout=120) as response:  # noqa: S310
            value = response.read()
        target.write_bytes(value)
    actual = sha256_bytes(value)
    if actual != source["sha256"]:
        raise ValueError(
            f"pinned corpus hash mismatch for {source['source']}: "
            f"expected {source['sha256']}, got {actual}"
        )
    return value


def _record(*, case_id: str, family: str, prompt: str, source: Mapping[str, str]) -> dict[str, str]:
    return {
        "id": case_id,
        "family": family,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "source": source["source"],
        "source_revision": source["revision"],
        "source_license": source["license"],
    }


def build_tuning40(raw: bytes) -> list[dict[str, str]]:
    source_rows = json.loads(raw)
    if not isinstance(source_rows, list) or len(source_rows) != 40:
        raise ValueError("pinned tuning source must contain exactly 40 prompts")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(source_rows):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"invalid tuning prompt at index {index}")
        family, prompt = map(str, item)
        rows.append(
            _record(
                case_id=f"tuning-{index:03d}", family=family, prompt=prompt, source=TUNING_SOURCE
            )
        )
    return rows


def _render_alpaca(item: Mapping[str, Any]) -> str:
    instruction = str(item.get("instruction", "")).strip()
    context = str(item.get("input", "")).strip()
    if not instruction:
        raise ValueError("Alpaca row has an empty instruction")
    if context:
        return f"{instruction}\n\nInput:\n{context}"
    return instruction


def build_validation100(raw: bytes) -> list[dict[str, str]]:
    source_rows = json.loads(raw)
    if not isinstance(source_rows, list):
        raise ValueError("validation source must be a JSON list")
    used: set[int] = set()
    rows: list[dict[str, str]] = []
    for family, pattern in FAMILY_PATTERNS:
        selected = 0
        compiled = re.compile(pattern)
        for index, item in enumerate(source_rows):
            if index in used or not isinstance(item, Mapping):
                continue
            instruction = str(item.get("instruction", "")).lower()
            if not compiled.search(instruction):
                continue
            prompt = _render_alpaca(item)
            rows.append(
                _record(
                    case_id=f"validation-{family}-{selected:02d}",
                    family=family,
                    prompt=prompt,
                    source=VALIDATION_SOURCE,
                )
            )
            used.add(index)
            selected += 1
            if selected == 10:
                break
        if selected != 10:
            raise ValueError(f"validation source supplied only {selected}/10 prompts for {family}")
    if len(rows) != 100 or len({row["prompt_sha256"] for row in rows}) != 100:
        raise ValueError("validation corpus is not 100 unique prompts")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = canonical_json(row) + "\n"
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


def validate_frozen_hashes(result: Mapping[str, Mapping[str, str]], protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = protocol.get("decode_corpora")
    if not isinstance(frozen, Mapping):
        raise ValueError("protocol lacks decode_corpora hashes")
    expected = {
        "tuning40": frozen.get("tuning40_sha256"),
        "validation100": frozen.get("validation100_sha256"),
    }
    mismatches = [
        name
        for name, digest in expected.items()
        if not isinstance(digest, str) or result[name]["sha256"] != digest
    ]
    if mismatches:
        raise ValueError(f"generated corpora differ from frozen protocol: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".validation-v2/cache"))
    parser.add_argument("--out-dir", type=Path, default=Path(".validation-v2/corpora"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.json"),
    )
    args = parser.parse_args()
    tuning = build_tuning40(fetch_pinned(TUNING_SOURCE, args.cache))
    validation = build_validation100(fetch_pinned(VALIDATION_SOURCE, args.cache))
    result = {
        "tuning40": {
            "path": str(args.out_dir / "tuning40.jsonl"),
            "sha256": write_jsonl(args.out_dir / "tuning40.jsonl", tuning),
        },
        "validation100": {
            "path": str(args.out_dir / "validation100.jsonl"),
            "sha256": write_jsonl(args.out_dir / "validation100.jsonl", validation),
        },
    }
    validate_frozen_hashes(result, args.protocol)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
