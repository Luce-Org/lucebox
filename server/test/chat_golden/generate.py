"""Render cases.json with each model's official chat template; write expected/<model>.json.

The reference for a Qwen model is what its own chat template renders, prepared the way vLLM
prepares an OpenAI Chat Completions request for it: tool-call arguments decoded to objects,
`reasoning` read as `reasoning_content`, the request's `chat_template_kwargs` and
`reasoning_effort` passed to the template. The template runs in transformers' sandboxed Jinja.

DeepSeek V4.1 has no Jinja template: its reference is a Python encoder (encoding/encoding.py in
the model repo). This script does not run it; it takes the encoder's own golden pairs
(encoding/tests/test_input_N.json, test_output_N.txt) as data instead.

    uv run --project server python server/test/chat_golden/generate.py

Needs the Hugging Face Hub (or its cache). test/unit/test_server_unit.cpp (test_chat_golden_*) reads the output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

HERE = Path(__file__).resolve().parent

# Model name (as in share/model_cards) -> where its reference lives, pinned to a Hub revision.
MODELS: dict[str, dict[str, str]] = {
    "qwen3.8-27b": {
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "kind": "jinja",
    },
    # Flash-Next ships the same chat template and tokenizer as the 27B.
    "qwen3.8-flash-next": {
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "kind": "jinja",
    },
    "deepseek-v4.1-flash": {
        "repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "kind": "deepseek-v41-fixtures",
    },
}


def load_cases() -> list[dict[str, Any]]:
    corpus = json.loads((HERE / "cases.json").read_text())
    cases = []
    for case in corpus["cases"]:
        case = copy.deepcopy(case)
        if case["body"].get("tools") == "$tools":
            case["body"]["tools"] = copy.deepcopy(corpus["tools"])
        cases.append(case)
    return cases


def vllm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation as vLLM hands it to a chat template."""
    out = []
    for message in copy.deepcopy(messages):
        if "reasoning" in message and "reasoning_content" not in message:
            message["reasoning_content"] = message.pop("reasoning")
        if message.get("content") is None:
            message["content"] = ""
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments) if arguments else {}
        out.append(message)
    return out


def download(spec: dict[str, str], filename: str) -> Path:
    return Path(hf_hub_download(spec["repo"], filename, revision=spec["revision"]))


class JinjaReference:
    def __init__(self, spec: dict[str, str]):
        config = json.loads(download(spec, "tokenizer_config.json").read_text())
        self.template = config["chat_template"]
        tokenizer_file = str(download(spec, "tokenizer.json"))
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file, chat_template=self.template
        )
        self.raw = Tokenizer.from_file(tokenizer_file)

    def render(self, body: dict[str, Any]) -> str:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if "reasoning_effort" in body:
            kwargs["reasoning_effort"] = body["reasoning_effort"]
        return self.tokenizer.apply_chat_template(
            vllm_messages(body["messages"]),
            tools=body.get("tools"),
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )

    def cases(self, cases: list[dict[str, Any]], model: str) -> dict[str, Any]:
        out = {}
        for case in cases:
            if model not in case.get("models", MODELS):
                continue
            text = self.render(case["body"])
            tokens = self.raw.encode(text, add_special_tokens=False).ids
            out[case["id"]] = {"text": text, "tokens": tokens}
        return out

    def source_digest(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()


class DeepseekV41Fixtures:
    def __init__(self, spec: dict[str, str]):
        self.spec = spec
        self.raw = Tokenizer.from_file(str(download(spec, "tokenizer.json")))
        self.readme = download(spec, "encoding/README.md").read_bytes()

    def cases(self, _cases: list[dict[str, Any]], _model: str) -> dict[str, Any]:
        out = {}
        n = 1
        while True:
            try:
                given = download(self.spec, f"encoding/tests/test_input_{n}.json")
                expected = download(self.spec, f"encoding/tests/test_output_{n}.txt")
            except EntryNotFoundError as error:
                # The numbered pairs end at the first one the Hub does not have. Not in the
                # cache, offline, is not that: it would drop the cases after it.
                if isinstance(error, LocalEntryNotFoundError):
                    raise
                break
            text = expected.read_text()
            out[f"official_{n}"] = {
                "input": json.loads(given.read_text()),
                "text": text,
                "tokens": self.raw.encode(text, add_special_tokens=False).ids,
            }
            n += 1
        return out

    def source_digest(self) -> str:
        return hashlib.sha256(self.readme).hexdigest()


def dump_expected(out: dict[str, Any]) -> str:
    """expected/<model>.json with each case on one line, so a file stays a few hundred lines."""
    fields = []
    for key, value in out.items():
        if key == "cases":
            rows = ",\n".join(
                f"  {json.dumps(name, ensure_ascii=False)}: {json.dumps(case, ensure_ascii=False)}"
                for name, case in value.items()
            )
            body = "{\n" + rows + "\n }"
        else:
            body = json.dumps(value, ensure_ascii=False, indent=1).replace("\n", "\n ")
        fields.append(f" {json.dumps(key)}: {body}")
    return "{\n" + ",\n".join(fields) + "\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    args = parser.parse_args()

    cases = load_cases()
    (HERE / "expected").mkdir(exist_ok=True)
    for name in args.models:
        spec = MODELS[name]
        reference: Any = (
            JinjaReference(spec) if spec["kind"] == "jinja" else DeepseekV41Fixtures(spec)
        )
        out = {
            "model": name,
            "reference": {
                "repo": spec["repo"],
                # hf_hub_download resolves the revision into the snapshot directory name.
                "revision": download(spec, "tokenizer.json").parent.name,
                "kind": spec["kind"],
                "source_sha256": reference.source_digest(),
            },
            "cases": reference.cases(cases, name),
        }
        if spec["kind"] == "jinja":
            # The server's Jinja path (Flash-Next) renders with this source.
            out["template"] = reference.template
        path = HERE / "expected" / f"{name}.json"
        path.write_text(dump_expected(out))
        print(f"{name}: {len(out['cases'])} cases -> {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
