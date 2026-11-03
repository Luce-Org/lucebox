"""Render cases.json with each model's official chat template; write expected/<model>.json.

The reference for a Qwen model is what its own chat template renders, prepared the way vLLM
prepares an OpenAI Chat Completions request for it: tool-call arguments decoded to objects,
`reasoning` read as `reasoning_content`, the request's `chat_template_kwargs` and
`reasoning_effort` passed to the template. The template runs in transformers' sandboxed Jinja.

DeepSeek V4 and V4.1 have no Jinja template: their reference is DeepSeek's Python encoder
(encoding/encoding_dsv4.py and encoding/encoding.py in the model repos, which SGLang also serves
them with, through deepseek-recipe, and vLLM with adapted copies). Each case is prepared the way
vLLM's V4 or V4.1 tokenizer prepares a request (tools on the system message, thinking on unless
the request turns it off, effort "high" by default) and encoded by the encoder at the pinned
revision, loaded from its file. The encoder's own golden pairs (encoding/tests/test_input_N.json,
test_output_N.txt) are kept beside the cases as official_N.

    uv run --project server python server/test/chat_golden/generate.py

Needs the Hugging Face Hub (or its cache). test/unit/test_server_unit.cpp (test_chat_golden_*) reads the output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
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
    "qwen3.6-27b": {
        "repo": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "kind": "jinja",
    },
    # The 35B-A3B ships the same chat template.
    "qwen3.5-27b": {
        "repo": "Qwen/Qwen3.5-27B",
        "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "kind": "jinja",
    },
    "deepseek-v4-flash": {
        "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        "kind": "deepseek-v4-encoder",
    },
    "deepseek-v4.1-flash": {
        "repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "kind": "deepseek-v41-encoder",
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


def deepseek_v41_request(body: dict[str, Any]) -> dict[str, Any]:
    """encode_messages' arguments for a Chat Completions request, as vLLM's V4.1 tokenizer
    builds them (vllm/tokenizers/deepseek_v41.py)."""
    conversation = []
    for message in copy.deepcopy(body["messages"]):
        if "reasoning" in message:
            message["reasoning_content"] = message.pop("reasoning")
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = "\n\n".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        conversation.append(message)
    if body.get("tools"):
        system = next((m for m in conversation if m["role"] == "system"), None)
        if system is None:
            system = {"role": "system", "content": ""}
            conversation.insert(0, system)
        system["tools"] = copy.deepcopy(body["tools"])
    kwargs = body.get("chat_template_kwargs") or {}
    thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
    if "thinking" not in kwargs and "enable_thinking" not in kwargs:
        thinking = True
    effort = body.get("reasoning_effort", kwargs.get("reasoning_effort"))
    if effort == "none":
        thinking, effort = False, None
    return {
        "messages": conversation,
        "thinking_mode": "thinking" if thinking else "chat",
        "drop_thinking": kwargs.get("drop_thinking", True),
        "reasoning_effort": "high" if effort is None else effort,
    }


def deepseek_v4_request(body: dict[str, Any]) -> dict[str, Any]:
    """encode_messages' arguments for a Chat Completions request, as vLLM's V4 tokenizer builds
    them (vllm/tokenizers/deepseek_v4.py), from the conversation its generic renderer hands it
    (text parts joined with '\n')."""
    conversation = []
    for message in copy.deepcopy(body["messages"]):
        if "reasoning" in message:
            message["reasoning_content"] = message.pop("reasoning")
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = "\n".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        conversation.append(message)
    if body.get("tools"):
        system = next((m for m in conversation if m["role"] == "system"), None)
        if system is None:
            system = {"role": "system"}
            conversation.insert(0, system)
        system["tools"] = copy.deepcopy(body["tools"])
    kwargs = body.get("chat_template_kwargs") or {}
    thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
    if "thinking" not in kwargs and "enable_thinking" not in kwargs:
        thinking = True
    mode = "thinking" if thinking else "chat"
    effort = body.get("reasoning_effort", kwargs.get("reasoning_effort"))
    if not isinstance(effort, str):
        effort = "high" if thinking else None
    elif effort == "none":
        mode, effort = "chat", None
    elif effort in ("low", "minimal", "medium"):
        effort = "low"
    elif effort != "max":
        effort = "high"
    return {
        "messages": conversation,
        "thinking_mode": mode,
        "drop_thinking": kwargs.get("drop_thinking", True),
        "reasoning_effort": effort,
    }


class DeepseekReference:
    def __init__(self, spec: dict[str, str]):
        self.spec = spec
        v41 = spec["kind"] == "deepseek-v41-encoder"
        self.prepare = deepseek_v41_request if v41 else deepseek_v4_request
        self.raw = Tokenizer.from_file(str(download(spec, "tokenizer.json")))
        self.source = download(spec, "encoding/encoding.py" if v41 else "encoding/encoding_dsv4.py")
        # Imported from its file: nothing from the download's directory joins sys.path.
        module_spec = importlib.util.spec_from_file_location("deepseek_encoding", self.source)
        self.encoding = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.encoding)

    def cases(self, cases: list[dict[str, Any]], model: str) -> dict[str, Any]:
        out = {}
        for case in cases:
            if model not in case.get("models", MODELS):
                continue
            text = self.encoding.encode_messages(**self.prepare(case["body"]))
            tokens = self.raw.encode(text, add_special_tokens=False).ids
            out[case["id"]] = {"text": text, "tokens": tokens}
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
            given = json.loads(given.read_text())
            if self.spec["kind"] == "deepseek-v4-encoder":
                # As test_encoding_dsv4.py encodes them: thinking but for the 4th, at the encoder's
                # default effort (low: no effort prompt), case 1's tools on its first message.
                raw = given
                given = {
                    "messages": raw["messages"] if isinstance(raw, dict) else raw,
                    "thinking_mode": "chat" if n == 4 else "thinking",
                    "reasoning_effort": "low",
                }
                if isinstance(raw, dict) and "tools" in raw:
                    given["tools"] = raw["tools"]
            out[f"official_{n}"] = {
                "input": given,
                "text": text,
                "tokens": self.raw.encode(text, add_special_tokens=False).ids,
            }
            n += 1
        return out

    def source_digest(self) -> str:
        return hashlib.sha256(self.source.read_bytes()).hexdigest()


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
            JinjaReference(spec) if spec["kind"] == "jinja" else DeepseekReference(spec)
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
