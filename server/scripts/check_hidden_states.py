#!/usr/bin/env python3
"""Compare luce_server's /v1/hidden_states against Hugging Face transformers.

The server reports the residual stream after block L (before the final norm)
at the last prompt token and, with pooling=both, its mean over the prompt.
This script renders the same chat with the HF tokenizer, runs the HF model,
reads each requested decoder block's output through a forward hook (HF's
output_hidden_states puts block L at hidden_states[L+1], but its last entry
already has the final norm applied, so hooks are used for every layer), and
prints the cosine similarity per layer. A correct server gives > 0.99 (the
GGUF is quantized, so values are not bit-identical).

Needs the base HF weights of the served GGUF, for example:
    hf download Qwen/Qwen3.5-2B --local-dir ~/models/hf/Qwen3.5-2B

Example:
    python3 server/scripts/check_hidden_states.py \
        --url http://127.0.0.1:8080 --model qwen3.5-2b \
        --hf ~/models/hf/Qwen3.5-2B --layers 0 11 -1 --bare
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request

ASSISTANT_HEADER = "<|im_start|>assistant\n"

DEFAULT_PROMPTS = [
    "Prove that the square root of 2 is irrational.",
    "Write a haiku about autumn.",
    "def fib(n):\n    # complete this function and explain its complexity",
]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/hidden_states",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else float("nan")


def find_decoder_layers(model, n_hint: int | None):
    import torch

    best = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and name.endswith("layers"):
            if "vision" in name or "visual" in name:
                continue
            if best is None or len(module) > len(best[1]):
                best = (name, module)
    if best is None:
        sys.exit("could not find the decoder layer list in the HF model")
    if n_hint is not None and len(best[1]) != n_hint:
        print(f"warning: HF has {len(best[1])} layers, server reports {n_hint}")
    return best[1]


def load_model(path: str, dtype):
    import transformers

    last: Exception | None = None

    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(path, torch_dtype=dtype)
        except (ValueError, KeyError, OSError) as exc:
            last = exc
    sys.exit(f"could not load {path}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen35-2b", help="server model block name")
    ap.add_argument("--hf", required=True, help="HF checkpoint directory or repo id")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 11, -1])
    ap.add_argument("--prompt", action="append", help="user message (repeatable)")
    ap.add_argument("--bare", action="store_true", help='generation_prompt="bare"')
    ap.add_argument("--thinking", action="store_true", help="enable_thinking=true")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    dtype = getattr(torch, args.dtype)
    hf_path = os.path.expanduser(args.hf)
    tok = AutoTokenizer.from_pretrained(hf_path)
    model = load_model(hf_path, dtype).to(args.device).eval()

    worst = 1.0
    for prompt in args.prompt or DEFAULT_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        resp = post(args.url, {
            "model": args.model,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": args.thinking},
            "layers": args.layers,
            "pooling": "both",
            "generation_prompt": "bare" if args.bare else "template",
        })
        n_layers = resp["n_layers"]

        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.thinking)
        if args.bare:
            text = text[: text.rindex(ASSISTANT_HEADER) + len(ASSISTANT_HEADER)]
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] != resp["prompt_tokens"]:
            print(f"warning: HF prompt has {ids.shape[1]} tokens, server "
                  f"{resp['prompt_tokens']}; templates differ, compare with care")
        hf_last_tok = tok.decode(ids[0, -1:])
        if hf_last_tok != resp["last_token"]:
            print(f"warning: last token HF={hf_last_tok!r} server={resp['last_token']!r}")

        blocks = find_decoder_layers(model, n_layers)
        captured: dict[int, torch.Tensor] = {}
        hooks = []
        for requested in args.layers:
            block = requested % len(blocks)

            def hook(_mod, _inp, out, block=block):
                h = out[0] if isinstance(out, (tuple, list)) else out
                captured[block] = h.detach().float()[0]

            hooks.append(blocks[block].register_forward_hook(hook))
        with torch.no_grad():
            model(input_ids=ids.to(args.device))
        for h in hooks:
            h.remove()

        print(f"\nprompt={prompt[:48]!r} tokens={resp['prompt_tokens']} "
              f"last_token={resp['last_token']!r} prefill_ms={resp['timings']['prefill_ms']:.1f}")
        for requested in args.layers:
            block = requested % len(blocks)
            ref = captured[block]
            srv = resp["layers"][str(requested)]
            c_last = cosine(srv["last"], ref[-1].tolist())
            c_mean = cosine(srv["mean"], ref.mean(dim=0).tolist())
            # A zero or non-finite vector has no cosine (nan); min() would skip it.
            worst = min(worst, *(c if math.isfinite(c) else -math.inf for c in (c_last, c_mean)))
            print(f"  layer {requested:>4} (block {block:>2}): "
                  f"cos(last)={c_last:.5f} cos(mean)={c_mean:.5f}")

    ok = worst > args.threshold
    print(f"\nworst cosine {worst:.5f} -> {'PASS' if ok else 'FAIL'} (threshold {args.threshold})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
