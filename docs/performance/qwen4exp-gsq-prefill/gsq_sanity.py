#!/usr/bin/env python3
"""Greedy short and exact-16366-token GSQ sanity checks."""

import argparse
import json
import sys

import requests

sys.path.insert(0, "/tmp")
from t13_runner import prompt_exact_16366


def request(port, prompt, max_tokens):
    payload = {
        "model": "dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning": {"effort": "none"},
    }
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=1800
    )
    response.raise_for_status()
    return response.json()


parser = argparse.ArgumentParser()
parser.add_argument("port", type=int)
parser.add_argument("output")
args = parser.parse_args()

short = request(args.port, "Complete this factual sentence: The capital of France is", 24)
long = request(args.port, prompt_exact_16366(), 32)
short_text = short["choices"][0]["message"].get("content", "")
long_text = long["choices"][0]["message"].get("content", "")
result = {
    "short": short,
    "long": long,
    "short_pass": "paris" in short_text.lower(),
    "long_pass": "QUINCE-AMBER-7731" in long_text.upper(),
}
with open(args.output, "w") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
print(json.dumps({
    "short_pass": result["short_pass"],
    "long_pass": result["long_pass"],
    "short_text": short_text,
    "long_text": long_text,
    "long_usage": long.get("usage"),
}, indent=2))
if not result["short_pass"] or not result["long_pass"]:
    raise SystemExit(1)
