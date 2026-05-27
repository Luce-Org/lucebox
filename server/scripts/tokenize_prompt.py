"""
Tokenize a prompt string using the Qwen3.5 HF tokenizer (via transformers)
and emit the token IDs as a flat int32 binary file.

We depend on Python only for the tokenizer — the C++ library consumes the
int32 file directly. This keeps the standalone lib free of a BPE impl.

Usage:
    python tokenize_prompt.py --out /tmp/prompt.bin --prompt "The capital of France is"
"""

import argparse
import os
import sys
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B",
                    help="HF repo id whose tokenizer to use")
    ap.add_argument("--add-bos", action="store_true", help="Prepend BOS token")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ids = tok.encode(args.prompt, add_special_tokens=args.add_bos)
    print(f"tokenized {len(ids)} tokens: {ids}")

    with open(args.out, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))

    print(f"wrote {args.out} ({len(ids) * 4} bytes)")


if __name__ == "__main__":
    main()
