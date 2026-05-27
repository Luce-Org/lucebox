"""Read int32 token IDs from a file and print them as decoded text."""

import argparse
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--slice", default=None,
                    help="optional 'start:end' (token indices, exclusive end)")
    args = ap.parse_args()

    with open(args.inp, "rb") as f:
        raw = f.read()
    ids = list(struct.unpack(f"<{len(raw) // 4}i", raw))
    print(f"read {len(ids)} tokens")

    if args.slice:
        s, e = args.slice.split(":")
        ids = ids[int(s) if s else None : int(e) if e else None]

    print(f"ids = {ids}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = tok.decode(ids, skip_special_tokens=False)
    print(f"text = {text!r}")


if __name__ == "__main__":
    main()
