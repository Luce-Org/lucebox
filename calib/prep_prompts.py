#!/usr/bin/env python3
"""Tokenize calibration prompts to int32 .bin files for test_dflash.

Pulls prompts from the same datasets the benchmark uses (HumanEval / GSM8K /
MATH-500), applies the Qwen3.6 chat template, and writes one packed-int32 file
per prompt into the output dir. Each file is a prompt test_dflash can decode.

  python3 calib/prep_prompts.py --n 200 --out calib --thinking off

Deps: transformers, datasets, jinja2 (CPU only). Network access to HF for the
tokenizer + datasets (set HF_TOKEN to avoid rate limits).
"""
import argparse, struct, sys
from pathlib import Path

DATASETS = [
    # (label, hf_id, config, split, field-extractor)
    ("he",   "openai/openai_humaneval", None,   "test", lambda x: x["prompt"]),
    ("gsm",  "openai/gsm8k",            "main", "test", lambda x: f"Question: {x['question']}\nAnswer: "),
    ("math", "HuggingFaceH4/MATH-500",  None,   "test", lambda x: f"Problem: {x['problem']}\nSolution: "),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="total prompts (split across datasets)")
    ap.add_argument("--out", default="calib", help="output dir for .bin files")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-prompt-tok", type=int, default=1024,
                    help="skip prompts longer than this (keeps verify ctx small)")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    per = max(1, args.n // len(DATASETS))
    written = 0
    for label, hf_id, cfg, split, extract in DATASETS:
        ds = load_dataset(hf_id, cfg, split=split).shuffle(seed=args.seed)
        take = min(per, len(ds))
        for i in range(take):
            raw = extract(ds[i])
            text = tok.apply_chat_template(
                [{"role": "user", "content": raw}], tokenize=False,
                add_generation_prompt=True, enable_thinking=(args.thinking == "on"))
            ids = tok.encode(text, add_special_tokens=False)
            if not ids or len(ids) > args.max_prompt_tok:
                continue
            with open(out / f"{label}_{i:04d}.bin", "wb") as f:
                for t in ids:
                    f.write(struct.pack("<i", int(t)))
            written += 1
    print(f"wrote {written} prompt bins to {out.resolve()}", file=sys.stderr)

if __name__ == "__main__":
    main()
