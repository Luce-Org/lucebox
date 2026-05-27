"""Generate RULER-style multi-hop arithmetic chain test cases at any context size.

Fast path: uses ~4 chars/token proxy; tokenizer called at most 3 times per case.
"""
import argparse, json, os, random, sys, time
from transformers import AutoTokenizer

FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")

OPERATIONS = [
    ("plus", lambda a, b: a + b),
    ("minus", lambda a, b: a - b),
    ("multiplied by", lambda a, b: a * b),
]

CHARS_PER_TOKEN = 4  # proxy; English BPE tokenizers cluster around 3.7-4.2


def _positive_int(s):
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {s!r}")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {v}")
    return v


def _build_chain(seed: int, hops: int):
    """Build a K-link arithmetic chain. Returns links, final value, question, var_names."""
    rng = random.Random(seed)
    suffix = rng.randint(10, 99)
    letters = "XYZWVUTSRQPONMLKJIH"
    var_names = [f"{letters[i]}{suffix}" for i in range(hops + 1)]

    base_val = rng.randint(2, 9)
    values = [base_val]
    links = [(var_names[0], f"The value of {var_names[0]} is {base_val}.")]

    for i in range(1, hops):
        op_name, op_fn = OPERATIONS[rng.randint(0, len(OPERATIONS) - 1)]
        operand = rng.randint(2, 5)
        new_val = op_fn(values[-1], operand)
        if new_val <= 0:
            # fallback: switch to plus so result is always positive
            op_name, op_fn = "plus", lambda a, b: a + b
            new_val = op_fn(values[-1], operand)
        values.append(new_val)
        links.append((var_names[i], f"Let {var_names[i]} equal {var_names[i-1]} {op_name} {operand}."))

    op_name, op_fn = OPERATIONS[rng.randint(0, 1)]
    operand = rng.randint(2, 5)
    final_val = op_fn(values[-1], operand)
    if final_val <= 0:
        op_name, op_fn = "plus", lambda a, b: a + b
        final_val = op_fn(values[-1], operand)
    values.append(final_val)
    links.append((var_names[hops], f"Let {var_names[hops]} equal {var_names[hops-1]} {op_name} {operand}."))

    question = f"What is the value of {var_names[hops]}? Reply with just the integer."
    return links, final_val, question, var_names


def _build_prompt(link_sentences: list, fracs: list, target_chars: int, question: str) -> str:
    """Assemble filler + spliced needle sentences into the full prompt string."""
    target_chars = max(0, target_chars)
    filler_total = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
    positions = [int(target_chars * f) for f in fracs]
    text = filler_total
    for pos, sent in zip(sorted(positions, reverse=True), reversed(link_sentences)):
        pos = min(pos, len(text))
        text = text[:pos] + " " + sent + " " + text[pos:]
    return (
        "Below is a long passage. Answer the question at the end based ONLY on information in the passage.\n\n"
        f"{text}\n\nQuestion: {question}\nAnswer:"
    )


def gen_one(seed: int, target_tokens: int, hops: int, tokenizer):
    """Generate one multi-hop case. Tokenizer called at most 3 times (trim loop).

    Tolerance: ±5% of target_tokens. Always <= target_tokens on return.
    """
    rng = random.Random(seed)
    links, final_val, question, var_names = _build_chain(seed, hops)
    link_sentences = [s for _, s in links]

    fracs = sorted([
        rng.uniform(0.05 + i * 0.8 / hops, 0.05 + (i + 1) * 0.8 / hops)
        for i in range(len(links))
    ])

    # Step 1: build at char-proxy size (no tokenizer call)
    target_chars = target_tokens * CHARS_PER_TOKEN
    prompt = _build_prompt(link_sentences, fracs, target_chars, question)

    # Step 2: single tokenizer call to measure actual tokens
    actual = len(tokenizer.encode(prompt))

    # Step 3: trim if over (max 3 iterations, each trims 1000 chars from filler region)
    for _ in range(3):
        if actual <= target_tokens:
            break
        overshoot_tokens = actual - target_tokens
        trim_chars = max(1000, overshoot_tokens * CHARS_PER_TOKEN)
        target_chars = max(0, target_chars - trim_chars)
        prompt = _build_prompt(link_sentences, fracs, target_chars, question)
        actual = len(tokenizer.encode(prompt))

    if actual > target_tokens:
        raise ValueError(
            f"target_tokens={target_tokens} still exceeded after trim "
            f"(actual={actual}). Scaffold floor too large."
        )

    return {
        "prompt": prompt,
        "answer": final_val,
        "hops": hops,
        "n_tokens": actual,
        "var_names": var_names,
        "chain": link_sentences,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--context", type=_positive_int, default=8192)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    tmp_path = f"{args.out}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            for i in range(args.n):
                t0 = time.monotonic()
                try:
                    ex = gen_one(
                        seed=args.seed_base + i,
                        target_tokens=args.context,
                        hops=args.hops,
                        tokenizer=tok,
                    )
                except ValueError as e:
                    sys.exit(f"[error] case {i}: {e}")
                assert ex["n_tokens"] <= args.context, (
                    f"case {i}: n_tokens={ex['n_tokens']} exceeds --context={args.context}")
                f.write(json.dumps(ex) + "\n")
                elapsed = time.monotonic() - t0
                print(f"  case {i}: ntok={ex['n_tokens']} ans={ex['answer']} "
                      f"chain={ex['chain']} ({elapsed:.2f}s)")
        os.replace(tmp_path, args.out)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    print(f"saved {args.n} cases to {args.out}")


if __name__ == "__main__":
    main()
