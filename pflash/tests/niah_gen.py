"""Generate NIAH single-needle test cases at any context size."""
import argparse, json, os, random, sys
from transformers import AutoTokenizer


def _positive_int(s):
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {s!r}")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {v}")
    return v

FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
NEEDLE_TMPL = "The special magic {key} number is: {value}."
QUESTION_TMPL = "What is the special magic {key} number? Answer in one short sentence."


def gen_one(seed: int, target_tokens: int, tokenizer, tolerance: float = 0.005):
    """Generate a NIAH case whose tokenized length is within `tolerance` of
    `target_tokens` *and never exceeds it*, regardless of which tokenizer is
    passed in.

    Strategy: binary-search the chars-per-token ratio against the actual
    tokenizer until we land within tolerance, then hard-trim chars off the
    filler if we still overshoot by even one token. Token counts include
    special tokens (BOS etc) so the cap matches what downstream consumers
    see when they tokenize the prompt with default settings.

    Raises ValueError if `target_tokens` is below the scaffold floor (the
    fixed intro + needle + question alone exceed the budget).
    """
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    rng = random.Random(seed)
    key = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8))
    value = "".join(rng.choices("0123456789", k=7))
    needle = NEEDLE_TMPL.format(key=key, value=value)
    question = QUESTION_TMPL.format(key=key)
    insert_frac = rng.uniform(0.25, 0.75)  # capture once for determinism

    def build(target_chars: int) -> str:
        target_chars = max(0, target_chars)
        filler = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
        insert = int(target_chars * insert_frac)
        body = filler[:insert] + " " + needle + " " + filler[insert:]
        return (
            "Below is a long passage. Answer the question at the end based ONLY on information in the passage.\n\n"
            f"{body}\n\nQuestion: {question}\nAnswer:"
        )

    def n_tokens(prompt: str) -> int:
        # Default add_special_tokens=True matches what downstream consumers
        # (bench_niah_cpp.py via tokenizer(prompt)) actually see, so the cap
        # below correctly accounts for any BOS/EOS the tokenizer prepends.
        return len(tokenizer.encode(prompt))

    # Binary-search chars-per-token. Bounds [2.0, 6.0] cover every common
    # tokenizer (Qwen ~3.7, Llama/Mistral ~4.0, GPT-2 BPE ~4.2, byte-level ~5+).
    lo, hi = 2.0, 6.0
    target_chars = int(target_tokens * (lo + hi) / 2)
    prompt = build(target_chars)
    actual = n_tokens(prompt)
    for _ in range(20):
        if abs(actual - target_tokens) / target_tokens < tolerance:
            break
        if actual > target_tokens:
            hi = (lo + hi) / 2
        else:
            lo = (lo + hi) / 2
        target_chars = int(target_tokens * (lo + hi) / 2)
        prompt = build(target_chars)
        actual = n_tokens(prompt)

    # Hard-trim: shrink filler until actual <= target_tokens. Step by larger
    # chunks first to keep the worst case bounded (~few re-tokenizations).
    for step in (256, 64, 16, 1):
        while actual > target_tokens and target_chars >= step:
            target_chars -= step
            prompt = build(target_chars)
            actual = n_tokens(prompt)

    if actual > target_tokens:
        # filler at zero and the fixed scaffold (intro + needle + question +
        # any specials) alone exceeds target_tokens. Surface this clearly
        # rather than returning an over-limit prompt.
        raise ValueError(
            f"target_tokens={target_tokens} is below the scaffold floor "
            f"(empty filler + needle + question + specials = {actual} tokens). "
            f"Pick a larger target."
        )

    return {"prompt": prompt, "answer": value, "key": key, "n_tokens": actual}


def main():
    ap = argparse.ArgumentParser()
    # --n stays type=int so --n 0 keeps producing an empty JSONL (preserves
    # prior CLI behavior). --ctx is positive-only because the convergence
    # loop divides by it.
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ctx", type=_positive_int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, default=42,
                    help="Base seed for case generation (default 42). "
                         "Vary this to produce different needle key/value "
                         "pairs without changing --n.")
    # Default matches bench_niah_cpp.py's --drafter-tokenizer, since that is
    # the tokenizer the downstream NIAH bench uses to size case["prompt"]
    # for the drafter forward. Override for any other harness.
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Write to a sibling temp file and atomically rename on success. Avoids
    # truncating an existing args.out when gen_one fails mid-run (e.g. on
    # an impossibly small --ctx for case 0).
    tmp_path = f"{args.out}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            for i in range(args.n):
                try:
                    ex = gen_one(seed=args.seed_base + i, target_tokens=args.ctx, tokenizer=tok)
                except ValueError as e:
                    sys.exit(f"[error] case {i}: {e}")
                assert ex["n_tokens"] <= args.ctx, (
                    f"case {i}: n_tokens={ex['n_tokens']} exceeds --ctx={args.ctx}")
                f.write(json.dumps(ex) + "\n")
                print(f"  case {i}: ntok={ex['n_tokens']} key={ex['key']} ans={ex['answer']}")
        os.replace(tmp_path, args.out)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    print(f"saved {args.n} cases to {args.out}")


if __name__ == "__main__":
    main()
