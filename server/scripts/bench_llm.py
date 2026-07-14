"""
10 prompts per dataset, AR + DFlash per prompt.

    python3 scripts/bench_llm.py

Paths resolve from the repo root by default. Override with env vars:
    DFLASH_TARGET    path to target Qwen3.6-27B-Q4_K_M.gguf (or 3.5)
    DFLASH_DRAFT     path to DFlash draft GGUF or model.safetensors
    DFLASH_BIN       path to build/test_dflash
    DFLASH_BIN_AR    path to build/test_generate
    DFLASH_TOKENIZER HF tokenizer repo (default Qwen/Qwen3.5-27B; matches run.py)
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# Shared math-scoring helpers (canonical copy in harness/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harness"))
from math_scoring import _extract_boxed, _math_equiv, _normalize_math

ROOT = Path(__file__).resolve().parent.parent
BIN_SUFFIX = ".exe" if os.name == "nt" else ""
TARGET = os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
)
_LOCAL_DRAFT_FILE = ROOT / "models" / "draft" / "dflash-draft-3.6-q4_k_m.gguf"
_LOCAL_DRAFT_ROOT = ROOT / "models" / "draft"
DRAFT = None
TEST_DFLASH = os.environ.get("DFLASH_BIN", str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"))
TEST_GENERATE = os.environ.get("DFLASH_BIN_AR", str(ROOT / "build" / f"test_generate{BIN_SUFFIX}"))
TOKENIZER = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.6-27B")
TMPDIR = Path(tempfile.gettempdir()) / "dflash_bench"
TMPDIR.mkdir(parents=True, exist_ok=True)

N_GEN = 256
BUDGET = 22  # default; overridden by --budget CLI arg
N_SAMPLE = int(os.environ.get("DFLASH_N_SAMPLE", "10"))
# Optional sampler tail for the DFlash run, "temp,top_p,top_k,rep_pen,seed[,freq,pres]".
# When set, exercises the sample_logits chain (and its GPU port, on by default,
# opt out with DFLASH_GPU_SAMPLE=0) instead of the default greedy path. AR
# (test_generate) is greedy-only and ignores this.
SAMP = os.environ.get("DFLASH_SAMP", "").strip()

def _gsm_gold(x):
    """Extract numeric answer after #### from GSM8K answer field."""
    ans = x["answer"]
    idx = ans.rfind("####")
    if idx >= 0:
        return ans[idx + 4:].strip().replace(",", "")
    return ans.strip()


BENCHES = [
    ("HumanEval", "openai/openai_humaneval", None, "test", lambda x: x["prompt"], None, N_GEN),
    ("GSM8K", "openai/gsm8k", "main", "test", lambda x: f"Question: {x['question']}\nAnswer: ", _gsm_gold, 1024),
    ("Math500", "HuggingFaceH4/MATH-500", None, "test", lambda x: f"Problem: {x['problem']}\nSolution: Put your final answer in \\boxed{{}}.\n", lambda x: x["answer"], 2048),
]


def _find_draft_model(root: Path) -> str | None:
    if root.is_file():
        return str(root)
    if not root.is_dir():
        return None
    for pattern in ("dflash-draft-*.gguf", "*.gguf", "model.safetensors"):
        matches = sorted(root.rglob(pattern))
        if matches:
            return str(matches[0])
    return None


def _resolve_draft() -> str:
    env = os.environ.get("DFLASH_DRAFT")
    if env:
        found = _find_draft_model(Path(env))
        if found:
            return found
        raise FileNotFoundError(f"DFLASH_DRAFT does not point to a DFlash draft GGUF or model.safetensors: {env}")

    for candidate in (_LOCAL_DRAFT_FILE, _LOCAL_DRAFT_ROOT):
        found = _find_draft_model(candidate)
        if found:
            return found

    raise FileNotFoundError(
        "DFlash draft GGUF or model.safetensors not found. Expected one of:\n"
        f"  - {_LOCAL_DRAFT_FILE}\n"
        "Download it as documented in the README, or set DFLASH_DRAFT to an explicit file or directory."
    )


def _require_file(path: str, label: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _run_checked(cmd, timeout: int, label: str) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "<no output>").strip()[-2000:]
        raise RuntimeError(f"{label} exited {r.returncode}: {tail}")
    return r


def tokenize(tok, p, path: Path):
    ids = tok.encode(p, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def run_ar(path: Path, n_gen: int = N_GEN):
    out_bin = TMPDIR / "ar_out.bin"
    r = _run_checked(
        [TEST_GENERATE, TARGET, str(path), str(n_gen), str(out_bin)],
        timeout=300,
        label="test_generate",
    )
    m = re.search(r"(\d+\.\d+)\s+tok/s", r.stdout)
    if not m:
        raise RuntimeError(f"test_generate output parse failed: {r.stdout[-1000:]}")
    return float(m.group(1))


def _auto_max_ctx(n_prompt, n_gen: int = N_GEN):
    # Auto-fit attention budget: prompt + gen + small verify pad, aligned to
    # FATTN_KQ_STRIDE=256. Oversizing max_ctx makes attention stride over
    # unused KV and can cost >20× prefill time (32K prompt + --kv-q4 +
    # max_ctx=131072 → 1035s vs 38s at max_ctx=32768). See scripts/run.py.
    pad = 64  # covers q_len=16 + ddtree verify overhead with margin
    return ((n_prompt + n_gen + pad + 255) // 256) * 256


def run_df(path: Path, n_prompt, n_gen: int = N_GEN):
    max_ctx = _auto_max_ctx(n_prompt, n_gen)
    out_bin = TMPDIR / f"df_out.bin"
    cmd = [
        TEST_DFLASH,
        TARGET,
        DRAFT,
        str(path),
        str(n_gen),
        str(out_bin),
        "--fast-rollback",
        "--ddtree",
        f"--ddtree-budget={BUDGET}",
        f"--max-ctx={max_ctx}",
    ]
    if SAMP:
        cmd.append(f"--samp={SAMP}")
    r = _run_checked(cmd, timeout=300, label="test_dflash")
    tps = re.search(r"(\d+(?:\.\d+)?)\s+tok/s", r.stdout)
    al = re.search(r"avg commit/step=(\d+(?:\.\d+)?)", r.stdout)
    if not (tps and al):
        raise RuntimeError(f"test_dflash output parse failed: {r.stdout[-1500:]}")
    return float(tps.group(1)), float(al.group(1)), out_bin


def _read_ids(path: Path):
    """Read a binary file of packed int32 token IDs."""
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data)//4}i", data))


def score_math(output_bin: Path, gold_answer: str, tok) -> tuple[bool, str]:
    """Score a Math500 output against the gold answer. Returns (correct, detail_str)."""
    ids = _read_ids(output_bin)
    text = tok.decode(ids)

    think_end = text.rfind("</think>")
    answer_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    pred = _extract_boxed(answer_text)

    # Fallback: "the answer is **X**" patterns
    if pred is None:
        bold_pattern = re.compile(
            r'(?:answer\s+is|there\s+are|result\s+is|equals?|=)\s*\*\*(.+?)\*\*',
            re.IGNORECASE)
        m = bold_pattern.search(answer_text)
        if m:
            pred = m.group(1).strip().rstrip(".")

    # Fallback: last $...$ expression
    if pred is None:
        matches = re.findall(r'\$([^$]+)\$', answer_text)
        if matches:
            pred = matches[-1].strip()

    correct = _math_equiv(pred, gold_answer)
    pred_short = (pred[:60] + "…") if pred and len(pred) > 60 else pred
    gold_short = (gold_answer[:60] + "…") if len(gold_answer) > 60 else gold_answer
    if correct:
        detail = f"🎯 {pred_short}"
    elif pred:
        detail = f"✗ pred={pred_short} gold={gold_short}"
    else:
        detail = f"✗ no answer found, gold={gold_short}"
    return correct, detail


def score_gsm(output_bin: Path, gold_answer: str, tok) -> tuple[bool, str]:
    """Score a GSM8K output against the gold numeric answer. Returns (correct, detail_str)."""
    ids = _read_ids(output_bin)
    text = tok.decode(ids)

    think_end = text.rfind("</think>")
    answer_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    pred = None

    # \boxed{<number>}
    boxed = _extract_boxed(answer_text)
    if boxed:
        cleaned = boxed.replace(",", "").replace("$", "").strip()
        if re.match(r'^[+-]?\d+\.?\d*$', cleaned):
            pred = cleaned

    # #### <number>
    if pred is None:
        m = re.search(r'####\s*\$?([+-]?\d[\d,]*\.?\d*)', answer_text)
        if m:
            pred = m.group(1).replace(",", "")

    # "the answer is **X**"
    if pred is None:
        m = re.search(
            r'(?:answer\s+is|result\s+is|equals?|there\s+are|we\s+get)\s*\*?\*?\$?([+-]?\d[\d,]*\.?\d*)',
            answer_text, re.IGNORECASE)
        if m:
            pred = m.group(1).replace(",", "")

    # **<number>** or **$<number>**
    if pred is None:
        m = re.search(r'\*\*\$?([+-]?\d[\d,]*\.?\d*)\*\*', answer_text)
        if m:
            pred = m.group(1).replace(",", "")

    # Last standalone number
    if pred is None:
        nums = re.findall(r'(?<![.\d])([+-]?\d[\d,]*\.?\d*)(?![.\d])', answer_text)
        if nums:
            pred = nums[-1].replace(",", "")

    correct = False
    if pred is not None:
        try:
            correct = abs(float(pred) - float(gold_answer)) < 1e-6
        except (ValueError, TypeError):
            correct = pred.strip() == gold_answer.strip()

    if correct:
        detail = f"🎯 {pred}"
    elif pred:
        detail = f"✗ pred={pred} gold={gold_answer}"
    else:
        detail = f"✗ no answer found, gold={gold_answer}"
    return correct, detail



def main():
    global DRAFT, BUDGET

    parser = argparse.ArgumentParser(description="DFlash LLM benchmark suite")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help=f"DDTree budget (default {BUDGET})")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Wrap prompts in chat template with enable_thinking=False")
    parser.add_argument("--bench", nargs="+", choices=["HumanEval", "GSM8K", "Math500"],
                        help="Run only specified benchmarks (default: all)")
    args = parser.parse_args()
    BUDGET = args.budget

    DRAFT = _resolve_draft()
    _require_file(TARGET, "target GGUF")
    _require_file(TEST_DFLASH, "test_dflash binary")
    _require_file(TEST_GENERATE, "test_generate binary")

    print(f"[bench] target    = {TARGET}", flush=True)
    print(f"[bench] draft     = {DRAFT}", flush=True)
    print(f"[bench] ar bin    = {TEST_GENERATE}", flush=True)
    print(f"[bench] df bin    = {TEST_DFLASH}", flush=True)
    print(f"[bench] tokenizer = {TOKENIZER}", flush=True)
    print(f"[bench] budget    = {BUDGET}", flush=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

    bench_filter = set(args.bench) if args.bench else None

    if args.no_thinking and not getattr(tok, "chat_template", None):
        parser.error(
            f"--no-thinking requires a tokenizer with a chat template, "
            f"but {TOKENIZER!r} has none. Use a Qwen3 tokenizer or drop --no-thinking."
        )

    def _wrap_prompt(raw_prompt: str) -> str:
        # Instruct/thinking models require the chat template. Feeding a raw
        # prompt makes the model ramble and never emit a scorable answer
        # (issue #191: Math500 scored 0/10). Always apply the template when the
        # tokenizer has one; --no-thinking only toggles Qwen's <think> block.
        if not getattr(tok, "chat_template", None):
            return raw_prompt
        return tok.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=not args.no_thinking,
        )

    results = {}
    for name, ds_name, cfg, split, extract, gold_extract, gen in BENCHES:
        if bench_filter and name not in bench_filter:
            continue
        print(f"\n[bench] ==== {name} (n={N_SAMPLE}, n_gen={gen}) ====", flush=True)
        ds = load_dataset(ds_name, cfg, split=split)
        ds_selected = ds.shuffle(seed=42).select(range(N_SAMPLE))
        prompt_list = [extract(s) for s in ds_selected]
        gold_list = [gold_extract(s) for s in ds_selected] if gold_extract else [None] * len(prompt_list)

        ar_tps, df_tps, df_al = [], [], []
        n_score_correct, n_scored = 0, 0
        for i, (p, gold) in enumerate(zip(prompt_list, gold_list)):
            path = TMPDIR / f"b_{name}_{i:02d}.bin"
            n = tokenize(tok, _wrap_prompt(p), path)
            if n == 0 or n > 3500:
                continue
            try:
                ar = run_ar(path, gen)
                df, al, df_bin = run_df(path, n, gen)
            except Exception as e:
                print(f"  [{i+1:02d}/{N_SAMPLE}] n_tok={n:4d}  FAILED: {e}", flush=True)
                continue

            score_detail = ""
            if gold is not None:
                if name == "GSM8K":
                    correct, score_detail = score_gsm(df_bin, gold, tok)
                else:
                    correct, score_detail = score_math(df_bin, gold, tok)
                n_scored += 1
                if correct:
                    n_score_correct += 1
                score_detail = f"  {score_detail}"

            if ar > 0:
                ar_tps.append(ar)
            if df > 0:
                df_tps.append(df)
                df_al.append(al)
            print(f"  [{i+1:02d}/{N_SAMPLE}] n_tok={n:4d}  AR={ar:6.2f}  DFlash={df:7.2f}  AL={al:5.2f}{score_detail}", flush=True)
        ar_m = sum(ar_tps) / len(ar_tps) if ar_tps else 0
        df_m = sum(df_tps) / len(df_tps) if df_tps else 0
        al_m = sum(df_al) / len(df_al) if df_al else 0
        score_str = f"{n_score_correct}/{n_scored}" if n_scored else ""
        results[name] = {"ar": ar_m, "dflash": df_m, "al": al_m,
                         "speedup": df_m / ar_m if ar_m else 0,
                         "score": score_str}
        summary = f"  {name} mean: AR={ar_m:.2f}  DFlash={df_m:.2f}  AL={al_m:.2f}  {results[name]['speedup']:.2f}x"
        if score_str:
            summary += f"  score={score_str} ({n_score_correct/n_scored*100:.0f}%)"
        print(summary, flush=True)

    print("\n[bench] === SUMMARY ===")
    print(f"{'Task':12s}  {'AR':>8s}  {'DFlash':>8s}  {'AL':>6s}  {'Speedup':>8s}  {'Score':>8s}")
    for name, r in results.items():
        print(f"{name:12s}  {r['ar']:8.2f}  {r['dflash']:8.2f}  {r['al']:6.2f}  {r['speedup']:7.2f}x  {r.get('score',''):>8s}")

    out_json = TMPDIR / "bench_llm_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
