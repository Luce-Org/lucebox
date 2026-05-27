#!/usr/bin/env python3
"""HumanEval-style bench through dflash_server HTTP API. Measures end-to-end
decode tok/s on code-completion prompts. Server-internal [gemma4-spec] log
captures the true decode speed + acceptance rate per request."""
import json, time, urllib.request, sys

PROMPTS = [
    ("has_close_elements",
     "from typing import List\n\n"
     "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
     '    """Check if in given list of numbers, are any two numbers closer to each other than\n'
     "    given threshold.\n"
     "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
     "    False\n"
     '    """\n    for'),
    ("separate_paren_groups",
     "from typing import List\n\n"
     "def separate_paren_groups(paren_string: str) -> List[str]:\n"
     '    """ Separate groups of nested parentheses into list of strings. """\n'
     "    result = []\n    current = ''\n    depth = 0\n    for"),
    ("truncate_number",
     "def truncate_number(number: float) -> float:\n"
     '    """ Return the decimal part of a floating-point number. """\n'
     "    return"),
    ("below_zero",
     "from typing import List\n\n"
     "def below_zero(operations: List[int]) -> bool:\n"
     '    """ Detect if balance falls below zero. """\n'
     "    balance = 0\n    for op in operations:\n"),
    ("mean_absolute_deviation",
     "from typing import List\n\n"
     "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
     '    """ For a given list of input numbers, calculate Mean Absolute Deviation. """\n'
     "    mean ="),
    ("intersperse",
     "from typing import List\n\n"
     "def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n"
     '    """ Insert delimiter between consecutive elements. """\n'
     "    if not numbers:\n        return []\n    result ="),
    ("parse_nested_parens",
     "from typing import List\n\n"
     "def parse_nested_parens(paren_string: str) -> List[int]:\n"
     '    """ For each group return deepest level of nesting. """\n'
     "    def parse(s):\n        depth = max_depth ="),
    ("filter_by_substring",
     "from typing import List\n\n"
     "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
     '    """ Filter strings that contain substring. """\n'
     "    return"),
    ("sum_product",
     "from typing import List, Tuple\n\n"
     "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
     '    """ Return tuple (sum, product). Empty list -> (0,1). """\n'
     "    s ="),
    ("rolling_max",
     "from typing import List\n\n"
     "def rolling_max(numbers: List[int]) -> List[int]:\n"
     '    """ Rolling maximum of running list. """\n'
     "    result = []\n    cur = -10**18\n    for n in numbers:\n"),
]

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080/v1/chat/completions"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 96

total_tok = 0
total_dt  = 0.0
print(f"[bench] url={URL} max_tokens={MAX_TOKENS} n_prompts={len(PROMPTS)}")
for name, prompt in PROMPTS:
    body = {
        "model": "dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": 0,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            r = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  {name:30s} FAIL {e}")
        continue
    dt = time.time() - t
    n  = r["usage"]["completion_tokens"]
    total_tok += n; total_dt += dt
    print(f"  {name:30s} N={n:3d} dt={dt:.3f}s tok/s={n/dt:6.2f}")

print(f"\n[bench] total tokens={total_tok}  total dt={total_dt:.2f}s")
print(f"[bench] avg tok/s = {total_tok/total_dt:.2f}")
