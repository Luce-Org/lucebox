# Swift IQ4_XS: controlled DFlash2 versus MTP comparison

## DFlash2 finished the article suite 27.2% sooner

With the frozen inputs and xhigh thinking, median total request time was **64.15s for Lucebox + DFlash2** and **88.13s for llama.cpp + MTP**. Median generation throughput was **90.29 versus 61.14 tokens/s**, including thinking.

Both passed **18/18 quality checks**: six distinct tasks repeated three times, with exact answers and strict JSON/tool-call formatting. No output reached the ceiling. This is a small correctness screen, not proof of equal quality on broader coding work.

The backends generated different numbers of tokens despite identical inputs. Total latency therefore reflects both output length and generation speed. This compares complete serving configurations, not the speculation algorithms in isolation.

## Suite medians (three repetitions)

| Suite | Backend | Input tokens | Generated tokens | Thinking text tokens | Prefill s | Generation s | Total s | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| quality | Lucebox + DFlash2 | 929 | 1,717 | 1,590 | 1.17 | 19.63 | 20.85 | 87.45 |
| speed | Lucebox + DFlash2 | 1,703 | 5,605 | 4,092 | 1.98 | 62.08 | 64.15 | 90.29 |
| quality | llama.cpp + MTP | 929 | 1,574 | 1,447 | 2.80 | 25.98 | 28.80 | 60.60 |
| speed | llama.cpp + MTP | 1,703 | 5,084 | 3,482 | 4.84 | 83.16 | 88.13 | 61.14 |

## Variation across the three runs

| Suite | Backend | Total-time range (s) | Decode-rate range (tok/s) |
|---|---|---:|---:|
| quality | Lucebox + DFlash2 | 20.56–20.86 | 87.38–88.69 |
| speed | Lucebox + DFlash2 | 63.57–64.21 | 90.21–91.09 |
| quality | llama.cpp + MTP | 28.39–28.82 | 60.58–61.49 |
| speed | llama.cpp + MTP | 86.99–88.29 | 60.95–61.89 |

## Reading the measurements

**Two suites, never pooled.** Quality has six fixed, known-answer tasks. Speed has the ten canonical Lucebox article prompts; generated code is not executed or scored for correctness. Each row sums one full suite, then takes the median across three repetitions. Per-prompt rows take medians separately, so their sums need not equal the suite median.

**Tokens.** Input and total generated counts come from each server. Total generated includes thinking, final answers, and any generated protocol tokens. Thinking text is retokenized with the same Swift tokenizer for both backends, without added special tokens. This excludes wrapper markers and may differ slightly from native reasoning-token counts; Lucebox's native counts are retained in the raw data. Do not subtract thinking text tokens from total generated and label the remainder exactly as visible-answer tokens.

**Time.** Prefill and generation are the backend's own reported phase times. Total time is externally measured HTTP wall time; it also includes parsing, transport and other request overhead. Model loading, a single unmeasured warmup, and post-run token counting are excluded. Decode tok/s divides all generated tokens by generation time. Three repeats quantify limited timing variation; fixed-seed repeats do not create new independent quality tasks.

## Frozen settings and controls

Same Swift IQ4_XS file and embedded Jinja template; xhigh thinking, temperature 0, seed 42, non-streaming requests, single stream, 65,536-token context, 64,000-token output ceiling, GPU weights and Q8 K/V. Forced thinking closure and prefix caching are disabled. The finite context prevents truly unlimited output; no response approached the ceiling. DFlash2 uses its existing Q8 drafter and block 16; MTP uses the embedded head and maximum three draft tokens. No prompt compression.

Run order: DFlash2, MTP, MTP, DFlash2, DFlash2, MTP. Fresh process and the same short warmup per repetition; GPU memory is checked between loads. Request bodies and prompt-token counts match across all six runs. All 96 responses contain thinking text, complete normally, and report zero cached input tokens.

## Scope and exclusions

The six quality prompts and rubric were frozen before measured runs. The JSON-array requirement is explicit in this version and identical for both backends. No answers were repaired or coerced. The exact text, expected objects and tool schema appear below.

Earlier exploratory results are excluded. A discarded setup run exposed a silent 32,768-token server clamp; startup was corrected and all measured repetitions restarted. The partial data remain in a separately named rejected-preflight folder and are not included here. Earlier startup recovery briefly encountered delayed GPU-memory release; measured runs used a release check between loads.

These results describe this machine, model, quantization and the two complete backends. Different numerical kernels and generated traces remain part of that comparison. The speed suite does not establish code correctness, and the quality suite is too small to establish broad model equivalence.


**Why earlier numbers differ.** This frozen protocol uses the exact embedded template (including its xhigh instructions), the full context/output settings above, and disabled prefix caching. Earlier exploratory runs used differing renderers, output caps, contexts and reasoning settings. Their throughput values are not pooled with or treated as replicates of this experiment.

## Exact quality inputs and frozen criteria

### quality: arithmetic

```text
A machine makes 17 parts every 6 minutes. It runs for 3 hours 24 minutes, but is stopped for 18 minutes during that interval. It starts a fresh cycle when restarted. Both active intervals are multiples of 6 minutes. How many parts? Return JSON only: {"parts": integer}.
```

Exact JSON object, values and types; no fences or extra text. Normal completion required.

Expected:
```json
{
  "parts": 527
}
```

### quality: constraints

```text
Schedule tasks A,B,C,D, each one hour, on one machine starting at hour 0. A must precede C, B must precede D, and D must precede A. Return JSON only with the unique order: {"order":[...]}.
```

Exact JSON object, values and types; no fences or extra text. Normal completion required.

Expected:
```json
{
  "order": [
    "B",
    "D",
    "A",
    "C"
  ]
}
```

### quality: code_aliasing

```text
What does Python print?
a = [[0]] * 3
a[0].append(1)
a[1] = [2]
a[2][0] = 9
print(a)
Return JSON only with result as an array of arrays of integers, not a string: {"result": [[...], ...]}.
```

Exact JSON object, values and types; no fences or extra text. Normal completion required.

Expected:
```json
{
  "result": [
    [
      9,
      1
    ],
    [
      2
    ],
    [
      9,
      1
    ]
  ]
}
```

### quality: code_intervals

```text
A function merges CLOSED integer intervals when they overlap, but NOT merely when adjacent. Given [[8,10],[1,3],[3,6],[7,7],[12,12],[10,11]], return the sorted merged list as JSON only: {"intervals":[...]}.
```

Exact JSON object, values and types; no fences or extra text. Normal completion required.

Expected:
```json
{
  "intervals": [
    [
      1,
      6
    ],
    [
      7,
      7
    ],
    [
      8,
      11
    ],
    [
      12,
      12
    ]
  ]
}
```

### quality: extraction

```text
Data: [{"id":"a","active":true,"score":7},{"id":"b","active":false,"score":99},{"id":"c","active":true,"score":7},{"id":"d","active":true,"score":4}]. Select active items, sort by score descending then id descending, take first two. Return JSON only: {"ids":[...]}.
```

Exact JSON object, values and types; no fences or extra text. Normal completion required.

Expected:
```json
{
  "ids": [
    "c",
    "a"
  ]
}
```

### quality: tool_call

```text
Use lookup_weather to check Toronto in Celsius.
```

Exactly one lookup_weather call, exact argument object and types, finish_reason tool_calls.

Expected:
```json
{
  "city": "Toronto",
  "unit": "celsius"
}
```

Tool schema:
```json
[
  {
    "type": "function",
    "function": {
      "name": "lookup_weather",
      "description": "Look up weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string"
          },
          "unit": {
            "type": "string",
            "enum": [
              "celsius",
              "fahrenheit"
            ]
          }
        },
        "required": [
          "city",
          "unit"
        ]
      }
    }
  }
]
```

### speed: has_close_elements

```text
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for
```

Normal completion and nonempty final content; coding correctness is not scored.

### speed: separate_paren_groups

```text
from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    result = []
    current_string = []
    current_depth = 0
    for
```

Normal completion and nonempty final content; coding correctness is not scored.

### speed: truncate_number

```text
def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    return
```

Normal completion and nonempty final content; coding correctness is not scored.

### speed: below_zero

```text
from typing import List

def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
    balance = 0
    for op in
```

Normal completion and nonempty final content; coding correctness is not scored.

### speed: mean_absolute_deviation

```text
from typing import List

def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
    mean =
```

Normal completion and nonempty final content; coding correctness is not scored.

### speed: intersperse

```text
from typing import List

def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    result = []
    for i, n in
```

Normal completion and nonempty final content; coding correctness is not scored.

### speed: parse_nested_parens

```text
from typing import List

def parse_nested_parens(paren_string: str) -> List[int]:
    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
    For each of the group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.
    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
    def parse_paren_group(s):
        depth = 0
        max_depth = 0
        for c in
```

Normal completion and nonempty final content; coding correctness is not scored.

### speed: filter_by_substring

```text
from typing import List

def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
    return
```

Normal completion and nonempty final content; coding correctness is not scored.

### speed: sum_product

```text
from typing import List, Tuple

def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
    s = 0
    p = 1
    for n in
```

Normal completion and nonempty final content; coding correctness is not scored.

### speed: rolling_max

```text
from typing import List

def rolling_max(numbers: List[int]) -> List[int]:
    """ From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """
    result = []
    running_max = None
    for n in numbers:
        if running_max is
```

Normal completion and nonempty final content; coding correctness is not scored.

## Evidence

Protocol SHA-256: `f0b9120d5643f58ea6f40e9cdd9f183ff0e5f745317d49d838e59eae8fe85925`.

Raw responses, requests, native timings, launch flags, executable/model/template identities, per-prompt CSVs, and reproduction scripts are supplied in the evidence bundle. Production was restored after measurement; see the final health record.

## Regular Unsloth model: one additional run

The regular **Qwen3.8-27B-UD-IQ4_XS** model with DFlash2 finished the ten article prompts in **65.63s**, producing **6,190 tokens**, including **4,895 thinking-text tokens**, at **97.27 tokens/s**. It passed **6/6** fixed quality checks.

This is **one run**, compared with the three-run Swift medians above. The same 16 request bodies, scoring rules, xhigh settings, frozen Swift template, drafter and server binary were used. Input token counts match for every task. Prefix caching and compression remain disabled. The Unsloth model's native template differs; we intentionally retained the frozen template to hold the rendered instructions constant. The quantization files are both IQ4_XS-family but have different per-layer quantization layouts, so this is not a pure fine-tuning ablation.

Thinking text is counted with the same tokenizer as before; the models' vocabulary and merge hashes match. No response hit the 64,000-token ceiling. This single run does not establish timing variability. Article code remains unscored for correctness.

| Suite | Prefill s | Generation s | Total s | Generated tokens | Thinking tokens | Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| quality | 1.17 | 36.28 | 37.5 | 3073 | 2946 | 84.71 |
| speed | 1.91 | 63.64 | 65.63 | 6190 | 4895 | 97.27 |
