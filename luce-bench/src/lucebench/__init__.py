"""luce-bench — capability benchmarks for chat-completion endpoints.

Quick start:

    from lucebench.areas import ds4_eval
    from lucebench.runner import run_case

    cases = ds4_eval.load_cases()
    rows = [run_case(url="http://localhost:8080", case=c, model="dflash")
            for c in cases]

Or via CLI:

    lucebench --url http://localhost:8080 --area ds4-eval --model dflash

The package vendors evaluation fixtures from upstream MIT-licensed
projects (antirez/ds4, openai/human-eval, antoinezambelli/forge). See
NOTICE for attribution.
"""

__version__ = "0.2.4"
