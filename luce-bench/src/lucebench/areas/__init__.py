"""Per-area case loaders + graders.

Each module exposes:
  * a CASES constant (list[dict]) for that area's evaluation set
  * a grade_*(case, completion) helper returning a dict with at least
    {"pass": bool, "status": str}
  * load_cases() if dynamic loading is preferred

The dispatcher in lucebench.cli routes ``--area X`` to ``lucebench.areas.X``.
"""

from . import ds4_eval, gsm8k, hellaswag, humaneval, truthfulqa_mc1

__all__ = ["ds4_eval", "gsm8k", "hellaswag", "humaneval", "truthfulqa_mc1"]
