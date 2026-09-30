"""Brick capability classifier (regolo/brick-modernbert-capability-classifier), on CPU in-process.

Temporary choice: luce_server has no encoder architectures, so the ModernBERT classifier runs
here with torch (CPU wheels, `uv sync --extra brick`). No ONNX export of it is published.
Loading is lazy and cached per model directory. Calls are serialised on one worker thread so
the gateway's event loop is not blocked, and torch uses its own intra-op threads.

What the Brick runtime does (brick-SR1 `candle-binding/src/lib.rs`), which this copies:
* input: the texts of all user messages joined with "\\n", raw (no chat template or prefix),
  tokenised with special tokens and right-truncated to 512 tokens;
* output: softmax over the 6 logits, then L1 renormalisation. The model card says sigmoid
  (it was trained multi-label with BCE), but Brick routes on the softmax distribution;
* labels are reordered from the model's id2label order into brick_math.CAPABILITIES.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .brick_math import CAPABILITIES, normalise_probs
from .types import content_text

DEFAULT_MODEL_DIR = "/home/berto/models/routing/brick-capability"
MAX_LENGTH = 512

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="brick-capability")
_CACHE: dict[tuple[str, int], "CapabilityClassifier"] = {}
_CACHE_LOCK = threading.Lock()


def routing_text(messages: list[dict]) -> str:
    """Brick's routing text: every user message's text, joined with newlines."""
    return "\n".join(content_text(m.get("content")) for m in messages if m.get("role") == "user")


class CapabilityClassifier:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, max_length: int = MAX_LENGTH):
        self.model_dir = model_dir
        self.max_length = max_length
        self._model = None
        self._tok = None
        self._order: list[int] | None = None
        self._lock = threading.Lock()

    def _load(self):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on the optional extra
            raise RuntimeError("brick_skill needs the capability classifier: uv sync --extra brick") from e
        if not Path(self.model_dir).exists():
            raise RuntimeError(f"capability model not found at {self.model_dir} (see README: hf download)")
        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, dtype=torch.float32).eval()
        id2label = {int(k): v for k, v in self._model.config.id2label.items()}
        by_name = {v: k for k, v in id2label.items()}
        missing = set(CAPABILITIES) - set(by_name)
        if missing:
            raise RuntimeError(f"capability model lacks labels {sorted(missing)}")
        self._order = [by_name[c] for c in CAPABILITIES]

    def predict(self, text: str) -> dict[str, float]:
        import torch

        with self._lock:
            if self._model is None:
                self._load()
            with torch.inference_mode():
                enc = self._tok(text, return_tensors="pt", truncation=True, max_length=self.max_length)
                probs = torch.softmax(self._model(**enc).logits[0].float(), dim=-1).tolist()
        return normalise_probs({c: probs[i] for c, i in zip(CAPABILITIES, self._order)})


def get_classifier(model_dir: str = DEFAULT_MODEL_DIR, max_length: int = MAX_LENGTH) -> CapabilityClassifier:
    with _CACHE_LOCK:
        key = (model_dir, max_length)
        if key not in _CACHE:
            _CACHE[key] = CapabilityClassifier(model_dir, max_length)
        return _CACHE[key]


async def run_capability(fn, text: str) -> dict[str, float]:
    """Run a blocking ``fn(text) -> probs`` on the classifier thread."""
    return await asyncio.get_running_loop().run_in_executor(_EXECUTOR, fn, text)
