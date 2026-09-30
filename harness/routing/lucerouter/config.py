"""Load harness/routing/config/backends.json into plain dataclasses.

Schema (extra keys are kept in `extra` and otherwise ignored):

    {"models":   {"<name>": {"base_url": ".../v1", "kind": "luce_server"|"llama-server",
                             "rank": 0, "served_name": optional, "context": optional int,
                             "capacity": optional int, ...}},
     "services": {"<name>": {"base_url": "...", "kind": "...", "served_name": optional,
                             "prompt_suffix": optional str, ...}}}

`rank` orders models by size (0 = smallest). `served_name` is the `model`
value sent to the backend (defaults to the config key). `context` is the
model's context window in tokens (informational). `capacity` is the
concurrent request budget used when the backend's status endpoint does not
report one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def root_url(base_url: str) -> str:
    """Server root: base_url without a trailing `/v1`."""
    url = base_url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


@dataclass
class ModelBackend:
    name: str
    base_url: str
    kind: str
    rank: int
    served_name: str | None = None
    context: int | None = None
    capacity: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def root(self) -> str:
        return root_url(self.base_url)

    @property
    def request_model(self) -> str:
        return self.served_name or self.name


@dataclass
class Service:
    name: str
    base_url: str
    kind: str
    served_name: str | None = None
    prompt_suffix: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def root(self) -> str:
        return root_url(self.base_url)


@dataclass
class Config:
    models: dict[str, ModelBackend]
    services: dict[str, Service] = field(default_factory=dict)
    path: str | None = None

    def by_rank(self) -> list[ModelBackend]:
        return sorted(self.models.values(), key=lambda m: m.rank)

    def rank(self, model: str) -> int:
        return self.models[model].rank

    def smallest(self) -> str:
        return self.by_rank()[0].name

    def largest(self) -> str:
        return self.by_rank()[-1].name

    def service(self, name: str) -> Service:
        if name not in self.services:
            raise KeyError(f"service {name!r} not in config (have {sorted(self.services)})")
        return self.services[name]


_MODEL_KEYS = {"base_url", "kind", "rank", "served_name", "context", "capacity"}
_SERVICE_KEYS = {"base_url", "kind", "served_name", "prompt_suffix"}


def config_from_dict(data: dict, path: str | None = None) -> Config:
    models = {}
    for name, m in data.get("models", {}).items():
        models[name] = ModelBackend(
            name=name,
            base_url=m["base_url"],
            kind=m.get("kind", "luce_server"),
            rank=int(m["rank"]),
            served_name=m.get("served_name"),
            context=m.get("context"),
            capacity=m.get("capacity"),
            extra={k: v for k, v in m.items() if k not in _MODEL_KEYS},
        )
    services = {}
    for name, s in data.get("services", {}).items():
        services[name] = Service(
            name=name,
            base_url=s["base_url"],
            kind=s.get("kind", "luce_server"),
            served_name=s.get("served_name"),
            prompt_suffix=s.get("prompt_suffix", ""),
            extra={k: v for k, v in s.items() if k not in _SERVICE_KEYS},
        )
    if not models:
        raise ValueError("config has no models")
    return Config(models=models, services=services, path=path)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return config_from_dict(json.load(f), path=str(path))
