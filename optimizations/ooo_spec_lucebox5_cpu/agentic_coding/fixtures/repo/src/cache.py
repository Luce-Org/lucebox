"""Small deterministic cache module used by the coding-agent benchmark."""

from dataclasses import dataclass

DEFAULT_MAX_ENTRIES = 64
STALE_AFTER_SECONDS = 300


@dataclass(frozen=True)
class CacheRecord:
    key: str
    age_seconds: int


class CachePolicy:
    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.max_entries = max_entries

    def should_refresh(self, record: CacheRecord) -> bool:
        return record.age_seconds > STALE_AFTER_SECONDS


class BuildCache:
    def __init__(self) -> None:
        self.invalidated: list[str] = []

    def invalidate(self, key: str) -> None:
        self.invalidated.append(key)
