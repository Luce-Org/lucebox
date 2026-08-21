from .cache import BuildCache, CachePolicy, CacheRecord
from .paths import normalized_path


class BuildRunner:
    def __init__(self, cache: BuildCache, policy: CachePolicy) -> None:
        self.cache = cache
        self.policy = policy

    def prepare(self, raw_path: str, record: CacheRecord) -> str:
        key = normalized_path(raw_path)
        if self.policy.should_refresh(record):
            self.cache.invalidate(key)
        return key
