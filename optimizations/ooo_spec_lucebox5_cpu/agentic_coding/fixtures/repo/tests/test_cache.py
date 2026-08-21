from src.cache import BuildCache, CachePolicy, CacheRecord
from src.runner import BuildRunner


def test_exact_expiry_boundary_is_fresh() -> None:
    policy = CachePolicy()
    assert policy.should_refresh(CacheRecord("one", age_seconds=300)) is False


def test_stale_record_is_invalidated_before_build() -> None:
    cache = BuildCache()
    runner = BuildRunner(cache, CachePolicy())
    assert runner.prepare("./src/cache.py", CacheRecord("one", 301)) == "src/cache.py"
    assert cache.invalidated == ["src/cache.py"]
