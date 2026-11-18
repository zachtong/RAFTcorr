"""Tests for RenderCache byte-limited eviction."""
from server.render_cache import RenderCache


def test_evicts_by_entry_count():
    cache = RenderCache(max_entries=3, max_bytes=10_000_000)
    for i in range(5):
        cache.put(("k", i), b"x" * 100)
    assert len(cache) == 3


def test_evicts_by_total_bytes():
    cache = RenderCache(max_entries=1000, max_bytes=500)
    cache.put(("a",), b"x" * 200)
    cache.put(("b",), b"x" * 200)
    cache.put(("c",), b"x" * 200)
    # 600 bytes > 500 limit, oldest should be evicted
    assert cache.get(("a",)) is None
    assert cache.get(("c",)) is not None
    assert cache.total_bytes <= 500


def test_overwrite_updates_bytes():
    cache = RenderCache(max_entries=100, max_bytes=10_000)
    cache.put(("k",), b"x" * 100)
    assert cache.total_bytes == 100
    cache.put(("k",), b"y" * 300)
    assert cache.total_bytes == 300
    assert cache.get(("k",)) == b"y" * 300


def test_clear_resets_bytes():
    cache = RenderCache(max_entries=100, max_bytes=10_000)
    cache.put(("a",), b"x" * 500)
    cache.put(("b",), b"x" * 500)
    cache.clear()
    assert len(cache) == 0
    assert cache.total_bytes == 0
