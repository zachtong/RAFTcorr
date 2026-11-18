"""Simple LRU cache for rendered PNG bytes, shared across route modules."""

from collections import OrderedDict

# ---------------------------------------------------------------------------
# Render-cache defaults
# ---------------------------------------------------------------------------

# Default maximum number of cached rendered PNG blobs.
_DEFAULT_MAX_ENTRIES: int = 512

# Default byte budget for the render cache (300 MB).
_DEFAULT_MAX_BYTES: int = 300 * 1024 * 1024


def auto_cache_size(target_mb: int = 500, avg_item_kb: int = 200) -> int:
    """Compute max entry count. Used as fallback; byte limit is primary."""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        budget_mb = min(target_mb, available_mb * 0.1)
        return max(64, int(budget_mb * 1024 / avg_item_kb))
    except (ImportError, Exception):
        return _DEFAULT_MAX_ENTRIES


def auto_max_bytes(target_mb: int = 300) -> int:
    """Compute byte budget: min(target_mb, 5% available RAM)."""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        budget_mb = min(target_mb, available_mb * 0.05)
        return max(50, int(budget_mb)) * 1024 * 1024
    except (ImportError, Exception):
        return target_mb * 1024 * 1024


class RenderCache:
    """Bounded LRU cache mapping (result_id, frame_idx, params) -> PNG bytes.

    Evicts when either entry count OR total byte size is exceeded.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES, max_bytes: int = 0):
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._max = max_entries
        self._max_bytes = max_bytes if max_bytes > 0 else auto_max_bytes()
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def get(self, key: tuple) -> bytes | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple, data: bytes) -> None:
        if key in self._cache:
            self._total_bytes -= len(self._cache[key])
        self._cache[key] = data
        self._cache.move_to_end(key)
        self._total_bytes += len(data)
        self._evict()

    def _evict(self) -> None:
        while self._cache and (
            len(self._cache) > self._max
            or self._total_bytes > self._max_bytes
        ):
            _, evicted = self._cache.popitem(last=False)
            self._total_bytes -= len(evicted)

    def clear(self) -> None:
        self._cache.clear()
        self._total_bytes = 0

    def __len__(self) -> int:
        return len(self._cache)
