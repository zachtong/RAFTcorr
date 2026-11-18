"""Lightweight input validation utilities for API endpoints.

Provides simple validation functions that raise ValueError on bad input.
Route handlers catch these and return 400 responses with clear messages.
"""

from typing import Any, Collection, Optional, Tuple


def validate_positive(value: Any, name: str) -> float:
    """Validate that *value* is a positive number.

    Returns the float-cast value on success.
    Raises ValueError with a user-friendly message on failure.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


def validate_non_negative(value: Any, name: str) -> float:
    """Validate that *value* is >= 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if v < 0:
        raise ValueError(f"{name} must be non-negative, got {v}")
    return v


def validate_positive_int(value: Any, name: str) -> int:
    """Validate that *value* is a positive integer."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if v <= 0:
        raise ValueError(f"{name} must be a positive integer, got {v}")
    return v


def validate_range(
    value: Any,
    name: str,
    min_val: float,
    max_val: float,
) -> float:
    """Validate that *value* is within [min_val, max_val]."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if v < min_val or v > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {v}")
    return v


def validate_choice(value: Any, name: str, choices: Collection[str]) -> str:
    """Validate that *value* is one of the allowed *choices*."""
    v = str(value)
    if v not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of [{allowed}], got {v!r}")
    return v


def validate_odd_positive_int(value: Any, name: str, minimum: int = 1) -> int:
    """Validate that *value* is a positive odd integer >= *minimum*."""
    v = validate_positive_int(value, name)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    if v % 2 == 0:
        raise ValueError(f"{name} must be odd, got {v}")
    return v


def validate_frame_index(idx: int, total: int, name: str = "Frame index") -> int:
    """Validate that *idx* is within [0, total)."""
    if idx < 0 or idx >= total:
        raise ValueError(f"{name} out of range: {idx} (valid: 0..{total - 1})")
    return idx


def validate_coordinate_range(
    x: float,
    y: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    """Validate that (x, y) is within image bounds [0, width) x [0, height)."""
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(
            f"Coordinate ({x}, {y}) out of image bounds "
            f"(0..{width - 1}, 0..{height - 1})"
        )
    return (x, y)
