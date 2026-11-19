"""Reference-frame schedules for image sequences.

A sequence of ``n`` frames is processed as ``n - 1`` (reference, deformed) pairs.
``ref_indices[k - 1]`` is the 0-based reference frame of frame ``k``, with ``0 <= ref < k``.

* ``accumulative``: every frame against frame 0 (default; no error accumulation);
* ``incremental``: every frame against its predecessor;
* ``every_n``: a new reference every ``n`` frames;
* ``custom``: user-chosen reference (key) frames; each frame uses the latest key frame before it.

Every field is reported in frame-0 coordinates; incremental results are composed
(:mod:`raftcorr.engine.compose`).
"""

from __future__ import annotations

from dataclasses import dataclass

MODES: tuple[str, ...] = ("accumulative", "incremental", "every_n", "custom")


@dataclass(frozen=True)
class FrameSchedule:
    ref_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        refs = tuple(int(r) for r in self.ref_indices)
        object.__setattr__(self, "ref_indices", refs)
        for i, ref in enumerate(refs):
            if ref < 0 or ref > i:
                raise ValueError(f"frame {i + 1}: reference {ref} must be in [0, {i}]")

    # -- constructors -----------------------------------------------------------------------------
    @classmethod
    def from_mode(cls, mode: str, n_frames: int, every_n: int = 10, key_frames: tuple[int, ...] = ()) -> "FrameSchedule":
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "accumulative":
            return cls.accumulative(n_frames)
        if mode == "incremental":
            return cls.incremental(n_frames)
        if mode == "every_n":
            return cls.every_n(every_n, n_frames)
        return cls.custom(list(key_frames), n_frames)

    @classmethod
    def accumulative(cls, n_frames: int) -> "FrameSchedule":
        _check_n(n_frames)
        return cls(tuple(0 for _ in range(n_frames - 1)))

    @classmethod
    def incremental(cls, n_frames: int) -> "FrameSchedule":
        _check_n(n_frames)
        return cls(tuple(range(n_frames - 1)))

    @classmethod
    def every_n(cls, n: int, n_frames: int) -> "FrameSchedule":
        _check_n(n_frames)
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return cls(tuple((k - 1) // n * n for k in range(1, n_frames)))

    @classmethod
    def custom(cls, key_frames: list[int], n_frames: int) -> "FrameSchedule":
        _check_n(n_frames)
        for r in key_frames:
            if r < 0 or r > n_frames - 2:
                # The engine counts from 0 (the CLI's --keys too); the GUI checks its 1-based field first.
                raise ValueError(f"key frame {r} (0-based) out of range [0, {n_frames - 2}] "
                                 f"(the last frame, {n_frames - 1}, cannot be a reference)")
        keys = sorted(set(key_frames) | {0})
        refs = []
        for k in range(1, n_frames):
            refs.append(max(r for r in keys if r < k))
        return cls(tuple(refs))

    # -- queries ----------------------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return len(self.ref_indices) + 1

    def reference_of(self, frame: int) -> int:
        if frame < 1 or frame > len(self.ref_indices):
            raise IndexError(f"frame {frame} out of range [1, {len(self.ref_indices)}]")
        return self.ref_indices[frame - 1]

    def pairs(self) -> list[tuple[int, int]]:
        """``[(reference, deformed), ...]`` in processing order (frame order)."""
        return [(ref, k + 1) for k, ref in enumerate(self.ref_indices)]

    @property
    def reference_frames(self) -> list[int]:
        return sorted(set(self.ref_indices) | {0})

    @property
    def is_accumulative(self) -> bool:
        return all(r == 0 for r in self.ref_indices)

    def to_dict(self) -> dict:
        return {"ref_indices": list(self.ref_indices)}

    @classmethod
    def from_dict(cls, d: dict) -> "FrameSchedule":
        return cls(tuple(d["ref_indices"]))

    def __len__(self) -> int:
        return len(self.ref_indices)


def _check_n(n_frames: int) -> None:
    if n_frames < 2:
        raise ValueError(f"a sequence needs at least 2 frames, got {n_frames}")
