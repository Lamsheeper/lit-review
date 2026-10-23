"""Core progress."""

from __future__ import annotations


import sys


from typing import Any


try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - dependency is optional at runtime
    _tqdm = None


class NullProgress:
    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, count: int) -> None:
        return None


def progress_bar(total: int, desc: str, unit: str, show: bool) -> Any:
    if not show or _tqdm is None or not sys.stderr.isatty():
        return NullProgress()
    return _tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
    )
