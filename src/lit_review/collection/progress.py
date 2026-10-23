"""Collection progress."""

from __future__ import annotations


import sys


import time


from collections import Counter


from lit_review.collection.sources import Candidate

from lit_review.collection.downloads import LOGGER


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def download_status_counts(candidates: list[Candidate]) -> Counter[str]:
    return Counter(str(candidate.download.get("status") or "pending") for candidate in candidates)


class ProgressReporter:
    def __init__(
        self,
        total: int,
        *,
        enabled: bool = True,
        label: str = "Candidates",
        min_interval: float = 0.5,
    ) -> None:
        self.total = max(total, 0)
        self.enabled = enabled and self.total > 0 and sys.stderr.isatty()
        self.label = label
        self.min_interval = min_interval
        self.started_at = time.monotonic()
        self.current = 0
        self._last_render_at = 0.0
        self._line_length = 0

    def set_current(
        self,
        current: int,
        candidates: list[Candidate],
        *,
        force: bool = False,
    ) -> None:
        self.current = max(0, min(current, self.total))
        self.render(candidates, force=force)

    def advance(self, candidates: list[Candidate], step: int = 1) -> None:
        self.current = max(0, min(self.current + step, self.total))
        self.render(candidates)

    def render(self, candidates: list[Candidate], *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and self.current < self.total and now - self._last_render_at < self.min_interval:
            return
        self._last_render_at = now
        elapsed = max(now - self.started_at, 0.001)
        rate = self.current / elapsed
        remaining = self.total - self.current
        eta = remaining / rate if rate > 0 else 0
        percent = (self.current / self.total * 100.0) if self.total else 100.0
        counts = download_status_counts(candidates)
        summary_parts = []
        for key in ["downloaded", "already_exists", "abstract_only", "failed", "not_attempted"]:
            count = counts.get(key, 0)
            if count:
                summary_parts.append(f"{key}={count}")
        summary = " ".join(summary_parts) if summary_parts else "pending"
        line = (
            f"{self.label}: {self.current}/{self.total} "
            f"({percent:5.1f}%) | {rate * 60:5.1f}/min | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} | {summary}"
        )
        padding = " " * max(0, self._line_length - len(line))
        sys.stderr.write("\r" + line + padding)
        sys.stderr.flush()
        self._line_length = len(line)

    def clear(self) -> None:
        if not self.enabled or not self._line_length:
            return
        sys.stderr.write("\r" + " " * self._line_length + "\r")
        sys.stderr.flush()
        self._line_length = 0

    def done(self, candidates: list[Candidate]) -> None:
        if not self.enabled:
            return
        self.current = self.total
        self.render(candidates, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()
        self._line_length = 0


class PhaseProgressReporter:
    def __init__(
        self,
        total: int,
        *,
        enabled: bool = True,
        label: str = "Progress",
        min_interval: float = 0.5,
        log_interval: float = 30.0,
    ) -> None:
        self.total = max(total, 0)
        self.enabled = enabled and self.total > 0
        self.tty = sys.stderr.isatty()
        self.label = label
        self.min_interval = min_interval
        self.log_interval = log_interval
        self.started_at = time.monotonic()
        self.current = 0
        self.summary = ""
        self._last_render_at = 0.0
        self._last_log_at = 0.0
        self._line_length = 0

    def set_current(
        self,
        current: int,
        *,
        summary: str | None = None,
        force: bool = False,
    ) -> None:
        self.current = max(0, min(current, self.total))
        if summary is not None:
            self.summary = summary
        self.render(force=force)

    def advance(self, step: int = 1, *, summary: str | None = None) -> None:
        self.current = max(0, min(self.current + step, self.total))
        if summary is not None:
            self.summary = summary
        self.render()

    def progress_line(self) -> str:
        now = time.monotonic()
        elapsed = max(now - self.started_at, 0.001)
        rate = self.current / elapsed
        remaining = self.total - self.current
        eta = remaining / rate if rate > 0 else 0
        percent = (self.current / self.total * 100.0) if self.total else 100.0
        line = (
            f"{self.label}: {self.current}/{self.total} "
            f"({percent:5.1f}%) | {rate * 60:5.1f}/min | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
        )
        if self.summary:
            line += f" | {self.summary}"
        return line

    def render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if self.tty:
            if not force and self.current < self.total and now - self._last_render_at < self.min_interval:
                return
            self._last_render_at = now
            line = self.progress_line()
            padding = " " * max(0, self._line_length - len(line))
            sys.stderr.write("\r" + line + padding)
            sys.stderr.flush()
            self._line_length = len(line)
            return

        if force or self.current >= self.total or now - self._last_log_at >= self.log_interval:
            self._last_log_at = now
            LOGGER.info(self.progress_line())

    def done(self, *, summary: str | None = None) -> None:
        if not self.enabled:
            return
        self.current = self.total
        if summary is not None:
            self.summary = summary
        self.render(force=True)
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._line_length = 0
