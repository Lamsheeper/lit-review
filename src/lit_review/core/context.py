"""Explicit, task-local taxonomy and domain profile for stage services."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

TAXONOMY: ContextVar[dict[str, Any] | None] = ContextVar("taxonomy", default=None)
PROFILE: ContextVar[str | None] = ContextVar("profile", default=None)


@contextmanager
def research_context(taxonomy: dict[str, Any], profile: str | None = None):
    taxonomy_token = TAXONOMY.set(taxonomy)
    profile_token = PROFILE.set(profile)
    try:
        yield
    finally:
        PROFILE.reset(profile_token)
        TAXONOMY.reset(taxonomy_token)


def family_order() -> list[str]:
    return [f["id"] for f in (TAXONOMY.get() or {}).get("families", [])]
