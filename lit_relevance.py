"""Validation helpers for project-specific candidate relevance scoring profiles."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CRITERION_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FIELD_NAMES = ("title", "abstract", "matched_keywords")
METADATA_SIGNAL_NAMES = (
    "source_relevance",
    "citation_impact",
    "recency",
    "open_access",
    "doi",
)


def _bounded_number(value: Any, label: str, *, maximum: float = 10.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if number < 0 or number > maximum:
        raise ValueError(f"{label} must be between 0 and {maximum:g}.")
    return number


def _term_list(value: Any, label: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    terms: list[str] = []
    seen: set[str] = set()
    for raw in value:
        term = re.sub(r"\s+", " ", str(raw or "")).strip()
        key = term.lower()
        if term and key not in seen:
            terms.append(term)
            seen.add(key)
    if required and not terms:
        raise ValueError(f"{label} must contain at least one term.")
    return terms


def _weight_map(value: Any, names: tuple[str, ...], label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    unknown = sorted(set(value) - set(names))
    if unknown:
        raise ValueError(f"{label} contains unsupported keys: {', '.join(unknown)}")
    return {
        name: _bounded_number(value.get(name, 0), f"{label}.{name}")
        for name in names
    }


def validate_relevance_profile(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Relevance scoring profile must be a JSON object.")
    if payload.get("version", 1) != 1:
        raise ValueError("Relevance scoring profile version must be 1.")

    title = str(payload.get("title") or "").strip()
    description = str(payload.get("description") or "").strip()
    if not title:
        raise ValueError("Relevance scoring profile title is required.")
    if not description:
        raise ValueError("Relevance scoring profile description is required.")

    raw_criteria = payload.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise ValueError("Relevance scoring profile must contain at least one criterion.")
    criteria: list[dict[str, Any]] = []
    criterion_ids: set[str] = set()
    for index, raw in enumerate(raw_criteria, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Relevance criterion {index} must be an object.")
        criterion_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or "").strip()
        criterion_description = str(raw.get("description") or "").strip()
        if not CRITERION_ID_RE.fullmatch(criterion_id):
            raise ValueError(
                f"Relevance criterion {index} id must be stable snake_case starting with a letter."
            )
        if criterion_id in criterion_ids:
            raise ValueError(f"Duplicate relevance criterion id: {criterion_id}")
        if not label:
            raise ValueError(f"Relevance criterion {criterion_id} requires a label.")
        if not criterion_description:
            raise ValueError(f"Relevance criterion {criterion_id} requires a description.")
        criteria.append(
            {
                "id": criterion_id,
                "label": label,
                "description": criterion_description,
                "weight": _bounded_number(
                    raw.get("weight", 1), f"criteria.{criterion_id}.weight"
                ),
                "terms": _term_list(
                    raw.get("terms") or [],
                    f"criteria.{criterion_id}.terms",
                    required=True,
                ),
            }
        )
        criterion_ids.add(criterion_id)

    field_weights = _weight_map(
        payload.get("field_weights") or {},
        FIELD_NAMES,
        "field_weights",
    )
    metadata_weights = _weight_map(
        payload.get("metadata_weights") or {},
        METADATA_SIGNAL_NAMES,
        "metadata_weights",
    )
    if not any(field_weights.values()):
        raise ValueError("At least one field weight must be greater than zero.")
    if not any(item["weight"] for item in criteria) and not any(metadata_weights.values()):
        raise ValueError("At least one criterion or metadata weight must be greater than zero.")

    raw_exclusions = payload.get("exclusions") or []
    if not isinstance(raw_exclusions, list):
        raise ValueError("Relevance scoring exclusions must be a list.")
    exclusions: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_exclusions, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Relevance exclusion {index} must be an object.")
        term = re.sub(r"\s+", " ", str(raw.get("term") or "")).strip()
        reason = str(raw.get("reason") or "").strip()
        if not term:
            raise ValueError(f"Relevance exclusion {index} requires a term.")
        exclusions.append(
            {
                "term": term,
                "penalty": _bounded_number(
                    raw.get("penalty", 0.25),
                    f"exclusions.{index}.penalty",
                    maximum=1.0,
                ),
                "reason": reason,
            }
        )

    return {
        "version": 1,
        "title": title,
        "description": description,
        "criteria": criteria,
        "field_weights": field_weights,
        "metadata_weights": metadata_weights,
        "exclusions": exclusions,
    }


def load_relevance_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Relevance scoring profile does not exist: {path}")
    return validate_relevance_profile(json.loads(path.read_text(encoding="utf-8")))
