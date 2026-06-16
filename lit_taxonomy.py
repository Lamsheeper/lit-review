"""Shared validation and lookup helpers for user-defined feature taxonomies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


FAMILY_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def normalize_family_key(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_taxonomy(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Taxonomy must be a JSON object.")
    version = payload.get("version", 1)
    if version != 1:
        raise ValueError("Taxonomy version must be 1.")
    title = str(payload.get("title") or "").strip()
    if not title:
        raise ValueError("Taxonomy title is required.")
    raw_families = payload.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("Taxonomy must contain at least one family.")

    families: list[dict[str, Any]] = []
    ids: set[str] = set()
    lookup_keys: dict[str, str] = {}
    for index, raw in enumerate(raw_families, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Taxonomy family {index} must be an object.")
        family_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or "").strip()
        description = str(raw.get("description") or "").strip()
        aliases_raw = raw.get("aliases") or []
        if not FAMILY_ID_RE.fullmatch(family_id):
            raise ValueError(
                f"Taxonomy family {index} id must be stable snake_case starting with a letter."
            )
        if family_id in ids:
            raise ValueError(f"Duplicate taxonomy family id: {family_id}")
        if not label:
            raise ValueError(f"Taxonomy family {family_id} requires a label.")
        if not isinstance(aliases_raw, list):
            raise ValueError(f"Taxonomy family {family_id} aliases must be a list.")

        aliases: list[str] = []
        seen_aliases: set[str] = set()
        for value in aliases_raw:
            alias = str(value).strip()
            key = normalize_family_key(alias)
            if alias and key and key not in seen_aliases:
                aliases.append(alias)
                seen_aliases.add(key)

        family = {
            "id": family_id,
            "label": label,
            "description": description,
            "aliases": aliases,
        }
        for value in [family_id, label, *aliases]:
            key = normalize_family_key(value)
            owner = lookup_keys.get(key)
            if owner and owner != family_id:
                raise ValueError(
                    f"Taxonomy family label or alias {value!r} is shared by {owner} and {family_id}."
                )
            lookup_keys[key] = family_id
        ids.add(family_id)
        families.append(family)

    return {"version": 1, "title": title, "families": families}


def load_taxonomy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file does not exist: {path}")
    return validate_taxonomy(json.loads(path.read_text(encoding="utf-8")))


def taxonomy_category_lookup(taxonomy: dict[str, Any]) -> dict[str, str]:
    validated = validate_taxonomy(taxonomy)
    lookup: dict[str, str] = {}
    for family in validated["families"]:
        for value in [family["id"], family["label"], *family["aliases"]]:
            lookup[normalize_family_key(value)] = family["id"]
    return lookup


def taxonomy_prompt_payload(taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    validated = validate_taxonomy(taxonomy)
    return [
        {
            "id": family["id"],
            "label": family["label"],
            "description": family["description"],
            "aliases": family["aliases"],
        }
        for family in validated["families"]
    ]
