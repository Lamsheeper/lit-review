#!/usr/bin/env python3
"""Compare two extracted feature sets with Matplotlib plots.

The expected input is a LitExtract-style feature export: either JSONL rows or a
JSON file containing a list of feature rows. Rows should include feature names
and paper counts, with optional category and mention_count fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FeatureRecord:
    key: str
    display_name: str
    category: str
    category_key: str
    paper_count: int
    mention_count: int
    aliases: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    definitions: tuple[str, ...] = ()

    @property
    def plot_label(self) -> str:
        if self.category:
            return f"{self.display_name} ({self.category})"
        return self.display_name


@dataclass(frozen=True)
class FeatureSet:
    path: Path
    label: str
    records: list[FeatureRecord]

    @property
    def by_key(self) -> dict[str, FeatureRecord]:
        return {record.key: record for record in self.records}


@dataclass(frozen=True)
class ScatterPoint:
    key: str
    label: str
    count_1: int
    count_2: int


@dataclass(frozen=True)
class CategorySpec:
    key: str
    label: str
    slug: str


@dataclass(frozen=True)
class SemanticMatch:
    set_1_key: str
    set_2_key: str
    set_1_name: str
    set_2_name: str
    confidence: float
    reason: str


DEFAULT_CATEGORY_KEYS = ("persuasion", "moral framing", "sentiment affect")
DEFAULT_CATEGORY_LABELS = {
    "persuasion": "Persuasion",
    "moral framing": "Moral Framing",
    "sentiment affect": "Sentiment",
}
CATEGORY_ALIASES = {
    "moral": "moral framing",
    "moral framing": "moral framing",
    "moral foundation": "moral framing",
    "moral foundations": "moral framing",
    "moral_framing": "moral framing",
    "sentiment": "sentiment affect",
    "affect": "sentiment affect",
    "sentiment affect": "sentiment affect",
    "sentiment_affect": "sentiment affect",
}
MATCH_CONNECTOR_TOKENS = {"and"}
SINGULARIZATION_SKIP_SUFFIXES = ("ss", "us", "is")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def normalize_feature_name(value: str | None) -> str:
    if not value:
        return ""
    text = value.lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_match_name(value: str | None, match_normalization: str) -> str:
    normalized = normalize_feature_name(value)
    if match_normalization == "singular":
        return " ".join(
            singularize_token(token)
            for token in normalized.split()
            if token not in MATCH_CONNECTOR_TOKENS
        )
    return normalized


def singularize_token(token: str) -> str:
    if len(token) <= 3 or token.endswith(SINGULARIZATION_SKIP_SUFFIXES):
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ches") or token.endswith("shes") or token.endswith("xes") or token.endswith("zes"):
        return token[:-2]
    if token.endswith("ses") and not token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def coerce_text_list(value: Any, limit: int = 6) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()

    def add(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, list):
            for nested in item:
                add(nested)
            return
        text = re.sub(r"\s+", " ", str(item)).strip()
        if not text:
            return
        key = text.casefold()
        if key in seen:
            return
        seen.add(key)
        output.append(text)

    add(value)
    return tuple(output[:limit])


def read_feature_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return read_jsonl_rows(path, raw)

    if isinstance(parsed, list):
        return coerce_row_list(parsed, path)
    if isinstance(parsed, dict):
        for key in ("features", "rows", "items", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return coerce_row_list(value, path)
        if parsed.get("feature_name") or parsed.get("normalized_feature_name"):
            return [parsed]

    raise ValueError(
        f"{path} must be JSONL, a JSON list of feature rows, or a JSON object "
        "with a list under 'features', 'rows', 'items', or 'data'."
    )


def read_jsonl_rows(path: Path, raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object row.")
        rows.append(parsed)
    return rows


def coerce_row_list(values: list[Any], path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"{path} item {index} must be a JSON object row.")
        rows.append(value)
    return rows


def feature_record(
    row: dict[str, Any],
    match_scope: str,
    match_normalization: str,
) -> FeatureRecord | None:
    raw_feature_key = str(row.get("feature_key") or "")
    feature_key_category, feature_key_name = split_feature_key(raw_feature_key)
    display_name = str(
        row.get("feature_name")
        or row.get("canonical_feature_name")
        or row.get("normalized_feature_name")
        or row.get("canonical_normalized_feature_name")
        or feature_key_name
        or ""
    ).strip()
    display_normalized = normalize_feature_name(
        str(
            row.get("normalized_feature_name")
            or row.get("canonical_normalized_feature_name")
            or feature_key_name
            or display_name
        )
    )
    if not display_normalized:
        return None

    category = str(row.get("category") or feature_key_category or "").strip()
    category_key = normalize_feature_name(category)
    category_key = CATEGORY_ALIASES.get(category_key, category_key)
    match_name = normalize_match_name(display_normalized, match_normalization)
    key = match_name if match_scope == "name" else f"{category_key}::{match_name}"
    return FeatureRecord(
        key=key,
        display_name=display_name or display_normalized,
        category=category,
        category_key=category_key,
        paper_count=safe_int(row.get("paper_count")),
        mention_count=safe_int(row.get("mention_count")),
        aliases=coerce_text_list(
            [
                row.get("source_feature_names"),
                row.get("source_normalized_feature_names"),
                row.get("canonical_feature_name"),
                row.get("canonical_normalized_feature_name"),
            ]
        ),
        synonyms=coerce_text_list(row.get("synonyms")),
        definitions=coerce_text_list(row.get("definitions") or row.get("definition"), limit=3),
    )


def split_feature_key(value: str) -> tuple[str, str]:
    if "::" not in value:
        return "", ""
    category, name = value.split("::", 1)
    return category.strip(), name.strip()


def dedupe_and_sort_records(
    rows: Iterable[dict[str, Any]],
    match_scope: str,
    match_normalization: str,
) -> list[FeatureRecord]:
    by_key: dict[str, FeatureRecord] = {}
    for row in rows:
        record = feature_record(row, match_scope, match_normalization)
        if record is None:
            continue
        existing = by_key.get(record.key)
        if existing is None or record_sort_key(record) < record_sort_key(existing):
            by_key[record.key] = record

    return sorted(by_key.values(), key=record_sort_key)


def record_sort_key(record: FeatureRecord) -> tuple[int, int, str, str]:
    return (
        -record.paper_count,
        -record.mention_count,
        record.display_name.casefold(),
        record.key,
    )


def load_feature_set(
    path: Path,
    label: str | None,
    match_scope: str,
    match_normalization: str = "singular",
) -> FeatureSet:
    rows = read_feature_rows(path)
    records = dedupe_and_sort_records(rows, match_scope, match_normalization)
    return FeatureSet(
        path=path,
        label=label or path.stem,
        records=records,
    )


def filter_feature_set_by_category(feature_set: FeatureSet, category_key: str) -> FeatureSet:
    return FeatureSet(
        path=feature_set.path,
        label=feature_set.label,
        records=[record for record in feature_set.records if record.category_key == category_key],
    )


def parse_category_specs(value: str) -> list[CategorySpec]:
    if value.strip().lower() in {"", "default"}:
        keys = list(DEFAULT_CATEGORY_KEYS)
    else:
        keys = [category_key(part) for part in value.split(",") if part.strip()]

    specs: list[CategorySpec] = []
    seen: set[str] = set()
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        specs.append(
            CategorySpec(
                key=key,
                label=DEFAULT_CATEGORY_LABELS.get(key, key.replace("_", " ").title()),
                slug=sanitize_filename_part(key.replace(" ", "_")),
            )
        )
    return specs


def category_key(value: str) -> str:
    normalized = normalize_feature_name(value)
    return CATEGORY_ALIASES.get(normalized, normalized)


def comparison_category_keys(*feature_sets: FeatureSet) -> list[str]:
    present = {
        record.category_key
        for feature_set in feature_sets
        for record in feature_set.records
    }
    ordered = [key for key in DEFAULT_CATEGORY_KEYS if key in present]
    ordered.extend(sorted(key for key in present if key not in DEFAULT_CATEGORY_KEYS))
    return ordered


def top_keys_by_category(
    feature_set: FeatureSet,
    k: int,
    category_keys: Iterable[str] | None = None,
) -> set[str]:
    if k <= 0:
        return set()
    selected: set[str] = set()
    keys = list(category_keys) if category_keys is not None else comparison_category_keys(feature_set)
    for key in keys:
        category_records = [record for record in feature_set.records if record.category_key == key]
        selected.update(record.key for record in category_records[:k])
    return selected


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def jaccard_series(
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    top_a: int,
    top_b: int,
) -> list[tuple[int, float]]:
    category_keys = comparison_category_keys(feature_set_1, feature_set_2)
    return [
        (
            k,
            jaccard_similarity(
                top_keys_by_category(feature_set_1, k, category_keys),
                top_keys_by_category(feature_set_2, k, category_keys),
            ),
        )
        for k in range(top_a, top_b + 1)
    ]


def venn_counts(
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    k: int,
) -> tuple[int, int, int]:
    category_keys = comparison_category_keys(feature_set_1, feature_set_2)
    keys_1 = top_keys_by_category(feature_set_1, k, category_keys)
    keys_2 = top_keys_by_category(feature_set_2, k, category_keys)
    return (
        len(keys_1 - keys_2),
        len(keys_1 & keys_2),
        len(keys_2 - keys_1),
    )


def scatter_points(
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    k: int,
) -> list[ScatterPoint]:
    category_keys = comparison_category_keys(feature_set_1, feature_set_2)
    keys = top_keys_by_category(feature_set_1, k, category_keys) & top_keys_by_category(
        feature_set_2,
        k,
        category_keys,
    )
    by_key_1 = feature_set_1.by_key
    by_key_2 = feature_set_2.by_key
    points: list[ScatterPoint] = []

    for key in keys:
        record_1 = by_key_1.get(key)
        record_2 = by_key_2.get(key)
        points.append(
            ScatterPoint(
                key=key,
                label=record_1.plot_label if record_1 else key,
                count_1=record_1.paper_count if record_1 else 0,
                count_2=record_2.paper_count if record_2 else 0,
            )
        )

    return sorted(
        points,
        key=lambda point: (-max(point.count_1, point.count_2), point.label.casefold()),
    )


def feature_union_rows(feature_set_1: FeatureSet, feature_set_2: FeatureSet) -> list[dict[str, Any]]:
    by_key_1 = feature_set_1.by_key
    by_key_2 = feature_set_2.by_key
    rank_1 = {record.key: rank for rank, record in enumerate(feature_set_1.records, start=1)}
    rank_2 = {record.key: rank for rank, record in enumerate(feature_set_2.records, start=1)}
    rows: list[dict[str, Any]] = []

    for key in by_key_1.keys() | by_key_2.keys():
        record_1 = by_key_1.get(key)
        record_2 = by_key_2.get(key)
        records = [record for record in (record_1, record_2) if record is not None]
        display_record = record_1 or record_2
        in_set_1 = record_1 is not None
        in_set_2 = record_2 is not None
        if in_set_1 and in_set_2:
            membership = "both"
        elif in_set_1:
            membership = "set_1_only"
        else:
            membership = "set_2_only"

        rows.append(
            {
                "feature_key": key,
                "feature_name": display_record.display_name if display_record else key,
                "category": first_non_empty(record.category for record in records),
                "category_key": first_non_empty(record.category_key for record in records),
                "membership": membership,
                "belongs_to": [
                    label
                    for label, present in (
                        (feature_set_1.label, in_set_1),
                        (feature_set_2.label, in_set_2),
                    )
                    if present
                ],
                "in_set_1": in_set_1,
                "in_set_2": in_set_2,
                "set_1": feature_record_json(record_1, rank_1.get(key)),
                "set_2": feature_record_json(record_2, rank_2.get(key)),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -max(
                int((row.get("set_1") or {}).get("paper_count") or 0),
                int((row.get("set_2") or {}).get("paper_count") or 0),
            ),
            str(row.get("feature_name") or "").casefold(),
            str(row.get("feature_key") or ""),
        ),
    )


def feature_record_json(record: FeatureRecord | None, rank: int | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "rank": rank,
        "feature_name": record.display_name,
        "category": record.category,
        "category_key": record.category_key,
        "paper_count": record.paper_count,
        "mention_count": record.mention_count,
    }


def first_non_empty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def write_feature_union_json(
    path: Path,
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    match_scope: str,
    match_normalization: str,
) -> Path:
    rows = feature_union_rows(feature_set_1, feature_set_2)
    payload = {
        "feature_set_1": {
            "label": feature_set_1.label,
            "path": str(feature_set_1.path),
            "feature_count": len(feature_set_1.records),
        },
        "feature_set_2": {
            "label": feature_set_2.label,
            "path": str(feature_set_2.path),
            "feature_count": len(feature_set_2.records),
        },
        "match_scope": match_scope,
        "match_normalization": match_normalization,
        "feature_count": len(rows),
        "features": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


FINAL_FEATURE_CSV_FIELDS = [
    "category",
    "category_key",
    "peak_k",
    "peak_jaccard",
    "selection_k",
    "selection_k_source",
    "membership",
    "belongs_to",
    "feature_key",
    "feature_name",
    "selected_from_set_1_top_k",
    "selected_from_set_2_top_k",
    "set_1_feature_name",
    "set_1_paper_count",
    "set_1_mention_count",
    "set_1_category_rank",
    "set_2_feature_name",
    "set_2_paper_count",
    "set_2_mention_count",
    "set_2_category_rank",
    "manual_decision",
    "manual_notes",
]


def category_jaccard_series(
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    category_key: str,
    top_a: int,
    top_b: int,
) -> list[tuple[int, float]]:
    return [
        (
            k,
            jaccard_similarity(
                top_keys_by_category(feature_set_1, k, [category_key]),
                top_keys_by_category(feature_set_2, k, [category_key]),
            ),
        )
        for k in range(top_a, top_b + 1)
    ]


def peak_jaccard_for_category(
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    category_key: str,
    top_a: int,
    top_b: int,
) -> tuple[int, float, list[int]]:
    series = category_jaccard_series(feature_set_1, feature_set_2, category_key, top_a, top_b)
    if not series:
        return 0, 0.0, []
    peak_jaccard = max(score for _, score in series)
    peak_ks = [k for k, score in series if score == peak_jaccard]
    return min(peak_ks), peak_jaccard, peak_ks


def parse_final_k_overrides(values: Iterable[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values or []:
        for raw_part in str(value).split(","):
            part = raw_part.strip()
            if not part:
                continue
            separator = "=" if "=" in part else ":" if ":" in part else ""
            if not separator:
                raise ValueError(
                    "--final-category-k entries must look like category=K, for example sentiment=75."
                )
            raw_category, raw_k = part.split(separator, 1)
            key = category_key(raw_category)
            if not key:
                raise ValueError(f"--final-category-k has an empty category in {part!r}.")
            k = safe_int(raw_k)
            if k <= 0:
                raise ValueError(f"--final-category-k for {raw_category.strip()!r} must be a positive integer.")
            overrides[key] = k
    return overrides


def category_rank_by_key(feature_set: FeatureSet) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for key in comparison_category_keys(feature_set):
        records = [record for record in feature_set.records if record.category_key == key]
        for rank, record in enumerate(records, start=1):
            ranks[record.key] = rank
    return ranks


def final_feature_record_json(record: FeatureRecord | None, category_rank: int | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "feature_name": record.display_name,
        "category": record.category,
        "category_key": record.category_key,
        "paper_count": record.paper_count,
        "mention_count": record.mention_count,
        "category_rank": category_rank,
        "aliases": list(record.aliases),
        "synonyms": list(record.synonyms),
    }


def final_feature_candidates(
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    top_a: int,
    top_b: int,
    category_specs: list[CategorySpec],
    final_k_overrides: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key_1 = feature_set_1.by_key
    by_key_2 = feature_set_2.by_key
    rank_1 = category_rank_by_key(feature_set_1)
    rank_2 = category_rank_by_key(feature_set_2)
    final_k_overrides = final_k_overrides or {}
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for spec in category_specs:
        category_records_1 = [record for record in feature_set_1.records if record.category_key == spec.key]
        category_records_2 = [record for record in feature_set_2.records if record.category_key == spec.key]
        if not category_records_1 and not category_records_2:
            continue

        peak_k, peak_jaccard, peak_ks = peak_jaccard_for_category(
            feature_set_1,
            feature_set_2,
            spec.key,
            top_a,
            top_b,
        )
        selection_k = final_k_overrides.get(spec.key, peak_k)
        selection_k_source = "manual_override" if spec.key in final_k_overrides else "peak_jaccard"
        keys_1 = top_keys_by_category(feature_set_1, selection_k, [spec.key])
        keys_2 = top_keys_by_category(feature_set_2, selection_k, [spec.key])
        category_rows: list[dict[str, Any]] = []

        for key in keys_1 | keys_2:
            record_1 = by_key_1.get(key)
            record_2 = by_key_2.get(key)
            display_record = record_1 or record_2
            in_set_1 = record_1 is not None
            in_set_2 = record_2 is not None
            if in_set_1 and in_set_2:
                membership = "both"
            elif in_set_1:
                membership = "set_1_only"
            else:
                membership = "set_2_only"

            category_rows.append(
                {
                    "category": spec.label,
                    "category_key": spec.key,
                    "peak_k": peak_k,
                    "peak_jaccard": peak_jaccard,
                    "peak_tie_ks": peak_ks,
                    "selection_k": selection_k,
                    "selection_k_source": selection_k_source,
                    "feature_key": key,
                    "feature_name": display_record.display_name if display_record else key,
                    "membership": membership,
                    "belongs_to": [
                        label
                        for label, present in (
                            (feature_set_1.label, in_set_1),
                            (feature_set_2.label, in_set_2),
                        )
                        if present
                    ],
                    "selected_from_set_1_top_k": key in keys_1,
                    "selected_from_set_2_top_k": key in keys_2,
                    "in_set_1": in_set_1,
                    "in_set_2": in_set_2,
                    "set_1": final_feature_record_json(record_1, rank_1.get(key)),
                    "set_2": final_feature_record_json(record_2, rank_2.get(key)),
                    "manual_decision": "",
                    "manual_notes": "",
                }
            )

        category_rows = sorted(category_rows, key=final_feature_row_sort_key)
        summaries.append(
            {
                "category": spec.label,
                "category_key": spec.key,
                "peak_k": peak_k,
                "peak_jaccard": peak_jaccard,
                "peak_tie_ks": peak_ks,
                "selection_k": selection_k,
                "selection_k_source": selection_k_source,
                "set_1_top_k_count": len(keys_1),
                "set_2_top_k_count": len(keys_2),
                "top_k_intersection_count": len(keys_1 & keys_2),
                "top_k_union_count": len(keys_1 | keys_2),
                "feature_count": len(category_rows),
            }
        )
        rows.extend(category_rows)

    return rows, summaries


def final_feature_row_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    set_1 = row.get("set_1") or {}
    set_2 = row.get("set_2") or {}
    membership_priority = {"both": 0, "set_1_only": 1, "set_2_only": 2}
    return (
        -max(safe_int(set_1.get("paper_count")), safe_int(set_2.get("paper_count"))),
        membership_priority.get(str(row.get("membership") or ""), 99),
        str(row.get("feature_name") or "").casefold(),
        str(row.get("feature_key") or ""),
    )


def write_final_feature_set(
    json_path: Path,
    csv_path: Path,
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    match_scope: str,
    match_normalization: str,
    top_a: int,
    top_b: int,
    category_specs: list[CategorySpec],
    final_k_overrides: dict[str, int] | None = None,
) -> list[Path]:
    rows, summaries = final_feature_candidates(
        feature_set_1,
        feature_set_2,
        top_a,
        top_b,
        category_specs,
        final_k_overrides,
    )
    final_k_overrides = final_k_overrides or {}
    payload = {
        "feature_set_1": {
            "label": feature_set_1.label,
            "path": str(feature_set_1.path),
            "feature_count": len(feature_set_1.records),
        },
        "feature_set_2": {
            "label": feature_set_2.label,
            "path": str(feature_set_2.path),
            "feature_count": len(feature_set_2.records),
        },
        "match_scope": match_scope,
        "match_normalization": match_normalization,
        "top_a": top_a,
        "top_b": top_b,
        "final_k_overrides": final_k_overrides,
        "peak_tie_break": "smallest_k_at_peak_jaccard",
        "manual_review_fields": ["manual_decision", "manual_notes"],
        "category_count": len(summaries),
        "feature_count": len(rows),
        "categories": summaries,
        "features": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_FEATURE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(final_feature_csv_row(row))

    return [json_path, csv_path]


def final_feature_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    set_1 = row.get("set_1") or {}
    set_2 = row.get("set_2") or {}
    return {
        "category": row.get("category") or "",
        "category_key": row.get("category_key") or "",
        "peak_k": row.get("peak_k") or "",
        "peak_jaccard": row.get("peak_jaccard") if row.get("peak_jaccard") is not None else "",
        "selection_k": row.get("selection_k") or "",
        "selection_k_source": row.get("selection_k_source") or "",
        "membership": row.get("membership") or "",
        "belongs_to": " | ".join(str(label) for label in row.get("belongs_to") or []),
        "feature_key": row.get("feature_key") or "",
        "feature_name": row.get("feature_name") or "",
        "selected_from_set_1_top_k": row.get("selected_from_set_1_top_k"),
        "selected_from_set_2_top_k": row.get("selected_from_set_2_top_k"),
        "set_1_feature_name": set_1.get("feature_name") or "",
        "set_1_paper_count": set_1.get("paper_count") if set_1 else "",
        "set_1_mention_count": set_1.get("mention_count") if set_1 else "",
        "set_1_category_rank": set_1.get("category_rank") if set_1 else "",
        "set_2_feature_name": set_2.get("feature_name") or "",
        "set_2_paper_count": set_2.get("paper_count") if set_2 else "",
        "set_2_mention_count": set_2.get("mention_count") if set_2 else "",
        "set_2_category_rank": set_2.get("category_rank") if set_2 else "",
        "manual_decision": row.get("manual_decision") or "",
        "manual_notes": row.get("manual_notes") or "",
    }


def unique_feature_rows(
    union_rows: list[dict[str, Any]],
    membership: str,
    set_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in union_rows
        if row.get("membership") == membership and isinstance(row.get(set_key), dict)
    ]
    return sorted(
        rows,
        key=lambda row: (
            -int((row.get(set_key) or {}).get("paper_count") or 0),
            -int((row.get(set_key) or {}).get("mention_count") or 0),
            int((row.get(set_key) or {}).get("rank") or 10**9),
            str((row.get(set_key) or {}).get("feature_name") or "").casefold(),
        ),
    )[:limit]


def unique_features_markdown(
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    match_scope: str,
    match_normalization: str,
    limit: int,
) -> str:
    union_rows = feature_union_rows(feature_set_1, feature_set_2)
    set_1_only = unique_feature_rows(union_rows, "set_1_only", "set_1", limit)
    set_2_only = unique_feature_rows(union_rows, "set_2_only", "set_2", limit)

    lines = [
        "# Top Unique Features",
        "",
        f"Set 1: {feature_set_1.label}",
        f"Set 2: {feature_set_2.label}",
        f"Match scope: {match_scope}",
        f"Match normalization: {match_normalization}",
        "",
        f"## Top {limit} Only In {feature_set_1.label}",
        "",
        *feature_table_lines(set_1_only, "set_1"),
        "",
        f"## Top {limit} Only In {feature_set_2.label}",
        "",
        *feature_table_lines(set_2_only, "set_2"),
        "",
    ]
    return "\n".join(lines)


def feature_table_lines(rows: list[dict[str, Any]], set_key: str) -> list[str]:
    if not rows:
        return ["No unique features found."]

    lines = [
        "| Rank | Feature | Category | Paper count | Mention count | Source rank |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        details = row.get(set_key) or {}
        lines.append(
            "| "
            f"{rank} | "
            f"{md_escape(str(details.get('feature_name') or row.get('feature_name') or ''))} | "
            f"{md_escape(str(details.get('category') or row.get('category') or ''))} | "
            f"{int(details.get('paper_count') or 0)} | "
            f"{int(details.get('mention_count') or 0)} | "
            f"{int(details.get('rank') or 0)} |"
        )
    return lines


def md_escape(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def write_unique_features_md(
    path: Path,
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    match_scope: str,
    match_normalization: str,
    limit: int,
) -> Path:
    markdown = unique_features_markdown(
        feature_set_1,
        feature_set_2,
        match_scope,
        match_normalization,
        limit,
    )
    path.write_text(markdown, encoding="utf-8")
    return path


def build_semantic_matches(
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: float,
    match_scope: str,
    candidate_limit: int,
    batch_size: int,
    confidence_threshold: float,
) -> list[SemanticMatch]:
    groups = semantic_candidate_groups(
        feature_set_1,
        feature_set_2,
        match_scope=match_scope,
        candidate_limit=candidate_limit,
    )
    if not groups:
        return []

    matches: list[SemanticMatch] = []
    for batch in batched(groups, batch_size):
        prompt = build_semantic_match_prompt(batch)
        raw_response = call_gemini_json(
            prompt,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
        matches.extend(parse_semantic_match_response(raw_response, batch, confidence_threshold))

    return dedupe_semantic_matches(matches)


def semantic_candidate_groups(
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    *,
    match_scope: str,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    set_1_records = feature_set_1.records
    deterministic_keys = {record.key for record in set_1_records}
    groups: list[dict[str, Any]] = []
    for set_2_index, record_2 in enumerate(feature_set_2.records, start=1):
        if record_2.key in deterministic_keys:
            continue
        candidates = [
            record_1
            for record_1 in set_1_records
            if match_scope == "name" or record_1.category_key == record_2.category_key
        ]
        scored = sorted(
            (
                (semantic_candidate_score(record_1, record_2), set_1_index, record_1)
                for set_1_index, record_1 in enumerate(candidates, start=1)
            ),
            key=lambda item: (-item[0], item[2].display_name.casefold()),
        )[:candidate_limit]
        if not scored:
            continue
        groups.append(
            {
                "set_2_id": f"B{set_2_index}",
                "set_2_record": record_2,
                "candidates": [
                    {
                        "set_1_id": f"A{set_2_index}_{candidate_index}",
                        "set_1_record": record_1,
                        "score": score,
                    }
                    for candidate_index, (score, _, record_1) in enumerate(scored, start=1)
                ],
            }
        )
    return groups


def semantic_candidate_score(record_1: FeatureRecord, record_2: FeatureRecord) -> float:
    terms_1 = comparable_terms(record_1)
    terms_2 = comparable_terms(record_2)
    seq_score = max(
        (
            SequenceMatcher(None, term_1, term_2).ratio()
            for term_1 in terms_1
            for term_2 in terms_2
            if term_1 and term_2
        ),
        default=0.0,
    )
    tokens_1 = set(" ".join(terms_1).split())
    tokens_2 = set(" ".join(terms_2).split())
    overlap_score = len(tokens_1 & tokens_2) / len(tokens_1 | tokens_2) if tokens_1 or tokens_2 else 0.0
    subset_bonus = 0.12 if tokens_1 and tokens_2 and (tokens_1 <= tokens_2 or tokens_2 <= tokens_1) else 0.0
    return max(seq_score, overlap_score + subset_bonus)


def comparable_terms(record: FeatureRecord) -> list[str]:
    terms = [record.display_name, *record.aliases, *record.synonyms]
    normalized: list[str] = []
    seen: set[str] = set()
    for term in terms:
        value = normalize_match_name(term, "singular")
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def build_semantic_match_prompt(groups: list[dict[str, Any]]) -> str:
    payload = []
    for group in groups:
        payload.append(
            {
                "set_2": feature_prompt_record(group["set_2_id"], group["set_2_record"]),
                "candidate_set_1_matches": [
                    feature_prompt_record(candidate["set_1_id"], candidate["set_1_record"])
                    for candidate in group["candidates"]
                ],
            }
        )

    return f"""Compare taxonomy feature names from two literature-review feature sets.

For each set_2 feature, choose at most one candidate_set_1_matches item that is the same conceptual taxonomy feature.

Match when labels are clear synonyms, paraphrases, morphology variants, slash/and variants, or common taxonomy-name variants.
Do not match when one item is merely related, broader/narrower, an example/subtype, an antonym, or a different construct in the same family.
Prefer precision over recall. Only return high-confidence matches.

Return valid JSON only in this shape:
{{
  "matches": [
    {{
      "set_2_id": "B1",
      "set_1_id": "A1_1",
      "confidence": 0.0,
      "reason": "short reason"
    }}
  ]
}}

Omit set_2 items with no high-confidence match.

Feature groups:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def feature_prompt_record(record_id: str, record: FeatureRecord) -> dict[str, Any]:
    return {
        "id": record_id,
        "name": record.display_name,
        "category": record.category,
        "paper_count": record.paper_count,
        "aliases": list(record.aliases[:5]),
        "synonyms": list(record.synonyms[:5]),
        "definitions": [shorten_text(item, 220) for item in record.definitions[:2]],
    }


def call_gemini_json(
    prompt: str,
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: float,
) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    url = f"{base_url.rstrip('/')}/models/{model.removeprefix('models/')}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    try:
        parts = data["candidates"][0].get("content", {}).get("parts", [])
        return "".join(str(part.get("text") or "") for part in parts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini API response: {data}") from exc


def parse_semantic_match_response(
    raw_response: str,
    groups: list[dict[str, Any]],
    confidence_threshold: float,
) -> list[SemanticMatch]:
    parsed = parse_json_payload(raw_response)
    raw_matches = parsed.get("matches") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_matches, list):
        return []

    set_2_by_id = {group["set_2_id"]: group["set_2_record"] for group in groups}
    set_1_by_id: dict[str, FeatureRecord] = {}
    for group in groups:
        for candidate in group["candidates"]:
            set_1_by_id[candidate["set_1_id"]] = candidate["set_1_record"]

    matches: list[SemanticMatch] = []
    for item in raw_matches:
        if not isinstance(item, dict):
            continue
        confidence = safe_float(item.get("confidence"))
        if confidence < confidence_threshold:
            continue
        set_1_record = set_1_by_id.get(str(item.get("set_1_id") or ""))
        set_2_record = set_2_by_id.get(str(item.get("set_2_id") or ""))
        if set_1_record is None or set_2_record is None:
            continue
        matches.append(
            SemanticMatch(
                set_1_key=set_1_record.key,
                set_2_key=set_2_record.key,
                set_1_name=set_1_record.display_name,
                set_2_name=set_2_record.display_name,
                confidence=round(confidence, 4),
                reason=str(item.get("reason") or "").strip(),
            )
        )
    return matches


def parse_json_payload(raw_response: str) -> Any:
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw_response, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw_response[start : end + 1])
    raise ValueError("Gemini response did not contain valid JSON.")


def dedupe_semantic_matches(matches: list[SemanticMatch]) -> list[SemanticMatch]:
    best_by_set_2: dict[str, SemanticMatch] = {}
    used_set_1: set[str] = set()
    for match in sorted(matches, key=lambda item: (-item.confidence, item.set_2_name.casefold())):
        if match.set_2_key in best_by_set_2 or match.set_1_key in used_set_1:
            continue
        best_by_set_2[match.set_2_key] = match
        used_set_1.add(match.set_1_key)
    return sorted(best_by_set_2.values(), key=lambda item: (item.set_2_name.casefold(), item.set_1_name.casefold()))


def apply_semantic_matches(feature_set: FeatureSet, matches: list[SemanticMatch]) -> FeatureSet:
    key_map = {match.set_2_key: match.set_1_key for match in matches}
    if not key_map:
        return feature_set
    by_key: dict[str, FeatureRecord] = {}
    for record in feature_set.records:
        mapped = replace(record, key=key_map.get(record.key, record.key))
        existing = by_key.get(mapped.key)
        if existing is None or record_sort_key(mapped) < record_sort_key(existing):
            by_key[mapped.key] = mapped
    return FeatureSet(
        path=feature_set.path,
        label=feature_set.label,
        records=sorted(by_key.values(), key=record_sort_key),
    )


def write_semantic_matches_json(
    path: Path,
    matches: list[SemanticMatch],
    *,
    model: str,
    confidence_threshold: float,
) -> Path:
    payload = {
        "model": model,
        "confidence_threshold": confidence_threshold,
        "match_count": len(matches),
        "matches": [
            {
                "set_1_key": match.set_1_key,
                "set_2_key": match.set_2_key,
                "set_1_name": match.set_1_name,
                "set_2_name": match.set_2_name,
                "confidence": match.confidence,
                "reason": match.reason,
            }
            for match in matches
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def gemini_api_key(cli_value: str | None) -> str:
    api_key = cli_value or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Gemini matching requires --gemini-api-key, GEMINI_API_KEY, or GOOGLE_API_KEY.")
    return api_key


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), max(1, size)):
        yield values[start : start + max(1, size)]


def shorten_text(value: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 1]}..."


def require_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, ScalarFormatter
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to generate plots. Install it with "
            "`python3 -m pip install matplotlib` or add it to your uv environment."
        ) from exc
    return plt, MaxNLocator, ScalarFormatter


def plot_jaccard(
    series: list[tuple[int, float]],
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    output_path: Path,
    dpi: int,
    title_prefix: str = "",
) -> None:
    plt, MaxNLocator, _ = require_pyplot()
    xs = [item[0] for item in series]
    ys = [item[1] for item in series]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(xs, ys, color="#2f6fed", linewidth=2.2, marker="o", markersize=3.5)
    title = f"Top-k-per-category Jaccard Similarity: {feature_set_1.label} vs {feature_set_2.label}"
    ax.set_title(f"{title_prefix} {title}".strip())
    ax.set_xlabel("Top k features per category by paper_count")
    ax.set_ylabel("Jaccard similarity")
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_venn(
    counts: tuple[int, int, int],
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    k: int,
    output_path: Path,
    dpi: int,
    title_prefix: str = "",
) -> None:
    plt, _, _ = require_pyplot()
    from matplotlib.patches import Circle

    left_only, both, right_only = counts
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    circle_1 = Circle((0.42, 0.5), 0.31, color="#2f6fed", alpha=0.32)
    circle_2 = Circle((0.58, 0.5), 0.31, color="#f05d5e", alpha=0.32)
    ax.add_patch(circle_1)
    ax.add_patch(circle_2)

    left_only_x = 0.21
    intersection_x = 0.50
    right_only_x = 0.79

    ax.text(left_only_x, 0.5, str(left_only), ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(intersection_x, 0.5, str(both), ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(right_only_x, 0.5, str(right_only), ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.33, 0.84, feature_set_1.label, ha="center", va="center", fontsize=10, color="#1f4fb4")
    ax.text(0.67, 0.84, feature_set_2.label, ha="center", va="center", fontsize=10, color="#b5383c")
    ax.text(left_only_x, 0.15, "only set 1", ha="center", va="center", fontsize=9, color="#4b5563")
    ax.text(intersection_x, 0.15, "intersection", ha="center", va="center", fontsize=9, color="#4b5563")
    ax.text(right_only_x, 0.15, "only set 2", ha="center", va="center", fontsize=9, color="#4b5563")
    ax.set_title(f"{title_prefix} Top {k} Per-Category Feature Overlap".strip())
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.05, 0.95)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_scatter(
    points: list[ScatterPoint],
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    k: int,
    output_path: Path,
    dpi: int,
    annotate_top: int,
    scale: str,
    title_prefix: str = "",
) -> None:
    plt, MaxNLocator, ScalarFormatter = require_pyplot()
    xs = [point.count_1 for point in points]
    ys = [point.count_2 for point in points]
    plot_xs = [count + 1 for count in xs] if scale == "log" else xs
    plot_ys = [count + 1 for count in ys] if scale == "log" else ys
    r_value, fit_xs, fit_ys = scatter_fit_line(xs, ys, scale)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(plot_xs, plot_ys, s=36, alpha=0.75, color="#4056a1", edgecolors="white", linewidths=0.5)

    title = f"Paper Counts for Intersection of Top {k} Per-Category Features"
    ax.set_title(f"{title_prefix} {title}".strip())
    if scale == "log":
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{feature_set_1.label} paper_count + 1")
        ax.set_ylabel(f"{feature_set_2.label} paper_count + 1")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(left=0.9, right=max(plot_xs or [1]) * 1.12)
        ax.set_ylim(bottom=0.9, top=max(plot_ys or [1]) * 1.12)
        reference_max = max([*plot_xs, *plot_ys, 1])
        ax.plot([1, reference_max], [1, reference_max], color="#6b7280", linewidth=1, linestyle="--", alpha=0.75)
    else:
        ax.set_xlabel(f"{feature_set_1.label} paper_count")
        ax.set_ylabel(f"{feature_set_2.label} paper_count")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(*linear_axis_limits(xs))
        ax.set_ylim(*linear_axis_limits(ys))
        reference_max = max([*xs, *ys, 1])
        ax.plot([0, reference_max], [0, reference_max], color="#6b7280", linewidth=1, linestyle="--", alpha=0.75)
    if fit_xs and fit_ys:
        ax.plot(fit_xs, fit_ys, color="#d97706", linewidth=1.8, alpha=0.9)
    ax.text(
        0.03,
        0.97,
        f"r = {r_value:.2f}" if r_value is not None else "r = N/A",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85, "boxstyle": "round,pad=0.25"},
    )
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)

    if annotate_top > 0:
        for point in points[:annotate_top]:
            x = point.count_1 + 1 if scale == "log" else point.count_1
            y = point.count_2 + 1 if scale == "log" else point.count_2
            ax.annotate(
                shorten_label(point.label),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#374151",
                alpha=0.9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def scatter_fit_line(
    xs: list[int],
    ys: list[int],
    scale: str,
) -> tuple[float | None, list[float], list[float]]:
    if len(xs) < 2 or len(ys) < 2:
        return None, [], []
    transformed_xs = transform_scatter_values(xs, scale)
    transformed_ys = transform_scatter_values(ys, scale)
    fit = linear_fit(transformed_xs, transformed_ys)
    if fit is None:
        return None, [], []
    slope, intercept, r_value = fit
    x_min = min(transformed_xs)
    x_max = max(transformed_xs)
    if x_min == x_max:
        return r_value, [], []
    fit_transformed_xs = [x_min, x_max]
    fit_transformed_ys = [slope * x + intercept for x in fit_transformed_xs]
    if scale == "log":
        return (
            r_value,
            [10**x for x in fit_transformed_xs],
            [10**y for y in fit_transformed_ys],
        )
    return r_value, fit_transformed_xs, fit_transformed_ys


def transform_scatter_values(values: list[int], scale: str) -> list[float]:
    if scale == "log":
        return [math.log10(value + 1) for value in values]
    return [float(value) for value in values]


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float | None] | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    sum_xx = sum(value * value for value in dx)
    sum_yy = sum(value * value for value in dy)
    if sum_xx == 0:
        return None
    sum_xy = sum(x_delta * y_delta for x_delta, y_delta in zip(dx, dy))
    slope = sum_xy / sum_xx
    intercept = mean_y - slope * mean_x
    if sum_yy == 0:
        return slope, intercept, None
    return slope, intercept, sum_xy / math.sqrt(sum_xx * sum_yy)


def linear_axis_limits(values: list[int]) -> tuple[float, float]:
    max_value = max(values or [0])
    if max_value <= 0:
        return -0.5, 1.0
    pad = max(0.5, max_value * 0.06)
    return -pad, max_value + pad


def shorten_label(value: str, max_chars: int = 42) -> str:
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 1]}..."


def sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "comparison"


def positive_int(value: str) -> int:
    parsed = safe_int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for two extracted feature-set JSON/JSONL files.",
    )
    parser.add_argument("feature_set_1", type=Path, help="Path to the first feature JSON/JSONL file.")
    parser.add_argument("feature_set_2", type=Path, help="Path to the second feature JSON/JSONL file.")
    parser.add_argument("--name-1", "--label-1", dest="label_1", help="Display name for the first feature set.")
    parser.add_argument("--name-2", "--label-2", dest="label_2", help="Display name for the second feature set.")
    parser.add_argument(
        "--match-scope",
        choices=("category", "name"),
        default="category",
        help="Match features by category plus normalized name, or by normalized name only.",
    )
    parser.add_argument(
        "--match-normalization",
        choices=("singular", "exact"),
        default="singular",
        help="Normalize feature names before matching. 'singular' merges simple plural and slash/and variants.",
    )
    parser.add_argument(
        "--semantic-match",
        choices=("none", "gemini"),
        default="none",
        help="Optionally use Gemini to merge semantically equivalent features after deterministic matching.",
    )
    parser.add_argument("--gemini-api-key", help="Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY.")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model for semantic matching.")
    parser.add_argument(
        "--gemini-base-url",
        default=DEFAULT_GEMINI_BASE_URL,
        help="Gemini API base URL.",
    )
    parser.add_argument("--gemini-timeout", type=float, default=120.0, help="Gemini request timeout in seconds.")
    parser.add_argument(
        "--gemini-batch-size",
        type=positive_int,
        default=20,
        help="Number of set-2 features to ask Gemini to match per request.",
    )
    parser.add_argument(
        "--gemini-candidate-limit",
        type=positive_int,
        default=8,
        help="Candidate set-1 features shown to Gemini for each set-2 feature.",
    )
    parser.add_argument(
        "--semantic-confidence-threshold",
        type=float,
        default=0.78,
        help="Minimum Gemini confidence for accepting a semantic match.",
    )
    parser.add_argument(
        "--semantic-matches-json",
        type=Path,
        help="Path for accepted Gemini semantic matches. Defaults to <output-dir>/<prefix>_semantic_matches.json.",
    )
    parser.add_argument("--top-a", type=positive_int, default=1, help="Start per-category k for the Jaccard line plot.")
    parser.add_argument("--top-b", type=positive_int, default=100, help="End per-category k for the Jaccard line plot.")
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=100,
        help="Default per-category top-k cutoff for the Venn and scatter plots.",
    )
    parser.add_argument("--venn-k", type=positive_int, help="Per-category top-k cutoff for the Venn plot.")
    parser.add_argument(
        "--scatter-k",
        type=positive_int,
        help="Per-category top-k cutoff for the scatter intersection plot.",
    )
    parser.add_argument(
        "--scatter-scale",
        choices=("linear", "log"),
        default="linear",
        help="Scatter axis scale. Log scale plots paper_count + 1.",
    )
    parser.add_argument(
        "--annotate-top",
        type=int,
        default=0,
        help="Number of high-count scatter points to label. Defaults to 0.",
    )
    parser.add_argument(
        "--category-plots",
        dest="category_plots",
        action="store_true",
        default=True,
        help="Write additional per-category versions of each plot.",
    )
    parser.add_argument(
        "--no-category-plots",
        dest="category_plots",
        action="store_false",
        help="Only write the overall plots.",
    )
    parser.add_argument(
        "--categories",
        default="persuasion,moral_framing,sentiment_affect",
        help="Comma-separated categories for per-category plots.",
    )
    parser.add_argument(
        "--features-json",
        type=Path,
        help="Path for the union feature membership JSON. Defaults to <output-dir>/<prefix>_features.json.",
    )
    parser.add_argument(
        "--unique-md",
        type=Path,
        help="Path for the Markdown report of top unique features. Defaults to <output-dir>/<prefix>_unique_top<N>.md.",
    )
    parser.add_argument(
        "--unique-limit",
        type=positive_int,
        default=10,
        help="Number of top set-unique features to include in the Markdown report.",
    )
    parser.add_argument(
        "--final-features-json",
        type=Path,
        help="Path for peak-Jaccard final feature candidates JSON. Defaults to <output-dir>/<prefix>_final_features.json.",
    )
    parser.add_argument(
        "--final-features-csv",
        type=Path,
        help="Path for peak-Jaccard final feature candidates CSV. Defaults to <output-dir>/<prefix>_final_features.csv.",
    )
    parser.add_argument(
        "--final-category-k",
        action="append",
        default=[],
        metavar="CATEGORY=K",
        help=(
            "Override the final feature export k for one category, e.g. "
            "--final-category-k sentiment=75. Can be repeated or comma-separated."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_plots"), help="Directory for PNGs.")
    parser.add_argument("--prefix", help="Filename prefix for generated PNGs.")
    parser.add_argument("--dpi", type=positive_int, default=200, help="Output image DPI.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.top_a > args.top_b:
        raise ValueError("--top-a must be less than or equal to --top-b.")
    if args.annotate_top < 0:
        raise ValueError("--annotate-top must be 0 or greater.")
    if not 0.0 <= args.semantic_confidence_threshold <= 1.0:
        raise ValueError("--semantic-confidence-threshold must be between 0 and 1.")
    if args.gemini_timeout <= 0:
        raise ValueError("--gemini-timeout must be greater than 0.")
    parse_final_k_overrides(args.final_category_k)
    if not args.feature_set_1.exists():
        raise FileNotFoundError(f"Feature set 1 does not exist: {args.feature_set_1}")
    if not args.feature_set_2.exists():
        raise FileNotFoundError(f"Feature set 2 does not exist: {args.feature_set_2}")


def write_plot_group(
    *,
    feature_set_1: FeatureSet,
    feature_set_2: FeatureSet,
    output_dir: Path,
    prefix: str,
    top_a: int,
    top_b: int,
    venn_k: int,
    scatter_k: int,
    dpi: int,
    annotate_top: int,
    scatter_scale: str,
    title_prefix: str = "",
) -> list[Path]:
    jaccard_path = output_dir / f"{prefix}_jaccard_top_{top_a}_{top_b}.png"
    venn_path = output_dir / f"{prefix}_venn_top_{venn_k}.png"
    scatter_path = output_dir / f"{prefix}_scatter_top_{scatter_k}.png"

    plot_jaccard(
        jaccard_series(feature_set_1, feature_set_2, top_a, top_b),
        feature_set_1,
        feature_set_2,
        jaccard_path,
        dpi,
        title_prefix=title_prefix,
    )
    plot_venn(
        venn_counts(feature_set_1, feature_set_2, venn_k),
        feature_set_1,
        feature_set_2,
        venn_k,
        venn_path,
        dpi,
        title_prefix=title_prefix,
    )
    plot_scatter(
        scatter_points(feature_set_1, feature_set_2, scatter_k),
        feature_set_1,
        feature_set_2,
        scatter_k,
        scatter_path,
        dpi,
        annotate_top,
        scale=scatter_scale,
        title_prefix=title_prefix,
    )
    return [jaccard_path, venn_path, scatter_path]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
        feature_set_1 = load_feature_set(
            args.feature_set_1,
            args.label_1,
            args.match_scope,
            args.match_normalization,
        )
        feature_set_2 = load_feature_set(
            args.feature_set_2,
            args.label_2,
            args.match_scope,
            args.match_normalization,
        )
        if not feature_set_1.records:
            raise ValueError(f"No usable feature rows found in {args.feature_set_1}.")
        if not feature_set_2.records:
            raise ValueError(f"No usable feature rows found in {args.feature_set_2}.")

        venn_k = args.venn_k or args.top_k
        scatter_k = args.scatter_k or args.top_k
        prefix = args.prefix or (
            f"{sanitize_filename_part(feature_set_1.label)}_vs_{sanitize_filename_part(feature_set_2.label)}"
        )
        category_specs = parse_category_specs(args.categories)
        final_k_overrides = parse_final_k_overrides(args.final_category_k)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        output_paths: list[Path] = []
        if args.semantic_match == "gemini":
            semantic_matches = build_semantic_matches(
                feature_set_1,
                feature_set_2,
                api_key=gemini_api_key(args.gemini_api_key),
                model=args.gemini_model,
                base_url=args.gemini_base_url,
                timeout=args.gemini_timeout,
                match_scope=args.match_scope,
                candidate_limit=args.gemini_candidate_limit,
                batch_size=args.gemini_batch_size,
                confidence_threshold=args.semantic_confidence_threshold,
            )
            semantic_matches_path = (
                args.semantic_matches_json or args.output_dir / f"{prefix}_semantic_matches.json"
            )
            semantic_matches_path.parent.mkdir(parents=True, exist_ok=True)
            output_paths.append(
                write_semantic_matches_json(
                    semantic_matches_path,
                    semantic_matches,
                    model=args.gemini_model,
                    confidence_threshold=args.semantic_confidence_threshold,
                )
            )
            feature_set_2 = apply_semantic_matches(feature_set_2, semantic_matches)

        features_json_path = args.features_json or args.output_dir / f"{prefix}_features.json"
        features_json_path.parent.mkdir(parents=True, exist_ok=True)
        unique_md_path = args.unique_md or args.output_dir / f"{prefix}_unique_top{args.unique_limit}.md"
        unique_md_path.parent.mkdir(parents=True, exist_ok=True)
        final_features_json_path = args.final_features_json or args.output_dir / f"{prefix}_final_features.json"
        final_features_csv_path = args.final_features_csv or args.output_dir / f"{prefix}_final_features.csv"
        final_features_json_path.parent.mkdir(parents=True, exist_ok=True)
        final_features_csv_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths.extend(
            [
                write_feature_union_json(
                    features_json_path,
                    feature_set_1,
                    feature_set_2,
                    args.match_scope,
                    args.match_normalization,
                ),
                write_unique_features_md(
                    unique_md_path,
                    feature_set_1,
                    feature_set_2,
                    args.match_scope,
                    args.match_normalization,
                    args.unique_limit,
                ),
            ]
        )
        output_paths.extend(
            write_final_feature_set(
                final_features_json_path,
                final_features_csv_path,
                feature_set_1,
                feature_set_2,
                args.match_scope,
                args.match_normalization,
                args.top_a,
                args.top_b,
                category_specs,
                final_k_overrides,
            )
        )

        output_paths.extend(
            write_plot_group(
                feature_set_1=feature_set_1,
                feature_set_2=feature_set_2,
                output_dir=args.output_dir,
                prefix=prefix,
                top_a=args.top_a,
                top_b=args.top_b,
                venn_k=venn_k,
                scatter_k=scatter_k,
                dpi=args.dpi,
                annotate_top=args.annotate_top,
                scatter_scale=args.scatter_scale,
            )
        )

        if args.category_plots:
            for spec in category_specs:
                category_set_1 = filter_feature_set_by_category(feature_set_1, spec.key)
                category_set_2 = filter_feature_set_by_category(feature_set_2, spec.key)
                if not category_set_1.records and not category_set_2.records:
                    continue
                output_paths.extend(
                    write_plot_group(
                        feature_set_1=category_set_1,
                        feature_set_2=category_set_2,
                        output_dir=args.output_dir,
                        prefix=f"{prefix}_{spec.slug}",
                        top_a=args.top_a,
                        top_b=args.top_b,
                        venn_k=venn_k,
                        scatter_k=scatter_k,
                        dpi=args.dpi,
                        annotate_top=args.annotate_top,
                        scatter_scale=args.scatter_scale,
                        title_prefix=spec.label,
                    )
                )
    except Exception as exc:
        print(f"comparison_plots: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(feature_set_1.records)} features from {feature_set_1.path}")
    print(f"Loaded {len(feature_set_2.records)} features from {feature_set_2.path}")
    for output_path in output_paths:
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
