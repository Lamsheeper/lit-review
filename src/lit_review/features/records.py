"""Features records."""

from __future__ import annotations
from lit_review.core.text import normalize_feature_name

from lit_review.core.context import TAXONOMY, PROFILE
from lit_review.core.taxonomy import taxonomy_category_lookup


import csv


import json


import re


from collections import Counter


from pathlib import Path

from typing import Any


from lit_review.core.taxonomy import normalize_family_key


FEATURE_INVENTORY_JSON_FIELDS = {
    "annotation",
    "embedding_merge",
    "score",
    "authors",
    "pages",
    "chunk_ids",
    "parent_categories",
    "definitions",
    "synonyms",
    "examples",
    "source_paper_ids",
    "source_titles",
    "source_feature_names",
    "source_normalized_feature_names",
    "evidence_types",
    "annotation_synonyms",
    "annotation_examples",
    "score_neighbors",
}


FEATURE_INVENTORY_INT_FIELDS = {
    "year",
    "paper_count",
    "mention_count",
    "member_count",
    "annotation_paper_count",
    "annotation_mention_count",
    "score_literature_support",
    "score_annotation_reliability",
    "score_distinctiveness",
}


FEATURE_INVENTORY_FLOAT_FIELDS = {
    "confidence",
    "source_confidence_avg",
    "source_confidence_max",
    "curation_confidence",
    "annotation_confidence",
    "score_total",
    "score_confidence",
}


def canonical_category(value: str | None, taxonomy_lookup: dict[str, str] | None = None) -> str | None:
    if taxonomy_lookup is None and PROFILE.get() == "media_attributes":
        from lit_review.profiles.media_extraction import canonical_category as media_category

        return media_category(value)
    if taxonomy_lookup is None and TAXONOMY.get():
        taxonomy_lookup = taxonomy_category_lookup(TAXONOMY.get())
    return taxonomy_lookup.get(normalize_family_key(value)) if taxonomy_lookup is not None else None


def canonical_evidence_type(value: str | None, item: dict[str, Any] | None = None) -> str:
    if value:
        normalized = normalize_feature_name(value)
        evidence_map = {
            "definition": "definition",
            "defined": "definition",
            "taxonomy": "taxonomy_row",
            "taxonomy row": "taxonomy_row",
            "taxonomy label": "taxonomy_row",
            "label": "taxonomy_row",
            "annotation label": "taxonomy_row",
            "example": "example",
            "example usage": "example",
            "named mention": "named_mention",
            "mention": "named_mention",
            "name": "named_mention",
        }
        if normalized in evidence_map:
            return evidence_map[normalized]
    item = item or {}
    if item.get("definition"):
        return "definition"
    if item.get("examples"):
        return "example"
    if item.get("parent_category") or item.get("synonyms"):
        return "taxonomy_row"
    return "named_mention"


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, tuple):
        raw_values = list(value)
    elif isinstance(value, str):
        if not value.strip():
            return []
        raw_values = [value]
    else:
        raw_values = [value]
    output: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            output.append(text)
    return output


def truncate_text(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 15)].rstrip() + " [truncated]"


def trim_source_quote(value: Any, max_chars: int = 500) -> str:
    return truncate_text(str(value or ""), max_chars)


def normalize_feature_item(
    item: dict[str, Any],
    *,
    paper: dict[str, Any],
    chunk: dict[str, Any],
    run_id: str,
    model: str,
    extracted_at: str,
    taxonomy_lookup: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    category = canonical_category(str(item.get("category") or ""), taxonomy_lookup)
    feature_name = str(item.get("feature_name") or item.get("name") or item.get("label") or "").strip()
    normalized_feature = normalize_feature_name(feature_name)
    if not category or not normalized_feature:
        return None
    if taxonomy_lookup is None and PROFILE.get() == "media_attributes":
        from lit_review.profiles.media_extraction import is_taxonomy_relevant

        if not is_taxonomy_relevant(item, chunk_text=str(chunk.get("text") or ""), paper=paper):
            return None

    confidence = item.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.5
    confidence_value = max(0.0, min(1.0, confidence_value))

    evidence_type = canonical_evidence_type(str(item.get("evidence_type") or ""), item=item)
    source_quote = trim_source_quote(item.get("source_quote") or item.get("quote") or "")
    snippet = trim_source_quote(item.get("snippet") or source_quote or chunk.get("text") or "", 700)

    return {
        "paper_id": str(paper.get("paper_id") or chunk.get("paper_id") or ""),
        "title": str(paper.get("title") or chunk.get("title") or ""),
        "authors": coerce_str_list(paper.get("authors")),
        "year": paper.get("year"),
        "doi": paper.get("doi"),
        "venue": paper.get("venue"),
        "category": category,
        "feature_name": feature_name,
        "normalized_feature_name": normalized_feature,
        "parent_category": str(item.get("parent_category") or "").strip(),
        "definition": str(item.get("definition") or "").strip(),
        "synonyms": coerce_str_list(item.get("synonyms")),
        "examples": coerce_str_list(item.get("examples")),
        "evidence_type": evidence_type,
        "chunk_id": str(chunk.get("chunk_id") or item.get("chunk_id") or ""),
        "pages": coerce_pages(chunk.get("pages") or item.get("pages")),
        "snippet": snippet,
        "source_quote": source_quote,
        "confidence": round(confidence_value, 4),
        "notes": str(item.get("notes") or "").strip(),
        "model": model,
        "run_id": run_id,
        "extracted_at": extracted_at,
    }


def coerce_pages(value: Any) -> list[int]:
    pages: list[int] = []
    values = value if isinstance(value, list) else [value]
    for raw in values:
        if raw is None or raw == "":
            continue
        try:
            page = int(raw)
        except (TypeError, ValueError):
            continue
        if page not in pages:
            pages.append(page)
    return pages


def parse_jsonish_csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] in '[{"' or stripped in {"null", "true", "false"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return stripped


def csv_int_value(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return 0


def csv_float_value(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def rebuild_annotation_from_flat_fields(row: dict[str, Any]) -> None:
    if isinstance(row.get("annotation"), dict):
        return
    mapping = {
        "annotation_status": "status",
        "annotation_formal_definition": "formal_definition",
        "annotation_short_definition": "short_definition",
        "annotation_synonyms": "synonyms",
        "annotation_examples": "examples",
        "annotation_paper_count": "paper_count",
        "annotation_mention_count": "mention_count",
        "annotation_confidence": "confidence",
        "annotation_definition_basis": "definition_basis",
        "annotation_notes": "notes",
        "annotation_model": "model",
        "annotation_run_id": "run_id",
        "annotation_annotated_at": "annotated_at",
    }
    annotation: dict[str, Any] = {}
    for csv_field, annotation_field in mapping.items():
        value = row.get(csv_field)
        if value in (None, "", [], {}):
            continue
        annotation[annotation_field] = value
    if annotation:
        row["annotation"] = annotation


def rebuild_score_from_flat_fields(row: dict[str, Any]) -> None:
    if isinstance(row.get("score"), dict):
        return
    mapping = {
        "score_literature_support": "literature_support",
        "score_literature_support_rationale": "literature_support_rationale",
        "score_annotation_reliability": "annotation_reliability",
        "score_annotation_reliability_rationale": "annotation_reliability_rationale",
        "score_distinctiveness": "distinctiveness",
        "score_distinctiveness_rationale": "distinctiveness_rationale",
        "score_total": "total",
        "score_confidence": "confidence",
        "score_notes": "notes",
        "score_model": "model",
        "score_run_id": "run_id",
        "score_scored_at": "scored_at",
    }
    score: dict[str, Any] = {}
    for csv_field, score_field in mapping.items():
        value = row.get(csv_field)
        if value in (None, "", [], {}):
            continue
        score[score_field] = value
    if score:
        row["score"] = score


def read_csv_feature_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    list_json_fields = FEATURE_INVENTORY_JSON_FIELDS - {"annotation", "embedding_merge", "score"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for field, raw_value in raw_row.items():
                if field is None:
                    continue
                if field in FEATURE_INVENTORY_JSON_FIELDS:
                    value = parse_jsonish_csv_cell(raw_value)
                    if field in list_json_fields and isinstance(value, str):
                        value = coerce_str_list(value)
                    row[field] = value
                elif field in FEATURE_INVENTORY_INT_FIELDS:
                    row[field] = csv_int_value(raw_value)
                elif field in FEATURE_INVENTORY_FLOAT_FIELDS:
                    row[field] = csv_float_value(raw_value)
                else:
                    row[field] = "" if raw_value is None else raw_value
            rebuild_annotation_from_flat_fields(row)
            rebuild_score_from_flat_fields(row)
            rows.append(ensure_feature_identity(row))
    return rows


def read_feature_inventory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Feature inventory does not exist: {path}")
    if path.suffix.lower() == ".csv":
        return read_csv_feature_inventory(path)
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        return [ensure_feature_identity(row) for row in read_jsonl_rows_from_text(path, raw)]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [ensure_feature_identity(row) for row in read_jsonl_rows_from_text(path, raw)]
    return [ensure_feature_identity(row) for row in coerce_feature_inventory_rows(parsed, path)]


def read_jsonl_rows_from_text(path: Path, raw: str) -> list[dict[str, Any]]:
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


def coerce_feature_inventory_rows(parsed: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return coerce_feature_inventory_row_list(parsed, path)
    if isinstance(parsed, dict):
        for key in ("features", "rows", "items", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return coerce_feature_inventory_row_list(value, path)
        if any(key in parsed for key in ("feature_name", "normalized_feature_name", "feature_key")):
            return [parsed]
    raise ValueError(
        f"{path} must be JSONL, a JSON list, or a JSON object with feature rows "
        "under 'features', 'rows', 'items', or 'data'."
    )


def coerce_feature_inventory_row_list(values: list[Any], path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"{path} item {index} must be a JSON object row.")
        rows.append(value)
    return rows


def load_goal_text(goal_path: Path | None, max_chars: int = 12000) -> tuple[str, str | None]:
    if goal_path is None:
        return "", None
    goal_path = goal_path.expanduser()
    if not goal_path.exists():
        raise FileNotFoundError(f"Goal file does not exist: {goal_path}")
    return truncate_text(goal_path.read_text(encoding="utf-8", errors="replace"), max_chars), str(goal_path)


def choose_feature_name(rows: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        name = str(row.get("feature_name") or "").strip()
        if name:
            counts[name] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def unique_values(rows: list[dict[str, Any]], field: str) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None or item == "":
                continue
            key = json.dumps(item, sort_keys=True) if isinstance(item, (list, dict)) else str(item)
            if key not in seen:
                seen.add(key)
                output.append(item)
    return output


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_rank = {
        "definition": 4,
        "taxonomy_row": 3,
        "example": 2,
        "named_mention": 1,
    }
    return max(
        rows,
        key=lambda row: (
            evidence_rank.get(str(row.get("evidence_type")), 0),
            float(row.get("confidence") or 0.0),
            len(str(row.get("definition") or "")),
            len(str(row.get("source_quote") or "")),
        ),
    )


def aggregate_paper_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("paper_id") or ""),
            str(row.get("category") or ""),
            str(row.get("normalized_feature_name") or ""),
        )
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        chosen = best_row(group)
        output.append(
            {
                "paper_id": chosen.get("paper_id"),
                "title": chosen.get("title"),
                "authors": chosen.get("authors") or [],
                "year": chosen.get("year"),
                "doi": chosen.get("doi"),
                "venue": chosen.get("venue"),
                "category": chosen.get("category"),
                "feature_name": choose_feature_name(group),
                "normalized_feature_name": chosen.get("normalized_feature_name"),
                "parent_category": chosen.get("parent_category"),
                "definition": chosen.get("definition"),
                "synonyms": unique_values(group, "synonyms"),
                "examples": unique_values(group, "examples"),
                "evidence_types": unique_values(group, "evidence_type"),
                "mention_count": len(group),
                "chunk_ids": unique_values(group, "chunk_id"),
                "pages": sorted({page for row in group for page in coerce_pages(row.get("pages"))}),
                "best_snippet": chosen.get("snippet"),
                "best_source_quote": chosen.get("source_quote"),
                "confidence": round(
                    sum(float(row.get("confidence") or 0.0) for row in group) / len(group),
                    4,
                ),
                "notes": "; ".join(unique_values(group, "notes")),
                "model": chosen.get("model"),
                "run_id": chosen.get("run_id"),
                "extracted_at": chosen.get("extracted_at"),
            }
        )
    return output


def aggregate_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("category") or ""),
            str(row.get("normalized_feature_name") or ""),
        )
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        chosen = best_row(group)
        paper_ids = unique_values(group, "paper_id")
        output.append(
            {
                "category": chosen.get("category"),
                "feature_name": choose_feature_name(group),
                "normalized_feature_name": chosen.get("normalized_feature_name"),
                "parent_categories": unique_values(group, "parent_category"),
                "definitions": unique_values(group, "definition"),
                "synonyms": unique_values(group, "synonyms"),
                "examples": unique_values(group, "examples"),
                "paper_count": len(paper_ids),
                "mention_count": len(group),
                "source_paper_ids": paper_ids,
                "source_titles": unique_values(group, "title"),
                "evidence_types": unique_values(group, "evidence_type"),
                "confidence": round(
                    sum(float(row.get("confidence") or 0.0) for row in group) / len(group),
                    4,
                ),
                "notes": "; ".join(unique_values(group, "notes")),
                "model": chosen.get("model"),
                "run_id": chosen.get("run_id"),
                "extracted_at": chosen.get("extracted_at"),
            }
        )
    return output


def feature_key(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "").strip()
    normalized = normalize_feature_name(
        str(row.get("normalized_feature_name") or row.get("feature_name") or "")
    )
    return f"{category}::{normalized}"


def ensure_feature_identity(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    normalized = normalize_feature_name(
        str(output.get("normalized_feature_name") or output.get("feature_name") or "")
    )
    output["normalized_feature_name"] = normalized
    output["feature_key"] = f"{output.get('category') or ''}::{normalized}"
    return output


def compact_list(value: Any, limit: int, max_chars: int = 180) -> list[str]:
    return [truncate_text(item, max_chars) for item in coerce_str_list(value)[:limit]]


def compact_feature_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_key": row.get("feature_key") or feature_key(row),
        "category": row.get("category"),
        "feature_name": row.get("feature_name"),
        "normalized_feature_name": row.get("normalized_feature_name"),
        "parent_categories": compact_list(row.get("parent_categories"), 8),
        "definitions": compact_list(row.get("definitions"), 5, max_chars=240),
        "synonyms": compact_list(row.get("synonyms"), 12),
        "examples": compact_list(row.get("examples"), 8),
        "paper_count": row.get("paper_count"),
        "mention_count": row.get("mention_count"),
        "source_titles": compact_list(row.get("source_titles"), 8, max_chars=160),
        "evidence_types": compact_list(row.get("evidence_types"), 8),
        "notes": truncate_text(str(row.get("notes") or ""), 500),
    }
