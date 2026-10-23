"""Embeddings similarity."""

from __future__ import annotations


import copy

import json


from dataclasses import dataclass

from pathlib import Path

from typing import Any, Iterable

from lit_review.profiles.embedding_cleanup import EMBEDDING_TEXT_CLEANUP_BUILTINS
from lit_review.embeddings.core import (
    EmbeddingTextCleanup,
    FeatureItem,
    parse_jsonish_list,
    parse_text_fields,
    row_field_value,
)


MERGED_FIELD_PRIORITY = [
    "feature_key",
    "category",
    "feature_name",
    "normalized_feature_name",
    "parent_categories",
    "definitions",
    "synonyms",
    "examples",
    "paper_count",
    "mention_count",
    "source_paper_ids",
    "source_titles",
    "source_feature_names",
    "source_normalized_feature_names",
    "evidence_types",
    "annotation",
    "embedding_merge",
]


AGGREGATE_LIST_FIELDS = [
    "parent_categories",
    "definitions",
    "synonyms",
    "examples",
    "source_paper_ids",
    "source_titles",
    "source_feature_names",
    "source_normalized_feature_names",
    "evidence_types",
]


AGGREGATE_ANNOTATION_LIST_FIELDS = [
    "annotation.synonyms",
    "annotation.examples",
]


@dataclass(frozen=True)
class SimilarityPair:
    left_index: int
    right_index: int
    similarity: float


@dataclass(frozen=True)
class MergeGroup:
    group_id: str
    member_indices: list[int]
    canonical_index: int
    pairs: list[SimilarityPair]


@dataclass(frozen=True)
class CategoryMergeSpec:
    text_fields: list[str]
    threshold: float
    linkage: str = "complete"
    embedding_text_cleanup: EmbeddingTextCleanup | None = None


def validate_threshold_number(value: float, label: str = "threshold") -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{label} must be greater than 0 and at most 1.")


def load_category_merge_specs(path: Path) -> dict[str, CategoryMergeSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        isinstance(payload, dict)
        and set(payload) == {"categories"}
        and isinstance(payload.get("categories"), dict)
    ):
        payload = payload["categories"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must contain a non-empty JSON object keyed by category.")

    specs: dict[str, CategoryMergeSpec] = {}
    for raw_category, raw_spec in payload.items():
        category = str(raw_category).strip()
        if not category:
            raise ValueError(f"{path} contains an empty category name.")
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Category {category!r} in {path} must map to a JSON object.")
        if "text_fields" not in raw_spec:
            raise ValueError(f"Category {category!r} in {path} is missing text_fields.")
        if "threshold" not in raw_spec:
            raise ValueError(f"Category {category!r} in {path} is missing threshold.")
        fields = parse_text_fields(raw_spec["text_fields"])
        try:
            threshold = float(raw_spec["threshold"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Category {category!r} threshold must be a number.") from exc
        validate_threshold_number(threshold, f"threshold for category {category!r}")
        linkage = str(raw_spec.get("linkage") or "complete").strip().lower()
        if linkage not in {"single", "complete"}:
            raise ValueError(f"Category {category!r} linkage must be 'single' or 'complete'.")
        cleanup = parse_embedding_text_cleanup(raw_spec.get("embedding_text_cleanup"), path, category)
        specs[category] = CategoryMergeSpec(
            text_fields=fields,
            threshold=threshold,
            linkage=linkage,
            embedding_text_cleanup=cleanup,
        )
    return specs


def parse_embedding_text_cleanup(raw_cleanup: Any, path: Path, category: str) -> EmbeddingTextCleanup | None:
    if raw_cleanup in (None, ""):
        return None
    if not isinstance(raw_cleanup, dict):
        raise ValueError(f"Category {category!r} embedding_text_cleanup in {path} must be a JSON object.")

    raw_builtins = raw_cleanup.get("strip_builtins", [])
    if not isinstance(raw_builtins, list):
        raise ValueError(f"Category {category!r} embedding_text_cleanup.strip_builtins must be a list.")
    strip_builtins = tuple(str(item).strip() for item in raw_builtins if str(item).strip())
    unknown_builtins = sorted(set(strip_builtins) - set(EMBEDDING_TEXT_CLEANUP_BUILTINS))
    if unknown_builtins:
        raise ValueError(
            f"Category {category!r} embedding_text_cleanup has unknown strip_builtins: "
            + ", ".join(unknown_builtins)
        )

    raw_min_chars = raw_cleanup.get("min_chars", 0)
    try:
        min_chars = int(raw_min_chars)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Category {category!r} embedding_text_cleanup.min_chars must be an integer."
        ) from exc
    if min_chars < 0:
        raise ValueError(f"Category {category!r} embedding_text_cleanup.min_chars must be non-negative.")

    fallback = str(raw_cleanup.get("fallback") or "original").strip().lower()
    if fallback not in {"original", "empty"}:
        raise ValueError(
            f"Category {category!r} embedding_text_cleanup.fallback must be 'original' or 'empty'."
        )

    return EmbeddingTextCleanup(
        strip_builtins=strip_builtins,
        min_chars=min_chars,
        fallback=fallback,
    )


def normalized_matrix(embeddings: list[list[float]]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("NumPy is required for embedding merge similarity search.") from exc

    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Embeddings must be a non-empty 2D matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def category_merge_summary(
    items: list[FeatureItem],
    groups: list[MergeGroup],
    *,
    all_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    input_counts: dict[str, int] = {}
    if all_rows is None:
        for item in items:
            input_counts[item.category] = input_counts.get(item.category, 0) + 1
    else:
        for row in all_rows:
            category = str(row.get("category") or "").strip() or "uncategorized"
            input_counts[category] = input_counts.get(category, 0) + 1

    merged_away_counts = {category: 0 for category in input_counts}
    for group in groups:
        for index in group.member_indices:
            if index == group.canonical_index:
                continue
            category = items[index].category
            merged_away_counts[category] = merged_away_counts.get(category, 0) + 1

    return [
        {
            "category": category,
            "input_features": input_counts[category],
            "retained_features": input_counts[category] - merged_away_counts.get(category, 0),
            "merged_away_features": merged_away_counts.get(category, 0),
        }
        for category in sorted(input_counts, key=lambda value: (-input_counts[value], value.casefold()))
    ]


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def build_merge_groups(
    items: list[FeatureItem],
    pairs: list[SimilarityPair],
    *,
    linkage: str = "single",
) -> list[MergeGroup]:
    if linkage == "single":
        return build_single_link_merge_groups(items, pairs)
    if linkage == "complete":
        return build_complete_link_merge_groups(items, pairs)
    raise ValueError("linkage must be 'single' or 'complete'.")


def build_single_link_merge_groups(items: list[FeatureItem], pairs: list[SimilarityPair]) -> list[MergeGroup]:
    dsu = DisjointSet(len(items))
    for pair in pairs:
        dsu.union(pair.left_index, pair.right_index)

    components: dict[int, list[int]] = {}
    for index in range(len(items)):
        components.setdefault(dsu.find(index), []).append(index)

    pair_lookup: dict[frozenset[int], SimilarityPair] = {
        frozenset((pair.left_index, pair.right_index)): pair for pair in pairs
    }
    raw_groups: list[tuple[int, list[int], list[SimilarityPair]]] = []
    for member_indices in components.values():
        if len(member_indices) < 2:
            continue
        group_pairs = [
            pair
            for pair in pair_lookup.values()
            if pair.left_index in member_indices and pair.right_index in member_indices
        ]
        canonical_index = choose_canonical_index(items, member_indices)
        ordered_members = sorted(
            member_indices,
            key=lambda index: (0 if index == canonical_index else 1, member_sort_key(items[index])),
        )
        raw_groups.append((canonical_index, ordered_members, sorted_group_pairs(items, group_pairs)))

    return indexed_merge_groups(items, raw_groups)


def build_complete_link_merge_groups(
    items: list[FeatureItem], pairs: list[SimilarityPair]
) -> list[MergeGroup]:
    pair_lookup: dict[frozenset[int], SimilarityPair] = {
        frozenset((pair.left_index, pair.right_index)): pair for pair in pairs
    }
    clusters: list[list[int]] = []
    for item_index in sorted(range(len(items)), key=lambda index: member_sort_key(items[index])):
        compatible_clusters = [
            cluster
            for cluster in clusters
            if all(frozenset((item_index, member_index)) in pair_lookup for member_index in cluster)
        ]
        if not compatible_clusters:
            clusters.append([item_index])
            continue
        cluster = sorted(
            compatible_clusters,
            key=lambda members: (
                -len(members),
                member_sort_key(items[choose_canonical_index(items, members)]),
            ),
        )[0]
        cluster.append(item_index)

    raw_groups: list[tuple[int, list[int], list[SimilarityPair]]] = []
    for member_indices in clusters:
        if len(member_indices) < 2:
            continue
        canonical_index = choose_canonical_index(items, member_indices)
        ordered_members = sorted(
            member_indices,
            key=lambda index: (0 if index == canonical_index else 1, member_sort_key(items[index])),
        )
        group_pairs = [
            pair_lookup[frozenset((left_index, right_index))]
            for offset, left_index in enumerate(member_indices)
            for right_index in member_indices[offset + 1 :]
        ]
        raw_groups.append((canonical_index, ordered_members, sorted_group_pairs(items, group_pairs)))
    return indexed_merge_groups(items, raw_groups)


def indexed_merge_groups(
    items: list[FeatureItem],
    raw_groups: list[tuple[int, list[int], list[SimilarityPair]]],
) -> list[MergeGroup]:
    raw_groups.sort(key=lambda item: member_sort_key(items[item[0]]))
    return [
        MergeGroup(
            group_id=f"merge_group_{index:04d}",
            canonical_index=canonical_index,
            member_indices=member_indices,
            pairs=group_pairs,
        )
        for index, (canonical_index, member_indices, group_pairs) in enumerate(raw_groups, start=1)
    ]


def choose_canonical_index(items: list[FeatureItem], member_indices: Iterable[int]) -> int:
    return sorted(
        member_indices,
        key=lambda index: (
            -items[index].paper_count,
            -items[index].mention_count,
            len(items[index].label),
            items[index].row_index,
        ),
    )[0]


def member_sort_key(item: FeatureItem) -> tuple[int, int, int, str, int]:
    return (
        -item.paper_count,
        -item.mention_count,
        len(item.label),
        item.label.casefold(),
        item.row_index,
    )


def sorted_group_pairs(items: list[FeatureItem], pairs: list[SimilarityPair]) -> list[SimilarityPair]:
    return sorted(
        pairs,
        key=lambda pair: (
            -pair.similarity,
            items[pair.left_index].label.casefold(),
            items[pair.right_index].label.casefold(),
        ),
    )


def pair_json(pair: SimilarityPair, items: list[FeatureItem]) -> dict[str, Any]:
    left = items[pair.left_index]
    right = items[pair.right_index]
    return {
        "similarity": pair.similarity,
        "feature_key_1": left.feature_key,
        "feature_name_1": left.label,
        "category_1": left.category,
        "row_index_1": left.row_index,
        "paper_count_1": left.paper_count,
        "mention_count_1": left.mention_count,
        "feature_key_2": right.feature_key,
        "feature_name_2": right.label,
        "category_2": right.category,
        "row_index_2": right.row_index,
        "paper_count_2": right.paper_count,
        "mention_count_2": right.mention_count,
    }


def group_threshold(
    group: MergeGroup,
    items: list[FeatureItem],
    threshold: float | dict[str, float],
) -> float:
    if isinstance(threshold, dict):
        categories = {items[index].category for index in group.member_indices}
        if len(categories) != 1:
            raise ValueError("Per-category thresholds cannot be applied to a cross-category merge group.")
        return threshold[next(iter(categories))]
    return threshold


def group_json(
    group: MergeGroup,
    items: list[FeatureItem],
    threshold: float | dict[str, float],
    scope: str,
) -> dict[str, Any]:
    canonical = items[group.canonical_index]
    members = [items[index] for index in group.member_indices]
    pair_rows = [pair_json(pair, items) for pair in group.pairs]
    similarities = [pair.similarity for pair in group.pairs]
    return {
        "group_id": group.group_id,
        "threshold": group_threshold(group, items, threshold),
        "scope": scope,
        "canonical_feature_key": canonical.feature_key,
        "canonical_feature_name": canonical.label,
        "canonical_row_index": canonical.row_index,
        "member_count": len(members),
        "member_row_indices": [member.row_index for member in members],
        "member_feature_keys": [member.feature_key for member in members],
        "member_feature_names": [member.label for member in members],
        "member_categories": sorted({member.category for member in members}),
        "similarity_summary": similarity_summary(similarities),
        "pairs": pair_rows,
        "members": [
            {
                "row_index": member.row_index,
                "feature_key": member.feature_key,
                "feature_name": member.label,
                "category": member.category,
                "paper_count": member.paper_count,
                "mention_count": member.mention_count,
            }
            for member in members
        ],
    }


def similarity_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"edge_count": 0, "max": 0.0, "min": 0.0, "avg": 0.0}
    return {
        "edge_count": len(values),
        "max": round(max(values), 8),
        "min": round(min(values), 8),
        "avg": round(sum(values) / len(values), 8),
    }


def build_merged_features(
    items: list[FeatureItem],
    groups: list[MergeGroup],
    *,
    threshold: float | dict[str, float],
    scope: str,
    all_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    group_by_member: dict[int, MergeGroup] = {
        member_index: group for group in groups for member_index in group.member_indices
    }
    output: list[dict[str, Any]] = []
    emitted_groups: set[str] = set()
    item_index_by_row_index = {item.row_index: item_index for item_index, item in enumerate(items)}
    rows: Iterable[tuple[int, dict[str, Any]]]
    if all_rows is None:
        rows = ((item.row_index, item.raw_row) for item in items)
    else:
        rows = enumerate(all_rows, start=1)
    for row_index, raw_row in rows:
        index = item_index_by_row_index.get(row_index)
        if index is None:
            output.append(copy.deepcopy(raw_row))
            continue
        item = items[index]
        group = group_by_member.get(index)
        if group is None:
            row = copy.deepcopy(item.raw_row)
            row.setdefault("feature_key", item.feature_key)
            output.append(row)
            continue
        if group.group_id in emitted_groups or index != group.canonical_index:
            continue
        output.append(merged_group_row(items, group, threshold=threshold, scope=scope))
        emitted_groups.add(group.group_id)
    return output


def merged_group_row(
    items: list[FeatureItem],
    group: MergeGroup,
    *,
    threshold: float | dict[str, float],
    scope: str,
) -> dict[str, Any]:
    canonical = items[group.canonical_index]
    member_items = [items[index] for index in group.member_indices]
    member_rows = [item.raw_row for item in member_items]
    row = copy.deepcopy(canonical.raw_row)
    row.setdefault("feature_key", canonical.feature_key)

    for field in AGGREGATE_LIST_FIELDS:
        values = unique_text_values(row_field_value(member_row, field) for member_row in member_rows)
        if values:
            row[field] = values

    for field in AGGREGATE_ANNOTATION_LIST_FIELDS:
        values = unique_text_values(row_field_value(member_row, field) for member_row in member_rows)
        if values:
            set_nested_or_flat_annotation_value(row, field, values)

    source_paper_ids = unique_text_values(
        row_field_value(member_row, "source_paper_ids") for member_row in member_rows
    )
    row["paper_count"] = (
        len(source_paper_ids) if source_paper_ids else max(item.paper_count for item in member_items)
    )
    row["mention_count"] = sum(item.mention_count for item in member_items)
    if isinstance(row.get("annotation"), dict):
        row["annotation"]["paper_count"] = row["paper_count"]
        row["annotation"]["mention_count"] = row["mention_count"]

    applied_threshold = group_threshold(group, items, threshold)
    group_payload = group_json(group, items, threshold=applied_threshold, scope=scope)
    row["embedding_merge"] = {
        "group_id": group.group_id,
        "threshold": applied_threshold,
        "scope": scope,
        "canonical_feature_key": canonical.feature_key,
        "canonical_feature_name": canonical.label,
        "member_count": group_payload["member_count"],
        "member_row_indices": group_payload["member_row_indices"],
        "member_feature_keys": group_payload["member_feature_keys"],
        "member_feature_names": group_payload["member_feature_names"],
        "member_categories": group_payload["member_categories"],
        "similarity_summary": group_payload["similarity_summary"],
        "pairs": group_payload["pairs"],
    }
    return row


def unique_text_values(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in parse_jsonish_list(value):
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
    return output


def set_nested_or_flat_annotation_value(row: dict[str, Any], field: str, value: list[str]) -> None:
    flat_field = field.replace(".", "_")
    if isinstance(row.get("annotation"), dict) or flat_field not in row:
        annotation = row.get("annotation")
        if not isinstance(annotation, dict):
            annotation = {}
            row["annotation"] = annotation
        annotation[field.split(".", 1)[1]] = value
    if flat_field in row:
        row[flat_field] = value


def merged_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for row in rows:
        fields.update(row)
    ordered = [field for field in MERGED_FIELD_PRIORITY if field in fields]
    ordered.extend(sorted(field for field in fields if field not in set(ordered)))
    return ordered
