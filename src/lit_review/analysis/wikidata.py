#!/usr/bin/env python3
"""Analyze Wikidata/Wikipedia mappings for extracted feature inventories."""

from __future__ import annotations
from lit_review.core.text import normalize_feature_name

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_CATEGORY = "persuasion"
DEFAULT_TOP_K = (25, 50, 100, 250, 500, 1000)
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.8
CSV_JOIN = " | "
RELATIONSHIP_ORDER = ("same_article", "different_article")
CLOSENESS_ORDER = ("exact", "close", "broader/narrower", "related", "unknown")
CLOSENESS_RANKS = {
    "exact": 4,
    "close": 3,
    "broader": 2,
    "narrower": 2,
    "related": 1,
}
CLOSENESS_BY_RANK = {
    4: "exact",
    3: "close",
    2: "broader/narrower",
    1: "related",
    0: "unknown",
}


@dataclass(frozen=True)
class ScoredFeature:
    feature_key: str
    category: str
    feature_name: str
    normalized_feature_name: str
    score_total: float
    paper_count: int
    mention_count: int


@dataclass(frozen=True)
class MappedFeature:
    row_index: int
    input_record_id: str
    feature_name: str
    normalized_feature_name: str
    proposal_mode: str
    verified_qids: tuple[str, ...]
    verified_wikidata_labels: tuple[str, ...]
    mapping_types: tuple[str, ...]
    proposal_confidences: tuple[float | None, ...]
    proposal_rationales: tuple[str, ...]
    verification_warnings: tuple[str, ...]
    qid_suggestion_status: str


@dataclass(frozen=True)
class MappingCandidate:
    mapped_row_index: int
    candidate_index: int
    input_record_id: str
    feature_name: str
    normalized_feature_name: str
    proposal_mode: str
    qid: str
    wikidata_label: str
    mapping_type: str
    confidence: float | None
    rationale: str
    warning: str
    qid_suggestion_status: str


@dataclass(frozen=True)
class QidLink:
    qid: str
    wikidata_label: str
    mapping_type: str
    closeness_rank: int


@dataclass(frozen=True)
class EmbeddingItemRef:
    embedding_index: int
    feature_key: str
    feature_name: str
    normalized_feature_name: str
    category: str
    row_index: int
    paper_count: int
    mention_count: int


@dataclass(frozen=True)
class SimilarityPairRow:
    relationship: str
    similarity: float
    feature_key_1: str
    feature_name_1: str
    normalized_feature_name_1: str
    row_index_1: int
    feature_key_2: str
    feature_name_2: str
    normalized_feature_name_2: str
    row_index_2: int
    shared_qids: tuple[str, ...] = ()
    shared_wikidata_labels: tuple[str, ...] = ()
    pair_mapping_closeness: str = ""


def split_multivalue_cell(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    parts = [part.strip() for part in re.split(r"\r?\n|\|", value) if part.strip()]
    return tuple(parts)


def one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_total(row: dict[str, Any]) -> float:
    score = row.get("score")
    if isinstance(score, dict):
        parsed = safe_float(score.get("total"))
        return parsed if parsed is not None else 0.0
    parsed = safe_float(score)
    return parsed if parsed is not None else 0.0


def parse_top_k(value: str | Iterable[int]) -> tuple[int, ...]:
    values: Iterable[str | int]
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = value

    parsed: list[int] = []
    seen: set[int] = set()
    for item in values:
        if item == "":
            continue
        try:
            top_k = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("--top-k must be a comma-separated list of positive integers.") from exc
        if top_k <= 0:
            raise ValueError("--top-k values must be positive integers.")
        if top_k not in seen:
            seen.add(top_k)
            parsed.append(top_k)
    if not parsed:
        raise ValueError("--top-k must include at least one positive integer.")
    return tuple(parsed)


def confidence_text(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4g}"


def percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def read_scored_features(path: Path) -> list[ScoredFeature]:
    rows: list[ScoredFeature] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")

            feature_name = str(payload.get("feature_name") or "").strip()
            normalized_name = str(payload.get("normalized_feature_name") or "").strip()
            rows.append(
                ScoredFeature(
                    feature_key=str(payload.get("feature_key") or "").strip(),
                    category=str(payload.get("category") or "uncategorized").strip() or "uncategorized",
                    feature_name=feature_name,
                    normalized_feature_name=normalize_feature_name(normalized_name or feature_name),
                    score_total=score_total(payload),
                    paper_count=safe_int(payload.get("paper_count")),
                    mention_count=safe_int(payload.get("mention_count")),
                )
            )
    return rows


def read_mapped_features(path: Path) -> list[MappedFeature]:
    rows: list[MappedFeature] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "input_record_id",
            "feature_name",
            "proposal_mode",
            "verified_qids",
            "verified_wikidata_labels",
            "mapping_types",
            "proposal_confidences",
            "proposal_rationales",
            "verification_warnings",
            "qid_suggestion_status",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}.")

        for row_index, row in enumerate(reader, start=1):
            feature_name = str(row.get("feature_name") or "").strip()
            confidence_values = tuple(
                safe_float(value) for value in split_multivalue_cell(row.get("proposal_confidences"))
            )
            rows.append(
                MappedFeature(
                    row_index=row_index,
                    input_record_id=str(row.get("input_record_id") or "").strip(),
                    feature_name=feature_name,
                    normalized_feature_name=normalize_feature_name(feature_name),
                    proposal_mode=str(row.get("proposal_mode") or "").strip(),
                    verified_qids=split_multivalue_cell(row.get("verified_qids")),
                    verified_wikidata_labels=split_multivalue_cell(row.get("verified_wikidata_labels")),
                    mapping_types=split_multivalue_cell(row.get("mapping_types")),
                    proposal_confidences=confidence_values,
                    proposal_rationales=split_multivalue_cell(row.get("proposal_rationales")),
                    verification_warnings=split_multivalue_cell(row.get("verification_warnings")),
                    qid_suggestion_status=str(row.get("qid_suggestion_status") or "").strip(),
                )
            )
    return rows


def indexed_value(values: tuple[Any, ...], index: int, *, repeat_single: bool = False) -> Any:
    if index < len(values):
        return values[index]
    if repeat_single and len(values) == 1:
        return values[0]
    return ""


def explode_mapping_candidates(mapped_rows: Iterable[MappedFeature]) -> list[MappingCandidate]:
    candidates: list[MappingCandidate] = []
    for mapped in mapped_rows:
        candidate_count = max(
            len(mapped.verified_qids),
            len(mapped.verified_wikidata_labels),
            len(mapped.mapping_types),
            len(mapped.proposal_confidences),
            1,
        )
        for index in range(candidate_count):
            qid = str(indexed_value(mapped.verified_qids, index))
            label = str(indexed_value(mapped.verified_wikidata_labels, index))
            mapping_type = str(indexed_value(mapped.mapping_types, index))
            confidence_value = indexed_value(mapped.proposal_confidences, index)
            confidence = confidence_value if isinstance(confidence_value, float) else None
            candidates.append(
                MappingCandidate(
                    mapped_row_index=mapped.row_index,
                    candidate_index=index + 1,
                    input_record_id=mapped.input_record_id,
                    feature_name=mapped.feature_name,
                    normalized_feature_name=mapped.normalized_feature_name,
                    proposal_mode=mapped.proposal_mode,
                    qid=qid,
                    wikidata_label=label,
                    mapping_type=mapping_type,
                    confidence=confidence,
                    rationale=str(indexed_value(mapped.proposal_rationales, index, repeat_single=True)),
                    warning=str(indexed_value(mapped.verification_warnings, index, repeat_single=True)),
                    qid_suggestion_status=mapped.qid_suggestion_status,
                )
            )
    return candidates


def output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "report_md": output_dir / f"{prefix}_report.md",
        "feature_coverage_csv": output_dir / f"{prefix}_feature_coverage.csv",
        "feature_mapping_json": output_dir / f"{prefix}_feature_mapping.json",
        "candidate_mappings_csv": output_dir / f"{prefix}_candidate_mappings.csv",
        "qid_reuse_csv": output_dir / f"{prefix}_qid_reuse.csv",
        "audit_flags_csv": output_dir / f"{prefix}_audit_flags.csv",
        "coverage_plot_png": output_dir / f"{prefix}_coverage_topk.png",
        "mapping_types_plot_png": output_dir / f"{prefix}_mapping_types.png",
        "confidence_histogram_png": output_dir / f"{prefix}_confidence_histogram.png",
        "top_reused_qids_plot_png": output_dir / f"{prefix}_top_reused_qids.png",
    }


def embedding_similarity_output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "embedding_pair_similarities_csv": output_dir / f"{prefix}_embedding_pair_similarities.csv",
        "embedding_similarity_summary_csv": output_dir / f"{prefix}_embedding_similarity_summary.csv",
        "embedding_similarity_relationship_bar_png": (
            output_dir / f"{prefix}_embedding_similarity_relationship_bar.png"
        ),
        "embedding_similarity_relationship_zscore_bar_png": (
            output_dir / f"{prefix}_embedding_similarity_relationship_zscore_bar.png"
        ),
        "embedding_similarity_relationship_boxplot_png": (
            output_dir / f"{prefix}_embedding_similarity_relationship_boxplot.png"
        ),
        "embedding_similarity_relationship_overlay_histogram_png": (
            output_dir / f"{prefix}_embedding_similarity_relationship_overlay_histogram.png"
        ),
        "embedding_similarity_closeness_bar_png": output_dir
        / f"{prefix}_embedding_similarity_closeness_bar.png",
        "embedding_similarity_closeness_zscore_bar_png": (
            output_dir / f"{prefix}_embedding_similarity_closeness_zscore_bar.png"
        ),
    }


def feature_sort_key(feature: ScoredFeature) -> tuple[float, int, int, str]:
    return (
        feature.score_total,
        feature.paper_count,
        feature.mention_count,
        feature.feature_name.casefold(),
    )


def candidate_sort_key(candidate: MappingCandidate) -> tuple[str, int, int]:
    return (candidate.normalized_feature_name, candidate.mapped_row_index, candidate.candidate_index)


def build_top_k_coverage(
    selected_features: list[ScoredFeature],
    mapped_by_name: dict[str, list[MappedFeature]],
    top_k_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    sorted_features = sorted(selected_features, key=feature_sort_key, reverse=True)
    rows: list[dict[str, Any]] = []
    for requested_k in top_k_values:
        top_features = sorted_features[:requested_k]
        covered = sum(1 for feature in top_features if feature.normalized_feature_name in mapped_by_name)
        rows.append(
            {
                "top_k": requested_k,
                "feature_count": len(top_features),
                "covered_feature_count": covered,
                "coverage": percent(covered, len(top_features)),
            }
        )
    return rows


def build_feature_coverage_rows(
    selected_features: list[ScoredFeature],
    mapped_by_name: dict[str, list[MappedFeature]],
    candidates_by_name: dict[str, list[MappingCandidate]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for feature in sorted(selected_features, key=feature_sort_key, reverse=True):
        matched_rows = mapped_by_name.get(feature.normalized_feature_name, [])
        candidates = candidates_by_name.get(feature.normalized_feature_name, [])
        confidences = [candidate.confidence for candidate in candidates if candidate.confidence is not None]
        rows.append(
            {
                "category": feature.category,
                "feature_key": feature.feature_key,
                "feature_name": feature.feature_name,
                "normalized_feature_name": feature.normalized_feature_name,
                "mapped": "yes" if matched_rows else "no",
                "matched_mapped_row_count": str(len(matched_rows)),
                "candidate_count": str(len(candidates)),
                "verified_qids": CSV_JOIN.join(unique_values(candidate.qid for candidate in candidates)),
                "verified_wikidata_labels": CSV_JOIN.join(
                    unique_values(candidate.wikidata_label for candidate in candidates)
                ),
                "mapping_types": CSV_JOIN.join(
                    unique_values(candidate.mapping_type for candidate in candidates)
                ),
                "min_confidence": confidence_text(min(confidences)) if confidences else "",
                "max_confidence": confidence_text(max(confidences)) if confidences else "",
                "score_total": f"{feature.score_total:.4g}",
                "paper_count": str(feature.paper_count),
                "mention_count": str(feature.mention_count),
            }
        )
    return rows


def build_candidate_mapping_rows(
    category: str,
    features_by_name: dict[str, ScoredFeature],
    candidates: list[MappingCandidate],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in sorted(candidates, key=candidate_sort_key):
        feature = features_by_name.get(candidate.normalized_feature_name)
        rows.append(
            {
                "category": category,
                "feature_key": feature.feature_key if feature else "",
                "feature_name": candidate.feature_name,
                "normalized_feature_name": candidate.normalized_feature_name,
                "input_record_id": candidate.input_record_id,
                "mapped_row_index": str(candidate.mapped_row_index),
                "candidate_index": str(candidate.candidate_index),
                "qid": candidate.qid,
                "wikidata_label": candidate.wikidata_label,
                "mapping_type": candidate.mapping_type,
                "confidence": confidence_text(candidate.confidence),
                "proposal_mode": candidate.proposal_mode,
                "qid_suggestion_status": candidate.qid_suggestion_status,
                "warning": one_line(candidate.warning),
                "rationale": one_line(candidate.rationale),
                "score_total": f"{feature.score_total:.4g}" if feature else "",
                "paper_count": str(feature.paper_count) if feature else "",
                "mention_count": str(feature.mention_count) if feature else "",
            }
        )
    return rows


def wikidata_url(qid: str) -> str:
    return f"https://www.wikidata.org/wiki/{qid}" if qid else ""


def mapping_candidate_payload(candidate: MappingCandidate) -> dict[str, Any]:
    return {
        "mapped_row_index": candidate.mapped_row_index,
        "candidate_index": candidate.candidate_index,
        "input_record_id": candidate.input_record_id,
        "feature_name": candidate.feature_name,
        "normalized_feature_name": candidate.normalized_feature_name,
        "proposal_mode": candidate.proposal_mode,
        "qid": candidate.qid,
        "wikidata_url": wikidata_url(candidate.qid),
        "wikidata_label": candidate.wikidata_label,
        "mapping_type": candidate.mapping_type,
        "confidence": candidate.confidence,
        "rationale": one_line(candidate.rationale),
        "warning": one_line(candidate.warning),
        "qid_suggestion_status": candidate.qid_suggestion_status,
    }


def mapped_feature_payload(
    mapped: MappedFeature,
    candidates_by_mapped_row: dict[int, list[MappingCandidate]],
) -> dict[str, Any]:
    return {
        "mapped_row_index": mapped.row_index,
        "input_record_id": mapped.input_record_id,
        "feature_name": mapped.feature_name,
        "normalized_feature_name": mapped.normalized_feature_name,
        "proposal_mode": mapped.proposal_mode,
        "qid_suggestion_status": mapped.qid_suggestion_status,
        "verified_qids": [
            {
                "qid": qid,
                "wikidata_url": wikidata_url(qid),
                "wikidata_label": str(indexed_value(mapped.verified_wikidata_labels, index)),
                "mapping_type": str(indexed_value(mapped.mapping_types, index)),
                "confidence": (
                    mapped.proposal_confidences[index] if index < len(mapped.proposal_confidences) else None
                ),
                "rationale": one_line(
                    str(indexed_value(mapped.proposal_rationales, index, repeat_single=True))
                ),
                "warning": one_line(
                    str(indexed_value(mapped.verification_warnings, index, repeat_single=True))
                ),
            }
            for index, qid in enumerate(mapped.verified_qids)
        ],
        "candidate_mappings": [
            mapping_candidate_payload(candidate)
            for candidate in sorted(
                candidates_by_mapped_row.get(mapped.row_index, []),
                key=lambda candidate: candidate.candidate_index,
            )
        ],
    }


def qid_reuse_payload(row: dict[str, Any]) -> dict[str, Any]:
    qid = str(row["qid"])
    return {
        "qid": qid,
        "wikidata_url": wikidata_url(qid),
        "wikidata_label": str(row["wikidata_label"]),
        "feature_count": int(row["feature_count"]),
        "candidate_count": int(row["candidate_count"]),
        "mapped_row_count": int(row["mapped_row_count"]),
        "feature_names": split_multivalue_cell(str(row["feature_names"])),
        "wikidata_labels": split_multivalue_cell(str(row["wikidata_labels"])),
        "mapping_types": split_multivalue_cell(str(row["mapping_types"])),
        "min_confidence": row["min_confidence"],
        "max_confidence": row["max_confidence"],
        "mean_confidence": row["mean_confidence"],
    }


def audit_flag_payload(row: dict[str, str]) -> dict[str, Any]:
    qid = row["qid"]
    return {
        "flag_type": row["flag_type"],
        "feature_name": row["feature_name"],
        "normalized_feature_name": row["normalized_feature_name"],
        "qid": qid,
        "wikidata_url": wikidata_url(qid) if qid and CSV_JOIN not in qid else "",
        "wikidata_label": row["wikidata_label"],
        "mapping_type": row["mapping_type"],
        "confidence": safe_float(row["confidence"]),
        "score_total": safe_float(row["score_total"]),
        "paper_count": safe_int(row["paper_count"]),
        "mention_count": safe_int(row["mention_count"]),
        "details": row["details"],
    }


def build_feature_mapping_json_payload(
    *,
    category: str,
    mapped_csv: Path,
    scored_jsonl: Path,
    selected_features: list[ScoredFeature],
    mapped_rows: list[MappedFeature],
    matched_mapped_rows: list[MappedFeature],
    mapped_by_name: dict[str, list[MappedFeature]],
    candidates: list[MappingCandidate],
    candidates_by_name: dict[str, list[MappingCandidate]],
    qid_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, str]],
    top_k_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates_by_mapped_row: dict[int, list[MappingCandidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_mapped_row[candidate.mapped_row_index].append(candidate)

    covered_feature_count = sum(
        1 for feature in selected_features if feature.normalized_feature_name in mapped_by_name
    )
    feature_payloads: list[dict[str, Any]] = []
    for feature in sorted(selected_features, key=feature_sort_key, reverse=True):
        matched_rows = sorted(
            mapped_by_name.get(feature.normalized_feature_name, []),
            key=lambda mapped: mapped.row_index,
        )
        feature_candidates = sorted(
            candidates_by_name.get(feature.normalized_feature_name, []),
            key=lambda candidate: (candidate.mapped_row_index, candidate.candidate_index),
        )
        confidences = [
            candidate.confidence for candidate in feature_candidates if candidate.confidence is not None
        ]
        feature_payloads.append(
            {
                "feature_key": feature.feature_key,
                "category": feature.category,
                "feature_name": feature.feature_name,
                "normalized_feature_name": feature.normalized_feature_name,
                "score_total": feature.score_total,
                "paper_count": feature.paper_count,
                "mention_count": feature.mention_count,
                "mapping_status": {
                    "mapped": bool(matched_rows),
                    "matched_mapped_row_count": len(matched_rows),
                    "candidate_count": len(feature_candidates),
                    "verified_qids": [
                        {
                            "qid": qid,
                            "wikidata_url": wikidata_url(qid),
                        }
                        for qid in unique_values(candidate.qid for candidate in feature_candidates)
                    ],
                    "verified_wikidata_labels": unique_values(
                        candidate.wikidata_label for candidate in feature_candidates
                    ),
                    "mapping_types": unique_values(
                        candidate.mapping_type for candidate in feature_candidates
                    ),
                    "min_confidence": min(confidences) if confidences else None,
                    "max_confidence": max(confidences) if confidences else None,
                },
                "mapped_rows": [
                    mapped_feature_payload(mapped, candidates_by_mapped_row) for mapped in matched_rows
                ],
            }
        )

    return {
        "category": category,
        "description": "Feature-centric Wikidata mapping structure for the selected scored feature category.",
        "join_policy": "Scored features are joined to mapped CSV rows by normalized feature name.",
        "inputs": {
            "mapped_csv": str(mapped_csv),
            "scored_jsonl": str(scored_jsonl),
        },
        "summary": {
            "selected_scored_features": len(selected_features),
            "mapped_csv_rows": len(mapped_rows),
            "matched_mapped_rows": len(matched_mapped_rows),
            "covered_scored_features": covered_feature_count,
            "feature_coverage": percent(covered_feature_count, len(selected_features)),
            "candidate_mappings": len(candidates),
            "unique_qids": len(qid_rows),
            "reused_qids": sum(1 for row in qid_rows if int(row["feature_count"]) > 1),
            "audit_flags": len(audit_rows),
            "top_k_coverage": top_k_rows,
        },
        "features": feature_payloads,
        "qid_reuse": [qid_reuse_payload(row) for row in qid_rows],
        "audit_flags": [audit_flag_payload(row) for row in audit_rows],
    }


def qid_reuse_summary(candidates: list[MappingCandidate]) -> list[dict[str, Any]]:
    by_qid: dict[str, list[MappingCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.qid:
            by_qid[candidate.qid].append(candidate)

    rows: list[dict[str, Any]] = []
    for qid, qid_candidates in by_qid.items():
        feature_names = unique_values(candidate.feature_name for candidate in qid_candidates)
        labels = unique_values(candidate.wikidata_label for candidate in qid_candidates)
        mapping_types = unique_values(candidate.mapping_type for candidate in qid_candidates)
        confidences = [
            candidate.confidence for candidate in qid_candidates if candidate.confidence is not None
        ]
        rows.append(
            {
                "qid": qid,
                "wikidata_label": labels[0] if labels else "",
                "feature_count": len({candidate.normalized_feature_name for candidate in qid_candidates}),
                "candidate_count": len(qid_candidates),
                "mapped_row_count": len({candidate.mapped_row_index for candidate in qid_candidates}),
                "feature_names": CSV_JOIN.join(feature_names),
                "wikidata_labels": CSV_JOIN.join(labels),
                "mapping_types": CSV_JOIN.join(mapping_types),
                "min_confidence": min(confidences) if confidences else None,
                "max_confidence": max(confidences) if confidences else None,
                "mean_confidence": statistics.fmean(confidences) if confidences else None,
            }
        )
    return sorted(
        rows, key=lambda row: (-int(row["feature_count"]), -int(row["candidate_count"]), str(row["qid"]))
    )


def build_qid_reuse_rows(candidates: list[MappingCandidate]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in qid_reuse_summary(candidates):
        rows.append(
            {
                "qid": str(row["qid"]),
                "wikidata_label": str(row["wikidata_label"]),
                "feature_count": str(row["feature_count"]),
                "candidate_count": str(row["candidate_count"]),
                "mapped_row_count": str(row["mapped_row_count"]),
                "feature_names": str(row["feature_names"]),
                "wikidata_labels": str(row["wikidata_labels"]),
                "mapping_types": str(row["mapping_types"]),
                "min_confidence": confidence_text(row["min_confidence"]),
                "max_confidence": confidence_text(row["max_confidence"]),
                "mean_confidence": confidence_text(row["mean_confidence"]),
            }
        )
    return rows


def build_audit_rows(
    *,
    selected_features: list[ScoredFeature],
    mapped_by_name: dict[str, list[MappedFeature]],
    candidates: list[MappingCandidate],
    features_by_name: dict[str, ScoredFeature],
    duplicate_mapped_names: dict[str, list[MappedFeature]],
    qid_reuse_rows: list[dict[str, Any]],
    low_confidence_threshold: float,
    top_k_values: tuple[int, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add_candidate_flag(flag_type: str, candidate: MappingCandidate, details: str) -> None:
        feature = features_by_name.get(candidate.normalized_feature_name)
        rows.append(
            {
                "flag_type": flag_type,
                "feature_name": candidate.feature_name,
                "normalized_feature_name": candidate.normalized_feature_name,
                "qid": candidate.qid,
                "wikidata_label": candidate.wikidata_label,
                "mapping_type": candidate.mapping_type,
                "confidence": confidence_text(candidate.confidence),
                "score_total": f"{feature.score_total:.4g}" if feature else "",
                "paper_count": str(feature.paper_count) if feature else "",
                "mention_count": str(feature.mention_count) if feature else "",
                "details": details,
            }
        )

    for candidate in candidates:
        if candidate.confidence is not None and candidate.confidence < low_confidence_threshold:
            add_candidate_flag(
                "low_confidence",
                candidate,
                f"confidence below {low_confidence_threshold:.4g}",
            )
        if candidate.mapping_type and candidate.mapping_type != "exact":
            add_candidate_flag("non_exact", candidate, "mapping_type is not exact")
        if candidate.warning:
            add_candidate_flag("warning", candidate, one_line(candidate.warning))

    for normalized_name, mapped_rows in duplicate_mapped_names.items():
        feature = features_by_name.get(normalized_name)
        rows.append(
            {
                "flag_type": "duplicate_mapped_name",
                "feature_name": feature.feature_name if feature else mapped_rows[0].feature_name,
                "normalized_feature_name": normalized_name,
                "qid": CSV_JOIN.join(
                    unique_values(qid for mapped in mapped_rows for qid in mapped.verified_qids)
                ),
                "wikidata_label": CSV_JOIN.join(
                    unique_values(
                        label for mapped in mapped_rows for label in mapped.verified_wikidata_labels
                    )
                ),
                "mapping_type": CSV_JOIN.join(
                    unique_values(
                        mapping_type for mapped in mapped_rows for mapping_type in mapped.mapping_types
                    )
                ),
                "confidence": "",
                "score_total": f"{feature.score_total:.4g}" if feature else "",
                "paper_count": str(feature.paper_count) if feature else "",
                "mention_count": str(feature.mention_count) if feature else "",
                "details": f"{len(mapped_rows)} mapped CSV rows share this normalized feature name",
            }
        )

    for qid_row in qid_reuse_rows:
        if int(qid_row["feature_count"]) <= 1:
            continue
        rows.append(
            {
                "flag_type": "reused_qid",
                "feature_name": str(qid_row["feature_names"]),
                "normalized_feature_name": "",
                "qid": str(qid_row["qid"]),
                "wikidata_label": str(qid_row["wikidata_label"]),
                "mapping_type": str(qid_row["mapping_types"]),
                "confidence": confidence_text(qid_row["mean_confidence"]),
                "score_total": "",
                "paper_count": "",
                "mention_count": "",
                "details": f"QID is used by {qid_row['feature_count']} features",
            }
        )

    sorted_features = sorted(selected_features, key=feature_sort_key, reverse=True)
    top_window = max(top_k_values) if top_k_values else len(sorted_features)
    for feature in sorted_features[:top_window]:
        if feature.normalized_feature_name in mapped_by_name:
            continue
        rows.append(
            {
                "flag_type": "unmapped_top_ranked",
                "feature_name": feature.feature_name,
                "normalized_feature_name": feature.normalized_feature_name,
                "qid": "",
                "wikidata_label": "",
                "mapping_type": "",
                "confidence": "",
                "score_total": f"{feature.score_total:.4g}",
                "paper_count": str(feature.paper_count),
                "mention_count": str(feature.mention_count),
                "details": f"unmapped feature appears in top {top_window} by score_total",
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row["flag_type"],
            -(safe_float(row["score_total"]) or 0.0),
            row["feature_name"].casefold(),
            row["qid"],
        ),
    )


def closeness_rank(mapping_type: str) -> int:
    return CLOSENESS_RANKS.get(mapping_type.strip().lower(), 0)


def closeness_label(rank: int) -> str:
    return CLOSENESS_BY_RANK.get(rank, "unknown")


def resolve_trace_path(raw_path: Any, trace_path: Path) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute() or path.exists():
        return path
    candidate = trace_path.parent / path
    if candidate.exists():
        return candidate
    return path


def load_embedding_merge_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding merge trace does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    for key in ("input", "cache_path", "model", "task_type", "output_dimensionality", "category_config"):
        if key not in payload:
            raise ValueError(f"{path} is missing required trace key: {key}.")
    return payload


def category_specs_from_trace(trace: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    from lit_review.embeddings.similarity import (
        CategoryMergeSpec,
        load_category_merge_specs,
        parse_embedding_text_cleanup,
    )
    from lit_review.embeddings.core import parse_text_fields

    raw_config = trace.get("category_config")
    if isinstance(raw_config, str):
        return load_category_merge_specs(resolve_trace_path(raw_config, trace_path))
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(f"{trace_path} category_config must be a non-empty object or config path.")

    specs: dict[str, Any] = {}
    for raw_category, raw_spec in raw_config.items():
        category = str(raw_category).strip()
        if not category:
            raise ValueError(f"{trace_path} category_config contains an empty category.")
        if not isinstance(raw_spec, dict):
            raise ValueError(f"{trace_path} category_config[{category!r}] must be an object.")
        specs[category] = CategoryMergeSpec(
            text_fields=parse_text_fields(raw_spec.get("text_fields", [])),
            threshold=float(raw_spec.get("threshold", 1.0)),
            linkage=str(raw_spec.get("linkage") or "complete"),
            embedding_text_cleanup=parse_embedding_text_cleanup(
                raw_spec.get("embedding_text_cleanup"),
                trace_path,
                category,
            ),
        )
    return specs


def embedding_refs_by_name(
    *,
    embedding_merge_trace: Path,
    category: str,
    selected_names: set[str],
) -> tuple[dict[str, EmbeddingItemRef], Any, dict[str, Any]]:
    from lit_review.embeddings.similarity import normalized_matrix
    from lit_review.analysis.clustering import embedding_text_cleanup_payload
    from lit_review.analysis.clustering import prepare_category_embeddings

    trace = load_embedding_merge_trace(embedding_merge_trace)
    specs = category_specs_from_trace(trace, embedding_merge_trace)
    features_path = resolve_trace_path(trace["input"], embedding_merge_trace)
    cache_path = resolve_trace_path(trace["cache_path"], embedding_merge_trace)
    if category not in specs:
        raise ValueError(f"Embedding merge trace category_config does not include category {category!r}.")
    spec = specs[category]

    prepared = prepare_category_embeddings(
        features_path=features_path,
        specs=specs,
        cache_path=cache_path,
        client=None,
        model=str(trace["model"]),
        task_type=str(trace["task_type"]),
        output_dimensionality=int(trace["output_dimensionality"]),
        batch_size=50,
    )
    matrix = normalized_matrix(prepared.embeddings)
    refs: dict[str, EmbeddingItemRef] = {}
    duplicate_names: set[str] = set()
    for embedding_index, item in enumerate(prepared.items):
        if item.category != category:
            continue
        normalized_name = normalize_feature_name(item.label)
        if normalized_name not in selected_names:
            continue
        if normalized_name in refs:
            duplicate_names.add(normalized_name)
            continue
        refs[normalized_name] = EmbeddingItemRef(
            embedding_index=embedding_index,
            feature_key=item.feature_key,
            feature_name=item.label,
            normalized_feature_name=normalized_name,
            category=item.category,
            row_index=item.row_index,
            paper_count=item.paper_count,
            mention_count=item.mention_count,
        )

    metadata = {
        "trace_path": str(embedding_merge_trace),
        "features_path": str(features_path),
        "cache_path": str(cache_path),
        "model": str(trace["model"]),
        "task_type": str(trace["task_type"]),
        "output_dimensionality": int(trace["output_dimensionality"]),
        "embedding_text_fields": list(spec.text_fields),
        "embedding_threshold": spec.threshold,
        "embedding_linkage": spec.linkage,
        "embedding_text_cleanup": (
            embedding_text_cleanup_payload(spec.embedding_text_cleanup)
            if spec.embedding_text_cleanup is not None
            else None
        ),
        "embedded_features": len(prepared.items),
        "selected_embedded_features": len(refs),
        "duplicate_selected_embedding_names": sorted(duplicate_names),
        "cache": {
            "hits": prepared.cache_stats.cache_hits,
            "misses": prepared.cache_stats.api_misses,
            "api_batches": prepared.cache_stats.api_batches,
        },
    }
    return refs, matrix, metadata


def qid_links_by_name(candidates: list[MappingCandidate]) -> dict[str, dict[str, QidLink]]:
    links: dict[str, dict[str, QidLink]] = defaultdict(dict)
    for candidate in candidates:
        if not candidate.normalized_feature_name or not candidate.qid:
            continue
        rank = closeness_rank(candidate.mapping_type)
        existing = links[candidate.normalized_feature_name].get(candidate.qid)
        if existing is not None and existing.closeness_rank >= rank:
            continue
        links[candidate.normalized_feature_name][candidate.qid] = QidLink(
            qid=candidate.qid,
            wikidata_label=candidate.wikidata_label,
            mapping_type=candidate.mapping_type,
            closeness_rank=rank,
        )
    return links


def shared_qid_details(
    left_name: str,
    right_name: str,
    links_by_name: dict[str, dict[str, QidLink]],
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    left_links = links_by_name[left_name]
    right_links = links_by_name[right_name]
    shared_qids = tuple(sorted(set(left_links) & set(right_links)))
    if not shared_qids:
        return (), (), ""

    labels = unique_values(
        left_links[qid].wikidata_label or right_links[qid].wikidata_label for qid in shared_qids
    )
    pair_rank = max(
        min(left_links[qid].closeness_rank, right_links[qid].closeness_rank) for qid in shared_qids
    )
    return shared_qids, tuple(labels), closeness_label(pair_rank)


def build_embedding_similarity_pairs(
    *,
    candidates: list[MappingCandidate],
    embedding_refs: dict[str, EmbeddingItemRef],
    matrix: Any,
) -> list[SimilarityPairRow]:
    links_by_name = qid_links_by_name(candidates)
    covered_names = sorted(set(embedding_refs) & set(links_by_name))
    rows: list[SimilarityPairRow] = []
    for left_name, right_name in combinations(covered_names, 2):
        left_ref = embedding_refs[left_name]
        right_ref = embedding_refs[right_name]
        shared_qids, shared_labels, closeness = shared_qid_details(left_name, right_name, links_by_name)
        similarity = round(float(matrix[left_ref.embedding_index] @ matrix[right_ref.embedding_index]), 8)
        rows.append(
            SimilarityPairRow(
                relationship="same_article" if shared_qids else "different_article",
                similarity=similarity,
                feature_key_1=left_ref.feature_key,
                feature_name_1=left_ref.feature_name,
                normalized_feature_name_1=left_ref.normalized_feature_name,
                row_index_1=left_ref.row_index,
                feature_key_2=right_ref.feature_key,
                feature_name_2=right_ref.feature_name,
                normalized_feature_name_2=right_ref.normalized_feature_name,
                row_index_2=right_ref.row_index,
                shared_qids=shared_qids,
                shared_wikidata_labels=shared_labels,
                pair_mapping_closeness=closeness,
            )
        )
    return sorted(
        rows,
        key=lambda row: (
            RELATIONSHIP_ORDER.index(row.relationship)
            if row.relationship in RELATIONSHIP_ORDER
            else len(RELATIONSHIP_ORDER),
            -row.similarity,
            row.feature_name_1.casefold(),
            row.feature_name_2.casefold(),
        ),
    )


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "pair_count": 0,
            "mean_similarity": None,
            "median_similarity": None,
            "p25_similarity": None,
            "p75_similarity": None,
            "min_similarity": None,
            "max_similarity": None,
        }
    sorted_values = sorted(values)

    def quantile(q: float) -> float:
        index = round(q * (len(sorted_values) - 1))
        return sorted_values[index]

    return {
        "pair_count": len(sorted_values),
        "mean_similarity": statistics.fmean(sorted_values),
        "median_similarity": statistics.median(sorted_values),
        "p25_similarity": quantile(0.25),
        "p75_similarity": quantile(0.75),
        "min_similarity": sorted_values[0],
        "max_similarity": sorted_values[-1],
    }


def build_embedding_similarity_summary_rows(
    pair_rows: list[SimilarityPairRow],
) -> list[dict[str, str]]:
    relationship_values: dict[str, list[float]] = defaultdict(list)
    closeness_values: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        relationship_values[row.relationship].append(row.similarity)
        if row.relationship == "same_article" and row.pair_mapping_closeness:
            closeness_values[row.pair_mapping_closeness].append(row.similarity)

    rows: list[dict[str, str]] = []
    for relationship in RELATIONSHIP_ORDER:
        rows.append(summary_csv_row("relationship", relationship, relationship_values.get(relationship, [])))
    for closeness in CLOSENESS_ORDER:
        if closeness not in closeness_values:
            continue
        rows.append(summary_csv_row("pair_mapping_closeness", closeness, closeness_values[closeness]))
    return rows


def summary_csv_row(group_type: str, group: str, values: list[float]) -> dict[str, str]:
    summary = numeric_summary(values)
    return {
        "group_type": group_type,
        "group": group,
        "pair_count": str(summary["pair_count"]),
        "mean_similarity": fmt_number(summary["mean_similarity"]),
        "median_similarity": fmt_number(summary["median_similarity"]),
        "p25_similarity": fmt_number(summary["p25_similarity"]),
        "p75_similarity": fmt_number(summary["p75_similarity"]),
        "min_similarity": fmt_number(summary["min_similarity"]),
        "max_similarity": fmt_number(summary["max_similarity"]),
    }


def similarity_pair_csv_rows(pair_rows: list[SimilarityPairRow]) -> list[dict[str, str]]:
    return [
        {
            "relationship": row.relationship,
            "similarity": f"{row.similarity:.8f}",
            "feature_key_1": row.feature_key_1,
            "feature_name_1": row.feature_name_1,
            "normalized_feature_name_1": row.normalized_feature_name_1,
            "row_index_1": str(row.row_index_1),
            "feature_key_2": row.feature_key_2,
            "feature_name_2": row.feature_name_2,
            "normalized_feature_name_2": row.normalized_feature_name_2,
            "row_index_2": str(row.row_index_2),
            "shared_qids": CSV_JOIN.join(row.shared_qids),
            "shared_wikidata_labels": CSV_JOIN.join(row.shared_wikidata_labels),
            "pair_mapping_closeness": row.pair_mapping_closeness,
        }
        for row in pair_rows
    ]


def unique_values(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = one_line(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def confidence_summary(confidences: list[float]) -> dict[str, float | int | None]:
    if not confidences:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }

    sorted_values = sorted(confidences)

    def quantile(q: float) -> float:
        index = round(q * (len(sorted_values) - 1))
        return sorted_values[index]

    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p10": quantile(0.10),
        "p25": quantile(0.25),
        "median": statistics.median(sorted_values),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "max": sorted_values[-1],
        "mean": statistics.fmean(sorted_values),
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(markdown_cell(value) for value in row) + " |")
    return "\n".join(output)


def markdown_cell(value: Any) -> str:
    text = one_line(value)
    text = text.replace("|", "\\|")
    return text


def fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_report(
    *,
    path: Path,
    category: str,
    mapped_csv: Path,
    scored_jsonl: Path,
    output_map: dict[str, Path],
    selected_features: list[ScoredFeature],
    mapped_rows: list[MappedFeature],
    matched_mapped_rows: list[MappedFeature],
    candidates: list[MappingCandidate],
    top_k_rows: list[dict[str, Any]],
    mapping_type_counts: Counter[str],
    warning_counts: Counter[str],
    duplicate_mapped_names: dict[str, list[MappedFeature]],
    qid_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, str]],
    low_confidence_threshold: float,
    embedding_similarity: dict[str, Any] | None = None,
) -> None:
    covered_feature_count = sum(
        1
        for feature in selected_features
        if any(
            mapped.normalized_feature_name == feature.normalized_feature_name
            for mapped in matched_mapped_rows
        )
    )
    confidences = [candidate.confidence for candidate in candidates if candidate.confidence is not None]
    confidence_stats = confidence_summary(confidences)
    low_confidence_count = sum(
        1
        for candidate in candidates
        if candidate.confidence is not None and candidate.confidence < low_confidence_threshold
    )
    reused_qid_rows = [row for row in qid_rows if int(row["feature_count"]) > 1]
    non_exact_count = sum(
        1 for candidate in candidates if candidate.mapping_type and candidate.mapping_type != "exact"
    )
    warning_candidate_count = sum(1 for candidate in candidates if candidate.warning)
    duplicate_rows = [
        (
            rows[0].feature_name,
            len(rows),
            CSV_JOIN.join(unique_values(qid for row in rows for qid in row.verified_qids)),
        )
        for _, rows in sorted(
            duplicate_mapped_names.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:20]
    ]
    reused_qid_table_rows = [
        (row["qid"], row["wikidata_label"], row["feature_count"], row["feature_names"])
        for row in reused_qid_rows[:20]
    ]
    unmapped_table_rows = [
        (
            row["feature_name"],
            row["score_total"],
            row["paper_count"],
            row["mention_count"],
        )
        for row in audit_rows
        if row["flag_type"] == "unmapped_top_ranked"
    ][:20]
    embedding_similarity_section: list[str] = []
    if embedding_similarity is not None:
        relationship_rows = [
            row for row in embedding_similarity["summary_rows"] if row["group_type"] == "relationship"
        ]
        closeness_rows = [
            row
            for row in embedding_similarity["summary_rows"]
            if row["group_type"] == "pair_mapping_closeness"
        ]
        embedding_similarity_section = [
            "## Embedding Similarity",
            "",
            markdown_table(
                ["Metric", "Value"],
                [
                    ("embedding merge trace", embedding_similarity["trace_path"]),
                    ("embedding text fields", CSV_JOIN.join(embedding_similarity["embedding_text_fields"])),
                    (
                        "embedding text cleanup",
                        json.dumps(embedding_similarity["embedding_text_cleanup"] or {}),
                    ),
                    ("embedded selected features", embedding_similarity["selected_embedded_features"]),
                    ("mapped embedded features", embedding_similarity["mapped_embedded_features"]),
                    ("same-article pairs", embedding_similarity["same_article_pairs"]),
                    ("different-article pairs", embedding_similarity["different_article_pairs"]),
                ],
            ),
            "",
            markdown_table(
                ["Relationship", "Pairs", "Mean", "Median", "P25", "P75"],
                (
                    (
                        row["group"],
                        row["pair_count"],
                        row["mean_similarity"],
                        row["median_similarity"],
                        row["p25_similarity"],
                        row["p75_similarity"],
                    )
                    for row in relationship_rows
                ),
            ),
            "",
            markdown_table(
                ["Mapping Closeness", "Pairs", "Mean", "Median", "P25", "P75"],
                (
                    (
                        row["group"],
                        row["pair_count"],
                        row["mean_similarity"],
                        row["median_similarity"],
                        row["p25_similarity"],
                        row["p75_similarity"],
                    )
                    for row in closeness_rows
                ),
            ),
            "",
        ]

    lines = [
        f"# Wikidata Mapping Analysis: {category}",
        "",
        "## Inputs",
        "",
        f"- mapped CSV: `{mapped_csv}`",
        f"- scored JSONL: `{scored_jsonl}`",
        f"- category: `{category}`",
        "",
        "## Headline Counts",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ("selected scored features", len(selected_features)),
                ("mapped CSV rows", len(mapped_rows)),
                ("matched mapped rows", len(matched_mapped_rows)),
                ("covered scored features", covered_feature_count),
                ("feature coverage", fmt_percent(percent(covered_feature_count, len(selected_features)))),
                ("candidate mappings", len(candidates)),
                ("unique mapped QIDs", len(qid_rows)),
                ("reused QIDs", len(reused_qid_rows)),
                ("duplicate mapped names", len(duplicate_mapped_names)),
                (f"low-confidence candidates (< {low_confidence_threshold:.4g})", low_confidence_count),
                ("non-exact candidates", non_exact_count),
                ("warning-bearing candidates", warning_candidate_count),
            ],
        ),
        "",
        "## Top-K Coverage",
        "",
        markdown_table(
            ["Top K", "Covered", "Feature Count", "Coverage"],
            (
                (
                    row["top_k"],
                    row["covered_feature_count"],
                    row["feature_count"],
                    fmt_percent(float(row["coverage"])),
                )
                for row in top_k_rows
            ),
        ),
        "",
        "## Mapping Types",
        "",
        markdown_table(
            ["Mapping Type", "Candidate Count"],
            mapping_type_counts.most_common() or [("(none)", 0)],
        ),
        "",
        "## Confidence Summary",
        "",
        markdown_table(
            ["Metric", "Value"],
            [(key, fmt_number(value)) for key, value in confidence_stats.items()],
        ),
        "",
        "## Verification Warnings",
        "",
        markdown_table(
            ["Warning", "Candidate Count"],
            warning_counts.most_common() or [("(none)", 0)],
        ),
        "",
        "## Duplicate Mapped Names",
        "",
        markdown_table(
            ["Feature Name", "Mapped Rows", "QIDs"],
            duplicate_rows or [("(none)", 0, "")],
        ),
        "",
        "## Top Reused QIDs",
        "",
        markdown_table(
            ["QID", "Label", "Feature Count", "Feature Names"],
            reused_qid_table_rows or [("(none)", "", 0, "")],
        ),
        "",
        "## Highest-Scored Unmapped Features",
        "",
        markdown_table(
            ["Feature", "Score", "Papers", "Mentions"],
            unmapped_table_rows or [("(none)", "", "", "")],
        ),
        "",
        *embedding_similarity_section,
        "## Outputs",
        "",
        markdown_table(
            ["Artifact", "Path"],
            ((label, output_path) for label, output_path in output_map.items()),
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_no_data_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_coverage_plot(path: Path, top_k_rows: list[dict[str, Any]]) -> None:
    if not top_k_rows:
        write_no_data_plot(path, "Coverage By Top-K")
        return
    labels = [str(row["top_k"]) for row in top_k_rows]
    values = [float(row["coverage"]) * 100 for row in top_k_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(labels, values, marker="o", linewidth=2)
    ax.set_title("Coverage By Top-K Scored Features")
    ax.set_xlabel("Top K")
    ax.set_ylabel("Covered features (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_mapping_type_plot(path: Path, counts: Counter[str]) -> None:
    if not counts:
        write_no_data_plot(path, "Mapping Type Counts")
        return
    labels, values = zip(*counts.most_common())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title("Mapping Type Counts")
    ax.set_ylabel("Candidate count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_confidence_histogram(path: Path, confidences: list[float]) -> None:
    if not confidences:
        write_no_data_plot(path, "Confidence Histogram")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(confidences, bins=10, range=(0, 1), color="#59a14f", edgecolor="white")
    ax.set_title("Proposal Confidence Histogram")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Candidate count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_top_reused_qids_plot(path: Path, qid_rows: list[dict[str, Any]], limit: int = 15) -> None:
    reused = [row for row in qid_rows if int(row["feature_count"]) > 1][:limit]
    if not reused:
        write_no_data_plot(path, "Top Reused QIDs")
        return
    labels = [str(row["wikidata_label"] or "(unlabeled)") for row in reversed(reused)]
    labels = [label[:48] + "..." if len(label) > 51 else label for label in labels]
    values = [int(row["feature_count"]) for row in reversed(reused)]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(labels, values, color="#f28e2b")
    ax.set_title("Top Reused QIDs")
    ax.set_xlabel("Unique feature count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pair_values_by_relationship(pair_rows: list[SimilarityPairRow]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        values[row.relationship].append(row.similarity)
    return values


def pair_values_by_closeness(pair_rows: list[SimilarityPairRow]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        if row.relationship == "same_article" and row.pair_mapping_closeness:
            values[row.pair_mapping_closeness].append(row.similarity)
    return values


def zscores_for_group_means(
    *,
    values_by_group: dict[str, list[float]],
    groups: list[str],
    baseline_values: list[float],
) -> tuple[list[float], float, float] | None:
    if len(baseline_values) < 2:
        return None
    baseline_mean = statistics.fmean(baseline_values)
    baseline_stddev = statistics.pstdev(baseline_values)
    if baseline_stddev <= 0:
        return None
    zscores = [
        (statistics.fmean(values_by_group[group]) - baseline_mean) / baseline_stddev for group in groups
    ]
    return zscores, baseline_mean, baseline_stddev


def write_zscore_bar_plot(
    *,
    path: Path,
    title: str,
    ylabel: str,
    labels: list[str],
    groups: list[str],
    values_by_group: dict[str, list[float]],
    baseline_values: list[float],
    colors: list[str],
) -> None:
    zscore_result = zscores_for_group_means(
        values_by_group=values_by_group,
        groups=groups,
        baseline_values=baseline_values,
    )
    if zscore_result is None:
        write_no_data_plot(path, title)
        return
    zscores, baseline_mean, baseline_stddev = zscore_result
    counts = [len(values_by_group[group]) for group in groups]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, zscores, color=colors[: len(labels)])
    ax.bar_label(
        bars,
        labels=[f"{value:+.2f} z\n{count:,} pairs" for value, count in zip(zscores, counts)],
        padding=3,
    )
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(f"Baseline mean {baseline_mean:.3f}, SD {baseline_stddev:.3f}")
    z_min = min(zscores)
    z_max = max(zscores)
    if z_min >= 0:
        ax.set_ylim(0, z_max + 0.35)
    elif z_max <= 0:
        ax.set_ylim(z_min - 0.35, 0)
    else:
        ax.set_ylim(z_min - 0.25, z_max + 0.35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_embedding_relationship_bar_plot(path: Path, pair_rows: list[SimilarityPairRow]) -> None:
    values_by_group = pair_values_by_relationship(pair_rows)
    groups = [group for group in RELATIONSHIP_ORDER if values_by_group.get(group)]
    if not groups:
        write_no_data_plot(path, "Embedding Similarity by Article Relationship")
        return
    means = [statistics.fmean(values_by_group[group]) for group in groups]
    counts = [len(values_by_group[group]) for group in groups]
    labels = [group.replace("_", " ") for group in groups]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, means, color=["#4c78a8", "#9c755f"][: len(labels)])
    ax.bar_label(bars, labels=[f"{value:.3f}\n{count:,} pairs" for value, count in zip(means, counts)])
    ax.set_title("Mean Embedding Similarity by Article Relationship")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_ylim(0, min(1.05, max(means) + 0.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_embedding_relationship_zscore_bar_plot(path: Path, pair_rows: list[SimilarityPairRow]) -> None:
    values_by_group = pair_values_by_relationship(pair_rows)
    groups = [group for group in RELATIONSHIP_ORDER if values_by_group.get(group)]
    if not groups:
        write_no_data_plot(path, "Mean Similarity Z-Score by Article Relationship")
        return
    baseline_values = [value for group in groups for value in values_by_group[group]]
    write_zscore_bar_plot(
        path=path,
        title="Mean Similarity Z-Score by Article Relationship",
        ylabel="Group mean z-score",
        labels=[group.replace("_", " ") for group in groups],
        groups=groups,
        values_by_group=values_by_group,
        baseline_values=baseline_values,
        colors=["#4c78a8", "#9c755f"],
    )


def write_embedding_relationship_boxplot(path: Path, pair_rows: list[SimilarityPairRow]) -> None:
    values_by_group = pair_values_by_relationship(pair_rows)
    groups = [group for group in RELATIONSHIP_ORDER if values_by_group.get(group)]
    if not groups:
        write_no_data_plot(path, "Embedding Similarity Distribution by Article Relationship")
        return
    values = [values_by_group[group] for group in groups]
    labels = [f"{group.replace('_', ' ')}\n{len(values_by_group[group]):,} pairs" for group in groups]
    positions = [1.0 + index * 0.55 for index in range(len(labels))]
    fig, ax = plt.subplots(figsize=(4.8, 6.2))
    ax.boxplot(values, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
    ax.set_xticks(positions, labels=labels)
    ax.set_xlim(min(positions) - 0.45, max(positions) + 0.45)
    ax.set_title("Embedding Similarity Distribution by Article Relationship")
    ax.set_ylabel("Cosine similarity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_embedding_relationship_overlay_histogram(path: Path, pair_rows: list[SimilarityPairRow]) -> None:
    values_by_group = pair_values_by_relationship(pair_rows)
    groups = [group for group in RELATIONSHIP_ORDER if values_by_group.get(group)]
    if not groups:
        write_no_data_plot(path, "Embedding Similarity Overlay Histogram")
        return

    x_min = 0.75
    x_max = 1.0
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = {
        "same_article": "#4c78a8",
        "different_article": "#f28e2b",
    }
    plotted = False
    for group in groups:
        values = [value for value in values_by_group[group] if x_min <= value <= x_max]
        if not values:
            continue
        plotted = True
        ax.hist(
            values,
            bins=30,
            range=(x_min, x_max),
            density=True,
            alpha=0.5,
            color=colors.get(group),
            edgecolor="white",
            linewidth=0.3,
            label=f"{group.replace('_', ' ')} ({len(values):,} pairs)",
        )
    if not plotted:
        plt.close(fig)
        write_no_data_plot(path, "Embedding Similarity Overlay Histogram")
        return
    ax.set_title("Embedding Similarity Distribution by Article Relationship")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_embedding_closeness_zscore_bar_plot(path: Path, pair_rows: list[SimilarityPairRow]) -> None:
    values_by_group = pair_values_by_closeness(pair_rows)
    groups = [group for group in CLOSENESS_ORDER if values_by_group.get(group)]
    if not groups:
        write_no_data_plot(path, "Same-Article Similarity Z-Score by Mapping Closeness")
        return
    baseline_values = [row.similarity for row in pair_rows]
    write_zscore_bar_plot(
        path=path,
        title="Same-Article Similarity Z-Score by Mapping Closeness",
        ylabel="Closeness bucket mean z-score",
        labels=groups,
        groups=groups,
        values_by_group=values_by_group,
        baseline_values=baseline_values,
        colors=["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#bab0ac"],
    )


def write_embedding_closeness_bar_plot(path: Path, pair_rows: list[SimilarityPairRow]) -> None:
    values_by_group = pair_values_by_closeness(pair_rows)
    groups = [group for group in CLOSENESS_ORDER if values_by_group.get(group)]
    if not groups:
        write_no_data_plot(path, "Same-Article Similarity by Mapping Closeness")
        return
    means = [statistics.fmean(values_by_group[group]) for group in groups]
    counts = [len(values_by_group[group]) for group in groups]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(groups, means, color="#59a14f")
    ax.bar_label(bars, labels=[f"{value:.3f}\n{count:,}" for value, count in zip(means, counts)])
    ax.set_title("Same-Article Mean Similarity by Mapping Closeness")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xlabel("Weakest feature-to-article mapping in the pair")
    ax.set_ylim(0, min(1.05, max(means) + 0.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_embedding_similarity_outputs(
    *,
    paths: dict[str, Path],
    pair_rows: list[SimilarityPairRow],
) -> list[dict[str, str]]:
    pair_csv_rows = similarity_pair_csv_rows(pair_rows)
    summary_rows = build_embedding_similarity_summary_rows(pair_rows)
    write_csv(
        paths["embedding_pair_similarities_csv"],
        pair_csv_rows,
        [
            "relationship",
            "similarity",
            "feature_key_1",
            "feature_name_1",
            "normalized_feature_name_1",
            "row_index_1",
            "feature_key_2",
            "feature_name_2",
            "normalized_feature_name_2",
            "row_index_2",
            "shared_qids",
            "shared_wikidata_labels",
            "pair_mapping_closeness",
        ],
    )
    write_csv(
        paths["embedding_similarity_summary_csv"],
        summary_rows,
        [
            "group_type",
            "group",
            "pair_count",
            "mean_similarity",
            "median_similarity",
            "p25_similarity",
            "p75_similarity",
            "min_similarity",
            "max_similarity",
        ],
    )
    write_embedding_relationship_bar_plot(paths["embedding_similarity_relationship_bar_png"], pair_rows)
    write_embedding_relationship_zscore_bar_plot(
        paths["embedding_similarity_relationship_zscore_bar_png"],
        pair_rows,
    )
    write_embedding_relationship_boxplot(paths["embedding_similarity_relationship_boxplot_png"], pair_rows)
    write_embedding_relationship_overlay_histogram(
        paths["embedding_similarity_relationship_overlay_histogram_png"],
        pair_rows,
    )
    write_embedding_closeness_bar_plot(paths["embedding_similarity_closeness_bar_png"], pair_rows)
    write_embedding_closeness_zscore_bar_plot(
        paths["embedding_similarity_closeness_zscore_bar_png"],
        pair_rows,
    )
    return summary_rows


def run_wikidata_mapping_analysis(
    *,
    mapped_csv: Path,
    scored_jsonl: Path,
    output_dir: Path,
    category: str = DEFAULT_CATEGORY,
    prefix: str | None = None,
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    embedding_merge_trace: Path | None = None,
) -> dict[str, Any]:
    if not mapped_csv.exists():
        raise FileNotFoundError(f"Mapped CSV does not exist: {mapped_csv}")
    if not scored_jsonl.exists():
        raise FileNotFoundError(f"Scored JSONL does not exist: {scored_jsonl}")
    if not 0 <= low_confidence_threshold <= 1:
        raise ValueError("--low-confidence-threshold must be between 0 and 1.")

    top_k_values = parse_top_k(top_k_values)
    prefix = prefix or f"{category}_wikidata_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir, prefix)
    if embedding_merge_trace is not None:
        paths.update(embedding_similarity_output_paths(output_dir, prefix))

    scored_features = read_scored_features(scored_jsonl)
    mapped_rows = read_mapped_features(mapped_csv)
    selected_features = [feature for feature in scored_features if feature.category == category]
    if not selected_features:
        available = sorted({feature.category for feature in scored_features})
        raise ValueError(
            f"No scored features found for category {category!r}. Available: {', '.join(available)}."
        )

    selected_names = {feature.normalized_feature_name for feature in selected_features}
    matched_mapped_rows = [row for row in mapped_rows if row.normalized_feature_name in selected_names]

    mapped_by_name: dict[str, list[MappedFeature]] = defaultdict(list)
    for row in matched_mapped_rows:
        mapped_by_name[row.normalized_feature_name].append(row)

    features_by_name = {feature.normalized_feature_name: feature for feature in selected_features}
    candidates = explode_mapping_candidates(matched_mapped_rows)
    candidates_by_name: dict[str, list[MappingCandidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_name[candidate.normalized_feature_name].append(candidate)

    duplicate_mapped_names = {
        normalized_name: rows for normalized_name, rows in mapped_by_name.items() if len(rows) > 1
    }
    qid_rows = qid_reuse_summary(candidates)
    top_k_rows = build_top_k_coverage(selected_features, mapped_by_name, top_k_values)
    mapping_type_counts = Counter(candidate.mapping_type or "(none)" for candidate in candidates)
    warning_counts = Counter(candidate.warning for candidate in candidates if candidate.warning)

    feature_rows = build_feature_coverage_rows(selected_features, mapped_by_name, candidates_by_name)
    candidate_rows = build_candidate_mapping_rows(category, features_by_name, candidates)
    qid_csv_rows = build_qid_reuse_rows(candidates)
    audit_rows = build_audit_rows(
        selected_features=selected_features,
        mapped_by_name=mapped_by_name,
        candidates=candidates,
        features_by_name=features_by_name,
        duplicate_mapped_names=duplicate_mapped_names,
        qid_reuse_rows=qid_rows,
        low_confidence_threshold=low_confidence_threshold,
        top_k_values=top_k_values,
    )
    feature_mapping_json = build_feature_mapping_json_payload(
        category=category,
        mapped_csv=mapped_csv,
        scored_jsonl=scored_jsonl,
        selected_features=selected_features,
        mapped_rows=mapped_rows,
        matched_mapped_rows=matched_mapped_rows,
        mapped_by_name=mapped_by_name,
        candidates=candidates,
        candidates_by_name=candidates_by_name,
        qid_rows=qid_rows,
        audit_rows=audit_rows,
        top_k_rows=top_k_rows,
    )

    write_csv(
        paths["feature_coverage_csv"],
        feature_rows,
        [
            "category",
            "feature_key",
            "feature_name",
            "normalized_feature_name",
            "mapped",
            "matched_mapped_row_count",
            "candidate_count",
            "verified_qids",
            "verified_wikidata_labels",
            "mapping_types",
            "min_confidence",
            "max_confidence",
            "score_total",
            "paper_count",
            "mention_count",
        ],
    )
    paths["feature_mapping_json"].write_text(
        json.dumps(feature_mapping_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        paths["candidate_mappings_csv"],
        candidate_rows,
        [
            "category",
            "feature_key",
            "feature_name",
            "normalized_feature_name",
            "input_record_id",
            "mapped_row_index",
            "candidate_index",
            "qid",
            "wikidata_label",
            "mapping_type",
            "confidence",
            "proposal_mode",
            "qid_suggestion_status",
            "warning",
            "rationale",
            "score_total",
            "paper_count",
            "mention_count",
        ],
    )
    write_csv(
        paths["qid_reuse_csv"],
        qid_csv_rows,
        [
            "qid",
            "wikidata_label",
            "feature_count",
            "candidate_count",
            "mapped_row_count",
            "feature_names",
            "wikidata_labels",
            "mapping_types",
            "min_confidence",
            "max_confidence",
            "mean_confidence",
        ],
    )
    write_csv(
        paths["audit_flags_csv"],
        audit_rows,
        [
            "flag_type",
            "feature_name",
            "normalized_feature_name",
            "qid",
            "wikidata_label",
            "mapping_type",
            "confidence",
            "score_total",
            "paper_count",
            "mention_count",
            "details",
        ],
    )

    confidences = [candidate.confidence for candidate in candidates if candidate.confidence is not None]
    write_coverage_plot(paths["coverage_plot_png"], top_k_rows)
    write_mapping_type_plot(paths["mapping_types_plot_png"], mapping_type_counts)
    write_confidence_histogram(paths["confidence_histogram_png"], confidences)
    write_top_reused_qids_plot(paths["top_reused_qids_plot_png"], qid_rows)

    embedding_similarity: dict[str, Any] | None = None
    if embedding_merge_trace is not None:
        embedding_refs, matrix, embedding_metadata = embedding_refs_by_name(
            embedding_merge_trace=embedding_merge_trace,
            category=category,
            selected_names=selected_names,
        )
        pair_rows = build_embedding_similarity_pairs(
            candidates=candidates,
            embedding_refs=embedding_refs,
            matrix=matrix,
        )
        summary_rows = write_embedding_similarity_outputs(paths=paths, pair_rows=pair_rows)
        relationship_counts = Counter(row.relationship for row in pair_rows)
        mapped_embedded_names = set(embedding_refs) & set(qid_links_by_name(candidates))
        embedding_similarity = {
            **embedding_metadata,
            "summary_rows": summary_rows,
            "mapped_embedded_features": len(mapped_embedded_names),
            "same_article_pairs": relationship_counts.get("same_article", 0),
            "different_article_pairs": relationship_counts.get("different_article", 0),
            "pair_count": len(pair_rows),
        }

    write_report(
        path=paths["report_md"],
        category=category,
        mapped_csv=mapped_csv,
        scored_jsonl=scored_jsonl,
        output_map=paths,
        selected_features=selected_features,
        mapped_rows=mapped_rows,
        matched_mapped_rows=matched_mapped_rows,
        candidates=candidates,
        top_k_rows=top_k_rows,
        mapping_type_counts=mapping_type_counts,
        warning_counts=warning_counts,
        duplicate_mapped_names=duplicate_mapped_names,
        qid_rows=qid_rows,
        audit_rows=audit_rows,
        low_confidence_threshold=low_confidence_threshold,
        embedding_similarity=embedding_similarity,
    )

    covered_feature_count = sum(
        1 for feature in selected_features if feature.normalized_feature_name in mapped_by_name
    )
    trace = {
        "category": category,
        "inputs": {
            "mapped_csv": str(mapped_csv),
            "scored_jsonl": str(scored_jsonl),
            "embedding_merge_trace": str(embedding_merge_trace) if embedding_merge_trace else None,
        },
        "counts": {
            "scored_features": len(scored_features),
            "selected_features": len(selected_features),
            "mapped_rows": len(mapped_rows),
            "matched_mapped_rows": len(matched_mapped_rows),
            "covered_features": covered_feature_count,
            "candidate_mappings": len(candidates),
            "unique_qids": len(qid_rows),
            "reused_qids": sum(1 for row in qid_rows if int(row["feature_count"]) > 1),
            "duplicate_mapped_names": len(duplicate_mapped_names),
            "audit_flags": len(audit_rows),
        },
        "coverage": {
            "feature_coverage": percent(covered_feature_count, len(selected_features)),
            "top_k": top_k_rows,
        },
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    if embedding_similarity is not None:
        trace["embedding_similarity"] = {
            key: value for key, value in embedding_similarity.items() if key != "summary_rows"
        }
    return trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Wikidata/Wikipedia mapping quality for a scored feature category.",
    )
    parser.add_argument("mapped_csv", type=Path, help="Wikidata mapped features CSV.")
    parser.add_argument("scored_jsonl", type=Path, help="Scored features JSONL.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="Feature category to analyze.")
    parser.add_argument("--output-dir", type=Path, help="Directory for report, CSVs, and plots.")
    parser.add_argument("--prefix", help="Output filename prefix. Defaults to <category>_wikidata_mapping.")
    parser.add_argument(
        "--top-k",
        default=",".join(str(value) for value in DEFAULT_TOP_K),
        help="Comma-separated score-rank cutoffs for coverage analysis.",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        help="Audit candidates below this confidence.",
    )
    parser.add_argument(
        "--embedding-merge-trace",
        type=Path,
        help="Optional embedding-merge trace JSON used to add cache-only embedding similarity analysis.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = args.output_dir or args.mapped_csv.parent / "analysis" / args.category
        trace = run_wikidata_mapping_analysis(
            mapped_csv=args.mapped_csv,
            scored_jsonl=args.scored_jsonl,
            output_dir=output_dir,
            category=args.category,
            prefix=args.prefix,
            top_k_values=parse_top_k(args.top_k),
            low_confidence_threshold=args.low_confidence_threshold,
            embedding_merge_trace=args.embedding_merge_trace,
        )
    except Exception as exc:
        print(f"wikidata_mapping_analysis: {exc}", file=sys.stderr)
        return 1

    counts = trace["counts"]
    print(
        f"Analyzed {counts['selected_features']} {trace['category']} features: "
        f"{counts['covered_features']} covered "
        f"({trace['coverage']['feature_coverage'] * 100:.1f}%)."
    )
    for label, path in trace["outputs"].items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
