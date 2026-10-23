"""Features extraction_artifacts."""

from __future__ import annotations


from pathlib import Path

from typing import Any


from lit_review.features.records import aggregate_features, aggregate_paper_features


from lit_review.features.filtering import FEATURE_FIELDNAMES, write_csv
from lit_review.core.io import write_jsonl


CSV_FIELDNAMES = [
    "paper_id",
    "title",
    "authors",
    "year",
    "doi",
    "venue",
    "category",
    "feature_name",
    "normalized_feature_name",
    "parent_category",
    "definition",
    "synonyms",
    "examples",
    "evidence_type",
    "chunk_id",
    "pages",
    "snippet",
    "source_quote",
    "confidence",
    "notes",
    "model",
    "run_id",
    "extracted_at",
]


PAPER_FEATURE_FIELDNAMES = [
    "paper_id",
    "title",
    "authors",
    "year",
    "doi",
    "venue",
    "category",
    "feature_name",
    "normalized_feature_name",
    "parent_category",
    "definition",
    "synonyms",
    "examples",
    "evidence_types",
    "mention_count",
    "chunk_ids",
    "pages",
    "best_snippet",
    "best_source_quote",
    "confidence",
    "notes",
    "model",
    "run_id",
    "extracted_at",
]


def write_artifacts(output_dir: Path, mentions: list[dict[str, Any]]) -> dict[str, int]:
    mentions = dedupe_mentions(mentions)
    paper_features = aggregate_paper_features(mentions)
    features = aggregate_features(mentions)

    write_jsonl(output_dir / "feature_mentions.jsonl", mentions)
    write_csv(output_dir / "feature_mentions.csv", mentions, CSV_FIELDNAMES)
    write_jsonl(output_dir / "paper_features.jsonl", paper_features)
    write_csv(output_dir / "paper_features.csv", paper_features, PAPER_FEATURE_FIELDNAMES)
    write_jsonl(output_dir / "features.jsonl", features)
    write_csv(output_dir / "features.csv", features, FEATURE_FIELDNAMES)
    return {
        "feature_mentions": len(mentions),
        "paper_features": len(paper_features),
        "features": len(features),
    }


def dedupe_mentions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("paper_id") or ""),
            str(row.get("category") or ""),
            str(row.get("normalized_feature_name") or ""),
            str(row.get("chunk_id") or ""),
            str(row.get("evidence_type") or ""),
            str(row.get("source_quote") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output
