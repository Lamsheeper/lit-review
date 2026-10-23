"""Compare SemEval-2020 Task 11 final features against the expected labels."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED = Path("sanity_checks/semeval_2020_task11/expected_features.csv")
DEFAULT_FEATURE_CANDIDATES = [
    Path(
        "review_runs/semeval_2020_task11_sanity/feature_extract_pdf/"
        "final_curated_features_llm_top100_by_category.jsonl"
    ),
    Path(
        "review_runs/semeval_2020_task11_sanity/feature_extract_pdf/"
        "final_curated_features_llm_top100_by_category.csv"
    ),
    Path(
        "review_runs/semeval_2020_task11_sanity/feature_extract_pdf/"
        "final_curated_features_top100_by_category.csv"
    ),
]
NAME_FIELDS = (
    "feature_name",
    "normalized_feature_name",
    "source_feature_names",
    "source_normalized_feature_names",
    "synonyms",
)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def parse_jsonish_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return [str(value).strip()] if str(value).strip() else []

    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    if "|" in value:
        return [part.strip() for part in value.split("|") if part.strip()]
    if ";" in value:
        return [part.strip() for part in value.split(";") if part.strip()]
    return [value]


def load_expected(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = []
    for row in rows:
        aliases = [
            str(row.get("canonical_feature_name") or "").strip(),
            *parse_jsonish_list(row.get("matching_aliases")),
            # Support older expected inventories.
            *parse_jsonish_list(row.get("aliases")),
        ]
        expected.append(
            {
                "feature_id": row.get("feature_id", ""),
                "canonical": row.get("canonical_feature_name", ""),
                "aliases": sorted({normalize(alias) for alias in aliases if normalize(alias)}),
            }
        )
    return expected


def load_feature_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def row_terms(row: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for field in NAME_FIELDS:
        for value in parse_jsonish_list(row.get(field)):
            normalized = normalize(value)
            if normalized:
                terms.add(normalized)
    return terms


def row_label(row: dict[str, Any]) -> str:
    return str(row.get("feature_name") or row.get("normalized_feature_name") or "").strip()


def default_features_path() -> Path:
    for path in DEFAULT_FEATURE_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_FEATURE_CANDIDATES[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare final extracted features with expected SemEval-2020 Task 11 labels."
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
        help=f"Expected feature inventory CSV. Default: {DEFAULT_EXPECTED}",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=default_features_path(),
        help="Final feature CSV or JSONL to compare. Defaults to the SemEval PDF final output.",
    )
    parser.add_argument(
        "--show-unmatched",
        action="store_true",
        help="Also list final feature rows that did not match any expected label.",
    )
    parser.add_argument(
        "--fail-under",
        type=int,
        default=0,
        help="Exit with status 1 if fewer than this many expected labels match.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.expected.exists():
        raise FileNotFoundError(f"Expected inventory does not exist: {args.expected}")
    if not args.features.exists():
        raise FileNotFoundError(f"Feature output does not exist: {args.features}")

    expected = load_expected(args.expected)
    rows = load_feature_rows(args.features)
    indexed_rows = [(row, row_terms(row)) for row in rows]
    all_terms = sorted({term for _, terms in indexed_rows for term in terms})

    matches: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    missing: list[dict[str, Any]] = []
    matched_row_ids: set[int] = set()

    for item in expected:
        found = None
        for row, terms in indexed_rows:
            overlap = set(item["aliases"]) & terms
            if overlap:
                label_alias = normalize(row_label(row))
                alias = label_alias if label_alias in overlap else sorted(overlap, key=lambda x: (-len(x), x))[0]
                found = (row, alias)
                break
        if found:
            row, alias = found
            matches.append((item, row, alias))
            matched_row_ids.add(id(row))
        else:
            missing.append(item)

    total = len(expected)
    print(f"Expected coverage: {len(matches)}/{total} ({len(matches) / total:.1%})")
    print(f"Expected: {args.expected}")
    print(f"Features: {args.features}")
    print()

    print("Matched")
    for item, row, alias in matches:
        paper_count = row.get("paper_count", "")
        mention_count = row.get("mention_count", "")
        counts = []
        if paper_count not in {"", None}:
            counts.append(f"papers={paper_count}")
        if mention_count not in {"", None}:
            counts.append(f"mentions={mention_count}")
        count_text = f" ({', '.join(counts)})" if counts else ""
        print(f"- {item['canonical']} -> {row_label(row)} [alias={alias}]{count_text}")

    print()
    print("Missing")
    if not missing:
        print("- none")
    for item in missing:
        suggestions = difflib.get_close_matches(item["aliases"][0], all_terms, n=3, cutoff=0.72)
        suggestion_text = f" near: {', '.join(suggestions)}" if suggestions else ""
        print(f"- {item['canonical']}{suggestion_text}")

    if args.show_unmatched:
        print()
        print("Unmatched Final Features")
        for row, _ in indexed_rows:
            if id(row) not in matched_row_ids:
                print(f"- {row_label(row)}")

    if args.fail_under and len(matches) < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
