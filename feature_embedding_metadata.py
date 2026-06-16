#!/usr/bin/env python3
"""Generate clean, independent metadata for feature embedding experiments."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from feature_embedding_tsne import (
    DEFAULT_GEMINI_BASE_URL,
    gemini_api_key,
    read_feature_rows,
    sanitize_filename_part,
)
from known_feature_similarity import (
    ATTRIBUTE_GENERATION_MODEL,
    apply_generated_attributes,
    feature_identity,
    generate_attribute_metadata,
    generated_attributes_are_complete,
    normalize_generated_attributes,
    write_jsonl,
)
from lit_extract import make_llm_client


def parse_categories(value: str | Iterable[str] | None) -> set[str]:
    if value is None:
        return set()
    values = value.split(",") if isinstance(value, str) else value
    return {str(category).strip() for category in values if str(category).strip()}


def embedding_metadata_output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "features_json": output_dir / f"{prefix}_embedding_metadata_features.json",
        "features_jsonl": output_dir / f"{prefix}_embedding_metadata_features.jsonl",
        "generated_attributes_json": output_dir / f"{prefix}_embedding_metadata_generated_attributes.json",
        "trace_json": output_dir / f"{prefix}_embedding_metadata_trace.json",
    }


def load_embedding_metadata_overrides(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows") or payload.get("features") or payload.get("overrides")
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list or an object with rows, features, or overrides.")

    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path} override {index} must be a JSON object.")
        label = str(row.get("feature_name") or "").strip()
        category = str(row.get("category") or "").strip()
        if not label or not category:
            raise ValueError(f"{path} override {index} requires feature_name and category.")
        attributes = normalize_generated_attributes(row, label, category)
        if not generated_attributes_are_complete(attributes):
            raise ValueError(
                f"{path} override {index} for {label!r} requires one definition, "
                "at least one close synonym, and at least two examples."
            )
        overrides[feature_identity(label, category)] = attributes
    return overrides


def run_embedding_metadata_generation(
    *,
    features_path: Path,
    output_dir: Path,
    prefix: str,
    cache_path: Path,
    client: Any | None,
    categories: Iterable[str] = (),
    model: str = ATTRIBUTE_GENERATION_MODEL,
    refresh: bool = False,
    overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
    overrides_path: Path | None = None,
) -> dict[str, Any]:
    rows = read_feature_rows(features_path)
    selected_categories = parse_categories(categories)
    available_categories = {
        str(row.get("category") or "uncategorized").strip() or "uncategorized"
        for row in rows
    }
    missing_categories = sorted(selected_categories - available_categories)
    if missing_categories:
        raise ValueError(f"Categories absent from input: {', '.join(missing_categories)}.")

    selected_rows = [
        row
        for row in rows
        if not selected_categories
        or (str(row.get("category") or "uncategorized").strip() or "uncategorized") in selected_categories
    ]
    if not selected_rows:
        raise ValueError("No feature rows selected for embedding metadata generation.")

    manual_overrides = overrides or {}
    selected_identities = {
        feature_identity(row.get("feature_name"), row.get("category") or "uncategorized")
        for row in selected_rows
    }
    unknown_overrides = sorted(set(manual_overrides) - selected_identities)
    if unknown_overrides:
        raise ValueError(
            "Embedding metadata overrides do not match selected input rows: "
            + ", ".join(f"{name} ({category})" for name, category in unknown_overrides)
            + "."
        )
    rows_to_generate = [
        row
        for row in selected_rows
        if feature_identity(row.get("feature_name"), row.get("category") or "uncategorized") not in manual_overrides
    ]
    generated, generation_trace = generate_attribute_metadata(
        rows_to_generate,
        cache_path=cache_path,
        client=client,
        model=model,
        refresh=refresh,
    )
    generated.update(copy.deepcopy(manual_overrides))
    generated_rows = apply_generated_attributes(selected_rows, generated, model=model)
    generated_by_identity = {
        (
            str(row.get("feature_key") or ""),
            str(row.get("feature_name") or ""),
            str(row.get("category") or "uncategorized"),
        ): row
        for row in generated_rows
    }
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        identity = (
            str(row.get("feature_key") or ""),
            str(row.get("feature_name") or ""),
            str(row.get("category") or "uncategorized"),
        )
        output_rows.append(copy.deepcopy(generated_by_identity.get(identity, row)))

    paths = embedding_metadata_output_paths(output_dir, prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["features_json"].write_text(
        json.dumps(
            {
                "input": str(features_path),
                "feature_count": len(output_rows),
                "generated_feature_count": len(generated_rows),
                "features": output_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(paths["features_jsonl"], output_rows)
    paths["generated_attributes_json"].write_text(
        json.dumps(
            {
                "row_count": len(generated),
                "rows": sorted(
                    generated.values(),
                    key=lambda row: (
                        str(row.get("category") or "").casefold(),
                        str(row.get("feature_name") or "").casefold(),
                    ),
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    trace = {
        "project": "FeatureEmbeddingMetadata",
        "input": str(features_path),
        "categories": sorted(selected_categories) if selected_categories else sorted(available_categories),
        "model": model,
        "cache_path": str(cache_path),
        "overrides_path": str(overrides_path) if overrides_path else None,
        "counts": {
            "input_features": len(rows),
            "selected_features": len(selected_rows),
            "generated_features": len(generated_rows),
            "manual_overrides": len(manual_overrides),
            "output_features": len(output_rows),
        },
        "generation": generation_trace,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["trace_json"].write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate independent definitions, close synonyms, and examples from feature name "
            "and category for embedding experiments."
        ),
    )
    parser.add_argument("features", type=Path, help="Feature CSV, JSONL, or JSON file.")
    parser.add_argument(
        "--categories",
        help="Comma-separated categories to generate. Defaults to every category.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for generated inventory and trace.")
    parser.add_argument("--prefix", help="Output filename prefix. Defaults to the input stem.")
    parser.add_argument("--cache", type=Path, help="Generated-attribute cache JSONL path.")
    parser.add_argument(
        "--overrides",
        type=Path,
        help="Optional reviewed JSON metadata overrides for labels that cannot be generated automatically.",
    )
    parser.add_argument("--model", default=ATTRIBUTE_GENERATION_MODEL, help="Gemini generation model.")
    parser.add_argument("--gemini-api-key", help="Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY.")
    parser.add_argument("--base-url", default=DEFAULT_GEMINI_BASE_URL, help="Gemini API base URL.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Gemini request timeout in seconds.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate metadata even when matching cache entries exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.features.exists():
            raise FileNotFoundError(f"Feature file does not exist: {args.features}")
        if args.timeout <= 0:
            raise ValueError("--timeout must be greater than 0.")
        if args.overrides and not args.overrides.exists():
            raise FileNotFoundError(f"Override file does not exist: {args.overrides}")
        output_dir = args.output_dir or args.features.parent / "embedding_metadata"
        prefix = sanitize_filename_part(args.prefix or args.features.stem)
        cache_path = args.cache or output_dir / f"{prefix}_embedding_metadata_cache.jsonl"
        api_key = gemini_api_key(args.gemini_api_key)
        client = None
        if api_key:
            client = make_llm_client(
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                reasoning_effort="low",
                api_provider="gemini",
                input_mode="text",
                timeout=args.timeout,
            )
        trace = run_embedding_metadata_generation(
            features_path=args.features,
            output_dir=output_dir,
            prefix=prefix,
            cache_path=cache_path,
            client=client,
            categories=parse_categories(args.categories),
            model=args.model,
            refresh=args.refresh,
            overrides=load_embedding_metadata_overrides(args.overrides),
            overrides_path=args.overrides,
        )
    except Exception as exc:
        print(f"feature_embedding_metadata: {exc}", file=sys.stderr)
        return 1

    counts = trace["counts"]
    print(
        f"Generated embedding metadata for {counts['generated_features']} of "
        f"{counts['input_features']} features"
    )
    for label, path in trace["outputs"].items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
