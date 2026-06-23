#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_DIR="review_runs/known_feature_similarity/geneva_emotion_wheel"
BASELINE_DIR="$BASE_DIR/v2"
OUT_DIR="$BASE_DIR/v3_lexical_cut_controlled"

SYNONYM_REVIEW="$BASELINE_DIR/expected_features_synonym_review.json"
SOURCE_ATTRIBUTE_CACHE="$BASELINE_DIR/expected_features_generated_attribute_cache.jsonl"
SOURCE_EMBEDDING_CACHE="$BASELINE_DIR/expected_features_gemini_embedding_cache.jsonl"

ATTRIBUTE_CACHE="$OUT_DIR/expected_features_generated_attribute_cache.seeded_from_v2.jsonl"
EMBEDDING_CACHE="$OUT_DIR/expected_features_gemini_embedding_cache.seeded_from_v2.jsonl"

mkdir -p "$OUT_DIR"

if [ ! -f "$ATTRIBUTE_CACHE" ]; then
  cp "$SOURCE_ATTRIBUTE_CACHE" "$ATTRIBUTE_CACHE"
fi

if [ ! -f "$EMBEDDING_CACHE" ]; then
  cp "$SOURCE_EMBEDDING_CACHE" "$EMBEDDING_CACHE"
fi

uv run known-feature-similarity \
  sanity_checks/geneva_emotion_wheel_1_0/expected_features.csv \
  --annotated-features review_runs/media_attributes_v3/feature_extract_pdf/annotated/annotated_features.json \
  --embed-attribute-comparison \
  --synonym-review "$SYNONYM_REVIEW" \
  --attribute-generation-cache "$ATTRIBUTE_CACHE" \
  --cache "$EMBEDDING_CACHE" \
  --embedding-text-cleanup-builtin sentiment_affect_high_confidence_boilerplate \
  --embedding-text-cleanup-fields definitions \
  --embedding-text-cleanup-min-chars 24 \
  --embedding-text-cleanup-fallback original \
  --output-dir "$OUT_DIR"

uv run python - "$BASELINE_DIR" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

baseline = Path(sys.argv[1])
current = Path(sys.argv[2])
non_definition_combinations = [
    "name",
    "synonyms",
    "examples",
    "name_synonyms",
    "name_examples",
    "synonyms_examples",
]
filenames = [
    "expected_features_pairwise_similarities.jsonl",
    "expected_features_synonym_pairwise_similarities.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


for combination in non_definition_combinations:
    for filename in filenames:
        left = read_jsonl(baseline / "combinations" / combination / filename)
        right = read_jsonl(current / "combinations" / combination / filename)
        if left != right:
            raise SystemExit(
                f"Non-definition control changed: {combination}/{filename}"
            )

print("Non-definition control combinations match v2 exactly.")
PY
