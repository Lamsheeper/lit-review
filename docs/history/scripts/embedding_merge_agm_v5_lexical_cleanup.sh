#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT="review_runs/media_attributes_v3/feature_extract_pdf/embedding_metadata/all_categories/annotated_features_embedding_metadata_features.json"
CONFIG="merge_configs/agm_v5.json"
OUT_DIR="${1:-review_runs/media_attributes_v3/feature_extract_pdf/embedding_merge/agm_v5_lexical_cleanup}"
CACHE="$OUT_DIR/annotated_features_embedding_metadata_features_gemini_embedding_cache.jsonl"

mkdir -p "$OUT_DIR"

uv run embedding-merge \
  "$INPUT" \
  --category-config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --cache "$CACHE"
