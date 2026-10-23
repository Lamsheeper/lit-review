#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAPPED_CSV="review_runs/media_attributes_v3/feature_extract_pdf/wikidata_map/wikidata_mapped_features.csv"
SCORED_JSONL="review_runs/media_attributes_v3/feature_extract_pdf/scored/scored_features.jsonl"
EMBEDDING_TRACE="review_runs/media_attributes_v3/feature_extract_pdf/embedding_merge/agm_v5_lexical_cleanup/annotated_features_embedding_metadata_features_embedding_merge_trace.json"
OUT_DIR="${1:-review_runs/media_attributes_v3/feature_extract_pdf/wikidata_map/analysis/persuasion}"

mkdir -p "$OUT_DIR"

uv run wikidata-mapping-analysis \
  "$MAPPED_CSV" \
  "$SCORED_JSONL" \
  --category persuasion \
  --output-dir "$OUT_DIR" \
  --embedding-merge-trace "$EMBEDDING_TRACE"
