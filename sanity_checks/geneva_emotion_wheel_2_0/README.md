# Geneva Emotion Wheel 2.0 Inventory

This known set contains the 40 atomic emotion terms from prototype version 2.0
of the Geneva Emotion Wheel (GEW). The University of Geneva's rating-study
report describes the terms as arranged into 20 emotion families.

Each row in `expected_features.csv` is one separate analysis feature. The
original pair is retained in `source_family` as provenance only; family
membership never creates synonym controls. Each term also retains its predicted
valence and control quadrant.

The report found that the valence placements were generally well supported,
although compassion was rated positive rather than its predicted negative
placement. Control placements were less consistently supported. Those
dimensions are descriptive metadata here and do not affect matching or
embedding text unless explicitly requested with `--text-fields`.

Sources:

- University of Geneva, Swiss Center for Affective Sciences:
  <https://www.unige.ch/cisa/gew>
- Sacharin, V., Schlegel, K., & Scherer, K. R. (2012). *Geneva Emotion Wheel
  Rating Study*. University of Geneva, Swiss Center for Affective Sciences:
  <https://www.unige.ch/cisa/download_file/view/136/245>

Run a distinct-feature-only experiment:

```bash
uv run known-feature-similarity \
  sanity_checks/geneva_emotion_wheel_2_0/expected_features.csv \
  --annotated-features review_runs/media_attributes_v3/feature_extract_pdf/annotated/annotated_features.json \
  --text-fields feature_name \
  --distinct-only \
  --output-dir review_runs/known_feature_similarity/geneva_emotion_wheel_2_0/name_only
```

Run an experiment using names, definitions, and synonyms. The command
automatically starts or resumes the required human synonym review:

```bash
uv run known-feature-similarity \
  sanity_checks/geneva_emotion_wheel_2_0/expected_features.csv \
  --annotated-features review_runs/media_attributes_v3/feature_extract_pdf/annotated/annotated_features.json \
  --text-fields feature_name,definitions,synonyms \
  --synonym-review review_runs/known_feature_similarity/geneva_emotion_wheel_2_0/expected_features_synonym_review.json \
  --output-dir review_runs/known_feature_similarity/geneva_emotion_wheel_2_0/name_def_synonyms
```
