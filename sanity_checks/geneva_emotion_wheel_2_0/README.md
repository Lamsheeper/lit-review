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

Preview a name-only calibration experiment through the consolidated toolkit:

```bash
uv run lit-review analysis calibrate \
  --output-root runs/benchmarks/geneva_emotion_wheel_2_0 \
  --set known_set=sanity_checks/geneva_emotion_wheel_2_0/expected_features.csv \
  --set text_fields=feature_name \
  --set distinct_only=true \
  --dry-run
```

Remove `--dry-run` to execute with configured embedding credentials. For approved synonym controls and attribute-combination experiments, see [analysis configuration](../../docs/workflows.md#analysis). Source-related labels in this inventory remain provenance, not automatically approved synonyms.
