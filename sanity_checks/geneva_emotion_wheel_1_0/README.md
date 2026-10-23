# Geneva Emotion Wheel 1.0 Inventory

This known set contains the 16 atomic emotion terms in prototype version 1.0
of the Geneva Emotion Wheel. Each term is a separate analysis feature.

The 2012 University of Geneva rating-study report distinguishes this
16-term first prototype from GEW 2.0, which contains 40 terms arranged into
20 emotion families.

Source:

- Sacharin, V., Schlegel, K., & Scherer, K. R. (2012). *Geneva Emotion Wheel
  Rating Study*. University of Geneva, Swiss Center for Affective Sciences:
  <https://www.unige.ch/cisa/download_file/view/136/245>

Preview a name-only calibration experiment through the consolidated toolkit:

```bash
uv run lit-review analysis calibrate \
  --output-root runs/benchmarks/geneva_emotion_wheel_1_0 \
  --set known_set=sanity_checks/geneva_emotion_wheel_1_0/expected_features.csv \
  --set text_fields=feature_name \
  --set distinct_only=true \
  --dry-run
```

Remove `--dry-run` to execute with configured embedding credentials. For approved synonym controls and attribute-combination experiments, see [analysis configuration](../../docs/workflows.md#analysis). Source-related labels in this inventory remain provenance, not automatically approved synonyms.
