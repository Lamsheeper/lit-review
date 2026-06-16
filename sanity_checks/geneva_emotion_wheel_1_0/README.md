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

Run the full attribute comparison. It automatically starts or resumes the
durable human synonym review before continuing:

```bash
uv run known-feature-similarity \
  sanity_checks/geneva_emotion_wheel_1_0/expected_features.csv \
  --embed-attribute-comparison \
  --output-dir review_runs/known_feature_similarity/geneva_emotion_wheel_1_0
```
