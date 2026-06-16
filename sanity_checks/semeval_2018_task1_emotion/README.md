# SemEval-2018 Task 1 Emotion Inventory

This known set contains the eleven emotion labels used by the multi-label
emotion classification subtask (E-c) of SemEval-2018 Task 1: Affect in Tweets:

- anger
- anticipation
- disgust
- fear
- joy
- love
- optimism
- pessimism
- sadness
- surprise
- trust

The set is close in size to the fourteen-label official merged SemEval-2020
Task 11 persuasion inventory and provides a well-known affective NLP benchmark
for comparing embedding-similarity distributions.

The `related_terms` in `expected_features.csv` come from the task's official
annotation instructions, where each emotion is followed by terms it "also
includes." They are retained as provenance only and are not matching aliases or
synonym controls. Synonym controls require separate human approval.

Source:

- Saif Mohammad, Felipe Bravo-Marquez, Mohammad Salameh, and Svetlana
  Kiritchenko. 2018. "SemEval-2018 Task 1: Affect in Tweets."
- ACL Anthology: <https://aclanthology.org/S18-1001/>
- DOI: <https://doi.org/10.18653/v1/S18-1001>

Run a name-only experiment:

```bash
uv run known-feature-similarity \
  sanity_checks/semeval_2018_task1_emotion/expected_features.csv \
  --annotated-features review_runs/media_attributes_v3/feature_extract_pdf/annotated/annotated_features.json \
  --text-fields feature_name \
  --distinct-only \
  --output-dir review_runs/known_feature_similarity/semeval_2018_task1_emotion/name_only
```

Run an experiment using names, definitions, and synonyms. The command
automatically starts or resumes the required human synonym review:

```bash
uv run known-feature-similarity \
  sanity_checks/semeval_2018_task1_emotion/expected_features.csv \
  --annotated-features review_runs/media_attributes_v3/feature_extract_pdf/annotated/annotated_features.json \
  --text-fields feature_name,definitions,synonyms \
  --synonym-review review_runs/known_feature_similarity/semeval_2018_task1_emotion/expected_features_synonym_review.json \
  --output-dir review_runs/known_feature_similarity/semeval_2018_task1_emotion/name_def_synonyms
```
