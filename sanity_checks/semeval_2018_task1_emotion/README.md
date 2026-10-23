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

Preview a name-only calibration experiment through the consolidated toolkit:

```bash
uv run lit-review analysis calibrate \
  --output-root runs/benchmarks/semeval_2018_task1_emotion \
  --set known_set=sanity_checks/semeval_2018_task1_emotion/expected_features.csv \
  --set text_fields=feature_name \
  --set distinct_only=true \
  --dry-run
```

Remove `--dry-run` to execute with configured embedding credentials. For approved synonym controls and attribute-combination experiments, see [analysis configuration](../../docs/workflows.md#analysis). Source-related labels in this inventory remain provenance, not automatically approved synonyms.
