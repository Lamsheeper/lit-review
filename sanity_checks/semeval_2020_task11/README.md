# SemEval-2020 Task 11 sanity check

The expected inventory supports coverage checks and embedding calibration against known propaganda-technique labels. Keep it separate from model prompts when evaluating feature recovery.

The 18-row analysis inventory decomposes most official merged labels. `Name calling, labeling` and `Exaggeration or minimization` remain combined to match the research inventory, and `Dictatorship` is excluded. Original merged labels are retained in `source_merged_label`.

Use the [media profile pipeline](../../examples/media_attributes/pipeline.json) with your own SemEval corpus. Its checked-in paper is only a synthetic software fixture, so it does not establish benchmark coverage. Preview the configuration first:

```bash
uv run lit-review pipeline run --config examples/media_attributes/pipeline.json --dry-run
```

After running on a real corpus, compare its merged output with the expected inventory:

```bash
uv run python sanity_checks/semeval_2020_task11/compare_expected_features.py \
  --features runs/media_attributes/stages/merge/llm_merged_features.jsonl \
  --show-unmatched
```

Use `--fail-under N` to require a minimum number of recovered expected labels. Compare raw or filtered inventory paths with the same command to determine where labels were lost. Missing raw labels indicate collection/extraction gaps; later losses require inspection of filtering, annotation, and merge decisions.

Older stage-comparison plotting code is retained under `docs/history/benchmarks/` for its historical flat output layout. Current general comparison and visualization commands are documented in [workflows](../../docs/workflows.md).
