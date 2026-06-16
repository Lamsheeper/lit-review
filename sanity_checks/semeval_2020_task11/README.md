# SemEval-2020 Task 11 Sanity Check

This run checks whether the extraction pipeline can recover the official
SemEval-2020 Task 11 propaganda-technique inventory from a corpus collected
using the task itself as the search target, while keeping the main project
intent intact.

The harvest config uses the root `draft.md`; it just fills all six allowed
queries with SemEval-2020 Task 11 searches, so the collected corpus stays
limited to that task. The extraction, curation, and finalization configs use
the root `goal.md`. Use `expected_features.csv` only after extraction, when
scoring coverage.

## Commands

Preview the harvest queries without network calls:

```bash
uv run lit-harvest --config harvest_configs/config.semeval_2020_task11_sanity.json --dry-run
```

Run the harvest:

```bash
uv run lit-harvest --config harvest_configs/config.semeval_2020_task11_sanity.json
```

Convert and index the collected PDFs:

```bash
uv run lit-synthesize --config synth_configs/00_convert_semeval_2020_task11_sanity.json
uv run lit-synthesize --config synth_configs/01_index_semeval_2020_task11_sanity.json
```

Preview the extraction workload:

```bash
uv run lit-extract --config extract_configs/00_extract_semeval_2020_task11_pdf.json --dry-run
```

Run extraction, curation, and finalization:

```bash
uv run lit-extract --config extract_configs/00_extract_semeval_2020_task11_pdf.json
uv run lit-extract --config extract_configs/01_curate_semeval_2020_task11_pdf.json
uv run lit-extract --config extract_configs/02_finalize_semeval_2020_task11_pdf.json
```

## Expected Artifacts

- `collected_semeval_2020_task11_papers/logs/manifest.json`
- `collected_semeval_2020_task11_papers/logs/candidates.csv`
- `review_runs/semeval_2020_task11_sanity/paper_index.json`
- `review_runs/semeval_2020_task11_sanity/index.sqlite`
- `review_runs/semeval_2020_task11_sanity/feature_extract_pdf/features.csv`
- `review_runs/semeval_2020_task11_sanity/feature_extract_pdf/curated_features.csv`
- `review_runs/semeval_2020_task11_sanity/feature_extract_pdf/final_curated_features_llm_top100_by_category.csv`

## Coverage Check

After finalization, compare the extracted feature names against the expected
18-row analysis inventory. Most official merged labels are decomposed, while
`Name calling, labeling` and `Exaggeration or minimization` remain combined to
match the project inventory, and `Dictatorship` is excluded. Original merged
labels are retained in `source_merged_label`:

```bash
uv run python sanity_checks/semeval_2020_task11/compare_expected_features.py
```

To compare a different feature artifact:

```bash
uv run python sanity_checks/semeval_2020_task11/compare_expected_features.py \
  --features review_runs/semeval_2020_task11_sanity/feature_extract_pdf/final_curated_features_top100_by_category.csv
```

## Figures

Generate matplotlib charts for coverage, final label support, and category mix:

```bash
uv run python sanity_checks/semeval_2020_task11/plot_expected_features.py
```

Figures are written to:

```text
review_runs/semeval_2020_task11_sanity/feature_extract_pdf/figures/
```

## Interpretation

Green: final curated output recovers most or all official labels, including
Loaded language, Name calling, Repetition, and Appeal to fear/prejudice.

Yellow: labels appear in raw `features.csv` but are lost during curation or
finalization.

Red: common labels such as Loaded language, Name calling, Repetition, or
Appeal to fear/prejudice are absent from raw extraction.
