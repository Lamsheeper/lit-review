# Migrating from 0.1

Version 0.2 is a breaking release. Old module imports, standalone executable names, flat extraction configs, and implicit working-directory inputs are replaced by the installed package and explicit stage/pipeline configuration.

| Previous tool | New command |
|---|---|
| `lit-harvest` | `lit-review collect search` or `collect citations` |
| `lit-author-harvest` | `lit-review collect authors` |
| `lit-log-summary` | `lit-review collect summary` |
| `lit-synthesize prepare/convert/index/search/write` | `lit-review evidence prepare/convert/index/search/write` |
| `lit-extract run/curate/annotate/merge/score` | `lit-review features extract/filter/annotate/merge/score` |
| `embedding-merge` | `lit-review analysis cluster` |
| `known-feature-similarity` | `lit-review analysis calibrate` |
| `feature-embedding-metadata` | `lit-review analysis metadata` |
| `feature-embedding-tsne` | `lit-review analysis tsne` |
| `comparison-plots` | `lit-review analysis compare` |
| `wikidata-mapping-analysis` | `lit-review analysis wikidata` |
| `lit-review-ui` | `lit-review ui` |

Fixed-count `finalize` is removed. Filtering has no merge, canonicalization, or cap behavior. `curated_features.*` compatibility exports and `merge_style` options are removed. LLM merging accepts category text-field/cleanup settings; cosine thresholds and linkage belong to `analysis cluster`.

## Import deliberately

Preview before choosing a destination:

```bash
uv run lit-review import review_runs/my-old-run --destination runs/imported --dry-run
```

The destination must not exist and must be outside the source. Ambiguous stage outputs from multiple experiments are rejected; select a narrower source directory. Supported project definitions, collection manifests, paper/chunk indexes, feature inventories, and decision artifacts are validated before copying. The import report records source paths and hashes, along with unsupported state.

Import does not restore old SQLite projects or model batch checkpoints. Old PDF/Markdown references may still point to the original corpus. Keep that corpus available when using imported indexes. New runs record their own current stage state; imported artifacts are inputs, not proof that the new pipeline executed successfully.

For example, continue from a filtered inventory using a new taxonomy and run root:

```bash
uv run lit-review features annotate \
  --output-root runs/continued \
  --taxonomy examples/taxonomies/media_attributes.json \
  --goal examples/media_attributes/goal.md \
  --input-file runs/imported/stages/filter/filtered_features.jsonl \
  --set profile=media_attributes
```

Existing `review_runs/`, `ui_projects/`, corpora, faculty inventories, and ignored local configurations are left in place. New UI projects use `projects/`. Historical configurations in `docs/history/` document older experiments and are not runnable 0.2 recipes.
