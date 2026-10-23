# Portable examples

Preview every recipe with `lit-review pipeline run --config <recipe> --dry-run`.

- `collection/pipeline.json`: a small scholarly search/download run and service summary; actual execution uses network APIs.
- `features/pipeline.json`: general-taxonomy feature discovery using the synthetic ergonomics paper; actual execution needs generation and embedding credentials.
- `media_attributes/pipeline.json`: explicit media domain rules and the historical research goal.
- `review/pipeline.json`: local conversion, indexing, and offline review writing; no model key required.
- `analysis/pipeline.json`: local feature-set comparison; install `uv sync --extra analysis`, no credentials required.

`fixtures/papers/workplace.pdf` is an original synthetic software fixture. It is not a published paper and its contents are not research evidence. Small JSONL inventories support offline analysis examples. Output directories resolve to `runs/` at the repository root.
