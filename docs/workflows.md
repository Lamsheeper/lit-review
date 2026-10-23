# Workflows and configuration

## Pipeline recipes

A pipeline JSON document has `version: 1`, `output_root`, optional `inputs`, optional `providers`, optional `profile`, and ordered `stages`. Each stage has a unique snake-case `id`, an `operation`, and optional `options`.

```json
{
  "version": 1,
  "output_root": "../../runs/features",
  "inputs": {
    "papers": "../fixtures/papers",
    "taxonomy": "../taxonomies/ergonomics.json",
    "goal": "../fixtures/goal.md"
  },
  "providers": {"provider": "gemini", "model": "gemini-2.5-flash"},
  "stages": [
    {"id": "prepare", "operation": "evidence.prepare"},
    {"id": "extract", "operation": "features.extract"},
    {"id": "filter", "operation": "features.filter"},
    {"id": "annotate", "operation": "features.annotate"},
    {"id": "merge", "operation": "features.merge"},
    {"id": "score", "operation": "features.score"}
  ]
}
```

Relative paths are interpreted relative to the configuration file. Shared inputs supply initial artifacts; subsequent stages use the latest applicable predecessor. Stage options override shared settings. Stage commands accept an options JSON file plus explicit feature input flags and repeated `--set KEY=VALUE` overrides. Relative paths in CLI overrides use the working directory. `--output-root` overrides a pipeline's configured run root.

Unknown options and options that do not apply to an operation are errors. Credentials use `api_key_env` and `embedding_api_key_env`; literal API keys are not accepted in pipeline configuration.

## Collection

`collect search` accepts a draft, source selection, relevance profile, query/download limits, date limits, and the collection controls documented by the typed execution configuration. `collect citations` accepts `core_pdf` paths or `core_pdf_dir` and the citation extractor/provider settings. Collection uses scholarly metadata APIs and open PDF resolution.

`collect authors` accepts `roster`, institution-aware matching settings, and opt-in `download`. It writes a publication inventory and a standalone HTML browser. `collect summary` accepts `logs_dir` and writes a per-service summary.

The new execution service defaults to full PDFs; abstract surrogate files require an explicit option. Optional provider credentials are read from the environment.

## Evidence

- `prepare`: `papers`, optional collection `manifest`; builds the lightweight index for direct-PDF extraction.
- `convert`: the same inputs plus an optional extractor; creates Markdown and a paper index.
- `index`: `run_dir`, with chunk size/overlap controls; writes a separate indexed stage containing the source artifacts.
- `search`: `run_dir`, `query`, optional `limit`; writes retrieved evidence.
- `write`: indexed `run_dir`, `goal`, and optional generation/retrieval controls; writes `report.md`. Without a generation key it uses the existing offline evidence-report path.

The review recipe can run without model credentials. Its small synthetic fixture demonstrates software behavior, not research findings.

## Features

All feature stages require `taxonomy`; `goal` and the explicit media profile are optional. Supported generation settings include provider, model, base URL, reasoning effort, temperature, timeout, token limit, and the credential environment-variable name.

- `extract`: prepared/indexed `run_dir`; `input_mode` is `pdf` by default or `chunks` with an indexed corpus. PDF/chunk feature limits and batching controls are operation-specific.
- `filter`: `input_dir` containing `features.jsonl`; retains original feature identities and writes rejected rows and decision traces.
- `annotate`: `input_file`; writes definitions, aliases, examples, exclusions, and review statuses.
- `merge`: annotated `input_file`; supports `top_k`, pair batching, an optional similarity floor, category-specific embedding fields/cleanup, and embedding provider settings. Only accepted merge decisions combine rows.
- `score`: merged `input_file`; writes support, annotation-reliability, and distinctiveness scores. It is optional and does not gate inventory export.

Source paper IDs/titles, evidence fields, annotation metadata, and merge membership survive consolidation. Required inventories are JSONL; stages also emit CSV and, where applicable, JSON inventories and analysis figures.

## Analysis

Install the optional research dependencies with `uv sync --extra analysis` (or `pip install 'lit-review[analysis]'` for an existing wheel). They enable analysis commands and supplementary feature-stage figures. Collection, review writing, feature inventories, and the UI work with the default installation; inventory exports do not require plotting.

Analysis operations retain their individual algorithm controls, supplied as JSON options or `--set` values. Use argument destination names with underscores. Positional inputs become named options in JSON:

| Operation | Inputs |
|---|---|
| `calibrate` | `known_set`, optional `annotated_features` |
| `metadata` | `features` |
| `cluster` | `features`, threshold or sweep/category controls |
| `compare` | `feature_set_1`, `feature_set_2` |
| `tsne` | `features` |
| `wikidata` | `mapped_csv`, `scored_jsonl`, optional `embedding_merge_trace` |

For example, plot a fixture inventory with configured embeddings:

```bash
uv run lit-review analysis tsne --output-root runs/tsne \
  --set features=examples/fixtures/features_b.jsonl --set html=true --dry-run
```

The example has only two rows; use at least three usable features for an actual t-SNE run. No credential is needed for the preview. Wikidata analysis consumes existing mapping artifacts; it does not perform new entity mapping.
