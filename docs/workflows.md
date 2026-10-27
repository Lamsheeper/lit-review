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

Unknown options and options that do not apply to an operation are errors. API key credentials use `api_key_env` and `embedding_api_key_env`; Google Cloud uses ADC. Literal credentials are not accepted in pipeline configuration.

## Collection

`collect search` accepts a draft, source selection, relevance profile, query/download limits, date limits, and the collection controls documented by the typed execution configuration. `collect citations` accepts `core_pdf` paths or `core_pdf_dir` and the citation extractor/provider settings. Collection uses scholarly metadata APIs and open PDF resolution.

`collect authors` accepts `roster`, institution-aware matching settings, and opt-in `download`. It writes a publication inventory and a standalone HTML browser. `collect summary` accepts `logs_dir` and writes a per-service summary.

The new execution service defaults to full PDFs; abstract surrogate files require an explicit option. Optional provider credentials are read from the environment.

## Evidence

- `prepare`: `papers`, optional collection `manifest`; builds the lightweight index for direct-PDF extraction.
- `convert`: the same inputs plus an optional extractor; creates Markdown and a paper index.
- `index`: `run_dir`, with chunk size/overlap controls; writes a separate indexed stage containing the source artifacts.
- `search`: `run_dir`, `query`, optional `limit`; writes retrieved evidence.
- `write`: indexed `run_dir`, `goal`, and optional generation/retrieval controls; writes `report.md`. Without configured generation credentials it uses the existing offline evidence-report path.

The review recipe can run without model credentials. Its small synthetic fixture demonstrates software behavior, not research findings.

## Authentication

Gemini generation (including PDF extraction, citation collection, review writing, and feature processing) and embeddings support API keys and Vertex AI ADC. OpenAI-compatible providers continue to use API keys. The default installation includes the Google authentication library; the gcloud executable is needed only to set up a local user login, not for each model request.

For local Google Cloud use, run `gcloud auth application-default login`, and set `GOOGLE_CLOUD_PROJECT` and optionally `GOOGLE_CLOUD_LOCATION` (default `us-central1`) before starting the app or CLI. Use a project with billing, the Vertex AI API enabled, and an account with Vertex AI permissions. `gcloud auth login` alone does not configure ADC. Google documents [ADC setup](https://cloud.google.com/docs/authentication/provide-credentials-adc) and [Vertex AI authentication](https://cloud.google.com/vertex-ai/docs/authentication).

Provider configuration accepts:

```json
{
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "auth_mode": "adc",
  "vertex_project": "your-project-id",
  "vertex_location": "us-central1"
}
```

Put these fields under `providers` in a pipeline or at the top level of a feature stage config. For a ready-to-run structure, copy [the Vertex AI recipe](../examples/features/pipeline.vertex.json) and set your project environment variable. Existing API key recipes also use Vertex AI when `GOOGLE_GENAI_USE_VERTEXAI=true` is set.

Authentication selection is deterministic:

- `auth_mode: "api_key"` forces API key authentication, even when Google Cloud environment variables are set.
- `auth_mode: "adc"` forces Vertex AI and requires `vertex_project` or `GOOGLE_CLOUD_PROJECT`; API keys are ignored.
- `auth_mode: "auto"` (default) selects Vertex AI when an explicit `vertex_project` is supplied, when `GOOGLE_GENAI_USE_VERTEXAI=true` is set, or when no Gemini key is available and `GOOGLE_CLOUD_PROJECT` is set. Otherwise it uses the API key.

Explicit project and location fields override environment values. Generation and embedding model availability varies by region. `global` uses the global endpoint for models that support it; use a supported region such as `us-central1` for `gemini-embedding-001`. Never put access tokens or credential JSON into pipeline files. ADC also supports `GOOGLE_APPLICATION_CREDENTIALS`, workload identity, and attached service accounts without a gcloud installation.

Merge stages inherit generation authentication and accept independent `embedding_auth_mode`, `embedding_vertex_project`, and `embedding_vertex_location` overrides. For example, an OpenAI generation provider can use Vertex embeddings by setting `embedding_auth_mode: "adc"` and `embedding_vertex_project`. `embedding_api_key_env` continues to select an embedding API key environment variable.

Citation collection uses `llm_provider: "gemini"`, `llm_model`, and the same `auth_mode` / `vertex_project` / `vertex_location` fields in standalone configs. Pipeline citation stages inherit the shared provider settings. Standalone review writing accepts `provider: "gemini"` with these same fields; writing without any credentials still uses the offline report path.

Analysis operations `calibrate`, `metadata`, `cluster`, `compare`, and `tsne` accept `auth_mode`, `vertex_project`, and `vertex_location` through JSON or `--set`:

```bash
uv run lit-review analysis tsne --output-root runs/tsne \
  --set features=path/to/features.jsonl --set auth_mode=adc \
  --set vertex_project=your-project-id --set vertex_location=us-central1
```

Cache-only analysis still works without credentials. For API key authentication, use `GEMINI_API_KEY` or `GOOGLE_API_KEY` for Gemini and `OPENAI_API_KEY` for OpenAI; explicit `api_key_env` selects a different environment variable. UI session keys are passed to background jobs in memory.

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
