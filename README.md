# Discovering features from literature collection

Collect research papers and build a feature inventory with definitions and supporting evidence.

![The question “What emotions are described in psychology research?” stays visible through collection, PDF selection, extraction, filtering, annotation, and merging into an emotion inventory.](docs/assets/workflow.gif)

## Install and open the UI

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Node.js 22+ to build the UI from source.

```bash
git clone https://github.com/Lamsheeper/lit-review.git
cd lit-review
uv sync
npm --prefix src/lit_review/ui/frontend ci
npm --prefix src/lit_review/ui/frontend run build
uv run lit-review ui
```

Open **[localhost:8765](http://127.0.0.1:8765)**. Add your Gemini or OpenAI API key under **Credentials**; it stays in memory for the session. For terminal runs, set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) or `OPENAI_API_KEY` in your environment. Gemini generation and embeddings also support Google Cloud through Vertex AI and Application Default Credentials (ADC).

To use your Google Cloud login, enable the Vertex AI API in a project with billing and grant your account access to the models, then run:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI=true
uv run lit-review ui
```

Choose **Google Cloud (ADC)** in the Gemini authentication settings, or leave **Automatic** to use these environment settings. Project and region can also be entered in the UI. Merging uses the same authentication for embeddings unless you override its embedding settings. ADC tokens refresh automatically; workloads can also use `GOOGLE_APPLICATION_CREDENTIALS` or an attached service account. See [authentication and configuration](docs/workflows.md#authentication) for CLI examples and precedence.

Create a project, define your question and taxonomy, then follow **Collect → Select/download → Extract → Filter → Annotate → Merge**. Inspect decisions and supporting evidence as you go; export the merged inventory to CSV or JSONL. Scoring is optional.

![Lit Review Studio showing an example project.](docs/assets/studio.png)

## Run a pipeline from the terminal

Start with a [ready-to-edit recipe](examples/README.md) for collection, review writing, feature discovery, or analysis. Advanced analysis and supplementary plots are optional: `uv sync --extra analysis`.

```bash
# Preview inputs and planned outputs without calling providers.
uv run lit-review pipeline run --config examples/features/pipeline.json --dry-run

# Run it; reuse matching completed stages on subsequent runs.
uv run lit-review pipeline run --config examples/features/pipeline.json
```

Use your own PDFs and taxonomy, or try the bundled synthetic example. Each run saves stage outputs and a manifest under its configured output root.

[Commands and configuration](docs/workflows.md) · [Local API](http://127.0.0.1:8765/docs) · [Importing older runs](docs/migration.md) · [Development](docs/development.md) · [Architecture](docs/architecture.md)
