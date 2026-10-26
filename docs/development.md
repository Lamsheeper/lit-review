# Development

Install the development dependencies, then build the frontend:

```bash
uv sync --extra dev --extra analysis
npm --prefix src/lit_review/ui/frontend ci
npm --prefix src/lit_review/ui/frontend run build
```

Run the release checks:

```bash
uv run ruff check src tests
uv run python -m unittest discover -s tests
npm --prefix src/lit_review/ui/frontend test
uv build
uv run python scripts/verify_release.py dist
```

The installation check installs the wheel outside the checkout, exercises the command and bundled UI, and rebuilds a wheel from the source archive. Live provider calls are mocked in the tests.

Regenerate the README workflow animation with `uv run --extra dev python scripts/render_readme.py`. The animation follows the question “What emotions are described in psychology research?” through collection, PDF selection, extraction, filtering, annotation, and merging. It ends with an emotion inventory and fictional author labels such as Zhang et al. The question moves into a persistent panel so the goal stays visible through every step. All displayed records, author labels, and quotations are fictional examples. The UI image is a capture of the running application with an isolated example project, not a live collection session.

Local projects default to `projects/`; override this with `LIT_REVIEW_UI_PROJECTS_DIR`. Keep corpora, credentials, caches, and generated reports outside version control. [Historical material](history/README.md) and [benchmark inventories](../sanity_checks/) are separate from current usage instructions.
