# LitHarvest

## Project name
**LitHarvest: Draft-to-PDF Literature Collection Pipeline**

## Purpose
LitHarvest is a focused document-collection project for literature reviews. Its job is not to summarize papers or run downstream extraction. Its job is to:

1. read an initial paper draft,
2. extract relevant keywords and search phrases,
3. search scholarly metadata sources that do **not** require CAPTCHA bypassing,
4. identify likely relevant papers,
5. resolve open or directly downloadable PDF links when available, and
6. download those PDFs into a local folder for downstream processing.

The output of this project is a folder of collected PDFs plus a metadata manifest describing what was found, what was downloaded, and what could not be downloaded.

---

## Core constraint
This project **must not** rely on Google Scholar scraping or any workflow that requires solving CAPTCHAs.

Instead, it should use API-first or officially supported sources such as:

- **OpenAlex** for scholarly search and metadata
- **Semantic Scholar API** for paper search and related-paper discovery
- **Crossref REST API** for DOI-centric metadata lookup
- **Europe PMC** for biomedical and life-science literature
- **Unpaywall** for locating open-access full-text URLs by DOI
- **Zotero** as an optional organizer/export target, not as the primary search engine

---

## High-level workflow

### Step 1: Analyze the initial draft
Input:
- a paper draft in `.txt`, `.md`, or extracted plain text form

Goal:
- identify the most important topical signals from the draft

Outputs from this step:
- candidate keywords
- candidate noun phrases
- possible domain terms
- possible author-supplied terms from title, abstract, headings, and repeated terminology
- search queries suitable for metadata APIs

### Step 2: Build search queries
From the extracted terms, generate several query variants:

- exact technical phrases
- broader topic phrases
- narrower subtopic phrases
- title-like search phrases
- boolean-style combinations where supported

Example query buckets:
- **primary query**: closest to the draft's core topic
- **related query**: adjacent concepts or alternative naming
- **method query**: method-focused terms
- **application query**: domain/application-specific terms

### Step 3: Search non-CAPTCHA scholarly sources
Search these sources in parallel or in sequence:

1. **OpenAlex**
2. **Semantic Scholar**
3. **Crossref**
4. **Europe PMC** when relevant

For each source, capture:
- title
- authors
- year
- DOI
- abstract if available
- venue
- source API name
- landing page URL
- candidate PDF URL if available
- relevance score

### Step 4: Resolve downloadable PDFs
Preferred order:

1. direct PDF link returned by the source
2. open-access location resolved through DOI via **Unpaywall**
3. repository or full-text location provided in source metadata
4. skip if no allowed downloadable PDF is available

Important rule:
- The system should only download openly accessible or directly accessible PDFs from allowed sources.
- It should not attempt to bypass paywalls, CAPTCHAs, login walls, or robot protections.

### Step 5: Save PDFs to a folder
Each successfully downloaded PDF is stored in a target directory, for example:

```text
./collected_papers/
```

Suggested filename pattern:

```text
year_firstauthor_shorttitle_doihash.pdf
```

Also save:
- `manifest.json` with structured metadata for all candidates
- `download_log.json` with success/failure details

---

## Functional requirements

### Required inputs
- initial draft file path
- output folder path
- maximum number of queries to generate
- maximum papers to retrieve per source
- optional year filters
- optional domain/source filters
- optional email for polite API usage where recommended
- optional Unpaywall email
- optional Semantic Scholar API key
- optional Zotero library configuration

### Required outputs
- downloaded PDFs in target folder
- manifest JSON with all discovered candidates
- list of failed downloads and reasons
- deduplicated record set across all sources

### Required behaviors
- keyword extraction from input draft
- query generation from extracted terms
- API-based search across supported sources
- DOI-based deduplication
- PDF resolution and download
- safe filename generation
- timeout and retry handling
- rate-limiting awareness
- clean logging

### Nice-to-have behaviors
- citation chaining from seed paper DOI or references
- related-paper expansion
- reranking using text similarity to the draft
- optional Zotero export
- optional BibTeX export
- optional CSV report

---

## Non-goals
This project is **not** responsible for:

- OCR of downloaded PDFs
- feature extraction from the collected papers
- summarization of collected papers
- citation synthesis into the draft
- paywalled access automation
- browser automation against CAPTCHA-protected sites

Those can happen downstream after the PDFs are placed in the folder.

---

## Recommended architecture

```text
initial_draft.txt
    -> keyword extraction
    -> query generation
    -> source search
    -> candidate merge and dedupe
    -> PDF resolution
    -> PDF download
    -> collected_papers/
    -> manifest.json
```

### Modules

#### 1. `draft_reader`
Reads the draft and normalizes text.

#### 2. `keyword_extractor`
Extracts candidate phrases from the draft.
Initial simple options:
- title and heading extraction
- repeated noun-like phrases
- n-gram frequency with stopword filtering

Later options:
- KeyBERT
- YAKE
- spaCy noun chunks
- LLM-assisted term extraction

#### 3. `query_builder`
Turns extracted terms into a small set of strong search queries.

#### 4. `search_clients`
API clients for:
- OpenAlex
- Semantic Scholar
- Crossref
- Europe PMC

#### 5. `candidate_ranker`
Ranks papers by similarity to the draft text and keyword overlap.

#### 6. `pdf_resolver`
Attempts to find a downloadable PDF URL.

#### 7. `downloader`
Downloads PDFs and validates content type.

#### 8. `manifest_writer`
Writes structured JSON logs for downstream processing.

---

## Search source strategy

### OpenAlex
Use as a primary general-purpose scholarly metadata source.
Helpful for:
- broad paper search
- metadata enrichment
- open-access fields
- related entity lookups

### Semantic Scholar
Use for:
- paper search
- metadata enrichment
- relevance-oriented discovery
- optional related-paper expansion

### Crossref
Use for:
- DOI-centric metadata lookup
- bibliographic completion
- fallback metadata matching

### Europe PMC
Use when the topic is biomedical, life sciences, clinical, or adjacent domains.
Helpful because open full text is often easier to resolve in this domain.

### Unpaywall
Use when a DOI exists but no direct PDF was found from the first-pass search source.
This is the DOI-to-open-access resolver.

### Zotero
Use as an optional organizational layer.
Recommended uses:
- export collected metadata into a Zotero library
- maintain collections per project
- support human review

Not recommended as the only automated scholarly search layer for this project.

---

## Ranking and deduplication

### Deduplication priority
1. DOI exact match
2. normalized title exact match
3. normalized title + year + first author heuristic

### Suggested relevance score
Weighted combination of:
- keyword overlap with the draft
- title similarity to extracted queries
- abstract similarity to draft summary
- recency bonus if desired
- citation count or influence score if available

---

## Folder layout

```text
litharvest/
├── README.md
├── lit_harvest.py
├── collected_papers/
├── logs/
│   ├── manifest.json
│   └── download_log.json
└── config.example.json
```

---

## Example run

```bash
python lit_harvest.py \
  --draft ./seed_paper.txt \
  --output ./collected_papers \
  --max-queries 8 \
  --top-k-per-query 10 \
  --email you@example.com
```

Optional:

```bash
python lit_harvest.py \
  --draft ./seed_paper.txt \
  --output ./collected_papers \
  --unpaywall-email you@example.com \
  --semantic-scholar-api-key YOUR_KEY
```

---

## Suggested implementation phases

### Phase 1: Minimal viable collector
- plain-text draft input
- simple keyword extraction
- OpenAlex search
- Semantic Scholar search
- DOI dedupe
- download open PDFs into a folder
- write manifest

### Phase 2: Better recall and robustness
- Crossref enrichment
- Europe PMC support
- Unpaywall resolution
- retries and backoff
- better ranking

### Phase 3: Better research workflow integration
- optional Zotero export
- BibTeX output
- reference-seed expansion
- citation chaining

---

## Acceptance criteria
The project is successful if:

1. a user provides a draft,
2. the system extracts meaningful search terms,
3. the system searches non-CAPTCHA scholarly APIs,
4. the system downloads reachable PDFs into a folder,
5. all findings are logged in a machine-readable manifest, and
6. the output folder can be handed directly to the downstream document-processing pipeline.

---

## Risks and design cautions
- Many scholarly search sources return metadata but not downloadable PDFs.
- PDF links can be stale or redirect to HTML landing pages.
- Open-access availability varies by field.
- Keyword extraction quality heavily affects recall.
- A broad query strategy can produce many loosely relevant papers.

Mitigations:
- combine multiple APIs
- keep strong logging
- validate content type before saving
- use DOI-based OA resolution
- rank before downloading when result volume is large

---

## Python package requirements
Suggested baseline dependencies:

```text
requests
```

Optional upgrades:

```text
spacy
keybert
scikit-learn
python-dotenv
rapidfuzz
```

Keep Phase 1 lightweight. You can start with only `requests` and standard library code.

---

## Future extensions
- LLM-generated search strategy from the draft
- seed-reference mining from the draft bibliography
- citation graph expansion
- topic clustering of collected results
- local cache of already-seen DOIs
- configurable inclusion/exclusion rules
- project-specific source adapters

---

## References and official docs
- OpenAlex API overview and works search
- Semantic Scholar Academic Graph API
- Crossref REST API
- Europe PMC REST API
- Unpaywall API
- Zotero Web API

Use official API docs and respect their rate limits, authentication guidance, and terms.
