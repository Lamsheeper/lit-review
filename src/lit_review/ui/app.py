from __future__ import annotations

from lit_review.core.io import read_jsonl

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from lit_review.features.extraction import PDF_EXTRACTION_SYSTEM_PROMPT
from lit_review.features.prompts import build_pdf_extraction_prompt
from lit_review.features.records import load_goal_text
from lit_review.core.llm import make_llm_client
from lit_review.collection.harvest import candidate_from_dict, stable_candidate_id
from lit_review.evidence.pdf import prepare_direct_pdf_index
from lit_review.collection.relevance import validate_relevance_profile
from lit_review.evidence.review import extract_json_object
from lit_review.core.taxonomy import load_taxonomy, validate_taxonomy

from .jobs import JobManager
from .store import ProjectStore


FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
SESSION_CREDENTIALS: dict[str, str] = {}
store: ProjectStore
jobs: JobManager
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    global store, jobs
    if "store" not in globals():
        store = ProjectStore()
        jobs = JobManager(store)
    yield


app = FastAPI(title="Lit Review UI", version="0.2.0", lifespan=lifespan)


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class DocumentsPayload(BaseModel):
    draft_markdown: str
    goal_markdown: str


class GeneratePayload(BaseModel):
    request: str
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    reasoning_effort: str | None = "low"
    base_url: str | None = None


class TaxonomyPayload(BaseModel):
    version: int = 1
    title: str
    families: list[dict[str, Any]]


class CredentialsPayload(BaseModel):
    credentials: dict[str, str] = Field(default_factory=dict)


class CollectionConfig(BaseModel):
    sources: list[str] = Field(
        default_factory=lambda: ["openalex", "semantic_scholar", "crossref", "europe_pmc"]
    )
    year_from: int | None = None
    year_to: int | None = None
    max_queries: int = Field(default=8, ge=1)
    top_k_per_query: int = Field(default=10, ge=1)
    max_downloads: int = Field(default=50, ge=1)
    email: str = ""
    web_search_sources: list[str] = Field(default_factory=list)
    timeout: float = Field(default=30, gt=0)
    retries: int = Field(default=2, ge=0)
    rate_limit_delay: float = Field(default=0.1, ge=0)
    download_workers: int = Field(default=4, ge=1)
    refresh_candidates: bool = False
    retry_failed_downloads: bool = False
    extra_queries: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_years(self) -> "CollectionConfig":
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from cannot be later than year_to")
        return self


from lit_review.core.settings import ProviderSettings


class ExtractionConfig(ProviderSettings):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    reasoning_effort: str | None = "low"
    base_url: str | None = None
    max_features_per_pdf: int = Field(default=40, ge=1)
    temperature: float = Field(default=0.1, ge=0)
    timeout: float = Field(default=600, gt=0)
    profile: Literal["media_attributes"] | None = None

    @model_validator(mode="after")
    def validate_provider(self) -> "ExtractionConfig":
        if self.provider not in {"gemini", "openai"}:
            raise ValueError("provider must be gemini or openai")
        return self


def project_dir(project_id: str) -> Path:
    try:
        return store.project_dir(project_id)
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc


def read_json(path: Path, default: Any = None) -> Any:
    return store.read_json(path, default)


def write_config(path: Path, payload: dict[str, Any]) -> None:
    secret_keys = {
        "api_key",
        "llm_api_key",
        "semantic_scholar_api_key",
        "brave_search_api_key",
        "serpapi_api_key",
        "embedding_api_key",
        "gemini_api_key",
    }
    secret_free = {key: value for key, value in payload.items() if key not in secret_keys}
    store.write_json(path, secret_free)


def credential_env() -> dict[str, str]:
    return {
        key: value
        for key, value in SESSION_CREDENTIALS.items()
        if key in {"GEMINI_API_KEY", "OPENAI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"}
    }


def launch_job(
    project_id: str,
    kind: str,
    commands: list[list[str]],
) -> dict[str, Any]:
    try:
        return jobs.start(project_id, kind, commands, credential_env())
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def provider_key(provider: str) -> str | None:
    if provider == "gemini":
        return SESSION_CREDENTIALS.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    return SESSION_CREDENTIALS.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def default_base_url(provider: str) -> str:
    if provider == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta"
    return "https://api.openai.com/v1"


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, {}) or {}
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    output: list[dict[str, Any]] = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        candidate = candidate_from_dict(raw)
        row = dict(raw)
        row["candidate_id"] = stable_candidate_id(candidate)
        output.append(row)
    return output


def paginate_rows(
    rows: list[dict[str, Any]],
    *,
    page: int,
    page_size: int,
    query: str,
    sort: str,
    order: str,
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    query_lower = query.lower().strip()
    filters = filters or {}
    if query_lower:
        rows = [row for row in rows if query_lower in json.dumps(row, ensure_ascii=False).lower()]
    for key, value in filters.items():
        if value:
            rows = [row for row in rows if str(row.get(key) or "") == value]
    reverse = order.lower() == "desc"
    if sort:

        def sort_key(row: dict[str, Any]) -> tuple[bool, int, Any]:
            value = row.get(sort)
            if isinstance(value, (int, float)):
                return value is None, 0, float(value)
            return value is None, 1, str(value or "").lower()

        rows.sort(key=sort_key, reverse=reverse)
    total = len(rows)
    page = max(1, page)
    page_size = min(max(1, page_size), 500)
    start = (page - 1) * page_size
    return {"items": rows[start : start + page_size], "total": total, "page": page, "page_size": page_size}


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    return store.list_projects()


@app.post("/api/projects")
def create_project(payload: ProjectCreate) -> dict[str, Any]:
    try:
        return store.create_project(payload.name, payload.description)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    try:
        project = store.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc
    project["jobs"] = store.list_jobs(project_id)
    return project


@app.get("/api/projects/{project_id}/workflow")
def get_project_workflow(project_id: str) -> dict[str, Any]:
    return project_workflow_status(project_id)


@app.patch("/api/projects/{project_id}")
def update_project(project_id: str, payload: ProjectUpdate) -> dict[str, Any]:
    try:
        return store.update_project(project_id, payload.model_dump(exclude_none=True))
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, Any]:
    try:
        project = store.delete_project(project_id)
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"deleted": True, "project": {"id": project["id"], "name": project["name"]}}


@app.get("/api/projects/{project_id}/documents")
def get_documents(project_id: str) -> dict[str, str]:
    path = project_dir(project_id)
    return {
        "draft_markdown": (path / "draft.md").read_text(encoding="utf-8"),
        "goal_markdown": (path / "goal.md").read_text(encoding="utf-8"),
    }


@app.put("/api/projects/{project_id}/documents")
def save_documents(project_id: str, payload: DocumentsPayload) -> dict[str, bool]:
    path = project_dir(project_id)
    (path / "draft.md").write_text(payload.draft_markdown, encoding="utf-8")
    (path / "goal.md").write_text(payload.goal_markdown, encoding="utf-8")
    store.touch_project(project_id)
    return {"saved": True}


@app.post("/api/projects/{project_id}/documents/generate")
def generate_documents(project_id: str, payload: GeneratePayload) -> dict[str, Any]:
    path = project_dir(project_id)
    request_text = payload.request.strip()
    if not request_text:
        raise HTTPException(422, "A feature request is required.")
    api_key = provider_key(payload.provider)
    if not api_key:
        raise HTTPException(422, f"No {payload.provider} API key is available.")
    client = make_llm_client(
        api_key=api_key,
        model=payload.model,
        base_url=payload.base_url or default_base_url(payload.provider),
        reasoning_effort=payload.reasoning_effort,
        api_provider=payload.provider,
        input_mode="text",
        timeout=180,
    )
    prompt = f"""Turn the user's broad literature feature request into two actionable Markdown documents, a structured taxonomy, and a candidate-paper relevance scoring profile.

Return one strict JSON object with keys:
- draft_markdown: detailed scholarly search target with questions, search terms, inclusion criteria, and exclusion criteria
- goal_markdown: detailed direct-PDF extraction target and evidence requirements
- taxonomy: object with version=1, title, and families. Each family has a stable snake_case id, label, description, and aliases.
- relevance_scoring: a declarative algorithm for ranking candidate papers, using this exact shape:
  {{
    "version": 1,
    "title": "...",
    "description": "...",
    "criteria": [
      {{
        "id": "stable_snake_case_id",
        "label": "...",
        "description": "...",
        "weight": 0.0 to 10.0,
        "terms": ["domain term or phrase", "..."]
      }}
    ],
    "field_weights": {{"title": number, "abstract": number, "matched_keywords": number}},
    "metadata_weights": {{
      "source_relevance": number,
      "citation_impact": number,
      "recency": number,
      "open_access": number,
      "doi": number
    }},
    "exclusions": [{{"term": "...", "penalty": 0.0 to 1.0, "reason": "..."}}]
  }}

Families must be mutually understandable extraction buckets and cover the user's request without inventing unrelated domains.
The relevance criteria must distinguish papers that can answer this specific request from nearby but unhelpful papers. Terms within one criterion are interchangeable indicators: matching any one activates that criterion for a field, so put distinct requirements in separate criteria. Include useful synonyms and technical phrases. Assign metadata weights only when they are justified by the user's setting; topical relevance should normally dominate. Do not assume persuasion, propaganda, sentiment, or any other domain unless the user requested it.

USER REQUEST:
{request_text}
"""
    raw = client.chat(
        [{"role": "system", "content": "Return only valid JSON."}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=8000,
        response_format={"type": "json_object"},
    )
    try:
        generated = extract_json_object(raw)
        taxonomy = validate_taxonomy(generated.get("taxonomy"))
        relevance_scoring = validate_relevance_profile(generated.get("relevance_scoring"))
        draft = str(generated.get("draft_markdown") or "").strip()
        goal = str(generated.get("goal_markdown") or "").strip()
        if not draft or not goal:
            raise ValueError("Generated documents were empty.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(502, f"Generated documents were invalid: {exc}") from exc
    (path / "draft.md").write_text(draft + "\n", encoding="utf-8")
    (path / "goal.md").write_text(goal + "\n", encoding="utf-8")
    store.write_json(path / "taxonomy.json", taxonomy)
    store.write_json(path / "relevance_scoring.json", relevance_scoring)
    store.touch_project(project_id)
    return {
        "draft_markdown": draft,
        "goal_markdown": goal,
        "taxonomy": taxonomy,
        "relevance_scoring": relevance_scoring,
    }


@app.get("/api/projects/{project_id}/taxonomy")
def get_taxonomy(project_id: str) -> dict[str, Any] | None:
    return read_json(project_dir(project_id) / "taxonomy.json")


@app.put("/api/projects/{project_id}/taxonomy")
def save_taxonomy(project_id: str, payload: TaxonomyPayload) -> dict[str, Any]:
    try:
        taxonomy = validate_taxonomy(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    store.write_json(project_dir(project_id) / "taxonomy.json", taxonomy)
    store.touch_project(project_id)
    return taxonomy


@app.get("/api/projects/{project_id}/relevance-scoring")
def get_relevance_scoring(project_id: str) -> dict[str, Any] | None:
    return read_json(project_dir(project_id) / "relevance_scoring.json")


@app.put("/api/projects/{project_id}/relevance-scoring")
def save_relevance_scoring(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        profile = validate_relevance_profile(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    store.write_json(project_dir(project_id) / "relevance_scoring.json", profile)
    store.touch_project(project_id)
    return profile


@app.get("/api/settings/credential-status")
def credential_status() -> dict[str, Any]:
    keys = ["GEMINI_API_KEY", "OPENAI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"]
    return {
        key: {
            "available": bool(SESSION_CREDENTIALS.get(key) or os.getenv(key)),
            "source": "session"
            if SESSION_CREDENTIALS.get(key)
            else ("environment" if os.getenv(key) else None),
        }
        for key in keys
    }


@app.put("/api/settings/session-credentials")
def set_session_credentials(payload: CredentialsPayload) -> dict[str, Any]:
    allowed = {"GEMINI_API_KEY", "OPENAI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"}
    for key, value in payload.credentials.items():
        if key not in allowed:
            raise HTTPException(422, f"Unsupported credential: {key}")
        if value:
            SESSION_CREDENTIALS[key] = value
        else:
            SESSION_CREDENTIALS.pop(key, None)
    return credential_status()


@app.get("/api/model-presets")
def model_presets() -> dict[str, Any]:
    return {
        "document_generation": [
            {"provider": "gemini", "model": "gemini-2.5-flash", "reasoning_effort": "low"},
            {"provider": "openai", "model": "gpt-4o-mini", "reasoning_effort": "low"},
        ],
        "direct_pdf_extraction": [
            {"provider": "gemini", "model": "gemini-2.5-flash", "reasoning_effort": "low"},
            {"provider": "openai", "model": "gpt-4o-mini", "reasoning_effort": "low"},
        ],
    }


@app.get("/api/projects/{project_id}/collection/config")
def get_collection_config(project_id: str) -> dict[str, Any]:
    path = project_dir(project_id)
    return read_json(
        path / "configs/collection.json",
        {
            "sources": ["openalex", "semantic_scholar", "crossref", "europe_pmc"],
            "year_from": None,
            "year_to": None,
            "max_queries": 8,
            "top_k_per_query": 10,
            "max_downloads": 50,
            "email": os.getenv("LITHARVEST_EMAIL", ""),
            "web_search_sources": [],
            "timeout": 30,
            "retries": 2,
            "rate_limit_delay": 0.1,
            "download_workers": 4,
            "refresh_candidates": False,
            "retry_failed_downloads": False,
        },
    )


@app.put("/api/projects/{project_id}/collection/config")
def save_collection_config(project_id: str, payload: CollectionConfig) -> dict[str, Any]:
    path = project_dir(project_id) / "configs/collection.json"
    write_config(path, payload.model_dump())
    return read_json(path)


def harvest_config(
    project_id: str, *, metadata_only: bool, selected_ids: list[str] | None = None
) -> dict[str, Any]:
    path = project_dir(project_id)
    config = dict(get_collection_config(project_id))
    config.update(
        {
            "harvest_mode": "draft",
            "draft": str(path / "draft.md"),
            "output": str(path / "run/stages/collect"),
            "logs": str(path / "run/stages/collect/logs"),
            "metadata_only": metadata_only,
            "allow_abstract_fallback": False,
            "max_pdf_mb": 50,
            "progress": True,
            "csv_report": True,
            "bibtex": True,
        }
    )
    relevance_path = path / "relevance_scoring.json"
    if relevance_path.exists():
        config["relevance_scoring"] = str(relevance_path)
    if selected_ids is not None:
        config["selected_candidate_ids"] = selected_ids
        config["refresh_candidates"] = False
    return config


@app.get("/api/projects/{project_id}/collection/candidates")
def list_candidates(
    project_id: str,
    page: int = 1,
    page_size: int = 50,
    query: str = "",
    sort: str = "relevance_score",
    order: str = "desc",
    download_status: str = "",
) -> dict[str, Any]:
    path = project_dir(project_id)
    rows = load_candidates(candidate_path(path))
    manifest = read_json(path / "run/stages/collect/logs/manifest.json", {}) or {}
    status_by_id = {}
    for raw in manifest.get("candidates", []):
        if isinstance(raw, dict):
            candidate = candidate_from_dict(raw)
            status_by_id[stable_candidate_id(candidate)] = (raw.get("download") or {}).get("status")
    selection_path = path / "selection.json"
    selection = read_json(selection_path)
    candidate_ids = {row["candidate_id"] for row in rows}
    if selection and selection.get("candidate_ids"):
        valid_ids = [value for value in selection["candidate_ids"] if value in candidate_ids]
        if valid_ids != selection["candidate_ids"]:
            selection = {"candidate_ids": valid_ids} if valid_ids else None
            if selection is None:
                selection_path.unlink(missing_ok=True)
            else:
                store.write_json(selection_path, selection)
    if selection is None and rows:
        maximum = int(get_collection_config(project_id).get("max_downloads") or 50)
        preselected = [row["candidate_id"] for row in rows[:maximum]]
        selection = {"candidate_ids": preselected}
        store.write_json(selection_path, selection)
    selected = set((selection or {"candidate_ids": []})["candidate_ids"])
    for row in rows:
        row["download_status"] = status_by_id.get(row["candidate_id"], "")
        row["selected"] = row["candidate_id"] in selected
    return paginate_rows(
        rows,
        page=page,
        page_size=page_size,
        query=query,
        sort=sort,
        order=order,
        filters={"download_status": download_status},
    )


@app.put("/api/projects/{project_id}/collection/selection")
def save_selection(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = project_dir(project_id)
    valid_ids = {row["candidate_id"] for row in load_candidates(candidate_path(path))}
    ids = list(dict.fromkeys(str(value) for value in payload.get("candidate_ids", [])))
    unknown = [value for value in ids if value not in valid_ids]
    if unknown:
        raise HTTPException(422, f"Unknown candidate IDs: {', '.join(unknown[:5])}")
    store.write_json(path / "selection.json", {"candidate_ids": ids})
    return {"candidate_ids": ids, "count": len(ids)}


@app.get("/api/projects/{project_id}/collection/selection")
def get_selection(project_id: str) -> dict[str, Any]:
    ids = read_json(project_dir(project_id) / "selection.json", {"candidate_ids": []})["candidate_ids"]
    return {"candidate_ids": ids, "count": len(ids)}


@app.get("/api/projects/{project_id}/extraction/config")
def get_extraction_config(project_id: str) -> dict[str, Any]:
    return read_json(
        project_dir(project_id) / "configs/extraction.json",
        {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "reasoning_effort": "low",
            "max_features_per_pdf": 40,
            "temperature": 0.1,
            "timeout": 600,
        },
    )


@app.put("/api/projects/{project_id}/extraction/config")
def save_extraction_config(project_id: str, payload: ExtractionConfig) -> dict[str, Any]:
    path = project_dir(project_id) / "configs/extraction.json"
    write_config(path, payload.model_dump())
    return read_json(path)


@app.post("/api/projects/{project_id}/extraction/prompt-preview")
def preview_extraction_prompt(project_id: str, payload: ExtractionConfig) -> dict[str, Any]:
    path = project_dir(project_id)
    goal_path = path / "goal.md"
    taxonomy_path = path / "taxonomy.json"
    if not taxonomy_path.exists():
        raise HTTPException(422, "Define a valid taxonomy first.")
    if not goal_path.exists() or not goal_path.read_text(encoding="utf-8").strip():
        raise HTTPException(422, "Define an extraction goal first.")

    try:
        taxonomy = load_taxonomy(taxonomy_path)
        goal_text, _ = load_goal_text(goal_path)
        manifest_path = path / "run/stages/collect/logs/manifest.json"
        with tempfile.TemporaryDirectory(prefix="lit-review-prompt-") as temp:
            paper_index = prepare_direct_pdf_index(
                path / "run/stages/collect",
                manifest_path if manifest_path.exists() else None,
                Path(temp),
            )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc

    papers = [
        paper for paper in paper_index.get("papers", []) if isinstance(paper, dict) and paper.get("pdf_path")
    ]
    if not papers:
        raise HTTPException(422, "Download at least one full PDF first.")
    paper = papers[0]
    return {
        "system_prompt": PDF_EXTRACTION_SYSTEM_PROMPT,
        "user_prompt": build_pdf_extraction_prompt(
            paper,
            goal_text=goal_text,
            max_features_per_pdf=payload.max_features_per_pdf,
            taxonomy=taxonomy,
        ),
        "attachment_filename": Path(str(paper.get("pdf_path") or "")).name,
        "paper": paper,
        "paper_count": len(papers),
        "max_features_per_pdf": payload.max_features_per_pdf,
    }


def result_list(
    project_id: str,
    filename: str,
    page: int,
    page_size: int,
    query: str,
    sort: str,
    order: str,
    category: str = "",
) -> dict[str, Any]:
    rows = read_jsonl(project_dir(project_id) / "run/stages/extract" / filename)
    return paginate_rows(
        rows,
        page=page,
        page_size=page_size,
        query=query,
        sort=sort,
        order=order,
        filters={"category": category},
    )


@app.get("/api/projects/{project_id}/results/features")
def results_features(
    project_id: str,
    page: int = 1,
    page_size: int = 50,
    query: str = "",
    sort: str = "paper_count",
    order: str = "desc",
    category: str = "",
) -> dict[str, Any]:
    return result_list(project_id, "features.jsonl", page, page_size, query, sort, order, category)


@app.get("/api/projects/{project_id}/results/mentions")
def results_mentions(
    project_id: str,
    page: int = 1,
    page_size: int = 50,
    query: str = "",
    sort: str = "confidence",
    order: str = "desc",
    category: str = "",
) -> dict[str, Any]:
    return result_list(project_id, "feature_mentions.jsonl", page, page_size, query, sort, order, category)


@app.get("/api/projects/{project_id}/results/errors")
def results_errors(project_id: str, page: int = 1, page_size: int = 50, query: str = "") -> dict[str, Any]:
    rows = read_jsonl(project_dir(project_id) / "run/stages/extract/extraction_errors.jsonl")
    return paginate_rows(rows, page=page, page_size=page_size, query=query, sort="created_at", order="desc")


@app.get("/api/projects/{project_id}/results/papers")
def results_papers(project_id: str, page: int = 1, page_size: int = 50, query: str = "") -> dict[str, Any]:
    payload = read_json(project_dir(project_id) / "run/stages/prepare/paper_index.json", {}) or {}
    return paginate_rows(
        payload.get("papers", []), page=page, page_size=page_size, query=query, sort="title", order="asc"
    )


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job not found.") from exc


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        return jobs.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job not found.") from exc


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    try:
        store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job not found.") from exc

    async def stream():
        position = 0
        while True:
            if await request.is_disconnected():
                break
            current = store.get_job(job_id)
            log_path = Path(current["log_path"])
            if log_path.exists():
                with log_path.open(encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
                if chunk:
                    yield "event: log\ndata: " + json.dumps({"text": chunk}) + "\n\n"
            yield "event: status\ndata: " + json.dumps(current) + "\n\n"
            if current["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                break
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")


from lit_review.core.settings import PipelineConfig, StageSpec
from lit_review.pipeline import compile_pipeline, run_pipeline, stage_current
from lit_review.services import ARTIFACTS, FEATURE_FIELDS, feature_options

PROCESSING = ("extract", "filter", "annotate", "merge", "score")
RESULT_FILES = {
    "extract": {
        "features.jsonl",
        "features.csv",
        "feature_mentions.jsonl",
        "paper_features.jsonl",
        "extraction_errors.jsonl",
    },
    "filter": {
        "filtered_features.jsonl",
        "filtered_features.csv",
        "rejected_features.jsonl",
        "curation_decisions.jsonl",
    },
    "annotate": {
        "annotated_features.jsonl",
        "annotated_features.csv",
        "excluded_annotations.jsonl",
        "annotation_decisions.jsonl",
    },
    "merge": {
        "llm_merged_features.jsonl",
        "llm_merged_features.csv",
        "llm_merge_candidates.jsonl",
        "llm_merge_decisions.jsonl",
    },
    "score": {"scored_features.jsonl", "scored_features.csv", "scoring_decisions.jsonl"},
}


def candidate_path(path: Path) -> Path:
    downloaded = path / "run/stages/collect/logs/candidates.json"
    searched = path / "run/stages/search/logs/candidates.json"
    existing = [path for path in (downloaded, searched) if path.exists()]
    return max(existing, key=lambda path: path.stat().st_mtime_ns) if existing else searched


def feature_pipeline_config(project_id: str) -> PipelineConfig:
    path = project_dir(project_id)
    extraction = get_extraction_config(project_id)
    stages = [
        StageSpec(
            id="prepare", operation="evidence.prepare", options={"selection": str(path / "selection.json")}
        )
    ]
    for stage in PROCESSING:
        op = "features." + stage
        allowed = FEATURE_FIELDS[op]
        options = {key: value for key, value in extraction.items() if key in allowed}
        options.update(read_json(path / f"configs/{stage}.json", {}) or {})
        stages.append(StageSpec(id=stage, operation=op, options=options))
    return PipelineConfig(
        output_root=path / "run",
        inputs={
            "papers": str(path / "run/stages/collect"),
            "manifest": str(path / "run/stages/collect/logs/manifest.json"),
            "taxonomy": str(path / "taxonomy.json"),
            "goal": str(path / "goal.md"),
        },
        stages=stages,
    )


def processing_status(project_id: str) -> dict:
    path = project_dir(project_id)
    try:
        records = read_json(path / "run/manifest.json", {}).get("stages", {})
    except (ValueError, OSError, AttributeError):
        return {}
    try:
        plan = compile_pipeline(feature_pipeline_config(project_id))
    except (ValueError, OSError):
        plan = []
    current = True
    result = {}
    for stage in plan:
        current = current and stage_current(stage, records.get(stage["id"]))
        result[stage["id"]] = {
            "complete": bool(current),
            "stale": bool(records.get(stage["id"]) and not current),
        }
    return result


def project_workflow_status(project_id: str) -> dict:
    path = project_dir(project_id)
    try:
        validate_taxonomy(read_json(path / "taxonomy.json"))
        validate_relevance_profile(read_json(path / "relevance_scoring.json"))
        defined = all(
            (path / name).exists() and (path / name).read_text().strip() for name in ("draft.md", "goal.md")
        )
    except (ValueError, TypeError, OSError):
        defined = False
    try:
        candidates = load_candidates(candidate_path(path))
    except (ValueError, TypeError):
        candidates = []
    selected = set((read_json(path / "selection.json", {}) or {}).get("candidate_ids", []))
    manifest = read_json(path / "run/stages/collect/logs/manifest.json", {}) or {}
    downloaded = {
        stable_candidate_id(candidate_from_dict(row))
        for row in manifest.get("candidates", [])
        if (row.get("download") or {}).get("status") in {"downloaded", "already_exists"}
    }
    active = store.active_job(project_id) or {}
    status = processing_status(project_id)

    def step(complete, kind, label, materials=None, missing=None, stale=False):
        running = active.get("kind") == kind
        return {
            "complete": bool(complete) and not running,
            "running": running,
            "stale": stale,
            "missing": [] if complete else (missing or ["successful upstream stage and current artifacts"]),
            "materials": materials or {},
            "action_label": label,
        }

    uses_cached = bool(candidates) and not get_collection_config(project_id).get("refresh_candidates")
    steps = {
        "define": step(
            defined,
            "definition",
            "Generate project definition",
            missing=["saved draft, goal, taxonomy, and relevance profile"],
        ),
        "collect": step(
            bool(candidates),
            "collection_search",
            "Re-score cached candidates" if uses_cached else "Find and rank papers",
            {
                "candidate_count": len(candidates),
                "uses_cached_candidates": uses_cached,
                "run_mode": "rescore_cached" if uses_cached else "search_metadata",
            },
            missing=["paper search results"],
        ),
        "candidates": step(
            bool(selected & downloaded),
            "collection_download",
            "Download selected PDFs",
            {
                "selected_count": len(selected),
                "downloaded_count": len(downloaded),
                "selected_downloaded_count": len(selected & downloaded),
            },
            missing=["a selected, downloaded PDF"],
        ),
    }
    for stage in PROCESSING:
        info = status.get(stage, {})
        artifact = path / "run/stages" / stage / ARTIFACTS["features." + stage][0]
        try:
            count = len(read_jsonl(artifact)) if artifact.exists() else 0
        except (ValueError, OSError):
            count = 0
        kind = "direct_pdf_extraction" if stage == "extract" else stage
        steps[stage] = step(
            info.get("complete"),
            kind,
            "Run " + stage,
            {"feature_count": count, "selected_downloaded_count": len(selected & downloaded)},
            stale=info.get("stale", False),
        )
    steps["results"] = step(
        steps["merge"]["complete"], "results", "Download merged features", steps["merge"]["materials"]
    )
    return {"steps": steps}


def pipeline_job(project_id: str, kind: str, config: PipelineConfig, stages: list[str]) -> dict:
    from uuid import uuid4

    path = project_dir(project_id) / "configs" / (kind + "_" + uuid4().hex + ".json")
    write_config(path, config.model_dump(mode="json"))
    return launch_job(
        project_id,
        kind,
        [
            [
                sys.executable,
                "-m",
                "lit_review",
                "pipeline",
                "run",
                "--config",
                str(path),
                "--stages",
                ",".join(stages),
            ]
        ],
    )


@app.post("/api/projects/{project_id}/collection/search")
def start_search(project_id: str) -> dict:
    path = project_dir(project_id)
    if not (path / "draft.md").read_text().strip():
        raise HTTPException(422, "Save a collection draft first")
    options = harvest_config(project_id, metadata_only=True)
    options = {key: value for key, value in options.items() if key not in {"output", "logs", "harvest_mode"}}
    cache = candidate_path(path)
    if cache.exists() and not options.get("refresh_candidates"):
        snapshot = path / "cache/search_candidates.json"
        snapshot.parent.mkdir(exist_ok=True)
        from shutil import copy2

        copy2(cache, snapshot)
        options["candidates_json"] = str(snapshot)
    config = PipelineConfig(
        output_root=path / "run", stages=[StageSpec(id="search", operation="collect.search", options=options)]
    )
    return pipeline_job(project_id, "collection_search", config, ["search"])


@app.post("/api/projects/{project_id}/collection/download")
def start_download(project_id: str) -> dict:
    path = project_dir(project_id)
    selected = (read_json(path / "selection.json", {}) or {}).get("candidate_ids", [])
    if not selected:
        raise HTTPException(422, "Select at least one paper")
    options = harvest_config(project_id, metadata_only=False, selected_ids=selected)
    options = {key: value for key, value in options.items() if key not in {"output", "logs", "harvest_mode"}}
    snapshot = path / "cache/download_candidates.json"
    snapshot.parent.mkdir(exist_ok=True)
    from shutil import copy2

    copy2(candidate_path(path), snapshot)
    options["candidates_json"] = str(snapshot)
    config = PipelineConfig(
        output_root=path / "run",
        stages=[StageSpec(id="collect", operation="collect.search", options=options)],
    )
    return pipeline_job(project_id, "collection_download", config, ["collect"])


@app.post("/api/projects/{project_id}/extraction/run")
def start_extraction(project_id: str) -> dict:
    if not project_workflow_status(project_id)["steps"]["candidates"]["complete"]:
        raise HTTPException(422, "Download at least one selected full PDF first")
    config = get_extraction_config(project_id)
    if not provider_key(config.get("provider", "gemini")):
        raise HTTPException(422, "Configure the selected provider credential")
    return pipeline_job(
        project_id, "direct_pdf_extraction", feature_pipeline_config(project_id), ["prepare", "extract"]
    )


@app.get("/api/projects/{project_id}/stages/{stage}/config")
def get_stage_config(project_id: str, stage: str) -> dict:
    if stage not in PROCESSING[1:]:
        raise HTTPException(404, "Unknown processing stage")
    config = feature_pipeline_config(project_id)
    options = next(spec.options for spec in config.stages if spec.id == stage)
    return options


@app.put("/api/projects/{project_id}/stages/{stage}/config")
def save_stage_config(project_id: str, stage: str, payload: dict) -> dict:
    if stage not in PROCESSING[1:]:
        raise HTTPException(404, "Unknown processing stage")
    path = project_dir(project_id)
    try:
        config = feature_options(
            "features." + stage, dict(payload, taxonomy=path / "taxonomy.json", goal=path / "goal.md")
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    options = {
        key: value
        for key, value in config.model_dump(mode="json").items()
        if key in FEATURE_FIELDS["features." + stage] and key not in {"taxonomy", "goal"}
    }
    write_config(path / f"configs/{stage}.json", options)
    return options


@app.post("/api/projects/{project_id}/stages/{stage}/preview")
def preview_stage(project_id: str, stage: str) -> dict:
    if stage not in PROCESSING:
        raise HTTPException(404, "Unknown processing stage")
    try:
        return run_pipeline(feature_pipeline_config(project_id), dry_run=True, stages=[stage])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/projects/{project_id}/stages/{stage}/run")
def run_stage(project_id: str, stage: str) -> dict:
    if stage not in PROCESSING[1:]:
        raise HTTPException(404, "Unknown processing stage")
    previous = PROCESSING[PROCESSING.index(stage) - 1]
    workflow = project_workflow_status(project_id)
    if not workflow["steps"][previous]["complete"]:
        raise HTTPException(409, f"Complete {previous} with current inputs first")
    config = get_stage_config(project_id, stage)
    if not provider_key(config.get("provider", "gemini")):
        raise HTTPException(422, "Configure the selected provider credential")
    return pipeline_job(project_id, stage, feature_pipeline_config(project_id), [stage])


@app.get("/api/projects/{project_id}/stages/{stage}/results/{artifact}")
def stage_results(
    project_id: str,
    stage: str,
    artifact: str,
    page: int = 1,
    page_size: int = 100,
    query: str = "",
    category: str = "",
) -> dict:
    if artifact not in RESULT_FILES.get(stage, set()) or not artifact.endswith(".jsonl"):
        raise HTTPException(404, "Unknown stage artifact")
    rows = read_jsonl(project_dir(project_id) / "run/stages" / stage / artifact)
    return paginate_rows(
        rows,
        page=page,
        page_size=page_size,
        query=query,
        sort="paper_count",
        order="desc",
        filters={"category": category},
    )


@app.get("/api/projects/{project_id}/stages/{stage}/exports/{artifact}")
def export_stage(project_id: str, stage: str, artifact: str) -> FileResponse:
    if artifact not in RESULT_FILES.get(stage, set()):
        raise HTTPException(404, "Unknown stage artifact")
    path = project_dir(project_id) / "run/stages" / stage / artifact
    if not path.is_file():
        raise HTTPException(404, "Artifact does not exist yet")
    return FileResponse(path, filename=artifact)


@app.get("/api/projects/{project_id}/exports/{artifact}")
def export_artifact(project_id: str, artifact: str) -> FileResponse:
    if artifact in {"manifest.json", "candidates.csv", "candidates.bib"}:
        path = candidate_path(project_dir(project_id)).parent / artifact
        if not path.is_file():
            raise HTTPException(404, "Artifact does not exist yet")
        return FileResponse(path, filename=artifact)
    return export_stage(project_id, "extract", artifact)


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        target = (FRONTEND_DIST / path).resolve()
        if not target.is_relative_to(FRONTEND_DIST.resolve()):
            raise HTTPException(404, "Unknown frontend resource")
        return FileResponse(target if target.is_file() else FRONTEND_DIST / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run("lit_review.ui.app:app", host="127.0.0.1", port=8765, reload=False)
