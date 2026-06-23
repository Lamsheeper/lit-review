from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from lit_extract import (
    PDF_EXTRACTION_SYSTEM_PROMPT,
    build_pdf_extraction_prompt,
    load_goal_text,
    make_llm_client,
)
from lit_harvest import candidate_from_dict, stable_candidate_id
from lit_pdf_service import prepare_direct_pdf_index
from lit_relevance import validate_relevance_profile
from lit_synthesize import extract_json_object
from lit_taxonomy import load_taxonomy, validate_taxonomy

from .jobs import JobManager
from .store import ProjectStore


REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = REPO_ROOT / "lit_review_ui" / "frontend" / "dist"
SESSION_CREDENTIALS: dict[str, str] = {}
store = ProjectStore()
jobs = JobManager(store)
app = FastAPI(title="Lit Review UI", version="0.1.0")


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
    sources: list[str] = Field(default_factory=lambda: ["openalex", "semantic_scholar", "crossref", "europe_pmc"])
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


class ExtractionConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    reasoning_effort: str | None = "low"
    base_url: str | None = None
    max_features_per_pdf: int = Field(default=40, ge=1)
    temperature: float = Field(default=0.1, ge=0)
    timeout: float = Field(default=600, gt=0)

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
        "api_key", "llm_api_key", "semantic_scholar_api_key",
        "brave_search_api_key", "serpapi_api_key",
    }
    secret_free = {key: value for key, value in payload.items() if key not in secret_keys}
    store.write_json(path, secret_free)


def credential_env() -> dict[str, str]:
    return {
        key: value
        for key, value in SESSION_CREDENTIALS.items()
        if key in {"GEMINI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"}
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
        rows = [
            row for row in rows
            if query_lower in json.dumps(row, ensure_ascii=False).lower()
        ]
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def project_workflow_status(project_id: str) -> dict[str, Any]:
    path = project_dir(project_id)
    draft_ready = bool((path / "draft.md").read_text(encoding="utf-8").strip())
    goal_ready = bool((path / "goal.md").read_text(encoding="utf-8").strip())
    try:
        validate_taxonomy(read_json(path / "taxonomy.json"))
        taxonomy_ready = True
    except (json.JSONDecodeError, ValueError, TypeError):
        taxonomy_ready = False
    try:
        validate_relevance_profile(read_json(path / "relevance_scoring.json"))
        relevance_ready = True
    except (json.JSONDecodeError, ValueError, TypeError):
        relevance_ready = False

    try:
        candidates = load_candidates(path / "papers/logs/candidates.json")
    except (json.JSONDecodeError, ValueError, TypeError):
        candidates = []
    candidate_ids = {row["candidate_id"] for row in candidates}
    try:
        selection = read_json(path / "selection.json", {"candidate_ids": []}) or {}
    except (json.JSONDecodeError, TypeError):
        selection = {"candidate_ids": []}
    selected_ids = {
        str(value)
        for value in selection.get("candidate_ids", [])
        if str(value) in candidate_ids
    }
    try:
        manifest = read_json(path / "papers/logs/manifest.json", {}) or {}
    except (json.JSONDecodeError, TypeError):
        manifest = {}
    downloaded_ids: set[str] = set()
    for raw in manifest.get("candidates", []):
        if not isinstance(raw, dict):
            continue
        status = (raw.get("download") or {}).get("status")
        if status in {"downloaded", "already_exists"}:
            try:
                downloaded_ids.add(stable_candidate_id(candidate_from_dict(raw)))
            except ValueError:
                continue
    selected_downloaded = selected_ids & downloaded_ids
    feature_path = path / "feature_extract/features.jsonl"
    features_ready = feature_path.exists()
    try:
        feature_count = len(read_jsonl(feature_path))
    except (json.JSONDecodeError, ValueError, TypeError):
        feature_count = 0
        features_ready = False

    config = get_collection_config(project_id)
    refresh_candidates = bool(config.get("refresh_candidates"))
    collection_run_mode = (
        "search_metadata"
        if not candidates or refresh_candidates
        else "rescore_cached"
    )
    active_job = store.active_job(project_id)
    active_kind = str((active_job or {}).get("kind") or "")
    define_missing = [
        label
        for ready, label in [
            (draft_ready, "collection draft"),
            (goal_ready, "extraction goal"),
            (taxonomy_ready, "feature families"),
            (relevance_ready, "relevance scoring profile"),
        ]
        if not ready
    ]
    selection_missing = []
    if not selected_ids:
        selection_missing.append("at least one selected candidate")
    if not selected_downloaded:
        selection_missing.append("at least one selected full PDF")

    return {
        "steps": {
            "define": {
                "complete": not define_missing,
                "running": False,
                "missing": define_missing,
                "materials": {
                    "draft": draft_ready,
                    "goal": goal_ready,
                    "taxonomy": taxonomy_ready,
                    "relevance_scoring": relevance_ready,
                },
                "action_label": "Generate project definition",
            },
            "collect": {
                "complete": bool(candidates),
                "running": active_kind == "collection_search",
                "missing": [] if candidates else ["ranked paper candidates"],
                "materials": {
                    "candidate_count": len(candidates),
                    "run_mode": collection_run_mode,
                    "uses_cached_candidates": collection_run_mode == "rescore_cached",
                },
                "action_label": (
                    "Re-score cached candidates"
                    if collection_run_mode == "rescore_cached"
                    else "Find and rank papers"
                ),
            },
            "candidates": {
                "complete": bool(selected_downloaded),
                "running": active_kind == "collection_download",
                "missing": selection_missing,
                "materials": {
                    "candidate_count": len(candidates),
                    "selected_count": len(selected_ids),
                    "downloaded_count": len(downloaded_ids),
                    "selected_downloaded_count": len(selected_downloaded),
                },
                "action_label": "Download selected PDFs",
            },
            "extract": {
                "complete": features_ready,
                "running": active_kind == "direct_pdf_extraction",
                "missing": [] if features_ready else ["raw extracted features"],
                "materials": {
                    "selected_downloaded_count": len(selected_downloaded),
                    "feature_count": feature_count,
                },
                "action_label": "Extract raw features",
            },
            "results": {
                "complete": features_ready,
                "running": False,
                "missing": [] if features_ready else ["raw extracted features"],
                "materials": {"feature_count": feature_count},
                "action_label": "Download raw features",
            },
        }
    }


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
    keys = ["GEMINI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"]
    return {
        key: {
            "available": bool(SESSION_CREDENTIALS.get(key) or os.getenv(key)),
            "source": "session" if SESSION_CREDENTIALS.get(key) else ("environment" if os.getenv(key) else None),
        }
        for key in keys
    }


@app.put("/api/settings/session-credentials")
def set_session_credentials(payload: CredentialsPayload) -> dict[str, Any]:
    allowed = {"GEMINI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"}
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
    return read_json(path / "configs/collection.json", {
        "sources": ["openalex", "semantic_scholar", "crossref", "europe_pmc"],
        "year_from": None, "year_to": None, "max_queries": 8, "top_k_per_query": 10,
        "max_downloads": 50, "email": os.getenv("LITHARVEST_EMAIL", ""),
        "web_search_sources": [], "timeout": 30, "retries": 2, "rate_limit_delay": 0.1,
        "download_workers": 4, "refresh_candidates": False, "retry_failed_downloads": False,
    })


@app.put("/api/projects/{project_id}/collection/config")
def save_collection_config(project_id: str, payload: CollectionConfig) -> dict[str, Any]:
    path = project_dir(project_id) / "configs/collection.json"
    write_config(path, payload.model_dump())
    return read_json(path)


def harvest_config(project_id: str, *, metadata_only: bool, selected_ids: list[str] | None = None) -> dict[str, Any]:
    path = project_dir(project_id)
    config = dict(get_collection_config(project_id))
    config.update({
        "harvest_mode": "draft",
        "draft": str(path / "draft.md"),
        "output": str(path / "papers"),
        "logs": str(path / "papers/logs"),
        "metadata_only": metadata_only,
        "allow_abstract_fallback": False,
        "max_pdf_mb": 50,
        "progress": True,
        "csv_report": True,
        "bibtex": True,
    })
    relevance_path = path / "relevance_scoring.json"
    if relevance_path.exists():
        config["relevance_scoring"] = str(relevance_path)
    if selected_ids is not None:
        config["selected_candidate_ids"] = selected_ids
        config["refresh_candidates"] = False
    return config


@app.post("/api/projects/{project_id}/collection/search")
def start_search(project_id: str) -> dict[str, Any]:
    path = project_dir(project_id)
    if not (path / "draft.md").read_text(encoding="utf-8").strip():
        raise HTTPException(422, "Create or edit the collection draft first.")
    config_path = path / "configs/collection_search.json"
    write_config(config_path, harvest_config(project_id, metadata_only=True))
    return launch_job(
        project_id, "collection_search",
        [[sys.executable, str(REPO_ROOT / "lit_harvest.py"), "--config", str(config_path)]],
    )


@app.get("/api/projects/{project_id}/collection/candidates")
def list_candidates(
    project_id: str, page: int = 1, page_size: int = 50, query: str = "",
    sort: str = "relevance_score", order: str = "desc", download_status: str = "",
) -> dict[str, Any]:
    path = project_dir(project_id)
    rows = load_candidates(path / "papers/logs/candidates.json")
    manifest = read_json(path / "papers/logs/manifest.json", {}) or {}
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
        rows, page=page, page_size=page_size, query=query, sort=sort, order=order,
        filters={"download_status": download_status},
    )


@app.put("/api/projects/{project_id}/collection/selection")
def save_selection(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = project_dir(project_id)
    valid_ids = {
        row["candidate_id"] for row in load_candidates(path / "papers/logs/candidates.json")
    }
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


@app.post("/api/projects/{project_id}/collection/download")
def start_download(project_id: str) -> dict[str, Any]:
    path = project_dir(project_id)
    selected = read_json(path / "selection.json", {"candidate_ids": []})["candidate_ids"]
    if not selected:
        raise HTTPException(422, "Select at least one candidate.")
    config_path = path / "configs/collection_download.json"
    write_config(config_path, harvest_config(project_id, metadata_only=False, selected_ids=selected))
    return launch_job(
        project_id, "collection_download",
        [[sys.executable, str(REPO_ROOT / "lit_harvest.py"), "--config", str(config_path)]],
    )


@app.get("/api/projects/{project_id}/extraction/config")
def get_extraction_config(project_id: str) -> dict[str, Any]:
    return read_json(project_dir(project_id) / "configs/extraction.json", {
        "provider": "gemini", "model": "gemini-2.5-flash", "reasoning_effort": "low",
        "max_features_per_pdf": 40, "temperature": 0.1, "timeout": 600,
    })


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
        manifest_path = path / "papers/logs/manifest.json"
        with tempfile.TemporaryDirectory(prefix="lit-review-prompt-") as temp:
            paper_index = prepare_direct_pdf_index(
                path / "papers",
                manifest_path if manifest_path.exists() else None,
                Path(temp),
            )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc

    papers = [
        paper for paper in paper_index.get("papers", [])
        if isinstance(paper, dict) and paper.get("pdf_path")
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


@app.post("/api/projects/{project_id}/extraction/run")
def start_extraction(project_id: str) -> dict[str, Any]:
    path = project_dir(project_id)
    if not (path / "taxonomy.json").exists():
        raise HTTPException(422, "Define a valid taxonomy first.")
    if not (path / "goal.md").read_text(encoding="utf-8").strip():
        raise HTTPException(422, "Define an extraction goal first.")
    manifest = read_json(path / "papers/logs/manifest.json", {}) or {}
    downloaded = [
        candidate
        for candidate in manifest.get("candidates", [])
        if isinstance(candidate, dict)
        if (candidate.get("download") or {}).get("status") in {"downloaded", "already_exists"}
    ]
    if not downloaded:
        raise HTTPException(422, "Download at least one selected full PDF first.")
    config = get_extraction_config(project_id)
    provider = str(config.get("provider") or "gemini")
    if not provider_key(provider):
        raise HTTPException(422, f"No {provider} API key is available.")
    prepare_config = {
        "command": "prepare", "papers": str(path / "papers"),
        "manifest": str(path / "papers/logs/manifest.json"), "out": str(path / "review_run"),
    }
    extract_config = {
        "command": "run", "run": str(path / "review_run"), "output_dir": str(path / "feature_extract"),
        "goal": str(path / "goal.md"), "taxonomy": str(path / "taxonomy.json"),
        "model": config.get("model"), "base_url": config.get("base_url") or default_base_url(provider),
        "api_provider": provider, "reasoning_effort": config.get("reasoning_effort"),
        "api_key_env": "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY",
        "input_mode": "pdf", "max_features_per_pdf": config.get("max_features_per_pdf", 40),
        "temperature": config.get("temperature", 0.1), "timeout": config.get("timeout", 600),
    }
    prepare_path = path / "configs/prepare.json"
    extract_path = path / "configs/extraction_run.json"
    write_config(prepare_path, prepare_config)
    write_config(extract_path, extract_config)
    return launch_job(
        project_id, "direct_pdf_extraction",
        [
            [sys.executable, str(REPO_ROOT / "lit_synthesize.py"), "--config", str(prepare_path)],
            [sys.executable, str(REPO_ROOT / "lit_extract.py"), "--config", str(extract_path)],
        ],
    )


def result_list(project_id: str, filename: str, page: int, page_size: int, query: str, sort: str, order: str, category: str = "") -> dict[str, Any]:
    rows = read_jsonl(project_dir(project_id) / "feature_extract" / filename)
    return paginate_rows(
        rows, page=page, page_size=page_size, query=query, sort=sort, order=order,
        filters={"category": category},
    )


@app.get("/api/projects/{project_id}/results/features")
def results_features(project_id: str, page: int = 1, page_size: int = 50, query: str = "", sort: str = "paper_count", order: str = "desc", category: str = "") -> dict[str, Any]:
    return result_list(project_id, "features.jsonl", page, page_size, query, sort, order, category)


@app.get("/api/projects/{project_id}/results/mentions")
def results_mentions(project_id: str, page: int = 1, page_size: int = 50, query: str = "", sort: str = "confidence", order: str = "desc", category: str = "") -> dict[str, Any]:
    return result_list(project_id, "feature_mentions.jsonl", page, page_size, query, sort, order, category)


@app.get("/api/projects/{project_id}/results/errors")
def results_errors(project_id: str, page: int = 1, page_size: int = 50, query: str = "") -> dict[str, Any]:
    rows = read_jsonl(project_dir(project_id) / "feature_extract/extraction_errors.jsonl")
    return paginate_rows(rows, page=page, page_size=page_size, query=query, sort="created_at", order="desc")


@app.get("/api/projects/{project_id}/results/papers")
def results_papers(project_id: str, page: int = 1, page_size: int = 50, query: str = "") -> dict[str, Any]:
    payload = read_json(project_dir(project_id) / "review_run/paper_index.json", {}) or {}
    return paginate_rows(payload.get("papers", []), page=page, page_size=page_size, query=query, sort="title", order="asc")


@app.get("/api/projects/{project_id}/exports/{artifact}")
def export_artifact(project_id: str, artifact: str) -> FileResponse:
    allowed = {
        "features.jsonl", "features.csv", "feature_mentions.jsonl", "feature_mentions.csv",
        "paper_features.jsonl", "paper_features.csv", "extraction_errors.jsonl",
        "manifest.json", "candidates.csv", "candidates.bib",
    }
    if artifact not in allowed:
        raise HTTPException(404, "Unknown export artifact.")
    base = project_dir(project_id)
    path = (
        base / "papers/logs" / artifact
        if artifact in {"manifest.json", "candidates.csv", "candidates.bib"}
        else base / "feature_extract" / artifact
    )
    if not path.exists():
        raise HTTPException(404, "Artifact does not exist yet.")
    return FileResponse(path, filename=artifact)


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


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        target = FRONTEND_DIST / path
        return FileResponse(target if target.is_file() else FRONTEND_DIST / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run("lit_review_ui.app:app", host="127.0.0.1", port=8765, reload=False)
