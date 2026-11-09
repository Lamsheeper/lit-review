"""Requirements shared by startup and stage-specific UI checks."""

from __future__ import annotations
import os
from pathlib import Path
from lit_review.core.google_auth import use_vertex, vertex_model_url
from lit_review.evidence.pdf import resolve_download_path


def summarize(requirements):
    merged = {}
    for item in requirements:
        if item["id"] not in merged:
            merged[item["id"]] = item
        elif item["message"] not in merged[item["id"]]["message"]:
            if item["id"] == "GEMINI_API_KEY":
                merged[item["id"]]["message"] = (
                    "Add a Gemini API key for literature search, feature extraction, and concept merging."
                )
            else:
                merged[item["id"]]["message"] += " " + item["message"]
    return {
        "ready": not merged,
        "requirements": list(merged.values()),
        "message": "Required credentials and inputs are present."
        if not merged
        else "Complete the required inputs before starting.",
    }


def provider_requirements(config, resolve_key, purpose="text generation", embedding=False):
    provider = "gemini" if embedding else config.get("provider", "gemini")
    env_name = (
        config.get("embedding_api_key_env", "GEMINI_API_KEY")
        if embedding
        else config.get("api_key_env") or ("GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY")
    )
    key = resolve_key(env_name)
    mode = (
        (config.get("embedding_auth_mode") or config.get("auth_mode", "auto"))
        if embedding
        else config.get("auth_mode", "auto")
    )
    project = (
        (config.get("embedding_vertex_project") or config.get("vertex_project"))
        if embedding
        else config.get("vertex_project")
    )
    location = (
        (config.get("embedding_vertex_location") or config.get("vertex_location"))
        if embedding
        else config.get("vertex_location")
    )
    model = (
        config.get("embedding_model", "gemini-embedding-001")
        if embedding
        else config.get("model", "gemini-2.5-flash")
    )
    vertex = provider == "gemini" and use_vertex(key, auth_mode=mode, project=project)
    if not vertex:
        if key and key.strip():
            return []
        return [
            {
                "id": env_name,
                "field": env_name,
                "kind": "credential",
                "label": "Gemini API key"
                if env_name == "GEMINI_API_KEY"
                else "OpenAI API key"
                if env_name == "OPENAI_API_KEY"
                else env_name,
                "message": f"Add a Gemini API key for {purpose}."
                if env_name == "GEMINI_API_KEY"
                else f"Add an OpenAI API key for {purpose}."
                if env_name == "OPENAI_API_KEY"
                else f"Set {env_name} for {purpose}.",
            }
        ]
    try:
        vertex_model_url(project, location, model)
    except ValueError:
        return [
            {
                "id": "google_project",
                "kind": "configuration",
                "label": "Google Cloud project and region",
                "message": "Set a valid Google Cloud project and region before using Vertex AI.",
            }
        ]
    credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_file and not Path(credentials_file).expanduser().is_file():
        return [
            {
                "id": "google_adc_file",
                "kind": "file",
                "label": "Google Cloud credentials file",
                "message": f"The configured Google Cloud credentials file is missing: {credentials_file}. Correct GOOGLE_APPLICATION_CREDENTIALS or set up Application Default Credentials.",
            }
        ]
    try:
        import google.auth

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if credentials is None:
            raise ValueError()
    except Exception:
        return [
            {
                "id": "google_adc",
                "kind": "configuration",
                "label": "Google Cloud login",
                "message": "Open Credentials → Google Cloud to sign in and save your project, or run gcloud auth application-default login on the machine running this app.",
            }
        ]
    return []


def document_requirements(path, read_json, selected_id, deep=False):
    issues = []
    goal = path / "goal.md"
    try:
        if not goal.read_text(encoding="utf-8").strip():
            raise ValueError()
    except (OSError, ValueError):
        issues.append(
            {
                "id": "goal",
                "kind": "file",
                "label": "Extraction goal",
                "message": "Save a nonempty extraction goal before extracting PDFs.",
            }
        )
    try:
        from lit_review.core.taxonomy import validate_taxonomy

        validate_taxonomy(read_json(path / "taxonomy.json"))
    except (OSError, ValueError, TypeError):
        issues.append(
            {
                "id": "taxonomy",
                "kind": "file",
                "label": "Feature families",
                "message": "Prepare or save valid feature families before extracting PDFs.",
            }
        )
    manifest_path = path / "run/stages/collect/logs/manifest.json"
    try:
        selection = set((read_json(path / "selection.json", {}) or {}).get("candidate_ids", []))
        candidates = (read_json(manifest_path, {}) or {}).get("candidates", [])
    except (OSError, ValueError, TypeError):
        selection = set()
        candidates = []
    usable = set()
    invalid = []
    for candidate in candidates:
        qid = selected_id(candidate)
        if qid not in selection:
            continue
        download = candidate.get("download") or {}
        raw = download.get("path") or download.get("filename")
        if not raw or download.get("status") not in ("downloaded", "already_exists"):
            invalid.append(candidate.get("title", "Selected paper"))
            continue
        file = resolve_download_path(str(raw), path / "run/stages/collect", manifest_path)
        try:
            with file.open("rb") as handle:
                if b"%PDF-" not in handle.read(1024):
                    raise ValueError()
            if deep:
                from pypdf import PdfReader

                pdf = PdfReader(str(file))
                if pdf.is_encrypted and not pdf.decrypt(""):
                    raise ValueError()
                if not len(pdf.pages):
                    raise ValueError()
            usable.add(qid)
        except Exception:
            invalid.append(candidate.get("title", file.name))
    absent = selection - usable
    if not selection or absent:
        message = (
            "Select and download at least one full PDF before extracting."
            if not selection
            else f"{len(absent)} selected paper(s) have no readable, unlocked PDF. Download or replace those PDFs, or deselect the papers."
        )
        if invalid:
            message += " Check: " + ", ".join(invalid[:3]) + "."
        issues.append({"id": "pdfs", "kind": "file", "label": "Selected full PDFs", "message": message})
    return issues, usable
