"""Collection citations."""

from __future__ import annotations

from lit_review.core.io import write_json


import base64


import json


import re


import time

import urllib.error

import urllib.parse

import urllib.request


from dataclasses import dataclass


from pathlib import Path

from typing import Any, Callable


from lit_review.core.http import USER_AGENT_NAME, safe_int

from lit_review.collection.sources import (
    Candidate,
    SearchClient,
    SearchQuery,
    add_doi_pdf_urls,
    add_identifier_pdf_urls,
    add_url,
    clean_url,
    dedupe_preserve,
    doi_to_url,
    ensure_list,
    looks_like_pdf_url,
    normalize_arxiv_id,
    normalize_doi,
)
from lit_review.core.text import normalize_whitespace, strip_markup

from lit_review.collection.downloads import LOGGER
from lit_review.core.text import content_tokens, normalize_title

from lit_review.collection.progress import PhaseProgressReporter


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass(frozen=True)
class CitationLLMCallResult:
    parsed: dict[str, Any]
    response: dict[str, Any]
    raw_text: str
    input_tokens: int | None
    output_tokens: int | None
    schema_valid: bool
    validation_errors: list[str]


CITATION_REFERENCE_KEYS = [
    "ref_id",
    "raw_reference",
    "authors",
    "title",
    "year",
    "venue",
    "doi",
    "arxiv_id",
    "url",
]


def nullable_schema_type(type_name: str) -> dict[str, Any]:
    return {"anyOf": [{"type": type_name}, {"type": "null"}]}


CITATION_BIBLIOGRAPHY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "string"},
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ref_id": {"type": "string"},
                    "raw_reference": nullable_schema_type("string"),
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "title": nullable_schema_type("string"),
                    "year": nullable_schema_type("integer"),
                    "venue": nullable_schema_type("string"),
                    "doi": nullable_schema_type("string"),
                    "arxiv_id": nullable_schema_type("string"),
                    "url": nullable_schema_type("string"),
                },
                "required": CITATION_REFERENCE_KEYS,
            },
        },
    },
    "required": ["paper_id", "references"],
}


CITATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "citation_extraction",
    "strict": True,
    "schema": CITATION_BIBLIOGRAPHY_SCHEMA,
}


GEMINI_CITATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "propertyOrdering": ["paper_id", "references"],
    "properties": {
        "paper_id": {"type": "string"},
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "propertyOrdering": CITATION_REFERENCE_KEYS,
                "properties": {
                    "ref_id": {"type": "string"},
                    "raw_reference": {"type": ["string", "null"]},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "title": {"type": ["string", "null"]},
                    "year": {"type": ["integer", "null"]},
                    "venue": {"type": ["string", "null"]},
                    "doi": {"type": ["string", "null"]},
                    "arxiv_id": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                },
                "required": CITATION_REFERENCE_KEYS,
            },
        },
    },
    "required": ["paper_id", "references"],
}


CITATION_EXTRACTION_PROMPT = """\
You extract bibliography/reference-list entries from academic papers.

Rules:
- Return only JSON matching the requested schema.
- Extract entries from the bibliography, references, or works-cited section.
- Do not use Semantic Scholar, Crossref, OpenAlex, DBLP, search engines, or any external citation database.
- Use only the supplied PDF or extracted page text.
- Use null for unknown scalar fields and [] for unknown authors.
- Preserve the raw reference text as faithfully as possible in raw_reference.
- Use stable reference IDs R001, R002, R003, ... in document order.
- Do not include in-text citations unless they are part of the reference list.
"""


def safe_pdf_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return stem or "paper"


def make_citation_user_prompt(paper_id: str, source_description: str) -> str:
    return (
        f"paper_id: {paper_id}\n"
        f"source: {source_description}\n\n"
        "Extract the bibliography/reference list for this paper into the JSON schema."
    )


def empty_citation_bibliography(paper_id: str) -> dict[str, Any]:
    return {"paper_id": paper_id, "references": []}


def validate_citation_bibliography(data: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["top-level output must be an object"]

    extra_top_level = sorted(set(data) - {"paper_id", "references"})
    if extra_top_level:
        errors.append(f"top-level output has extra keys: {', '.join(extra_top_level)}")

    if not isinstance(data.get("paper_id"), str):
        errors.append("paper_id must be a string")

    references = data.get("references")
    if not isinstance(references, list):
        errors.append("references must be a list")
        return errors

    nullable_strings = {"raw_reference", "title", "venue", "doi", "arxiv_id", "url"}
    for index, reference in enumerate(references):
        prefix = f"references[{index}]"
        if not isinstance(reference, dict):
            errors.append(f"{prefix} must be an object")
            continue
        extra_keys = sorted(set(reference) - set(CITATION_REFERENCE_KEYS))
        if extra_keys:
            errors.append(f"{prefix} has extra keys: {', '.join(extra_keys)}")
        missing = [key for key in CITATION_REFERENCE_KEYS if key not in reference]
        if missing:
            errors.append(f"{prefix} missing keys: {', '.join(missing)}")

        if not isinstance(reference.get("ref_id"), str):
            errors.append(f"{prefix}.ref_id must be a string")

        authors = reference.get("authors")
        if not isinstance(authors, list) or not all(isinstance(author, str) for author in authors):
            errors.append(f"{prefix}.authors must be a list of strings")

        year = reference.get("year")
        if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
            errors.append(f"{prefix}.year must be an integer or null")

        for key in nullable_strings:
            value = reference.get(key)
            if value is not None and not isinstance(value, str):
                errors.append(f"{prefix}.{key} must be a string or null")

    return errors


def parse_citation_llm_json(raw_text: str, paper_id: str) -> tuple[dict[str, Any], list[str]]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return empty_citation_bibliography(paper_id), [f"model output was not valid JSON: {exc}"]

    errors = validate_citation_bibliography(parsed)
    if errors:
        return parsed if isinstance(parsed, dict) else empty_citation_bibliography(paper_id), errors
    return parsed, []


def retry_after_delay(exc: urllib.error.HTTPError, fallback: float) -> float:
    retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), fallback)
        except ValueError:
            return fallback
    return fallback


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT_NAME,
    }
    request_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(retry_after_delay(exc, sleep_seconds * (attempt + 1)))
                continue
            raise RuntimeError(f"LLM request failed ({exc.code}): {error_body}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))
                continue
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    raise RuntimeError(f"LLM request failed: {last_error}")


def extract_openai_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    refusals: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            content_type = content.get("type")
            if content_type in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
            elif content_type == "refusal" and isinstance(content.get("refusal"), str):
                refusals.append(content["refusal"])

    if refusals:
        raise RuntimeError(f"Model refusal: {' '.join(refusals)}")
    if not parts:
        raise RuntimeError("OpenAI response did not contain output text.")
    return "\n".join(parts)


def openai_usage_tokens(response: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = response.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int):
        input_tokens = None
    if not isinstance(output_tokens, int):
        output_tokens = None
    return input_tokens, output_tokens


def gemini_usage_tokens(response: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = response.get("usageMetadata") or {}
    input_tokens = usage.get("promptTokenCount")
    output_tokens = usage.get("candidatesTokenCount")
    if not isinstance(input_tokens, int):
        input_tokens = None
    if not isinstance(output_tokens, int):
        output_tokens = None
    return input_tokens, output_tokens


def gemini_content_part(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "input_text":
        return {"text": item.get("text") or ""}
    if item_type == "input_file":
        return {
            "inline_data": {
                "mime_type": "application/pdf",
                "data": item.get("file_data") or "",
            }
        }
    raise RuntimeError(f"Unsupported Gemini citation input item: {item_type}")


def extract_gemini_output_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    if texts:
        return "\n".join(texts)
    prompt_feedback = response.get("promptFeedback")
    if prompt_feedback:
        raise RuntimeError(f"Gemini prompt feedback: {prompt_feedback}")
    raise RuntimeError("Gemini response did not contain output text.")


def call_openai_citation_extraction(
    content: list[dict[str, Any]],
    config: dict[str, Any],
    paper_id: str,
) -> CitationLLMCallResult:
    payload: dict[str, Any] = {
        "model": config["llm_model"],
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": CITATION_EXTRACTION_PROMPT}],
            },
            {"role": "user", "content": content},
        ],
        "text": {"format": CITATION_RESPONSE_FORMAT},
    }
    if config.get("max_output_tokens"):
        payload["max_output_tokens"] = int(config["max_output_tokens"])

    response = post_json(
        OPENAI_RESPONSES_URL,
        payload,
        headers={"Authorization": f"Bearer {config['llm_api_key']}"},
        timeout=float(config["timeout"]),
        retries=int(config["retries"]),
        sleep_seconds=float(config["backoff"]),
    )
    raw_text = extract_openai_output_text(response)
    parsed, validation_errors = parse_citation_llm_json(raw_text, paper_id)
    input_tokens, output_tokens = openai_usage_tokens(response)
    return CitationLLMCallResult(
        parsed=parsed,
        response=response,
        raw_text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        schema_valid=not validation_errors,
        validation_errors=validation_errors,
    )


def call_gemini_citation_extraction(
    content: list[dict[str, Any]],
    config: dict[str, Any],
    paper_id: str,
) -> CitationLLMCallResult:
    parts = [{"text": CITATION_EXTRACTION_PROMPT}]
    parts.extend(gemini_content_part(item) for item in content)
    generation_config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseJsonSchema": GEMINI_CITATION_RESPONSE_SCHEMA,
    }
    if config.get("max_output_tokens"):
        generation_config["maxOutputTokens"] = int(config["max_output_tokens"])

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }
    model_name = str(config["llm_model"]).removeprefix("models/")
    url = (
        f"{GEMINI_GENERATE_URL.format(model=urllib.parse.quote(model_name, safe=''))}"
        f"?{urllib.parse.urlencode({'key': config['llm_api_key']})}"
    )
    response = post_json(
        url,
        payload,
        headers={},
        timeout=float(config["timeout"]),
        retries=int(config["retries"]),
        sleep_seconds=float(config["backoff"]),
    )
    raw_text = extract_gemini_output_text(response)
    parsed, validation_errors = parse_citation_llm_json(raw_text, paper_id)
    input_tokens, output_tokens = gemini_usage_tokens(response)
    return CitationLLMCallResult(
        parsed=parsed,
        response=response,
        raw_text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        schema_valid=not validation_errors,
        validation_errors=validation_errors,
    )


def call_llm_citation_extraction(
    content: list[dict[str, Any]],
    config: dict[str, Any],
    paper_id: str,
) -> CitationLLMCallResult:
    provider = config["llm_provider"]
    if provider == "gemini":
        return call_gemini_citation_extraction(content, config, paper_id)
    if provider == "openai":
        return call_openai_citation_extraction(content, config, paper_id)
    raise RuntimeError(f"Unsupported LLM provider: {provider}")


def require_pymupdf():
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for citation extractor pymupdf_markdown.") from exc
    return fitz


def extract_pdf_text_for_citations(pdf_path: Path) -> tuple[str, int]:
    fitz = require_pymupdf()
    pages: list[str] = []
    with fitz.open(pdf_path) as document:
        page_count = document.page_count
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            pages.append(f"# Page {page_number}\n\n{text.strip()}\n")
    return "\n".join(pages), page_count


def maybe_truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def build_direct_pdf_citation_content(
    pdf_path: Path, paper_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pdf_data = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    content = [
        {"type": "input_text", "text": make_citation_user_prompt(paper_id, "the attached PDF file")},
        {"type": "input_file", "filename": pdf_path.name, "file_data": pdf_data},
    ]
    metadata = {
        "source_description": "the attached PDF file",
        "pdf_bytes": pdf_path.stat().st_size,
    }
    return content, metadata


def build_pymupdf_markdown_citation_content(
    pdf_path: Path,
    paper_id: str,
    max_text_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    markdown, page_count = extract_pdf_text_for_citations(pdf_path)
    text_for_model, truncated = maybe_truncate_text(markdown, max_text_chars)
    prompt = make_citation_user_prompt(paper_id, "PyMuPDF-extracted page text")
    if truncated:
        prompt += (
            f"\n\nNote: the extracted text was truncated to {max_text_chars} characters "
            "by LitHarvest before citation extraction."
        )
    content = [{"type": "input_text", "text": f"{prompt}\n\n{text_for_model}"}]
    metadata = {
        "source_description": "PyMuPDF-extracted page text",
        "num_pages": page_count,
        "text_chars_total": len(markdown),
        "text_chars_sent": len(text_for_model),
        "text_truncated": truncated,
        "markdown": markdown,
    }
    return content, metadata


def collect_core_pdfs(config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    raw_core_pdfs = config.get("core_pdf") or []
    if isinstance(raw_core_pdfs, str):
        raw_core_pdfs = [raw_core_pdfs]
    for value in raw_core_pdfs:
        if value:
            paths.append(Path(str(value)).expanduser())

    core_pdf_dir = config.get("core_pdf_dir")
    if core_pdf_dir:
        directory = Path(str(core_pdf_dir)).expanduser()
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Core PDF directory does not exist: {directory}")
        paths.extend(sorted(directory.glob("*.pdf")))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)

    missing = [path for path in deduped if not path.exists() or not path.is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Core PDF not found: {formatted}")
    if not deduped:
        raise ValueError("Citation harvest mode requires at least one --core-pdf or --core-pdf-dir PDF.")
    return deduped


def citation_extraction_paths(pdf_path: Path, logs_dir: Path) -> tuple[str, Path, Path, Path]:
    paper_id = safe_pdf_stem(pdf_path)
    artifact_dir = logs_dir / "citation_extractions" / paper_id
    return (
        paper_id,
        artifact_dir,
        artifact_dir / "references.json",
        artifact_dir / "metadata.json",
    )


def extract_citations_from_core_pdf(
    pdf_path: Path,
    logs_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    paper_id, artifact_dir, output_path, metadata_path = citation_extraction_paths(
        pdf_path,
        logs_dir,
    )
    extractor = str(config["citation_extractor"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if extractor == "direct_pdf":
        content, extraction_metadata = build_direct_pdf_citation_content(pdf_path, paper_id)
    elif extractor == "pymupdf_markdown":
        content, extraction_metadata = build_pymupdf_markdown_citation_content(
            pdf_path,
            paper_id,
            int(config.get("max_text_chars") or 0),
        )
        markdown = str(extraction_metadata.pop("markdown"))
        markdown_path = artifact_dir / "extracted_text.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        extraction_metadata["markdown_path"] = str(markdown_path)
    else:
        raise ValueError(f"Unsupported citation extractor: {extractor}")

    start = time.perf_counter()
    call = call_llm_citation_extraction(content, config, paper_id)
    elapsed = time.perf_counter() - start

    raw_response_path = artifact_dir / "response.json"
    raw_text_path = artifact_dir / "response.txt"
    write_json(output_path, call.parsed)
    write_json(raw_response_path, call.response)
    raw_text_path.write_text(call.raw_text, encoding="utf-8")

    metadata = {
        "core_pdf": str(pdf_path),
        "paper_id": paper_id,
        "extractor": extractor,
        "llm_provider": config["llm_provider"],
        "llm_model": config["llm_model"],
        "wall_clock_seconds": round(elapsed, 6),
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "schema_valid": call.schema_valid,
        "validation_errors": call.validation_errors,
        "output_path": str(output_path),
        "raw_response_path": str(raw_response_path),
        "raw_text_path": str(raw_text_path),
    }
    metadata.update(extraction_metadata)
    write_json(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path)

    return {
        "core_pdf": str(pdf_path),
        "paper_id": paper_id,
        "references": call.parsed.get("references", []),
        "output": call.parsed,
        "metadata": metadata,
    }


def extracted_reference_to_candidate(
    reference: dict[str, Any],
    *,
    core_pdf: str,
    paper_id: str,
) -> Candidate | None:
    raw_reference = reference.get("raw_reference")
    raw_reference = normalize_whitespace(str(raw_reference)) if raw_reference else None
    title = strip_markup(reference.get("title")) or raw_reference
    if not title:
        return None

    authors = [
        normalize_whitespace(str(author))
        for author in ensure_list(reference.get("authors"))
        if normalize_whitespace(str(author))
    ]
    doi = normalize_doi(reference.get("doi"))
    url = clean_url(reference.get("url"))
    arxiv_id = normalize_arxiv_id(reference.get("arxiv_id")) or normalize_arxiv_id(url)
    pdf_urls: list[str] = []
    landing_urls: list[str] = []
    if looks_like_pdf_url(url):
        add_url(pdf_urls, url)
    else:
        add_url(landing_urls, url)
    add_doi_pdf_urls(pdf_urls, doi)
    if arxiv_id:
        add_identifier_pdf_urls(pdf_urls, arxiv_id, kind="arxiv")

    source_ids: dict[str, str] = {}
    if doi:
        source_ids["citation:doi"] = doi
    if arxiv_id:
        source_ids["citation:arxiv"] = arxiv_id

    return Candidate(
        title=title,
        authors=dedupe_preserve(authors),
        year=safe_int(reference.get("year")),
        doi=doi,
        venue=strip_markup(reference.get("venue")),
        landing_page_url=doi_to_url(doi) or (url if url and not looks_like_pdf_url(url) else None),
        source_apis=["citation_extraction"],
        source_ids=source_ids,
        candidate_pdf_urls=dedupe_preserve(pdf_urls),
        candidate_landing_page_urls=dedupe_preserve(landing_urls),
        discovered_via=[
            {
                "source": "citation_extraction",
                "core_pdf": core_pdf,
                "paper_id": paper_id,
                "ref_id": reference.get("ref_id"),
                "raw_reference": raw_reference,
            }
        ],
    )


def candidates_from_citation_extractions(extractions: list[dict[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for extraction in extractions:
        core_pdf = str(extraction.get("core_pdf") or "")
        paper_id = str(extraction.get("paper_id") or "")
        for reference in extraction.get("references") or []:
            if not isinstance(reference, dict):
                continue
            candidate = extracted_reference_to_candidate(
                reference,
                core_pdf=core_pdf,
                paper_id=paper_id,
            )
            if candidate:
                candidates.append(candidate)
    return candidates


def candidate_arxiv_id(candidate: Candidate) -> str | None:
    if candidate.doi and candidate.doi.lower().startswith("10.48550/arxiv."):
        return normalize_arxiv_id(candidate.doi.split("arxiv.", 1)[1])
    for key, value in candidate.source_ids.items():
        if "arxiv" in key.lower():
            arxiv_id = normalize_arxiv_id(value)
            if arxiv_id:
                return arxiv_id
    return None


def citation_lookup_query(candidate: Candidate) -> str | None:
    if candidate.doi:
        return candidate.doi
    arxiv_id = candidate_arxiv_id(candidate)
    if arxiv_id:
        return f"arXiv:{arxiv_id}"
    title = normalize_whitespace(candidate.title)
    if not title:
        return None
    parts = [title]
    if candidate.year:
        parts.append(str(candidate.year))
    if candidate.authors:
        parts.append(candidate.authors[0])
    return " ".join(parts)


def first_author_last_name(candidate: Candidate) -> str | None:
    if not candidate.authors:
        return None
    tokens = normalize_title(candidate.authors[0]).split()
    if not tokens:
        return None
    return tokens[-1]


def title_token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(content_tokens(left))
    right_tokens = set(content_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def citation_match_reason(seed: Candidate, result: Candidate) -> str | None:
    if seed.doi and result.doi and normalize_doi(seed.doi) == normalize_doi(result.doi):
        return "doi"

    seed_arxiv = candidate_arxiv_id(seed)
    result_arxiv = candidate_arxiv_id(result)
    if seed_arxiv and result_arxiv and seed_arxiv.lower() == result_arxiv.lower():
        return "arxiv"

    seed_title = normalize_title(seed.title)
    result_title = normalize_title(result.title)
    if seed_title and result_title and seed_title == result_title:
        return "title"

    overlap = title_token_overlap_ratio(seed.title, result.title)
    seed_author = first_author_last_name(seed)
    result_author = first_author_last_name(result)
    author_matches = bool(seed_author and result_author and seed_author == result_author)
    year_matches = bool(seed.year and result.year and seed.year == result.year)
    if overlap >= 0.8 and (author_matches or year_matches):
        if author_matches and year_matches:
            return "title_overlap_author_year"
        if author_matches:
            return "title_overlap_author"
        return "title_overlap_year"
    return None


def citation_lookup_records(candidate: Candidate) -> list[dict[str, Any]]:
    return [
        item
        for item in candidate.discovered_via
        if item.get("source") == "citation_lookup" and isinstance(item.get("attempts"), list)
    ]


def citation_lookup_resume_query(candidate: Candidate) -> str | None:
    for record in reversed(citation_lookup_records(candidate)):
        if "query" in record:
            query = record.get("query")
            return str(query) if query is not None else None
    return citation_lookup_query(candidate)


def citation_lookup_record(candidate: Candidate, query_text: str | None) -> dict[str, Any]:
    for record in reversed(citation_lookup_records(candidate)):
        if record.get("query") == query_text:
            return record
    record = {
        "source": "citation_lookup",
        "query": query_text,
        "attempts": [],
    }
    candidate.discovered_via.append(record)
    return record


def citation_lookup_attempted_sources(candidate: Candidate, query_text: str | None) -> set[str]:
    sources: set[str] = set()
    for record in citation_lookup_records(candidate):
        if record.get("query") != query_text:
            continue
        for attempt in record.get("attempts") or []:
            if isinstance(attempt, dict) and attempt.get("source"):
                sources.add(str(attempt["source"]))
    return sources


def citation_lookup_has_skip(candidate: Candidate, query_text: str | None) -> bool:
    for record in citation_lookup_records(candidate):
        if record.get("query") != query_text:
            continue
        for attempt in record.get("attempts") or []:
            if (
                isinstance(attempt, dict)
                and attempt.get("source") == "citation_lookup"
                and attempt.get("status") == "skipped"
            ):
                return True
    return False


def citation_lookup_completed_attempt_count(
    candidate: Candidate,
    clients: list[SearchClient],
) -> int:
    query_text = citation_lookup_resume_query(candidate)
    if not query_text:
        return 1 if citation_lookup_has_skip(candidate, query_text) else 0
    client_names = {client.name for client in clients}
    return len(citation_lookup_attempted_sources(candidate, query_text) & client_names)


def citation_lookup_total_attempt_count(
    candidate: Candidate,
    clients: list[SearchClient],
) -> int:
    return len(clients) if citation_lookup_resume_query(candidate) else 1


def citation_lookup_accepted_count(candidates: list[Candidate]) -> int:
    total = 0
    for candidate in candidates:
        for record in citation_lookup_records(candidate):
            for attempt in record.get("attempts") or []:
                if isinstance(attempt, dict):
                    total += int(safe_int(attempt.get("accepted_count")) or 0)
    return total


def citation_lookup_skipped_count(candidates: list[Candidate]) -> int:
    skipped = 0
    for candidate in candidates:
        query_text = citation_lookup_resume_query(candidate)
        if not query_text and citation_lookup_has_skip(candidate, query_text):
            skipped += 1
    return skipped


def citation_lookup_progress_payload(
    candidates: list[Candidate],
    clients: list[SearchClient],
    lookup_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    total_attempts = sum(citation_lookup_total_attempt_count(candidate, clients) for candidate in candidates)
    completed_attempts = sum(
        citation_lookup_completed_attempt_count(candidate, clients) for candidate in candidates
    )
    completed_candidates = sum(
        1
        for candidate in candidates
        if citation_lookup_completed_attempt_count(candidate, clients)
        >= citation_lookup_total_attempt_count(candidate, clients)
    )
    return {
        "total_candidates": len(candidates),
        "completed_candidates": completed_candidates,
        "total_lookup_attempts": total_attempts,
        "completed_lookup_attempts": completed_attempts,
        "accepted_count": citation_lookup_accepted_count(candidates),
        "error_count": len(lookup_errors),
        "skipped_count": citation_lookup_skipped_count(candidates),
    }


def enrich_citation_candidates(
    candidates: list[Candidate],
    clients: list[SearchClient],
    config: dict[str, Any],
    checkpoint: Callable[[list[Candidate], list[dict[str, Any]], dict[str, Any]], None] | None = None,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    enriched: list[Candidate] = []
    lookup_errors: list[dict[str, Any]] = [
        error
        for error in ensure_list(config.get("_initial_citation_lookup_errors"))
        if isinstance(error, dict)
    ]
    total_attempts = sum(citation_lookup_total_attempt_count(candidate, clients) for candidate in candidates)
    completed_attempts = sum(
        citation_lookup_completed_attempt_count(candidate, clients) for candidate in candidates
    )
    progress = PhaseProgressReporter(
        total_attempts,
        enabled=bool(config.get("progress")),
        label="Citation lookup",
    )
    accepted_total = citation_lookup_accepted_count(candidates)
    skipped_total = citation_lookup_skipped_count(candidates)
    progress.set_current(
        completed_attempts,
        summary=f"accepted={accepted_total} errors={len(lookup_errors)} skipped={skipped_total}",
        force=True,
    )
    LOGGER.info(
        "Looking up %d citation candidates across %d sources (%d lookup attempts).",
        len(candidates),
        len(clients),
        total_attempts,
    )
    for candidate in candidates:
        query_text = citation_lookup_resume_query(candidate)
        lookup_record = citation_lookup_record(candidate, query_text)
        lookup_attempts = lookup_record["attempts"]
        if query_text:
            query = SearchQuery(
                bucket="citation_lookup", text=query_text, terms=content_tokens(candidate.title)
            )
            attempted_sources = citation_lookup_attempted_sources(candidate, query_text)
            for client in clients:
                if client.name in attempted_sources:
                    continue
                try:
                    results = client.search(
                        query,
                        top_k=int(config["top_k_per_query"]),
                        year_from=config.get("year_from"),
                        year_to=config.get("year_to"),
                    )
                    accepted = 0
                    accepted_reasons: list[str] = []
                    for result in results:
                        reason = citation_match_reason(candidate, result)
                        if not reason:
                            continue
                        accepted += 1
                        accepted_reasons.append(reason)
                        result.discovered_via = result.discovered_via + [
                            {
                                "source": "citation_lookup_match",
                                "client": client.name,
                                "query": query_text,
                                "match_reason": reason,
                            }
                        ]
                        merge_candidate_into(candidate, result)
                    accepted_total += accepted
                    lookup_attempts.append(
                        {
                            "source": client.name,
                            "query": query_text,
                            "result_count": len(results),
                            "accepted_count": accepted,
                            "accepted_reasons": dedupe_preserve(accepted_reasons),
                        }
                    )
                except Exception as exc:
                    error = {
                        "source": client.name,
                        "query": query_text,
                        "title": candidate.title,
                        "reason": str(exc),
                    }
                    lookup_errors.append(error)
                    lookup_attempts.append({**error, "status": "error"})
                attempted_sources.add(client.name)
                progress.advance(
                    summary=f"accepted={accepted_total} errors={len(lookup_errors)} skipped={skipped_total}"
                )
                if checkpoint:
                    checkpoint(
                        candidates,
                        lookup_errors,
                        citation_lookup_progress_payload(candidates, clients, lookup_errors),
                    )
        elif not citation_lookup_has_skip(candidate, query_text):
            skipped_total += 1
            lookup_attempts.append(
                {
                    "source": "citation_lookup",
                    "status": "skipped",
                    "reason": "No DOI, arXiv ID, title, or raw reference available.",
                }
            )
            progress.advance(
                summary=f"accepted={accepted_total} errors={len(lookup_errors)} skipped={skipped_total}"
            )
            if checkpoint:
                checkpoint(
                    candidates,
                    lookup_errors,
                    citation_lookup_progress_payload(candidates, clients, lookup_errors),
                )
        enriched.append(candidate)

    progress.done(summary=f"accepted={accepted_total} errors={len(lookup_errors)} skipped={skipped_total}")
    return merge_candidates(enriched), lookup_errors


def merge_candidates(candidates: list[Candidate]) -> list[Candidate]:
    merged: list[Candidate] = []
    by_key: dict[str, Candidate] = {}
    for candidate in candidates:
        keys = candidate_keys(candidate)
        existing = next((by_key[key] for key in keys if key in by_key), None)
        if existing is None:
            merged.append(candidate)
            for key in keys:
                by_key[key] = candidate
            continue
        merge_candidate_into(existing, candidate)
        for key in candidate_keys(existing):
            by_key[key] = existing
    return merged


def dedupe_discovered_via(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            key = json.dumps(item, sort_keys=True, default=str)
        except TypeError:
            key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def candidate_keys(candidate: Candidate) -> list[str]:
    keys: list[str] = []
    if candidate.doi:
        keys.append(f"doi:{candidate.doi}")
    arxiv_id = candidate_arxiv_id(candidate)
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id.lower()}")
    title_key = normalize_title(candidate.title)
    if title_key:
        keys.append(f"title:{title_key}")
        if candidate.year and candidate.authors:
            keys.append(
                "hint:"
                + "|".join(
                    [
                        title_key,
                        str(candidate.year),
                        normalize_title(candidate.authors[0]).split(" ")[-1],
                    ]
                )
            )
    return keys


def merge_candidate_into(target: Candidate, incoming: Candidate) -> None:
    target.source_apis = dedupe_preserve(target.source_apis + incoming.source_apis)
    target.source_ids.update({key: value for key, value in incoming.source_ids.items() if value})
    target.candidate_pdf_urls = dedupe_preserve(target.candidate_pdf_urls + incoming.candidate_pdf_urls)
    target.candidate_landing_page_urls = dedupe_preserve(
        target.candidate_landing_page_urls + incoming.candidate_landing_page_urls
    )
    target.matched_keywords = dedupe_preserve(target.matched_keywords + incoming.matched_keywords)
    target.discovered_via = dedupe_discovered_via(target.discovered_via + incoming.discovered_via)
    if not target.doi and incoming.doi:
        target.doi = incoming.doi
    if not target.year and incoming.year:
        target.year = incoming.year
    if not target.authors and incoming.authors:
        target.authors = incoming.authors
    if not target.venue and incoming.venue:
        target.venue = incoming.venue
    if not target.landing_page_url and incoming.landing_page_url:
        target.landing_page_url = incoming.landing_page_url
    if incoming.abstract and (not target.abstract or len(incoming.abstract) > len(target.abstract)):
        target.abstract = incoming.abstract
    if incoming.citation_count and (
        not target.citation_count or incoming.citation_count > target.citation_count
    ):
        target.citation_count = incoming.citation_count
    if incoming.source_score and (target.source_score is None or incoming.source_score > target.source_score):
        target.source_score = incoming.source_score
