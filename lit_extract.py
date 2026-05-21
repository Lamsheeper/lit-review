#!/usr/bin/env python3
"""Corpus-wide extraction of persuasion taxonomy feature mentions.

LitExtract is the database-building companion to LitSynth. It scans an existing
LitSynth run folder, sends Markdown-derived chunks or source PDFs to an
OpenAI-compatible or Gemini-native LLM endpoint, and writes normalized JSONL/CSV
artifacts for downstream taxonomy review.
"""

from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from lit_pdf_service import read_json, utc_now, write_json
from lit_synthesize import ChatCompletionClient, DEFAULT_MODEL

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - dependency is optional at runtime
    _tqdm = None


VERSION = "0.1.0"
PROMPT_VERSION = "lit_extract_v1"
LOGGER = logging.getLogger("litextract")
PDF_INPUT_MAX_BYTES = 50 * 1024 * 1024

CATEGORIES = {"persuasion", "moral_framing", "sentiment_affect"}
CATEGORY_EXPORT_ORDER = ["persuasion", "moral_framing", "sentiment_affect"]
EVIDENCE_TYPES = {"definition", "taxonomy_row", "example", "named_mention"}
FINAL_FEATURES_PER_CATEGORY = 100
TITLE_RANK_RULES = [
    (r"\btaxonom(?:y|ies)\b", 14),
    (r"\bclassification\b", 5),
    (r"\bannotation\b", 6),
    (r"\bdataset\b", 5),
    (r"\bbenchmark\b", 5),
    (r"\bpropaganda techniques?\b", 10),
    (r"\bpersua\w* techniques?\b", 9),
    (r"\bpropaganda\b", 16),
    (r"\bpersua\w*\b", 15),
    (r"\brhetoric\w*\b", 13),
    (r"\bfallac(?:y|ies)\b", 12),
    (r"\bframing\b", 7),
    (r"\bframe(?:s|d)?\b", 3),
    (r"\bmoral(?:ity)?\b", 7),
    (r"\bmoral foundations?\b", 9),
    (r"\bsentiment\b", 9),
    (r"\bemotion\w*\b", 9),
    (r"\baffect(?:ive|s)?\b", 8),
    (r"\bmedia bias\b", 11),
    (r"\bmedia analys(?:is|es)\b", 8),
    (r"\bnews media\b", 7),
    (r"\bjournalis\w*\b", 6),
    (r"\bdisinformation\b", 10),
    (r"\bmisinformation\b", 10),
    (r"\bnarrative(?:s)?\b", 6),
    (r"\bideolog(?:y|ical)\b", 5),
    (r"\bpolitical communication\b", 7),
    (r"\bdiscourse\b", 4),
    (r"\bstance\b", 4),
]
TITLE_PENALTY_RULES = [
    (r"\bpoint cloud\b", 8),
    (r"\bcompression\b", 5),
    (r"\bsensor\b", 4),
    (r"\bsemiconductor\b", 5),
    (r"\bcell atlas\b", 5),
    (r"\bclinical\b", 3),
]
CONFIG_ARG_FIELDS: dict[str, list[tuple[str, str]]] = {
    "run": [
        ("run", "--run"),
        ("output_dir", "--output-dir"),
        ("api_key", "--api-key"),
        ("model", "--model"),
        ("base_url", "--base-url"),
        ("api_provider", "--api-provider"),
        ("reasoning_effort", "--reasoning-effort"),
        ("input_mode", "--input-mode"),
        ("batch_chunks", "--batch-chunks"),
        ("paper_limit", "--paper-limit"),
        ("goal", "--goal"),
        ("max_features_per_chunk", "--max-features-per-chunk"),
        ("max_features_per_pdf", "--max-features-per-pdf"),
        ("max_chunk_chars", "--max-chunk-chars"),
        ("temperature", "--temperature"),
        ("timeout", "--timeout"),
        ("force", "--force"),
        ("dry_run", "--dry-run"),
        ("verbose", "--verbose"),
        ("quiet", "--quiet"),
    ],
    "curate": [
        ("input_dir", "--input-dir"),
        ("output_dir", "--output-dir"),
        ("goal", "--goal"),
        ("api_key", "--api-key"),
        ("model", "--model"),
        ("base_url", "--base-url"),
        ("reasoning_effort", "--reasoning-effort"),
        ("batch_features", "--batch-features"),
        ("merge_style", "--merge-style"),
        ("temperature", "--temperature"),
        ("timeout", "--timeout"),
        ("force", "--force"),
        ("dry_run", "--dry-run"),
        ("verbose", "--verbose"),
        ("quiet", "--quiet"),
    ],
    "finalize": [
        ("input_dir", "--input-dir"),
        ("output_dir", "--output-dir"),
        ("goal", "--goal"),
        ("api_key", "--api-key"),
        ("model", "--model"),
        ("base_url", "--base-url"),
        ("api_provider", "--api-provider"),
        ("reasoning_effort", "--reasoning-effort"),
        ("target_per_category", "--target-per-category"),
        ("candidate_pool_per_category", "--candidate-pool-per-category"),
        ("merge_style", "--merge-style"),
        ("temperature", "--temperature"),
        ("max_tokens", "--max-tokens"),
        ("timeout", "--timeout"),
        ("force", "--force"),
        ("dry_run", "--dry-run"),
        ("verbose", "--verbose"),
        ("quiet", "--quiet"),
    ],
}

CSV_FIELDNAMES = [
    "paper_id",
    "title",
    "authors",
    "year",
    "doi",
    "venue",
    "category",
    "feature_name",
    "normalized_feature_name",
    "parent_category",
    "definition",
    "synonyms",
    "examples",
    "evidence_type",
    "chunk_id",
    "pages",
    "snippet",
    "source_quote",
    "confidence",
    "notes",
    "model",
    "run_id",
    "extracted_at",
]

PAPER_FEATURE_FIELDNAMES = [
    "paper_id",
    "title",
    "authors",
    "year",
    "doi",
    "venue",
    "category",
    "feature_name",
    "normalized_feature_name",
    "parent_category",
    "definition",
    "synonyms",
    "examples",
    "evidence_types",
    "mention_count",
    "chunk_ids",
    "pages",
    "best_snippet",
    "best_source_quote",
    "confidence",
    "notes",
    "model",
    "run_id",
    "extracted_at",
]

FEATURE_FIELDNAMES = [
    "category",
    "feature_name",
    "normalized_feature_name",
    "parent_categories",
    "definitions",
    "synonyms",
    "examples",
    "paper_count",
    "mention_count",
    "source_paper_ids",
    "source_titles",
    "evidence_types",
    "confidence",
    "notes",
    "model",
    "run_id",
    "extracted_at",
]

CURATED_FEATURE_FIELDNAMES = [
    "category",
    "feature_name",
    "normalized_feature_name",
    "source_feature_names",
    "source_normalized_feature_names",
    "parent_categories",
    "definitions",
    "synonyms",
    "examples",
    "paper_count",
    "mention_count",
    "source_paper_ids",
    "source_titles",
    "evidence_types",
    "source_confidence_avg",
    "source_confidence_max",
    "curation_confidence",
    "member_count",
    "notes",
    "curation_notes",
    "model",
    "run_id",
    "curated_at",
]

CURATED_MEMBER_FIELDNAMES = [
    "feature_key",
    "category",
    "original_feature_name",
    "original_normalized_feature_name",
    "action",
    "canonical_feature_key",
    "canonical_feature_name",
    "canonical_normalized_feature_name",
    "rejection_reason",
    "curation_confidence",
    "curation_notes",
    "paper_count",
    "mention_count",
    "model",
    "run_id",
    "curated_at",
]

REJECTED_FEATURE_FIELDNAMES = [
    "feature_key",
    "category",
    "feature_name",
    "normalized_feature_name",
    "parent_categories",
    "definitions",
    "synonyms",
    "examples",
    "paper_count",
    "mention_count",
    "source_paper_ids",
    "source_titles",
    "evidence_types",
    "confidence",
    "notes",
    "rejection_reason",
    "curation_confidence",
    "curation_notes",
    "model",
    "run_id",
    "curated_at",
]

RELEVANCE_PATTERNS = [
    r"\bpersuasion\b",
    r"\bpersuasive\b",
    r"\bpropaganda\b",
    r"\bpropagandistic\b",
    r"\brhetoric(?:al)?\b",
    r"\bframing\b",
    r"\bmoral(?:ity)?\b",
    r"\bmoral foundations?\b",
    r"\bsentiment\b",
    r"\bemotion(?:al|s)?\b",
    r"\baffect(?:ive|s)?\b",
    r"\bideolog(?:y|ical)\b",
    r"\bmedia bias\b",
    r"\bdisinformation\b",
    r"\bmisinformation\b",
    r"\bnarrative(?:s)?\b",
    r"\bfallac(?:y|ies)\b",
    r"\bappeal(?:s)?\b",
    r"\bloaded language\b",
    r"\bname[- ]calling\b",
    r"\bwhataboutism\b",
    r"\bfear\b",
    r"\banger\b",
    r"\bdisgust\b",
    r"\bcare[/ -]?harm\b",
    r"\bfairness[/ -]?cheating\b",
    r"\bloyalty[/ -]?betrayal\b",
    r"\bauthority[/ -]?subversion\b",
    r"\bsanctity[/ -]?degradation\b",
    r"\bliberty[/ -]?oppression\b",
]
RELEVANCE_RE = re.compile("|".join(RELEVANCE_PATTERNS), flags=re.IGNORECASE)


@dataclass
class BatchResult:
    paper_id: str
    batch_index: int
    chunk_ids: list[str]
    features: list[dict[str, Any]]
    error: str | None = None
    raw_response: str | None = None


@dataclass
class ExtractionStats:
    paper_count: int = 0
    paper_completed_count: int = 0
    paper_skipped_count: int = 0
    pdf_count: int = 0
    chunk_count: int = 0
    batch_count: int = 0
    failed_batch_count: int = 0
    raw_feature_count: int = 0
    kept_feature_count: int = 0
    filtered_feature_count: int = 0
    repaired_json_count: int = 0


@dataclass
class CurationStats:
    feature_count: int = 0
    batch_count: int = 0
    completed_batch_count: int = 0
    skipped_batch_count: int = 0
    failed_batch_count: int = 0
    decision_count: int = 0
    repaired_json_count: int = 0


@dataclass
class FinalCurationStats:
    feature_count: int = 0
    category_count: int = 0
    candidate_feature_count: int = 0
    completed_category_count: int = 0
    skipped_category_count: int = 0
    failed_category_count: int = 0
    decision_count: int = 0
    repaired_json_count: int = 0


class NullProgress:
    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, count: int) -> None:
        return None


def progress_bar(total: int, desc: str, unit: str, show: bool) -> Any:
    if not show or _tqdm is None or not sys.stderr.isatty():
        return NullProgress()
    return _tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
    )


def chunk_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Extracting chunks", unit="chunk", show=show)


def extraction_progress(total: int, input_mode: str, show: bool) -> Any:
    if input_mode == "pdf":
        return progress_bar(total=total, desc="Extracting PDFs", unit="paper", show=show)
    return chunk_progress(total=total, show=show)


def title_score_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Scoring titles", unit="paper", show=show)


def curation_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Curating features", unit="feature", show=show)


def final_curation_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Finalizing features", unit="feature", show=show)


def clean_api_key(api_key: str) -> str:
    return api_key.strip().strip("\"'“”‘’")


def env_api_key() -> str | None:
    return (
        os.getenv("LIT_EXTRACT_OPENAI_API_KEY")
        or os.getenv("LIT_SYNTH_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def env_base_url() -> str:
    return (
        os.getenv("LIT_EXTRACT_LLM_BASE_URL")
        or os.getenv("LIT_SYNTH_LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )


def env_model() -> str:
    return os.getenv("LIT_EXTRACT_LLM_MODEL") or os.getenv("LIT_SYNTH_LLM_MODEL") or DEFAULT_MODEL


def native_gemini_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/openai"):
        base = base[: -len("/openai")]
    return base or "https://generativelanguage.googleapis.com/v1beta"


def is_gemini_base_url(base_url: str) -> bool:
    return "generativelanguage.googleapis.com" in base_url.lower()


def should_use_gemini_native_client(
    *,
    api_provider: str,
    base_url: str,
    input_mode: str,
) -> bool:
    if api_provider == "gemini":
        return True
    if api_provider == "openai":
        return False
    return input_mode == "pdf" and is_gemini_base_url(base_url)


def strip_data_url(value: str) -> str:
    if "," in value and value.lower().startswith("data:"):
        return value.split(",", 1)[1]
    return value


def gemini_thinking_config(model: str, reasoning_effort: str | None) -> dict[str, Any] | None:
    if not reasoning_effort:
        return None
    effort = reasoning_effort.strip().lower()
    model_name = model.lower()
    if "gemini-3" in model_name:
        if effort in {"minimal", "low", "medium", "high"}:
            return {"thinkingLevel": effort}
        return None
    budgets = {
        "none": 0,
        "minimal": 1024,
        "low": 1024,
        "medium": 8192,
        "high": 24576,
    }
    if effort in budgets:
        return {"thinkingBudget": budgets[effort]}
    return None


def gemini_parts_from_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append({"text": str(item)})
            continue
        item_type = str(item.get("type") or "")
        if item_type in {"text", "input_text"}:
            parts.append({"text": str(item.get("text") or "")})
            continue
        if item_type in {"file", "input_file"}:
            file_payload = item.get("file") if isinstance(item.get("file"), dict) else item
            file_data = str(file_payload.get("file_data") or "")
            file_uri = str(file_payload.get("file_uri") or file_payload.get("file_url") or "")
            mime_type = str(file_payload.get("mime_type") or "application/pdf")
            if file_data:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": strip_data_url(file_data),
                        }
                    }
                )
            elif file_uri:
                parts.append(
                    {
                        "file_data": {
                            "mime_type": mime_type,
                            "file_uri": file_uri,
                        }
                    }
                )
            else:
                raise ValueError("Gemini file input requires file_data or file_uri.")
            continue
        raise ValueError(f"Unsupported Gemini content part type: {item_type!r}.")
    return parts


def build_gemini_generate_content_payload(
    *,
    messages: list[dict[str, Any]],
    model: str,
    reasoning_effort: str | None,
    temperature: float,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        parts = gemini_parts_from_content(message.get("content") or "")
        if role in {"system", "developer"}:
            system_parts.extend(parts)
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            }
        )

    generation_config: dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if response_format and response_format.get("type") == "json_object":
        generation_config["responseMimeType"] = "application/json"
    thinking_config = gemini_thinking_config(model, reasoning_effort)
    if thinking_config is not None:
        generation_config["thinkingConfig"] = thinking_config

    payload: dict[str, Any] = {
        "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
        "generationConfig": generation_config,
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    return payload


class GeminiGenerateContentClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        reasoning_effort: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = clean_api_key(api_key)
        self.model = model
        self.base_url = native_gemini_base_url(base_url)
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload = build_gemini_generate_content_payload(
            messages=messages,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        request = urllib.request.Request(
            f"{self.base_url}/models/{self.model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        try:
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(str(part.get("text") or "") for part in parts)
            return text.strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini API response: {data}") from exc


def make_llm_client(
    *,
    api_key: str,
    model: str,
    base_url: str,
    reasoning_effort: str | None,
    api_provider: str = "auto",
    input_mode: str = "text",
    timeout: float | None = None,
) -> Any:
    client_timeout = 120.0 if timeout is None else max(1.0, float(timeout))
    if should_use_gemini_native_client(
        api_provider=api_provider,
        base_url=base_url,
        input_mode=input_mode,
    ):
        return GeminiGenerateContentClient(
            api_key=clean_api_key(api_key),
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            timeout=client_timeout,
        )
    return ChatCompletionClient(
        api_key=clean_api_key(api_key),
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        timeout=client_timeout,
    )


def normalize_feature_name(value: str | None) -> str:
    if not value:
        return ""
    text = value.lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_relevance_score(title: str | None) -> int:
    normalized = normalize_feature_name(title)
    if not normalized:
        return 0
    score = 0
    for pattern, weight in TITLE_RANK_RULES:
        if re.search(pattern, normalized):
            score += weight
    for pattern, penalty in TITLE_PENALTY_RULES:
        if re.search(pattern, normalized):
            score -= penalty
    return max(0, score)


def rank_paper_records(
    records: list[dict[str, Any]],
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    with title_score_progress(total=len(records), show=show_progress) as progress:
        for idx, record in enumerate(records):
            paper = dict(record)
            paper["title_relevance_score"] = title_relevance_score(str(paper.get("title") or ""))
            ranked.append((-int(paper["title_relevance_score"]), idx, paper))
            progress.update(1)
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [paper for _, _, paper in ranked]


def canonical_category(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_feature_name(value)
    category_map = {
        "persuasion": "persuasion",
        "persuasion technique": "persuasion",
        "persuasion techniques": "persuasion",
        "propaganda": "persuasion",
        "propaganda technique": "persuasion",
        "propaganda techniques": "persuasion",
        "rhetorical": "persuasion",
        "rhetorical technique": "persuasion",
        "rhetorical techniques": "persuasion",
        "moral": "moral_framing",
        "morality": "moral_framing",
        "moral framing": "moral_framing",
        "moral frame": "moral_framing",
        "moral frames": "moral_framing",
        "mft": "moral_framing",
        "moral foundation": "moral_framing",
        "moral foundations": "moral_framing",
        "sentiment": "sentiment_affect",
        "affect": "sentiment_affect",
        "affective": "sentiment_affect",
        "emotional": "sentiment_affect",
        "emotion": "sentiment_affect",
        "emotions": "sentiment_affect",
        "emotion affect": "sentiment_affect",
        "emotional affective targeting": "sentiment_affect",
        "sentiment affect": "sentiment_affect",
        "sentiment and affect": "sentiment_affect",
    }
    return category_map.get(normalized)


def canonical_evidence_type(value: str | None, item: dict[str, Any] | None = None) -> str:
    if value:
        normalized = normalize_feature_name(value)
        evidence_map = {
            "definition": "definition",
            "defined": "definition",
            "taxonomy": "taxonomy_row",
            "taxonomy row": "taxonomy_row",
            "taxonomy label": "taxonomy_row",
            "label": "taxonomy_row",
            "annotation label": "taxonomy_row",
            "example": "example",
            "example usage": "example",
            "named mention": "named_mention",
            "mention": "named_mention",
            "name": "named_mention",
        }
        if normalized in evidence_map:
            return evidence_map[normalized]
    item = item or {}
    if item.get("definition"):
        return "definition"
    if item.get("examples"):
        return "example"
    if item.get("parent_category") or item.get("synonyms"):
        return "taxonomy_row"
    return "named_mention"


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, tuple):
        raw_values = list(value)
    elif isinstance(value, str):
        if not value.strip():
            return []
        raw_values = [value]
    else:
        raw_values = [value]
    output: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            output.append(text)
    return output


def truncate_text(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 15)].rstrip() + " [truncated]"


def trim_source_quote(value: Any, max_chars: int = 500) -> str:
    return truncate_text(str(value or ""), max_chars)


def parse_features_json(content: str) -> list[dict[str, Any]]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    payload: Any
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        object_start = stripped.find("{")
        object_end = stripped.rfind("}")
        array_start = stripped.find("[")
        array_end = stripped.rfind("]")
        candidates: list[str] = []
        if object_start != -1 and object_end > object_start:
            candidates.append(stripped[object_start : object_end + 1])
        if array_start != -1 and array_end > array_start:
            candidates.append(stripped[array_start : array_end + 1])
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        else:
            raise last_error or ValueError("No JSON object or array found in model response.")

    if isinstance(payload, dict):
        features = payload.get("features", [])
    elif isinstance(payload, list):
        features = payload
    else:
        raise ValueError("Expected a JSON object with features or a feature array.")

    if not isinstance(features, list):
        raise ValueError("Expected features to be a list.")
    return [item for item in features if isinstance(item, dict)]


def repair_features_json(raw_content: str, client: ChatCompletionClient) -> str:
    return client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You repair malformed JSON. Return only one valid JSON object "
                    "with a features array. Do not add Markdown fences or explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Repair this extraction response into valid JSON with exactly one "
                    "top-level key, features. Preserve all feature objects and fields "
                    "that can be recovered. Escape embedded quotes and newlines in "
                    "string values. If no feature objects can be recovered, return "
                    '{"features": []}.\n\n'
                    f"MALFORMED RESPONSE:\n{raw_content[:20000]}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=5000,
        response_format={"type": "json_object"},
    )


def parse_features_json_with_repair(
    content: str,
    client: ChatCompletionClient | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    try:
        return parse_features_json(content), False
    except (json.JSONDecodeError, ValueError) as exc:
        if client is None:
            raise
        LOGGER.warning("Model returned malformed JSON; asking model to repair it: %s", exc)
        repaired = repair_features_json(content, client=client)
        try:
            return parse_features_json(repaired), True
        except (json.JSONDecodeError, ValueError) as repair_exc:
            raise ValueError(
                f"Model JSON parse failed: {exc}; repair failed: {repair_exc}"
            ) from repair_exc


def parse_curation_json(content: str) -> list[dict[str, Any]]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    payload: Any
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        object_start = stripped.find("{")
        object_end = stripped.rfind("}")
        if object_start == -1 or object_end <= object_start:
            raise
        payload = json.loads(stripped[object_start : object_end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object with decisions.")
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("Expected decisions to be a list.")
    return [item for item in decisions if isinstance(item, dict)]


def repair_curation_json(raw_content: str, client: ChatCompletionClient) -> str:
    return client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You repair malformed JSON. Return only one valid JSON object "
                    "with a decisions array. Do not add Markdown fences or explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Repair this curation response into valid JSON with exactly one "
                    "top-level key, decisions. Preserve all decision objects and fields "
                    "that can be recovered. Escape embedded quotes and newlines in "
                    "string values. If no decisions can be recovered, return "
                    '{"decisions": []}.\n\n'
                    f"MALFORMED RESPONSE:\n{raw_content[:20000]}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=5000,
        response_format={"type": "json_object"},
    )


def parse_curation_json_with_repair(
    content: str,
    client: ChatCompletionClient | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    try:
        return parse_curation_json(content), False
    except (json.JSONDecodeError, ValueError) as exc:
        if client is None:
            raise
        LOGGER.warning("Model returned malformed curation JSON; asking model to repair it: %s", exc)
        repaired = repair_curation_json(content, client=client)
        try:
            return parse_curation_json(repaired), True
        except (json.JSONDecodeError, ValueError) as repair_exc:
            raise ValueError(
                f"Model curation JSON parse failed: {exc}; repair failed: {repair_exc}"
            ) from repair_exc


def is_taxonomy_relevant(item: dict[str, Any], chunk_text: str, paper: dict[str, Any]) -> bool:
    haystack = "\n".join(
        [
            str(chunk_text or ""),
            str(item.get("feature_name") or ""),
            str(item.get("parent_category") or ""),
            str(item.get("definition") or ""),
            " ".join(coerce_str_list(item.get("synonyms"))),
            " ".join(coerce_str_list(item.get("examples"))),
            str(item.get("source_quote") or item.get("quote") or ""),
            str(item.get("snippet") or ""),
            str(paper.get("title") or ""),
            str(paper.get("venue") or ""),
        ]
    )
    return bool(RELEVANCE_RE.search(haystack))


def normalize_feature_item(
    item: dict[str, Any],
    *,
    paper: dict[str, Any],
    chunk: dict[str, Any],
    run_id: str,
    model: str,
    extracted_at: str,
) -> dict[str, Any] | None:
    category = canonical_category(str(item.get("category") or ""))
    feature_name = str(
        item.get("feature_name")
        or item.get("name")
        or item.get("label")
        or ""
    ).strip()
    normalized_feature = normalize_feature_name(feature_name)
    if not category or not normalized_feature:
        return None
    if category not in CATEGORIES:
        return None
    if not is_taxonomy_relevant(item, chunk_text=str(chunk.get("text") or ""), paper=paper):
        return None

    confidence = item.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.5
    confidence_value = max(0.0, min(1.0, confidence_value))

    evidence_type = canonical_evidence_type(
        str(item.get("evidence_type") or ""), item=item
    )
    source_quote = trim_source_quote(item.get("source_quote") or item.get("quote") or "")
    snippet = trim_source_quote(item.get("snippet") or source_quote or chunk.get("text") or "", 700)

    return {
        "paper_id": str(paper.get("paper_id") or chunk.get("paper_id") or ""),
        "title": str(paper.get("title") or chunk.get("title") or ""),
        "authors": coerce_str_list(paper.get("authors")),
        "year": paper.get("year"),
        "doi": paper.get("doi"),
        "venue": paper.get("venue"),
        "category": category,
        "feature_name": feature_name,
        "normalized_feature_name": normalized_feature,
        "parent_category": str(item.get("parent_category") or "").strip(),
        "definition": str(item.get("definition") or "").strip(),
        "synonyms": coerce_str_list(item.get("synonyms")),
        "examples": coerce_str_list(item.get("examples")),
        "evidence_type": evidence_type,
        "chunk_id": str(chunk.get("chunk_id") or item.get("chunk_id") or ""),
        "pages": coerce_pages(chunk.get("pages") or item.get("pages")),
        "snippet": snippet,
        "source_quote": source_quote,
        "confidence": round(confidence_value, 4),
        "notes": str(item.get("notes") or "").strip(),
        "model": model,
        "run_id": run_id,
        "extracted_at": extracted_at,
    }


def coerce_pages(value: Any) -> list[int]:
    pages: list[int] = []
    values = value if isinstance(value, list) else [value]
    for raw in values:
        if raw is None or raw == "":
            continue
        try:
            page = int(raw)
        except (TypeError, ValueError):
            continue
        if page not in pages:
            pages.append(page)
    return pages


def json_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp_path.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_csv_value(row.get(field)) for field in fieldnames})
    tmp_path.replace(path)


def load_papers(
    run_dir: Path,
    paper_limit: int | None = None,
    rank_by_title: bool = True,
    show_progress: bool = False,
    include_pdf_failures: bool = False,
) -> dict[str, dict[str, Any]]:
    paper_index_path = run_dir / "paper_index.json"
    if not paper_index_path.exists():
        raise FileNotFoundError(f"paper_index.json does not exist: {paper_index_path}")
    payload = read_json(paper_index_path)
    records: list[dict[str, Any]] = []
    for paper in payload.get("papers", []):
        if paper.get("conversion", {}).get("status") not in {
            "converted",
            "already_exists",
            "no_text",
        }:
            if not (include_pdf_failures and paper.get("pdf_path")):
                continue
        paper_id = str(paper.get("paper_id") or "")
        if not paper_id:
            continue
        records.append(dict(paper))
    if rank_by_title:
        records = rank_paper_records(records, show_progress=show_progress)
    if paper_limit is not None:
        records = records[:paper_limit]
    return {str(paper["paper_id"]): paper for paper in records}


def iter_chunks_by_paper(
    run_dir: Path,
    papers: dict[str, dict[str, Any]],
) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    db_path = run_dir / "index.sqlite"
    if db_path.exists():
        yield from iter_chunks_by_paper_sqlite(db_path, papers)
        return
    yield from iter_chunks_by_paper_jsonl(run_dir, papers)


def iter_chunks_by_paper_sqlite(
    db_path: Path,
    papers: dict[str, dict[str, Any]],
) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    with sqlite3.connect(db_path) as conn:
        for paper_id in papers:
            rows = conn.execute(
                """
                SELECT chunk_id, paper_id, title, year, pages_json, text, token_count
                FROM chunks
                WHERE paper_id = ?
                ORDER BY rowid
                """,
                (paper_id,),
            ).fetchall()
            chunks: list[dict[str, Any]] = []
            for chunk_id, row_paper_id, title, year, pages_json, text, token_count in rows:
                try:
                    pages = json.loads(pages_json)
                except json.JSONDecodeError:
                    pages = []
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "paper_id": row_paper_id,
                        "title": title,
                        "year": year,
                        "pages": pages,
                        "text": text,
                        "token_count": token_count,
                    }
                )
            yield paper_id, chunks


def iter_chunks_by_paper_jsonl(
    run_dir: Path,
    papers: dict[str, dict[str, Any]],
) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    chunks_path = run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl does not exist: {chunks_path}")

    chunks_by_paper = {paper_id: [] for paper_id in papers}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            paper_id = str(chunk.get("paper_id") or "")
            if paper_id in chunks_by_paper:
                chunks_by_paper[paper_id].append(chunk)
    for paper_id in papers:
        yield paper_id, chunks_by_paper.get(paper_id, [])


def count_chunks_for_papers(run_dir: Path, paper_ids: set[str]) -> int:
    if not paper_ids:
        return 0
    db_path = run_dir / "index.sqlite"
    if db_path.exists():
        total = 0
        paper_id_list = sorted(paper_ids)
        with sqlite3.connect(db_path) as conn:
            for idx in range(0, len(paper_id_list), 900):
                batch = paper_id_list[idx : idx + 900]
                placeholders = ",".join("?" for _ in batch)
                total += int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM chunks WHERE paper_id IN ({placeholders})",
                        batch,
                    ).fetchone()[0]
                )
        return total

    chunks_path = run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl does not exist: {chunks_path}")
    total = 0
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(chunk.get("paper_id") or "") in paper_ids:
                total += 1
    return total


def batch_chunks(chunks: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    size = max(1, batch_size)
    for idx in range(0, len(chunks), size):
        yield chunks[idx : idx + size]


def paper_prompt_metadata(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "authors": paper.get("authors") or [],
        "year": paper.get("year"),
        "doi": paper.get("doi"),
        "venue": paper.get("venue"),
    }


def chunk_prompt_block(chunk: dict[str, Any], max_chunk_chars: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "pages": chunk.get("pages") or [],
        "text": truncate_text(str(chunk.get("text") or ""), max_chunk_chars),
    }


def load_goal_text(goal_path: Path | None, max_chars: int = 12000) -> tuple[str, str | None]:
    if goal_path is None:
        default_goal = Path("goal.md")
        goal_path = default_goal if default_goal.exists() else None
    if goal_path is None:
        return "", None
    goal_path = goal_path.expanduser()
    if not goal_path.exists():
        raise FileNotFoundError(f"Goal file does not exist: {goal_path}")
    return truncate_text(goal_path.read_text(encoding="utf-8", errors="replace"), max_chars), str(goal_path)


def build_extraction_prompt(
    paper: dict[str, Any],
    chunks: list[dict[str, Any]],
    max_chunk_chars: int,
    goal_text: str = "",
    max_features_per_chunk: int = 12,
) -> str:
    prompt_payload = {
        "paper": paper_prompt_metadata(paper),
        "chunks": [chunk_prompt_block(chunk, max_chunk_chars) for chunk in chunks],
    }
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    return f"""Extract taxonomy feature mentions from the provided literature chunks.
{goal_section}

Return strict JSON only with this shape:
{{
  "features": [
    {{
      "category": "persuasion | moral_framing | sentiment_affect",
      "feature_name": "named feature exactly as supported by the chunk",
      "parent_category": "optional parent/group label or empty string",
      "definition": "definition from the chunk, or empty string",
      "synonyms": ["optional related terms"],
      "examples": ["optional example phrases from the chunk"],
      "evidence_type": "definition | taxonomy_row | example | named_mention",
      "chunk_id": "chunk id from input",
      "source_quote": "short quote/paraphrase from the chunk; escape internal quotes",
      "confidence": 0.0,
      "notes": "brief ambiguity note if needed"
    }}
  ]
}}

Extraction rules:
- Extract only features relevant to persuasion techniques, propaganda/rhetoric,
  moral framing, sentiment, emotion, affect, media framing, or media-analysis
  attributes.
- Use the goal context to preserve the distinction between persuasion techniques,
  moral framing, and sentiment/affect.
- Named mentions are valid, but use evidence_type="named_mention" when there is
  no definition, taxonomy row, or example.
- Do not extract generic ML/CV/signal-processing "features" unless the chunk
  clearly discusses persuasion, moral framing, sentiment, emotion, rhetoric,
  propaganda, framing, or media-analysis attributes.
- Do not infer feature names that are absent from the chunks.
- Use only the provided chunks and include the source chunk_id for every item.
- Prefer source_quote paraphrases without internal quotation marks when possible.
- Return at most {max_features_per_chunk} features per chunk. If a chunk contains
  more candidates, prioritize explicitly named taxonomy labels, definitions,
  annotation labels, and examples aligned with the goal context.
- If there are no supported features, return {{"features": []}}.

INPUT
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}
"""


def pdf_chunk_id(paper: dict[str, Any]) -> str:
    return f"{paper.get('paper_id') or 'paper'}:pdf"


def pdf_prompt_chunk(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": pdf_chunk_id(paper),
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "pages": [],
        "text": "",
    }


def resolve_pdf_path(paper: dict[str, Any], run_dir: Path) -> Path:
    raw_path = str(paper.get("pdf_path") or "").strip()
    if not raw_path:
        raise FileNotFoundError(f"Paper {paper.get('paper_id') or ''} has no pdf_path.")

    path = Path(raw_path).expanduser()
    if path.is_absolute() or path.exists():
        return path

    run_relative = run_dir / path
    if run_relative.exists():
        return run_relative
    return path


def pdf_file_data_url(pdf_path: Path, max_bytes: int = PDF_INPUT_MAX_BYTES) -> str:
    size = pdf_path.stat().st_size
    if size > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        size_mb = size / (1024 * 1024)
        raise ValueError(f"PDF is {size_mb:.1f} MB, above the {max_mb:.0f} MB direct-input limit.")
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def build_pdf_extraction_prompt(
    paper: dict[str, Any],
    goal_text: str = "",
    max_features_per_pdf: int = 40,
) -> str:
    chunk_id = pdf_chunk_id(paper)
    prompt_payload = {
        "paper": paper_prompt_metadata(paper),
        "pdf_input": {
            "chunk_id": chunk_id,
            "pages": "Use 1-based PDF page numbers when evidence is page-local.",
        },
    }
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    return f"""Extract taxonomy feature mentions from the attached academic PDF.
{goal_section}

Return strict JSON only with this shape:
{{
  "features": [
    {{
      "category": "persuasion | moral_framing | sentiment_affect",
      "feature_name": "named feature exactly as supported by the PDF",
      "parent_category": "optional parent/group label or empty string",
      "definition": "definition from the PDF, or empty string",
      "synonyms": ["optional related terms"],
      "examples": ["optional example phrases from the PDF"],
      "evidence_type": "definition | taxonomy_row | example | named_mention",
      "chunk_id": "{chunk_id}",
      "pages": [1],
      "source_quote": "short quote/paraphrase from the PDF; escape internal quotes",
      "snippet": "short local evidence snippet if helpful",
      "confidence": 0.0,
      "notes": "brief ambiguity note if needed"
    }}
  ]
}}

Extraction rules:
- Extract only features relevant to persuasion techniques, propaganda/rhetoric,
  moral framing, sentiment, emotion, affect, media framing, or media-analysis
  attributes.
- Use the goal context to preserve the distinction between persuasion techniques,
  moral framing, and sentiment/affect.
- Named mentions are valid, but use evidence_type="named_mention" when there is
  no definition, taxonomy row, or example.
- Do not extract generic ML/CV/signal-processing "features" unless the PDF
  clearly discusses persuasion, moral framing, sentiment, emotion, rhetoric,
  propaganda, framing, or media-analysis attributes.
- Do not infer feature names that are absent from the PDF.
- Use only the attached PDF and set chunk_id="{chunk_id}" for every item.
- Include 1-based page numbers when the PDF provides enough evidence to locate
  the mention.
- Prefer source_quote paraphrases without internal quotation marks when possible.
- Return at most {max_features_per_pdf} features from this PDF. If the PDF
  contains more candidates, prioritize explicitly named taxonomy labels,
  definitions, annotation labels, and examples aligned with the goal context.
- If there are no supported features, return {{"features": []}}.

INPUT
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}
"""


def build_pdf_extraction_messages(
    *,
    paper: dict[str, Any],
    pdf_path: Path,
    goal_text: str = "",
    max_features_per_pdf: int = 40,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an evidence-bound taxonomy extraction agent. Return only "
                "valid JSON. Extract named feature mentions from academic PDFs."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": pdf_path.name,
                        "file_data": pdf_file_data_url(pdf_path),
                    },
                },
                {
                    "type": "text",
                    "text": build_pdf_extraction_prompt(
                        paper,
                        goal_text=goal_text,
                        max_features_per_pdf=max_features_per_pdf,
                    ),
                },
            ],
        },
    ]


def extract_batch_features(
    *,
    client: ChatCompletionClient,
    paper: dict[str, Any],
    chunks: list[dict[str, Any]],
    model: str,
    run_id: str,
    max_chunk_chars: int,
    goal_text: str = "",
    max_features_per_chunk: int = 12,
    temperature: float,
    extracted_at: str,
) -> BatchResult:
    prompt = build_extraction_prompt(
        paper,
        chunks,
        max_chunk_chars=max_chunk_chars,
        goal_text=goal_text,
        max_features_per_chunk=max_features_per_chunk,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-bound taxonomy extraction agent. Return only "
                "valid JSON. Extract named feature mentions from academic-paper chunks."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    raw_response = client.chat(
        messages,
        temperature=temperature,
        max_tokens=3500,
        response_format={"type": "json_object"},
    )
    parsed_items, _ = parse_features_json_with_repair(raw_response, client=client)
    chunks_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
    fallback_chunk = chunks[0] if chunks else {}
    normalized: list[dict[str, Any]] = []
    for item in parsed_items:
        chunk_id = str(item.get("chunk_id") or "")
        chunk = chunks_by_id.get(chunk_id) or fallback_chunk
        row = normalize_feature_item(
            item,
            paper=paper,
            chunk=chunk,
            run_id=run_id,
            model=model,
            extracted_at=extracted_at,
        )
        if row is not None:
            normalized.append(row)
    return BatchResult(
        paper_id=str(paper.get("paper_id") or ""),
        batch_index=0,
        chunk_ids=[str(chunk.get("chunk_id") or "") for chunk in chunks],
        features=normalized,
        raw_response=raw_response,
    )


def choose_feature_name(rows: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        name = str(row.get("feature_name") or "").strip()
        if name:
            counts[name] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def unique_values(rows: list[dict[str, Any]], field: str) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None or item == "":
                continue
            key = json.dumps(item, sort_keys=True) if isinstance(item, (list, dict)) else str(item)
            if key not in seen:
                seen.add(key)
                output.append(item)
    return output


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_rank = {
        "definition": 4,
        "taxonomy_row": 3,
        "example": 2,
        "named_mention": 1,
    }
    return max(
        rows,
        key=lambda row: (
            evidence_rank.get(str(row.get("evidence_type")), 0),
            float(row.get("confidence") or 0.0),
            len(str(row.get("definition") or "")),
            len(str(row.get("source_quote") or "")),
        ),
    )


def aggregate_paper_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("paper_id") or ""),
            str(row.get("category") or ""),
            str(row.get("normalized_feature_name") or ""),
        )
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        chosen = best_row(group)
        output.append(
            {
                "paper_id": chosen.get("paper_id"),
                "title": chosen.get("title"),
                "authors": chosen.get("authors") or [],
                "year": chosen.get("year"),
                "doi": chosen.get("doi"),
                "venue": chosen.get("venue"),
                "category": chosen.get("category"),
                "feature_name": choose_feature_name(group),
                "normalized_feature_name": chosen.get("normalized_feature_name"),
                "parent_category": chosen.get("parent_category"),
                "definition": chosen.get("definition"),
                "synonyms": unique_values(group, "synonyms"),
                "examples": unique_values(group, "examples"),
                "evidence_types": unique_values(group, "evidence_type"),
                "mention_count": len(group),
                "chunk_ids": unique_values(group, "chunk_id"),
                "pages": sorted({page for row in group for page in coerce_pages(row.get("pages"))}),
                "best_snippet": chosen.get("snippet"),
                "best_source_quote": chosen.get("source_quote"),
                "confidence": round(
                    sum(float(row.get("confidence") or 0.0) for row in group) / len(group),
                    4,
                ),
                "notes": "; ".join(unique_values(group, "notes")),
                "model": chosen.get("model"),
                "run_id": chosen.get("run_id"),
                "extracted_at": chosen.get("extracted_at"),
            }
        )
    return output


def aggregate_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("category") or ""),
            str(row.get("normalized_feature_name") or ""),
        )
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        chosen = best_row(group)
        paper_ids = unique_values(group, "paper_id")
        output.append(
            {
                "category": chosen.get("category"),
                "feature_name": choose_feature_name(group),
                "normalized_feature_name": chosen.get("normalized_feature_name"),
                "parent_categories": unique_values(group, "parent_category"),
                "definitions": unique_values(group, "definition"),
                "synonyms": unique_values(group, "synonyms"),
                "examples": unique_values(group, "examples"),
                "paper_count": len(paper_ids),
                "mention_count": len(group),
                "source_paper_ids": paper_ids,
                "source_titles": unique_values(group, "title"),
                "evidence_types": unique_values(group, "evidence_type"),
                "confidence": round(
                    sum(float(row.get("confidence") or 0.0) for row in group) / len(group),
                    4,
                ),
                "notes": "; ".join(unique_values(group, "notes")),
                "model": chosen.get("model"),
                "run_id": chosen.get("run_id"),
                "extracted_at": chosen.get("extracted_at"),
            }
        )
    return output


def completed_paper_ids(output_dir: Path) -> set[str]:
    path = output_dir / "completed_papers.jsonl"
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper_id = str(payload.get("paper_id") or "")
            if paper_id:
                completed.add(paper_id)
    return completed


def load_existing_mentions(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "feature_mentions.jsonl"
    try:
        return read_jsonl(path)
    except json.JSONDecodeError:
        LOGGER.warning("Skipping malformed existing mention rows in %s", path)
        return []


def remove_existing_outputs(output_dir: Path) -> None:
    for name in [
        "feature_mentions.jsonl",
        "feature_mentions.csv",
        "paper_features.jsonl",
        "paper_features.csv",
        "features.jsonl",
        "features.csv",
        "completed_papers.jsonl",
        "extraction_errors.jsonl",
        "extraction_trace.json",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_artifacts(output_dir: Path, mentions: list[dict[str, Any]]) -> dict[str, int]:
    mentions = dedupe_mentions(mentions)
    paper_features = aggregate_paper_features(mentions)
    features = aggregate_features(mentions)

    write_jsonl(output_dir / "feature_mentions.jsonl", mentions)
    write_csv(output_dir / "feature_mentions.csv", mentions, CSV_FIELDNAMES)
    write_jsonl(output_dir / "paper_features.jsonl", paper_features)
    write_csv(output_dir / "paper_features.csv", paper_features, PAPER_FEATURE_FIELDNAMES)
    write_jsonl(output_dir / "features.jsonl", features)
    write_csv(output_dir / "features.csv", features, FEATURE_FIELDNAMES)
    return {
        "feature_mentions": len(mentions),
        "paper_features": len(paper_features),
        "features": len(features),
    }


def dedupe_mentions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("paper_id") or ""),
            str(row.get("category") or ""),
            str(row.get("normalized_feature_name") or ""),
            str(row.get("chunk_id") or ""),
            str(row.get("evidence_type") or ""),
            str(row.get("source_quote") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def feature_key(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "").strip()
    normalized = normalize_feature_name(
        str(row.get("normalized_feature_name") or row.get("feature_name") or "")
    )
    return f"{category}::{normalized}"


def ensure_feature_identity(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    normalized = normalize_feature_name(
        str(output.get("normalized_feature_name") or output.get("feature_name") or "")
    )
    output["normalized_feature_name"] = normalized
    output["feature_key"] = f"{output.get('category') or ''}::{normalized}"
    return output


def load_feature_rows(input_dir: Path) -> list[dict[str, Any]]:
    features_path = input_dir / "features.jsonl"
    if not features_path.exists():
        raise FileNotFoundError(f"features.jsonl does not exist: {features_path}")
    return [ensure_feature_identity(row) for row in read_jsonl(features_path)]


def load_curated_feature_rows(input_dir: Path) -> list[dict[str, Any]]:
    features_path = input_dir / "curated_features.jsonl"
    if not features_path.exists():
        raise FileNotFoundError(f"curated_features.jsonl does not exist: {features_path}")
    return [ensure_feature_identity(row) for row in read_jsonl(features_path)]


def curation_batch_id(category: str, batch_index: int, rows: list[dict[str, Any]]) -> str:
    digest_source = "\n".join(str(row.get("feature_key") or feature_key(row)) for row in rows)
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"{category}:{batch_index}:{digest}"


def prepare_curation_batches(
    features: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for feature in features:
        by_category.setdefault(str(feature.get("category") or "uncategorized"), []).append(feature)

    ordered_categories = [
        category for category in ["persuasion", "moral_framing", "sentiment_affect"]
        if category in by_category
    ]
    ordered_categories.extend(
        category
        for category in sorted(by_category)
        if category not in set(ordered_categories)
    )

    batches: list[dict[str, Any]] = []
    size = max(1, batch_size)
    for category in ordered_categories:
        category_features = sorted(
            by_category[category],
            key=lambda row: (
                str(row.get("normalized_feature_name") or ""),
                str(row.get("feature_name") or ""),
            ),
        )
        for batch_index, start in enumerate(range(0, len(category_features), size), start=1):
            rows = category_features[start : start + size]
            batches.append(
                {
                    "batch_id": curation_batch_id(category, batch_index, rows),
                    "category": category,
                    "batch_index": batch_index,
                    "features": rows,
                }
            )
    return batches


def compact_list(value: Any, limit: int, max_chars: int = 180) -> list[str]:
    return [truncate_text(item, max_chars) for item in coerce_str_list(value)[:limit]]


def compact_feature_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_key": row.get("feature_key") or feature_key(row),
        "category": row.get("category"),
        "feature_name": row.get("feature_name"),
        "normalized_feature_name": row.get("normalized_feature_name"),
        "parent_categories": compact_list(row.get("parent_categories"), 8),
        "definitions": compact_list(row.get("definitions"), 5, max_chars=240),
        "synonyms": compact_list(row.get("synonyms"), 12),
        "examples": compact_list(row.get("examples"), 8),
        "paper_count": row.get("paper_count"),
        "mention_count": row.get("mention_count"),
        "source_titles": compact_list(row.get("source_titles"), 8, max_chars=160),
        "evidence_types": compact_list(row.get("evidence_types"), 8),
        "notes": truncate_text(str(row.get("notes") or ""), 500),
    }


def build_curation_prompt(
    *,
    features: list[dict[str, Any]],
    category: str,
    goal_text: str,
    merge_style: str,
) -> str:
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    payload = {
        "category": category,
        "merge_style": merge_style,
        "features": [compact_feature_for_prompt(row) for row in features],
    }
    return f"""Curate extracted taxonomy feature candidates.
{goal_section}

Return strict JSON only with this shape:
{{
  "decisions": [
    {{
      "feature_key": "category::normalized feature name from input",
      "action": "keep_as_canonical | merge_into_canonical | reject",
      "canonical_feature_name": "canonical display name, empty for reject",
      "canonical_normalized_feature_name": "normalized canonical name, empty for reject",
      "rejection_reason": "reason if rejected, otherwise empty",
      "curation_confidence": 0.0,
      "notes": "short explanation"
    }}
  ]
}}

Curation rules:
- Work only within the input category: {category}.
- Merge style is {merge_style}. For moderate merging, combine clear aliases and
  close variants when names, definitions, parent categories, and evidence align.
- Do not merge concepts across distinct taxonomy roles.
- Reject features that are off-goal, overly broad, non-taxonomic, or artifacts of
  extraction noise.
- Keep useful canonical features even when their evidence is imperfect.
- Every input feature_key must appear exactly once in decisions.

INPUT
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def build_final_curation_prompt(
    *,
    features: list[dict[str, Any]],
    category: str,
    goal_text: str,
    merge_style: str,
    target_per_category: int,
) -> str:
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    payload = {
        "category": category,
        "merge_style": merge_style,
        "target_unique_features": target_per_category,
        "features_ranked_by_paper_count": [
            {
                "rank": idx,
                **compact_feature_for_prompt(row),
            }
            for idx, row in enumerate(features, start=1)
        ],
    }
    return f"""Final-curate ranked taxonomy features for a publication-ready export.
{goal_section}

Return strict JSON only with this shape:
{{
  "decisions": [
    {{
      "feature_key": "category::normalized feature name from input",
      "action": "keep_as_canonical | merge_into_canonical | reject",
      "canonical_feature_name": "canonical display name, empty for reject",
      "canonical_normalized_feature_name": "normalized canonical name, empty for reject",
      "rejection_reason": "reason if rejected, otherwise empty",
      "curation_confidence": 0.0,
      "notes": "short explanation"
    }}
  ]
}}

Final curation rules:
- Work only within the input category: {category}.
- Input features are already sorted by descending paper_count, then mention_count.
- Use merge style {merge_style}. Merge aliases, spelling variants, plural/singular
  variants, and near-duplicate labels that clearly describe the same taxonomy role.
- Reject off-goal, overly broad, non-taxonomic, ambiguous, or artifact-like labels.
- Prefer high paper_count features when choosing canonical names, unless a lower
  ranked label is clearer and supported by the merged evidence.
- Aim for exactly {target_per_category} unique canonical features when there are
  at least {target_per_category} valid concepts in the input; never create more
  than {target_per_category} canonical features.
- If more than {target_per_category} valid concepts remain, merge or reject the
  lower-priority ones first.
- Every input feature_key must appear exactly once in decisions.

INPUT
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def canonicalize_action(value: Any) -> str:
    normalized = normalize_feature_name(str(value or ""))
    if normalized in {"keep", "keep as canonical", "canonical", "keep canonical"}:
        return "keep_as_canonical"
    if normalized in {"merge", "merge into canonical", "synonym", "alias"}:
        return "merge_into_canonical"
    if normalized in {"reject", "drop", "remove", "irrelevant"}:
        return "reject"
    return str(value or "").strip()


def normalize_curation_decision(
    raw: dict[str, Any],
    feature: dict[str, Any],
    *,
    model: str,
    run_id: str,
    curated_at: str,
) -> dict[str, Any]:
    action = canonicalize_action(raw.get("action"))
    if action not in {"keep_as_canonical", "merge_into_canonical", "reject"}:
        action = "keep_as_canonical"

    feature_key_value = str(raw.get("feature_key") or feature.get("feature_key") or feature_key(feature))
    category = str(feature.get("category") or "")
    original_name = str(feature.get("feature_name") or "")
    original_normalized = str(feature.get("normalized_feature_name") or normalize_feature_name(original_name))

    canonical_name = str(raw.get("canonical_feature_name") or "").strip()
    canonical_normalized = normalize_feature_name(
        str(raw.get("canonical_normalized_feature_name") or canonical_name)
    )
    if action == "reject":
        canonical_name = ""
        canonical_normalized = ""
    elif not canonical_normalized:
        canonical_name = original_name
        canonical_normalized = original_normalized
    elif not canonical_name:
        canonical_name = canonical_normalized.title()

    try:
        confidence = float(raw.get("curation_confidence"))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    canonical_feature_key = f"{category}::{canonical_normalized}" if canonical_normalized else ""
    return {
        "feature_key": feature_key_value,
        "category": category,
        "original_feature_name": original_name,
        "original_normalized_feature_name": original_normalized,
        "action": action,
        "canonical_feature_key": canonical_feature_key,
        "canonical_feature_name": canonical_name,
        "canonical_normalized_feature_name": canonical_normalized,
        "rejection_reason": str(raw.get("rejection_reason") or "").strip(),
        "curation_confidence": round(confidence, 4),
        "curation_notes": str(raw.get("notes") or "").strip(),
        "paper_count": int(feature.get("paper_count") or 0),
        "mention_count": int(feature.get("mention_count") or 0),
        "model": model,
        "run_id": run_id,
        "curated_at": curated_at,
    }


def fallback_keep_decision(
    feature: dict[str, Any],
    *,
    model: str,
    run_id: str,
    curated_at: str,
) -> dict[str, Any]:
    return normalize_curation_decision(
        {
            "feature_key": feature.get("feature_key") or feature_key(feature),
            "action": "keep_as_canonical",
            "canonical_feature_name": feature.get("feature_name"),
            "canonical_normalized_feature_name": feature.get("normalized_feature_name"),
            "curation_confidence": 0.2,
            "notes": "No model decision returned; kept by fallback.",
        },
        feature,
        model=model,
        run_id=run_id,
        curated_at=curated_at,
    )


def normalize_curation_decisions(
    raw_decisions: list[dict[str, Any]],
    features: list[dict[str, Any]],
    *,
    model: str,
    run_id: str,
    curated_at: str,
) -> list[dict[str, Any]]:
    features_by_key = {str(feature.get("feature_key") or feature_key(feature)): feature for feature in features}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_decisions:
        key = str(raw.get("feature_key") or "")
        feature = features_by_key.get(key)
        if feature is None:
            continue
        output.append(
            normalize_curation_decision(
                raw,
                feature,
                model=model,
                run_id=run_id,
                curated_at=curated_at,
            )
        )
        seen.add(key)

    for key, feature in features_by_key.items():
        if key not in seen:
            output.append(
                fallback_keep_decision(
                    feature,
                    model=model,
                    run_id=run_id,
                    curated_at=curated_at,
                )
            )
    return output


def latest_decisions_by_feature(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        key = str(decision.get("feature_key") or "")
        if key:
            latest[key] = decision
    return latest


def unique_items(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            if item is None or item == "":
                continue
            key = json.dumps(item, sort_keys=True) if isinstance(item, (list, dict)) else str(item)
            if key not in seen:
                seen.add(key)
                output.append(item)
    return output


def avg_float(rows: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(field)))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def max_float(rows: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(field)))
        except (TypeError, ValueError):
            continue
    return round(max(values), 4) if values else 0.0


def build_curation_outputs(
    features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    features_by_key = {str(feature.get("feature_key") or feature_key(feature)): feature for feature in features}
    decision_map = latest_decisions_by_feature(decisions)
    members: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}

    for key, feature in features_by_key.items():
        decision = decision_map.get(key)
        if decision is None:
            continue
        member = {
            field: decision.get(field)
            for field in CURATED_MEMBER_FIELDNAMES
            if field in decision
        }
        members.append(member)
        if decision.get("action") == "reject":
            rejected_row = {
                **feature,
                "rejection_reason": decision.get("rejection_reason"),
                "curation_confidence": decision.get("curation_confidence"),
                "curation_notes": decision.get("curation_notes"),
                "model": decision.get("model"),
                "run_id": decision.get("run_id"),
                "curated_at": decision.get("curated_at"),
            }
            rejected.append(rejected_row)
            continue
        canonical_key = str(decision.get("canonical_feature_key") or key)
        groups.setdefault(canonical_key, []).append((feature, decision))

    curated: list[dict[str, Any]] = []
    for canonical_key in sorted(groups):
        group = groups[canonical_key]
        feature_rows = [feature for feature, _ in group]
        decision_rows = [decision for _, decision in group]
        representative = max(
            decision_rows,
            key=lambda row: (
                float(row.get("curation_confidence") or 0.0),
                len(str(row.get("canonical_feature_name") or "")),
            ),
        )
        paper_ids = unique_items(row.get("source_paper_ids") for row in feature_rows)
        source_feature_names = unique_items(row.get("feature_name") for row in feature_rows)
        curated.append(
            {
                "category": representative.get("category"),
                "feature_name": representative.get("canonical_feature_name"),
                "normalized_feature_name": representative.get("canonical_normalized_feature_name"),
                "source_feature_names": source_feature_names,
                "source_normalized_feature_names": unique_items(
                    row.get("normalized_feature_name") for row in feature_rows
                ),
                "parent_categories": unique_items(row.get("parent_categories") for row in feature_rows),
                "definitions": unique_items(row.get("definitions") for row in feature_rows),
                "synonyms": unique_items(
                    [
                        *[row.get("synonyms") for row in feature_rows],
                        source_feature_names,
                    ]
                ),
                "examples": unique_items(row.get("examples") for row in feature_rows),
                "paper_count": len(paper_ids),
                "mention_count": sum(int(row.get("mention_count") or 0) for row in feature_rows),
                "source_paper_ids": paper_ids,
                "source_titles": unique_items(row.get("source_titles") for row in feature_rows),
                "evidence_types": unique_items(row.get("evidence_types") for row in feature_rows),
                "source_confidence_avg": avg_float(feature_rows, "confidence"),
                "source_confidence_max": max_float(feature_rows, "confidence"),
                "curation_confidence": avg_float(decision_rows, "curation_confidence"),
                "member_count": len(feature_rows),
                "notes": "; ".join(str(item) for item in unique_items(row.get("notes") for row in feature_rows)),
                "curation_notes": "; ".join(
                    str(item) for item in unique_items(row.get("curation_notes") for row in decision_rows)
                ),
                "model": representative.get("model"),
                "run_id": representative.get("run_id"),
                "curated_at": representative.get("curated_at"),
            }
        )
    return curated, members, rejected


def curated_feature_sort_key(row: dict[str, Any]) -> tuple[int, int, float, int, str]:
    try:
        paper_count = int(row.get("paper_count") or 0)
    except (TypeError, ValueError):
        paper_count = 0
    try:
        mention_count = int(row.get("mention_count") or 0)
    except (TypeError, ValueError):
        mention_count = 0
    try:
        confidence = float(row.get("curation_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        member_count = int(row.get("member_count") or 0)
    except (TypeError, ValueError):
        member_count = 0
    return (
        -paper_count,
        -mention_count,
        -confidence,
        -member_count,
        str(row.get("feature_name") or "").casefold(),
    )


def select_top_curated_features_by_category(
    curated: list[dict[str, Any]],
    per_category: int = FINAL_FEATURES_PER_CATEGORY,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in curated:
        grouped.setdefault(str(row.get("category") or ""), []).append(row)

    output: list[dict[str, Any]] = []
    category_order = [
        *(category for category in CATEGORY_EXPORT_ORDER if category in grouped),
        *sorted(category for category in grouped if category not in CATEGORY_EXPORT_ORDER),
    ]
    for category in category_order:
        rows = grouped.get(category, [])
        output.extend(sorted(rows, key=curated_feature_sort_key)[:per_category])
    return output


def final_candidate_categories(
    curated: list[dict[str, Any]],
    candidate_pool_per_category: int = 0,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in curated:
        grouped.setdefault(str(row.get("category") or ""), []).append(row)

    category_order = [
        *(category for category in CATEGORY_EXPORT_ORDER if category in grouped),
        *sorted(category for category in grouped if category not in CATEGORY_EXPORT_ORDER),
    ]
    categories: list[dict[str, Any]] = []
    for category in category_order:
        rows = sorted(grouped.get(category, []), key=curated_feature_sort_key)
        if candidate_pool_per_category > 0:
            rows = rows[:candidate_pool_per_category]
        categories.append({"category": category, "features": rows})
    return categories


def final_curation_category_id(category: str, rows: list[dict[str, Any]]) -> str:
    digest_source = "\n".join(str(row.get("feature_key") or feature_key(row)) for row in rows)
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"{category}:final:{digest}"


def write_final_curation_artifacts(
    output_dir: Path,
    candidate_features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    target_per_category: int = FINAL_FEATURES_PER_CATEGORY,
) -> dict[str, int]:
    curated, members, rejected = build_curation_outputs(candidate_features, decisions)
    final_curated = select_top_curated_features_by_category(
        curated,
        per_category=target_per_category,
    )
    write_jsonl(output_dir / "final_curated_features_llm_top100_by_category.jsonl", final_curated)
    write_csv(
        output_dir / "final_curated_features_llm_top100_by_category.csv",
        final_curated,
        CURATED_FEATURE_FIELDNAMES,
    )
    write_jsonl(output_dir / "final_curated_feature_members_llm.jsonl", members)
    write_csv(output_dir / "final_curated_feature_members_llm.csv", members, CURATED_MEMBER_FIELDNAMES)
    write_jsonl(output_dir / "final_rejected_features_llm.jsonl", rejected)
    write_csv(output_dir / "final_rejected_features_llm.csv", rejected, REJECTED_FEATURE_FIELDNAMES)
    return {
        "final_curated_features_llm_top100_by_category": len(final_curated),
        "final_curated_feature_members_llm": len(members),
        "final_rejected_features_llm": len(rejected),
    }


def write_curation_artifacts(
    output_dir: Path,
    features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, int]:
    curated, members, rejected = build_curation_outputs(features, decisions)
    final_curated = select_top_curated_features_by_category(curated)
    write_jsonl(output_dir / "curated_features.jsonl", curated)
    write_csv(output_dir / "curated_features.csv", curated, CURATED_FEATURE_FIELDNAMES)
    write_jsonl(output_dir / "final_curated_features_top100_by_category.jsonl", final_curated)
    write_csv(
        output_dir / "final_curated_features_top100_by_category.csv",
        final_curated,
        CURATED_FEATURE_FIELDNAMES,
    )
    write_jsonl(output_dir / "curated_feature_members.jsonl", members)
    write_csv(output_dir / "curated_feature_members.csv", members, CURATED_MEMBER_FIELDNAMES)
    write_jsonl(output_dir / "rejected_features.jsonl", rejected)
    write_csv(output_dir / "rejected_features.csv", rejected, REJECTED_FEATURE_FIELDNAMES)
    return {
        "curated_features": len(curated),
        "final_curated_features_top100_by_category": len(final_curated),
        "curated_feature_members": len(members),
        "rejected_features": len(rejected),
    }


def completed_curation_batch_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    for row in read_jsonl(output_dir / "completed_curation_batches.jsonl"):
        batch_id = str(row.get("batch_id") or "")
        if batch_id:
            completed.add(batch_id)
    return completed


def load_existing_curation_decisions(output_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(output_dir / "curation_batch_decisions.jsonl")


def remove_curation_outputs(output_dir: Path) -> None:
    for name in [
        "curated_features.jsonl",
        "curated_features.csv",
        "final_curated_features_top100_by_category.jsonl",
        "final_curated_features_top100_by_category.csv",
        "curated_feature_members.jsonl",
        "curated_feature_members.csv",
        "rejected_features.jsonl",
        "rejected_features.csv",
        "curation_trace.json",
        "curation_errors.jsonl",
        "completed_curation_batches.jsonl",
        "curation_batch_decisions.jsonl",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def completed_final_curation_category_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    for row in read_jsonl(output_dir / "completed_final_curation_categories.jsonl"):
        category_id = str(row.get("category_id") or "")
        if category_id:
            completed.add(category_id)
    return completed


def load_existing_final_curation_decisions(output_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(output_dir / "final_curation_decisions.jsonl")


def remove_final_curation_outputs(output_dir: Path) -> None:
    for name in [
        "final_curated_features_llm_top100_by_category.jsonl",
        "final_curated_features_llm_top100_by_category.csv",
        "final_curated_feature_members_llm.jsonl",
        "final_curated_feature_members_llm.csv",
        "final_rejected_features_llm.jsonl",
        "final_rejected_features_llm.csv",
        "final_curation_trace.json",
        "final_curation_errors.jsonl",
        "completed_final_curation_categories.jsonl",
        "final_curation_decisions.jsonl",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def ranked_paper_preview(
    papers: dict[str, dict[str, Any]],
    limit: int = 25,
) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for paper in list(papers.values())[:limit]:
        preview.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "title_relevance_score": paper.get("title_relevance_score", 0),
            }
        )
    return preview


def dry_run_trace(
    *,
    run_dir: Path,
    output_dir: Path,
    papers: dict[str, dict[str, Any]],
    batch_chunks_count: int,
    input_mode: str,
    api_provider: str,
    model: str,
    run_id: str,
    goal_path: str | None = None,
    goal_text: str = "",
    max_features_per_chunk: int = 12,
    max_features_per_pdf: int = 40,
) -> dict[str, Any]:
    stats = ExtractionStats(paper_count=len(papers))
    if input_mode == "pdf":
        stats.pdf_count = len(papers)
        stats.paper_completed_count = len(papers)
        stats.batch_count = len(papers)
    else:
        for _, chunks in iter_chunks_by_paper(run_dir, papers):
            stats.paper_completed_count += 1
            stats.chunk_count += len(chunks)
            stats.batch_count += sum(1 for _ in batch_chunks(chunks, batch_chunks_count))
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": True,
        "model": model,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "goal_path": goal_path,
        "goal_used": bool(goal_text.strip()),
        "config": {
            "input_mode": input_mode,
            "api_provider": api_provider,
            "batch_chunks": batch_chunks_count,
            "paper_ranking": "title_relevance",
            "max_features_per_chunk": max_features_per_chunk,
            "max_features_per_pdf": max_features_per_pdf,
        },
        "top_ranked_papers": ranked_paper_preview(papers),
        "counts": dataclasses.asdict(stats),
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "extraction_trace.json", trace)
    return trace


def run_extraction(
    *,
    run_dir: Path,
    output_dir: Path,
    api_key: str | None,
    model: str,
    base_url: str,
    api_provider: str = "auto",
    reasoning_effort: str | None,
    batch_chunks_count: int,
    paper_limit: int | None,
    goal_path: Path | None = None,
    force: bool,
    dry_run: bool,
    max_chunk_chars: int,
    max_features_per_chunk: int = 12,
    max_features_per_pdf: int = 40,
    input_mode: str = "chunks",
    temperature: float,
    timeout: float | None = None,
    show_progress: bool = True,
    client: ChatCompletionClient | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser()
    output_dir = output_dir.expanduser()
    if input_mode not in {"chunks", "pdf"}:
        raise ValueError("input_mode must be 'chunks' or 'pdf'.")
    if api_provider not in {"auto", "openai", "gemini"}:
        raise ValueError("api_provider must be 'auto', 'openai', or 'gemini'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        remove_existing_outputs(output_dir)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    papers = load_papers(
        run_dir,
        paper_limit=paper_limit,
        rank_by_title=True,
        show_progress=show_progress,
        include_pdf_failures=input_mode == "pdf",
    )
    goal_text, resolved_goal_path = load_goal_text(goal_path)
    if dry_run:
        return dry_run_trace(
            run_dir=run_dir,
            output_dir=output_dir,
            papers=papers,
            batch_chunks_count=batch_chunks_count,
            input_mode=input_mode,
            api_provider=api_provider,
            model=model,
            run_id=run_id,
            goal_path=resolved_goal_path,
            goal_text=goal_text,
            max_features_per_chunk=max_features_per_chunk,
            max_features_per_pdf=max_features_per_pdf,
        )

    if client is None:
        if not api_key:
            raise ValueError("An API key is required unless --dry-run is used.")
        client = make_llm_client(
            api_key=api_key,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            api_provider=api_provider,
            input_mode=input_mode,
            timeout=timeout,
        )

    stats = ExtractionStats(paper_count=len(papers))
    completed = completed_paper_ids(output_dir)
    mentions = load_existing_mentions(output_dir)
    extracted_at = utc_now()
    failures: list[dict[str, Any]] = []
    pending_paper_ids = {paper_id for paper_id in papers if paper_id not in completed}
    total_pending_units = (
        len(pending_paper_ids)
        if input_mode == "pdf"
        else count_chunks_for_papers(run_dir, pending_paper_ids)
    )
    output_counts = write_artifacts(output_dir, mentions)

    with extraction_progress(
        total=total_pending_units,
        input_mode=input_mode,
        show=show_progress,
    ) as progress:
        if input_mode == "pdf":
            for paper_id, paper in papers.items():
                if paper_id in completed:
                    stats.paper_skipped_count += 1
                    continue

                stats.pdf_count += 1
                stats.batch_count += 1
                pseudo_chunk = pdf_prompt_chunk(paper)
                paper_mentions: list[dict[str, Any]] = []
                paper_failed = False
                raw_response = ""
                pdf_path_text = ""
                try:
                    pdf_path = resolve_pdf_path(paper, run_dir)
                    pdf_path_text = str(pdf_path)
                    if not pdf_path.exists():
                        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
                    LOGGER.info("Extracting %s from PDF: %s", paper_id, pdf_path)
                    raw_response = client.chat(
                        build_pdf_extraction_messages(
                            paper=paper,
                            pdf_path=pdf_path,
                            goal_text=goal_text,
                            max_features_per_pdf=max_features_per_pdf,
                        ),
                        temperature=temperature,
                        max_tokens=5000,
                        response_format={"type": "json_object"},
                    )
                    parsed, repaired_json = parse_features_json_with_repair(
                        raw_response,
                        client=client,
                    )
                    if repaired_json:
                        stats.repaired_json_count += 1
                    stats.raw_feature_count += len(parsed)
                    for item in parsed:
                        row = normalize_feature_item(
                            item,
                            paper=paper,
                            chunk=pseudo_chunk,
                            run_id=run_id,
                            model=model,
                            extracted_at=extracted_at,
                        )
                        if row is None:
                            stats.filtered_feature_count += 1
                            continue
                        stats.kept_feature_count += 1
                        paper_mentions.append(row)
                except Exception as exc:
                    paper_failed = True
                    stats.failed_batch_count += 1
                    error_row = {
                        "paper_id": paper_id,
                        "batch_index": 1,
                        "chunk_ids": [str(pseudo_chunk.get("chunk_id") or "")],
                        "pdf_path": pdf_path_text,
                        "error": str(exc),
                        "raw_response_excerpt": raw_response[:2000] if raw_response else "",
                        "run_id": run_id,
                        "created_at": utc_now(),
                    }
                    failures.append(error_row)
                    append_jsonl(output_dir / "extraction_errors.jsonl", error_row)
                    LOGGER.warning(
                        "PDF extraction failed for %s: %s",
                        paper_id,
                        exc,
                    )
                finally:
                    progress.update(1)

                mentions.extend(paper_mentions)
                output_counts = write_artifacts(output_dir, mentions)
                if paper_failed:
                    LOGGER.info("Leaving %s incomplete so failed PDF extraction can be retried.", paper_id)
                else:
                    append_jsonl(
                        output_dir / "completed_papers.jsonl",
                        {
                            "paper_id": paper_id,
                            "input_mode": input_mode,
                            "chunk_count": 0,
                            "pdf_path": pdf_path_text,
                            "feature_count": len(paper_mentions),
                            "run_id": run_id,
                            "completed_at": utc_now(),
                        },
                    )
                    completed.add(paper_id)
                    stats.paper_completed_count += 1
        else:
            for paper_id, chunks in iter_chunks_by_paper(run_dir, papers):
                if paper_id in completed:
                    stats.paper_skipped_count += 1
                    continue

                paper = papers[paper_id]
                stats.chunk_count += len(chunks)
                paper_mentions: list[dict[str, Any]] = []
                paper_failed = False
                LOGGER.info("Extracting %s with %d chunks.", paper_id, len(chunks))
                for batch_index, chunk_batch in enumerate(
                    batch_chunks(chunks, batch_chunks_count),
                    start=1,
                ):
                    stats.batch_count += 1
                    raw_response = ""
                    try:
                        raw_response = client.chat(
                            [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an evidence-bound taxonomy extraction agent. "
                                        "Return only valid JSON."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": build_extraction_prompt(
                                        paper,
                                        chunk_batch,
                                        max_chunk_chars=max_chunk_chars,
                                        goal_text=goal_text,
                                        max_features_per_chunk=max_features_per_chunk,
                                    ),
                                },
                            ],
                            temperature=temperature,
                            max_tokens=3500,
                            response_format={"type": "json_object"},
                        )
                        parsed, repaired_json = parse_features_json_with_repair(
                            raw_response,
                            client=client,
                        )
                        if repaired_json:
                            stats.repaired_json_count += 1
                        stats.raw_feature_count += len(parsed)
                        chunks_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunk_batch}
                        fallback_chunk = chunk_batch[0] if chunk_batch else {}
                        for item in parsed:
                            chunk_id = str(item.get("chunk_id") or "")
                            row = normalize_feature_item(
                                item,
                                paper=paper,
                                chunk=chunks_by_id.get(chunk_id) or fallback_chunk,
                                run_id=run_id,
                                model=model,
                                extracted_at=extracted_at,
                            )
                            if row is None:
                                stats.filtered_feature_count += 1
                                continue
                            stats.kept_feature_count += 1
                            paper_mentions.append(row)
                    except Exception as exc:
                        paper_failed = True
                        stats.failed_batch_count += 1
                        error_row = {
                            "paper_id": paper_id,
                            "batch_index": batch_index,
                            "chunk_ids": [str(chunk.get("chunk_id") or "") for chunk in chunk_batch],
                            "error": str(exc),
                            "raw_response_excerpt": raw_response[:2000] if raw_response else "",
                            "run_id": run_id,
                            "created_at": utc_now(),
                        }
                        failures.append(error_row)
                        append_jsonl(output_dir / "extraction_errors.jsonl", error_row)
                        LOGGER.warning(
                            "Extraction failed for %s batch %d: %s",
                            paper_id,
                            batch_index,
                            exc,
                        )
                    finally:
                        progress.update(len(chunk_batch))

                mentions.extend(paper_mentions)
                output_counts = write_artifacts(output_dir, mentions)
                if paper_failed:
                    LOGGER.info("Leaving %s incomplete so failed batches can be retried.", paper_id)
                else:
                    append_jsonl(
                        output_dir / "completed_papers.jsonl",
                        {
                            "paper_id": paper_id,
                            "input_mode": input_mode,
                            "chunk_count": len(chunks),
                            "feature_count": len(paper_mentions),
                            "run_id": run_id,
                            "completed_at": utc_now(),
                        },
                    )
                    completed.add(paper_id)
                    stats.paper_completed_count += 1

    errors_path = output_dir / "extraction_errors.jsonl"
    if not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": False,
        "model": model,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "goal_path": resolved_goal_path,
        "goal_used": bool(goal_text.strip()),
        "config": {
            "input_mode": input_mode,
            "api_provider": api_provider,
            "batch_chunks": batch_chunks_count,
            "paper_limit": paper_limit,
            "max_chunk_chars": max_chunk_chars,
            "max_features_per_chunk": max_features_per_chunk,
            "max_features_per_pdf": max_features_per_pdf,
            "temperature": temperature,
            "timeout": timeout,
            "force": force,
            "progress": show_progress,
            "paper_ranking": "title_relevance",
        },
        "top_ranked_papers": ranked_paper_preview(papers),
        "counts": dataclasses.asdict(stats),
        "outputs": output_counts,
        "failures": failures,
    }
    write_json(output_dir / "extraction_trace.json", trace)
    return trace


def dry_run_curation_trace(
    *,
    input_dir: Path,
    output_dir: Path,
    features: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    model: str,
    run_id: str,
    goal_path: str | None,
    goal_text: str,
    batch_features: int,
    merge_style: str,
) -> dict[str, Any]:
    stats = CurationStats(
        feature_count=len(features),
        batch_count=len(batches),
    )
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": True,
        "model": model,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "goal_path": goal_path,
        "goal_used": bool(goal_text.strip()),
        "config": {
            "batch_features": batch_features,
            "merge_style": merge_style,
        },
        "counts": dataclasses.asdict(stats),
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "curation_trace.json", trace)
    return trace


def run_curation(
    *,
    input_dir: Path,
    output_dir: Path,
    api_key: str | None,
    model: str,
    base_url: str,
    reasoning_effort: str | None,
    batch_features: int,
    merge_style: str,
    goal_path: Path | None = None,
    force: bool,
    dry_run: bool,
    temperature: float,
    timeout: float | None = None,
    show_progress: bool = True,
    client: ChatCompletionClient | None = None,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser()
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        remove_curation_outputs(output_dir)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    features = load_feature_rows(input_dir)
    batches = prepare_curation_batches(features, batch_size=batch_features)
    goal_text, resolved_goal_path = load_goal_text(goal_path)
    if dry_run:
        return dry_run_curation_trace(
            input_dir=input_dir,
            output_dir=output_dir,
            features=features,
            batches=batches,
            model=model,
            run_id=run_id,
            goal_path=resolved_goal_path,
            goal_text=goal_text,
            batch_features=batch_features,
            merge_style=merge_style,
        )

    if client is None:
        if not api_key:
            raise ValueError("An API key is required unless --dry-run is used.")
        client = ChatCompletionClient(
            api_key=clean_api_key(api_key),
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            timeout=120.0 if timeout is None else max(1.0, float(timeout)),
        )

    completed = completed_curation_batch_ids(output_dir)
    decisions = load_existing_curation_decisions(output_dir)
    stats = CurationStats(feature_count=len(features), batch_count=len(batches))
    output_counts = write_curation_artifacts(output_dir, features, decisions)
    failures: list[dict[str, Any]] = []
    pending_feature_count = sum(
        len(batch["features"])
        for batch in batches
        if str(batch["batch_id"]) not in completed
    )

    with curation_progress(total=pending_feature_count, show=show_progress) as progress:
        for batch in batches:
            batch_id = str(batch["batch_id"])
            batch_rows = list(batch["features"])
            if batch_id in completed:
                stats.skipped_batch_count += 1
                continue

            raw_response = ""
            try:
                raw_response = client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are an evidence-bound taxonomy curation agent. "
                                "Return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_curation_prompt(
                                features=batch_rows,
                                category=str(batch["category"]),
                                goal_text=goal_text,
                                merge_style=merge_style,
                            ),
                        },
                    ],
                    temperature=temperature,
                    max_tokens=5000,
                    response_format={"type": "json_object"},
                )
                raw_decisions, repaired_json = parse_curation_json_with_repair(
                    raw_response,
                    client=client,
                )
                if repaired_json:
                    stats.repaired_json_count += 1
                batch_decisions = normalize_curation_decisions(
                    raw_decisions,
                    batch_rows,
                    model=model,
                    run_id=run_id,
                    curated_at=utc_now(),
                )
                for decision in batch_decisions:
                    append_jsonl(
                        output_dir / "curation_batch_decisions.jsonl",
                        {
                            **decision,
                            "batch_id": batch_id,
                            "batch_index": batch.get("batch_index"),
                        },
                    )
                append_jsonl(
                    output_dir / "completed_curation_batches.jsonl",
                    {
                        "batch_id": batch_id,
                        "batch_index": batch.get("batch_index"),
                        "category": batch.get("category"),
                        "feature_count": len(batch_rows),
                        "decision_count": len(batch_decisions),
                        "run_id": run_id,
                        "completed_at": utc_now(),
                    },
                )
                completed.add(batch_id)
                decisions.extend(batch_decisions)
                stats.completed_batch_count += 1
                stats.decision_count += len(batch_decisions)
                output_counts = write_curation_artifacts(output_dir, features, decisions)
            except Exception as exc:
                stats.failed_batch_count += 1
                error_row = {
                    "batch_id": batch_id,
                    "batch_index": batch.get("batch_index"),
                    "category": batch.get("category"),
                    "feature_keys": [
                        str(row.get("feature_key") or feature_key(row))
                        for row in batch_rows
                    ],
                    "error": str(exc),
                    "raw_response_excerpt": raw_response[:2000] if raw_response else "",
                    "run_id": run_id,
                    "created_at": utc_now(),
                }
                failures.append(error_row)
                append_jsonl(output_dir / "curation_errors.jsonl", error_row)
                LOGGER.warning("Curation failed for %s: %s", batch_id, exc)
            finally:
                progress.update(len(batch_rows))

    errors_path = output_dir / "curation_errors.jsonl"
    if not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": False,
        "model": model,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "goal_path": resolved_goal_path,
        "goal_used": bool(goal_text.strip()),
        "config": {
            "batch_features": batch_features,
            "merge_style": merge_style,
            "temperature": temperature,
            "timeout": timeout,
            "force": force,
            "progress": show_progress,
        },
        "counts": dataclasses.asdict(stats),
        "outputs": output_counts,
        "failures": failures,
    }
    write_json(output_dir / "curation_trace.json", trace)
    return trace


def dry_run_final_curation_trace(
    *,
    input_dir: Path,
    output_dir: Path,
    curated_features: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    model: str,
    run_id: str,
    goal_path: str | None,
    goal_text: str,
    merge_style: str,
    target_per_category: int,
    candidate_pool_per_category: int,
    max_tokens: int,
    timeout: float | None,
) -> dict[str, Any]:
    candidate_count = sum(len(category["features"]) for category in categories)
    stats = FinalCurationStats(
        feature_count=len(curated_features),
        category_count=len(categories),
        candidate_feature_count=candidate_count,
    )
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": True,
        "model": model,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "goal_path": goal_path,
        "goal_used": bool(goal_text.strip()),
        "config": {
            "merge_style": merge_style,
            "target_per_category": target_per_category,
            "candidate_pool_per_category": candidate_pool_per_category,
            "max_tokens": max_tokens,
            "timeout": timeout,
        },
        "category_candidates": {
            str(category["category"]): len(category["features"])
            for category in categories
        },
        "counts": dataclasses.asdict(stats),
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "final_curation_trace.json", trace)
    return trace


def run_final_curation(
    *,
    input_dir: Path,
    output_dir: Path,
    api_key: str | None,
    model: str,
    base_url: str,
    api_provider: str = "auto",
    reasoning_effort: str | None,
    merge_style: str,
    goal_path: Path | None = None,
    target_per_category: int = FINAL_FEATURES_PER_CATEGORY,
    candidate_pool_per_category: int = 0,
    force: bool,
    dry_run: bool,
    temperature: float,
    max_tokens: int,
    timeout: float | None = None,
    show_progress: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser()
    output_dir = output_dir.expanduser()
    if api_provider not in {"auto", "openai", "gemini"}:
        raise ValueError("api_provider must be 'auto', 'openai', or 'gemini'.")
    target_per_category = max(1, target_per_category)
    candidate_pool_per_category = max(0, candidate_pool_per_category)
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        remove_final_curation_outputs(output_dir)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    curated_features = load_curated_feature_rows(input_dir)
    categories = final_candidate_categories(
        curated_features,
        candidate_pool_per_category=candidate_pool_per_category,
    )
    candidate_features = [
        row
        for category in categories
        for row in category["features"]
    ]
    goal_text, resolved_goal_path = load_goal_text(goal_path)
    if dry_run:
        return dry_run_final_curation_trace(
            input_dir=input_dir,
            output_dir=output_dir,
            curated_features=curated_features,
            categories=categories,
            model=model,
            run_id=run_id,
            goal_path=resolved_goal_path,
            goal_text=goal_text,
            merge_style=merge_style,
            target_per_category=target_per_category,
            candidate_pool_per_category=candidate_pool_per_category,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if client is None:
        if not api_key:
            raise ValueError("An API key is required unless --dry-run is used.")
        client = make_llm_client(
            api_key=api_key,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            api_provider=api_provider,
            input_mode="text",
            timeout=timeout,
        )

    completed = completed_final_curation_category_ids(output_dir)
    decisions = load_existing_final_curation_decisions(output_dir)
    stats = FinalCurationStats(
        feature_count=len(curated_features),
        category_count=len(categories),
        candidate_feature_count=len(candidate_features),
    )
    output_counts = write_final_curation_artifacts(
        output_dir,
        candidate_features,
        decisions,
        target_per_category=target_per_category,
    )
    failures: list[dict[str, Any]] = []
    pending_feature_count = sum(
        len(category["features"])
        for category in categories
        if final_curation_category_id(str(category["category"]), category["features"]) not in completed
    )

    with final_curation_progress(total=pending_feature_count, show=show_progress) as progress:
        for category_payload in categories:
            category = str(category_payload["category"])
            category_rows = list(category_payload["features"])
            category_id = final_curation_category_id(category, category_rows)
            if category_id in completed:
                stats.skipped_category_count += 1
                continue

            raw_response = ""
            try:
                raw_response = client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are an evidence-bound final taxonomy curation agent. "
                                "Return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_final_curation_prompt(
                                features=category_rows,
                                category=category,
                                goal_text=goal_text,
                                merge_style=merge_style,
                                target_per_category=target_per_category,
                            ),
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_decisions, repaired_json = parse_curation_json_with_repair(
                    raw_response,
                    client=client,
                )
                if repaired_json:
                    stats.repaired_json_count += 1
                category_decisions = normalize_curation_decisions(
                    raw_decisions,
                    category_rows,
                    model=model,
                    run_id=run_id,
                    curated_at=utc_now(),
                )
                for decision in category_decisions:
                    append_jsonl(
                        output_dir / "final_curation_decisions.jsonl",
                        {
                            **decision,
                            "category_id": category_id,
                            "final_category": category,
                        },
                    )
                append_jsonl(
                    output_dir / "completed_final_curation_categories.jsonl",
                    {
                        "category_id": category_id,
                        "category": category,
                        "candidate_feature_count": len(category_rows),
                        "decision_count": len(category_decisions),
                        "target_per_category": target_per_category,
                        "run_id": run_id,
                        "completed_at": utc_now(),
                    },
                )
                completed.add(category_id)
                decisions.extend(category_decisions)
                stats.completed_category_count += 1
                stats.decision_count += len(category_decisions)
                output_counts = write_final_curation_artifacts(
                    output_dir,
                    candidate_features,
                    decisions,
                    target_per_category=target_per_category,
                )
            except Exception as exc:
                stats.failed_category_count += 1
                error_row = {
                    "category_id": category_id,
                    "category": category,
                    "feature_keys": [
                        str(row.get("feature_key") or feature_key(row))
                        for row in category_rows
                    ],
                    "error": str(exc),
                    "raw_response_excerpt": raw_response[:2000] if raw_response else "",
                    "run_id": run_id,
                    "created_at": utc_now(),
                }
                failures.append(error_row)
                append_jsonl(output_dir / "final_curation_errors.jsonl", error_row)
                LOGGER.warning("Final curation failed for %s: %s", category, exc)
            finally:
                progress.update(len(category_rows))

    errors_path = output_dir / "final_curation_errors.jsonl"
    if not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": False,
        "model": model,
        "base_url": base_url,
        "api_provider": api_provider,
        "reasoning_effort": reasoning_effort,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "goal_path": resolved_goal_path,
        "goal_used": bool(goal_text.strip()),
        "config": {
            "merge_style": merge_style,
            "target_per_category": target_per_category,
            "candidate_pool_per_category": candidate_pool_per_category,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "force": force,
            "progress": show_progress,
            "paper_count_sort": "descending",
        },
        "category_candidates": {
            str(category["category"]): len(category["features"])
            for category in categories
        },
        "counts": dataclasses.asdict(stats),
        "outputs": output_counts,
        "failures": failures,
    }
    write_json(output_dir / "final_curation_trace.json", trace)
    return trace


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress logs.")
    parser.add_argument("--quiet", action="store_true", help="Show warnings and errors only.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = expand_config_argv(argv)
    parser = argparse.ArgumentParser(
        description="Extract persuasion taxonomy feature mentions from a LitSynth run."
    )
    parser.add_argument(
        "--config",
        help=(
            "Path to a JSON extraction config. Put this before the command, for example: "
            "lit-extract --config extract_configs/00_extract.json"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run corpus feature extraction.")
    add_common_args(run)
    run.add_argument("--run", required=True, help="LitSynth run folder with paper_index/chunks.")
    run.add_argument("--output-dir", required=True, help="Folder for extraction artifacts.")
    run.add_argument("--api-key", help="LLM API key. Defaults to env vars.")
    run.add_argument("--model", default=env_model(), help="LLM model name.")
    run.add_argument(
        "--base-url",
        default=env_base_url(),
        help="OpenAI-compatible base URL, or Gemini native v1beta base URL with --api-provider gemini.",
    )
    run.add_argument(
        "--api-provider",
        default="auto",
        choices=["auto", "openai", "gemini"],
        help="Transport to use. Auto keeps chunk mode OpenAI-compatible and uses Gemini-native PDFs for Gemini URLs.",
    )
    run.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        help="OpenAI-compatible reasoning effort/thinking level.",
    )
    run.add_argument(
        "--input-mode",
        default="chunks",
        choices=["chunks", "pdf"],
        help="Use indexed Markdown chunks or send each source PDF directly to the LLM.",
    )
    run.add_argument(
        "--batch-chunks",
        type=int,
        default=4,
        help="Number of chunks sent in each extraction request.",
    )
    run.add_argument(
        "--paper-limit",
        type=int,
        help="Optional limit for smoke tests; default scans all papers.",
    )
    run.add_argument(
        "--goal",
        help="Optional goal/taxonomy Markdown file. Defaults to ./goal.md when present.",
    )
    run.add_argument(
        "--max-features-per-chunk",
        type=int,
        default=12,
        help="Maximum feature objects the model should return for each chunk.",
    )
    run.add_argument(
        "--max-features-per-pdf",
        type=int,
        default=40,
        help="Maximum feature objects the model should return for each direct PDF request.",
    )
    run.add_argument(
        "--max-chunk-chars",
        type=int,
        default=2500,
        help="Maximum characters per chunk included in the LLM prompt.",
    )
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, help="LLM request timeout in seconds.")
    run.add_argument("--force", action="store_true", help="Overwrite existing extraction outputs.")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Count papers/chunks/batches without making LLM calls.",
    )

    curate = subparsers.add_parser("curate", help="Curate and consolidate extracted features.")
    add_common_args(curate)
    curate.add_argument("--input-dir", required=True, help="Folder containing first-pass outputs.")
    curate.add_argument(
        "--output-dir",
        help="Folder for curated artifacts. Defaults to --input-dir.",
    )
    curate.add_argument(
        "--goal",
        help="Optional goal/taxonomy Markdown file. Defaults to ./goal.md when present.",
    )
    curate.add_argument("--api-key", help="LLM API key. Defaults to env vars.")
    curate.add_argument("--model", default=env_model(), help="LLM model name.")
    curate.add_argument("--base-url", default=env_base_url(), help="OpenAI-compatible base URL.")
    curate.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        help="OpenAI-compatible reasoning effort/thinking level.",
    )
    curate.add_argument(
        "--batch-features",
        type=int,
        default=40,
        help="Number of corpus features sent in each curation request.",
    )
    curate.add_argument(
        "--merge-style",
        default="moderate",
        choices=["conservative", "moderate", "aggressive"],
        help="How aggressively to merge synonym or near-synonym features.",
    )
    curate.add_argument("--temperature", type=float, default=0.0)
    curate.add_argument("--timeout", type=float, help="LLM request timeout in seconds.")
    curate.add_argument("--force", action="store_true", help="Overwrite existing curation outputs.")
    curate.add_argument(
        "--dry-run",
        action="store_true",
        help="Count features/batches without making LLM calls.",
    )

    finalize = subparsers.add_parser(
        "finalize",
        help="LLM-finalize top curated features by category.",
    )
    add_common_args(finalize)
    finalize.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing curated_features.jsonl.",
    )
    finalize.add_argument(
        "--output-dir",
        help="Folder for final artifacts. Defaults to --input-dir.",
    )
    finalize.add_argument(
        "--goal",
        help="Optional goal/taxonomy Markdown file. Defaults to ./goal.md when present.",
    )
    finalize.add_argument("--api-key", help="LLM API key. Defaults to env vars.")
    finalize.add_argument("--model", default=env_model(), help="LLM model name.")
    finalize.add_argument(
        "--base-url",
        default=env_base_url(),
        help="OpenAI-compatible base URL, or Gemini native v1beta base URL with --api-provider gemini.",
    )
    finalize.add_argument(
        "--api-provider",
        default="auto",
        choices=["auto", "openai", "gemini"],
        help="Transport to use for the final LLM pass.",
    )
    finalize.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        help="OpenAI-compatible reasoning effort/thinking level.",
    )
    finalize.add_argument(
        "--target-per-category",
        type=int,
        default=FINAL_FEATURES_PER_CATEGORY,
        help="Maximum unique final features to keep in each category.",
    )
    finalize.add_argument(
        "--candidate-pool-per-category",
        type=int,
        default=0,
        help="Top ranked candidates per category to send; 0 sends all curated candidates.",
    )
    finalize.add_argument(
        "--merge-style",
        default="moderate",
        choices=["conservative", "moderate", "aggressive"],
        help="How aggressively to merge synonym or near-synonym features.",
    )
    finalize.add_argument("--temperature", type=float, default=0.0)
    finalize.add_argument(
        "--max-tokens",
        type=int,
        default=30000,
        help="Maximum output tokens for each category-level finalization request.",
    )
    finalize.add_argument("--timeout", type=float, help="LLM request timeout in seconds.")
    finalize.add_argument("--force", action="store_true", help="Overwrite existing final outputs.")
    finalize.add_argument(
        "--dry-run",
        action="store_true",
        help="Count category candidates without making LLM calls.",
    )
    return parser.parse_args(argv)


def load_extract_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Extraction config does not exist: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Extraction config must contain a JSON object.")
    return config


def extract_config_to_argv(config: dict[str, Any]) -> list[str]:
    command = str(config.get("command") or "").strip()
    if not command:
        raise ValueError("Extraction config must include a command.")
    if command not in CONFIG_ARG_FIELDS:
        allowed = ", ".join(sorted(CONFIG_ARG_FIELDS))
        raise ValueError(f"Unknown extraction config command {command!r}; expected one of {allowed}.")

    config = dict(config)
    api_key_env = config.get("api_key_env")
    if api_key_env and not config.get("api_key"):
        api_key = os.getenv(str(api_key_env))
        if api_key:
            config["api_key"] = api_key

    argv = [command]
    for key, flag in CONFIG_ARG_FIELDS[command]:
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    return argv


def expand_config_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if "--config" not in argv:
        return argv

    config_index = argv.index("--config")
    try:
        config_path = Path(argv[config_index + 1]).expanduser()
    except IndexError as exc:
        raise ValueError("--config requires a JSON config path.") from exc

    remaining = argv[:config_index] + argv[config_index + 2 :]
    config_argv = extract_config_to_argv(load_extract_config(config_path))
    return config_argv + remaining


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))
    try:
        if args.command == "run":
            api_key = args.api_key or env_api_key()
            trace = run_extraction(
                run_dir=Path(args.run),
                output_dir=Path(args.output_dir),
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                api_provider=args.api_provider,
                reasoning_effort=args.reasoning_effort,
                batch_chunks_count=args.batch_chunks,
                paper_limit=args.paper_limit,
                goal_path=Path(args.goal) if args.goal else None,
                force=args.force,
                dry_run=args.dry_run,
                max_chunk_chars=args.max_chunk_chars,
                max_features_per_chunk=args.max_features_per_chunk,
                max_features_per_pdf=args.max_features_per_pdf,
                input_mode=args.input_mode,
                temperature=args.temperature,
                timeout=args.timeout,
                show_progress=not args.quiet,
            )
            print(Path(args.output_dir) / "extraction_trace.json")
            LOGGER.info(
                "Extraction complete: %d mentions, %d paper features, %d corpus features.",
                trace.get("outputs", {}).get("feature_mentions", 0),
                trace.get("outputs", {}).get("paper_features", 0),
                trace.get("outputs", {}).get("features", 0),
            )
            return 0
        if args.command == "curate":
            api_key = args.api_key or env_api_key()
            output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_dir)
            trace = run_curation(
                input_dir=Path(args.input_dir),
                output_dir=output_dir,
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                reasoning_effort=args.reasoning_effort,
                batch_features=args.batch_features,
                merge_style=args.merge_style,
                goal_path=Path(args.goal) if args.goal else None,
                force=args.force,
                dry_run=args.dry_run,
                temperature=args.temperature,
                timeout=args.timeout,
                show_progress=not args.quiet,
            )
            print(output_dir / "curation_trace.json")
            LOGGER.info(
                "Curation complete: %d curated features, %d member mappings, %d rejected.",
                trace.get("outputs", {}).get("curated_features", 0),
                trace.get("outputs", {}).get("curated_feature_members", 0),
                trace.get("outputs", {}).get("rejected_features", 0),
            )
            return 0
        if args.command == "finalize":
            api_key = args.api_key or env_api_key()
            output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_dir)
            trace = run_final_curation(
                input_dir=Path(args.input_dir),
                output_dir=output_dir,
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                api_provider=args.api_provider,
                reasoning_effort=args.reasoning_effort,
                merge_style=args.merge_style,
                goal_path=Path(args.goal) if args.goal else None,
                target_per_category=args.target_per_category,
                candidate_pool_per_category=args.candidate_pool_per_category,
                force=args.force,
                dry_run=args.dry_run,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                show_progress=not args.quiet,
            )
            print(output_dir / "final_curation_trace.json")
            LOGGER.info(
                "Final curation complete: %d final features, %d member mappings, %d rejected.",
                trace.get("outputs", {}).get("final_curated_features_llm_top100_by_category", 0),
                trace.get("outputs", {}).get("final_curated_feature_members_llm", 0),
                trace.get("outputs", {}).get("final_rejected_features_llm", 0),
            )
            return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
