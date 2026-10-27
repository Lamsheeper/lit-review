#!/usr/bin/env python3
"""Visualize feature-set Gemini embeddings with t-SNE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from lit_review.core.google_auth import ADCAuth, use_vertex, vertex_model_url

from lit_review.profiles.embedding_cleanup import EMBEDDING_TEXT_CLEANUP_BUILTINS


DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TASK_TYPE = "CLUSTERING"
DEFAULT_OUTPUT_DIMENSIONALITY = 768
DEFAULT_EMBEDDING_RETRY_ATTEMPTS = 6
DEFAULT_EMBEDDING_RETRY_INITIAL_DELAY_SECONDS = 2.0
DEFAULT_EMBEDDING_RETRY_MAX_DELAY_SECONDS = 60.0
DEFAULT_TEXT_FIELDS = (
    "feature_name",
    "category",
    "annotation.formal_definition",
    "annotation.short_definition",
    "annotation.synonyms",
    "annotation.examples",
    "definitions",
    "synonyms",
)
DEFAULT_ANNOTATE_TOP = 80


@dataclass(frozen=True)
class EmbeddingTextCleanup:
    strip_builtins: tuple[str, ...] = ()
    min_chars: int = 0
    fallback: str = "original"


@dataclass(frozen=True)
class CacheStats:
    cache_hits: int
    api_misses: int
    api_batches: int


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text, preserving order."""


class GeminiEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        task_type: str,
        output_dimensionality: int,
        timeout: float,
    ) -> None:
        self.api_key = api_key
        self.model = model_resource_name(model)
        self.base_url = base_url.rstrip("/")
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        self.timeout = timeout

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        requests = [
            {
                "model": self.model,
                "content": {"parts": [{"text": text}]},
                "taskType": self.task_type,
                "outputDimensionality": self.output_dimensionality,
            }
            for text in texts
        ]
        payload = {"requests": requests}
        model_id = self.model.removeprefix("models/")
        url = f"{self.base_url}/models/{model_id}:batchEmbedContents"
        request = urllib.request.Request(
            url,
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
            raise RuntimeError(f"Gemini embeddings API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini embeddings API request failed: {exc}") from exc

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(f"Unexpected Gemini embeddings response: {data}")

        values: list[list[float]] = []
        for index, item in enumerate(embeddings, start=1):
            vector = item.get("values") if isinstance(item, dict) else None
            if not isinstance(vector, list):
                raise RuntimeError(f"Gemini embedding {index} did not contain a values list.")
            values.append([float(value) for value in vector])
        return values


class VertexEmbeddingClient:
    """Gemini embedding client for Vertex AI using gcloud ADC credentials."""

    def __init__(
        self,
        *,
        project: str,
        location: str,
        model: str,
        task_type: str,
        output_dimensionality: int,
        timeout: float,
    ) -> None:
        self.url = vertex_model_url(project, location, model) + ":predict"
        self.auth = ADCAuth()
        self.project = project.strip()
        self.location = location.strip()
        self.model = model_resource_name(model).removeprefix("models/")
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        self.timeout = timeout

    def access_token(self) -> str:
        # Kept for callers that used the original Vertex client helper.
        return self.auth.headers(self.url)["authorization"].removeprefix("Bearer ")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        url = self.url
        values: list[list[float]] = []
        # Vertex currently accepts one input per request for gemini-embedding-001.
        for text in texts:
            payload = {
                "instances": [{"content": text, "task_type": self.task_type}],
                "parameters": {"outputDimensionality": self.output_dimensionality},
            }
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    **self.auth.headers(url),
                    "Content-Type": "application/json; charset=utf-8",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Vertex AI embeddings returned HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Vertex AI embeddings request failed: {exc}") from exc
            predictions = data.get("predictions")
            embedding = (
                predictions[0].get("embeddings") if isinstance(predictions, list) and predictions else None
            )
            vector = embedding.get("values") if isinstance(embedding, dict) else None
            if not isinstance(vector, list):
                raise RuntimeError(f"Unexpected Vertex AI embeddings response: {data}")
            values.append([float(value) for value in vector])
        return values


def make_embedding_client(
    *, api_key: str | None, model: str, base_url: str,
    task_type: str, output_dimensionality: int, timeout: float,
    auth_mode: str = "auto", vertex_project: str | None = None,
    vertex_location: str | None = None,
) -> EmbeddingClient | None:
    if use_vertex(api_key, auth_mode=auth_mode, project=vertex_project):
        return VertexEmbeddingClient(
            project=vertex_project or os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            location=vertex_location or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1",
            model=model, task_type=task_type,
            output_dimensionality=output_dimensionality, timeout=timeout,
        )
    if not api_key:
        return None  # Analysis can run entirely from an existing embedding cache.
    return GeminiEmbeddingClient(
        api_key=api_key, model=model, base_url=base_url, task_type=task_type,
        output_dimensionality=output_dimensionality, timeout=timeout,
    )


def add_google_auth_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--auth-mode", choices=["auto", "api_key", "adc"], default="auto")
    parser.add_argument("--vertex-project", help="Vertex AI project (or GOOGLE_CLOUD_PROJECT).")
    parser.add_argument("--vertex-location", help="Vertex AI region (or GOOGLE_CLOUD_LOCATION; default us-central1).")


def read_feature_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        return read_jsonl_rows(path, raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return read_jsonl_rows(path, raw)
    return coerce_feature_rows(parsed, path)


def read_jsonl_rows(path: Path, raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object row.")
        rows.append(parsed)
    return rows


def coerce_feature_rows(parsed: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return coerce_row_list(parsed, path)
    if isinstance(parsed, dict):
        for key in ("features", "rows", "items", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return coerce_row_list(value, path)
        if any(key in parsed for key in ("feature_name", "normalized_feature_name", "feature_key")):
            return [parsed]
    raise ValueError(
        f"{path} must be CSV, JSONL, a JSON list, or a JSON object with rows under "
        "'features', 'rows', 'items', or 'data'."
    )


def coerce_row_list(values: list[Any], path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"{path} item {index} must be a JSON object row.")
        rows.append(value)
    return rows


def parse_text_fields(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        fields = [field.strip() for field in value.split(",")]
    else:
        fields = [str(field).strip() for field in value]
    parsed = [field for field in fields if field]
    if not parsed:
        raise ValueError("At least one embedding text field is required.")
    return parsed


def build_embedding_text(
    row: dict[str, Any],
    text_fields: Iterable[str],
    cleanup: EmbeddingTextCleanup | None = None,
    cleanup_fields: Iterable[str] | None = None,
) -> str:
    sections: list[str] = []
    seen_by_field: set[tuple[str, str]] = set()
    cleanup_field_set = (
        {str(field).strip() for field in cleanup_fields if str(field).strip()}
        if cleanup_fields is not None
        else None
    )
    for field in text_fields:
        values = []
        field_cleanup = cleanup if cleanup_field_set is None or field in cleanup_field_set else None
        for value in parse_jsonish_list(row_field_value(row, field)):
            cleaned_value = clean_embedding_text_value(value, field_cleanup)
            if not cleaned_value:
                continue
            normalized = cleaned_value.casefold()
            key = (field, normalized)
            if key in seen_by_field:
                continue
            seen_by_field.add(key)
            values.append(cleaned_value)
        if values:
            sections.append(f"{human_field_name(field)}: {'; '.join(values)}")
    return "\n".join(sections).strip()


def clean_embedding_text_value(value: str, cleanup: EmbeddingTextCleanup | None) -> str:
    original = re.sub(r"\s+", " ", value).strip()
    if cleanup is None or not cleanup.strip_builtins:
        return original

    cleaned = original
    phrases = cleanup_builtin_phrases(cleanup.strip_builtins)
    for _ in range(4):
        for phrase in phrases:
            stripped = strip_leading_phrase(cleaned, phrase)
            if stripped == cleaned:
                continue
            cleaned = normalize_after_leading_strip(stripped)
            break
        else:
            break

    if not cleaned or (cleanup.min_chars and len(cleaned) < cleanup.min_chars):
        return original if cleanup.fallback == "original" else ""
    return cleaned


def cleanup_builtin_phrases(strip_builtins: Iterable[str]) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for builtin in strip_builtins:
        if builtin not in EMBEDDING_TEXT_CLEANUP_BUILTINS:
            raise ValueError(f"Unknown embedding text cleanup builtin: {builtin}")
        for phrase in EMBEDDING_TEXT_CLEANUP_BUILTINS[builtin]:
            key = phrase.casefold()
            if key in seen:
                continue
            seen.add(key)
            phrases.append(phrase)
    return phrases


def strip_leading_phrase(text: str, phrase: str) -> str:
    flexible_phrase = r"\s+".join(re.escape(part) for part in phrase.split())
    pattern = re.compile(rf"^\s*{flexible_phrase}(?:\b|(?=\W))", flags=re.IGNORECASE)
    match = pattern.match(text)
    if not match:
        return text
    return text[match.end() :].lstrip(" \t,;:-.")


def normalize_after_leading_strip(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"^(?:and|or)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:that|where)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" \t,;:-")


def human_field_name(field: str) -> str:
    return field.replace(".", " ").replace("_", " ").strip().title()


def first_field_value(row: dict[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        value = row_field_value(row, field)
        if value not in (None, ""):
            return value
    return None


def load_feature_items(
    path: Path,
    text_fields: Iterable[str],
    cleanup: EmbeddingTextCleanup | None = None,
    cleanup_fields: Iterable[str] | None = None,
) -> list[FeatureItem]:
    rows = read_feature_rows(path)
    fields = parse_text_fields(text_fields)
    items: list[FeatureItem] = []
    for index, row in enumerate(rows, start=1):
        embedding_text = build_embedding_text(row, fields, cleanup=cleanup, cleanup_fields=cleanup_fields)
        if not embedding_text:
            continue
        label = feature_label(row)
        category = str(row.get("category") or "").strip() or "uncategorized"
        items.append(
            FeatureItem(
                row_index=index,
                feature_key=feature_key(row, label, category, index),
                label=label,
                category=category,
                paper_count=safe_int(first_field_value(row, ("paper_count", "annotation.paper_count"))),
                mention_count=safe_int(first_field_value(row, ("mention_count", "annotation.mention_count"))),
                embedding_text=embedding_text,
                raw_row=row,
            )
        )
    return items


def feature_label(row: dict[str, Any]) -> str:
    for field in ("feature_name", "canonical_feature_name", "normalized_feature_name", "feature_key"):
        value = str(row.get(field) or "").strip()
        if value:
            if field == "feature_key" and "::" in value:
                return value.split("::", 1)[1].strip() or value
            return value
    return "feature"


def feature_key(row: dict[str, Any], label: str, category: str, index: int) -> str:
    raw = str(row.get("feature_key") or "").strip()
    if raw:
        return raw
    normalized_name = normalize_key(row.get("normalized_feature_name") or label)
    normalized_category = normalize_key(category)
    if normalized_name:
        return f"{normalized_category}::{normalized_name}"
    return f"row::{index}"


def normalize_key(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"['\u2019]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def model_resource_name(model: str) -> str:
    model = model.strip()
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def embedding_cache_key(
    text: str,
    *,
    model: str,
    task_type: str,
    output_dimensionality: int,
) -> str:
    payload = {
        "model": model_resource_name(model),
        "task_type": task_type,
        "output_dimensionality": output_dimensionality,
        "text": text,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            key = str(item.get("cache_key") or "")
            embedding = item.get("embedding")
            if key and isinstance(embedding, list):
                cache[key] = [float(value) for value in embedding]
    return cache


def append_embedding_cache_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def is_retryable_embedding_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".casefold()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "resource exhausted" in message
        or "rate limit" in message
        or "quota" in message
    )


def embedding_retry_delay(
    retry_index: int,
    *,
    initial_delay_seconds: float,
    max_delay_seconds: float,
) -> float:
    initial = max(0.0, float(initial_delay_seconds))
    maximum = max(initial, float(max_delay_seconds))
    return min(maximum, initial * (2 ** max(0, retry_index - 1)))


def embed_text_batch_with_retries(
    client: EmbeddingClient,
    texts: list[str],
    *,
    retry_attempts: int,
    retry_initial_delay_seconds: float,
    retry_max_delay_seconds: float,
    retry_sleep: Callable[[float], None],
) -> list[list[float]]:
    attempts = max(1, int(retry_attempts))
    for attempt in range(1, attempts + 1):
        try:
            return client.embed_texts(texts)
        except Exception as exc:
            if attempt >= attempts or not is_retryable_embedding_error(exc):
                raise
            delay = embedding_retry_delay(
                attempt,
                initial_delay_seconds=retry_initial_delay_seconds,
                max_delay_seconds=retry_max_delay_seconds,
            )
            print(
                (
                    "Embedding request hit a retryable rate/quota error "
                    f"({attempt}/{attempts}); retrying in {delay:.1f}s."
                ),
                file=sys.stderr,
            )
            retry_sleep(delay)
    raise RuntimeError("Embedding retry loop exited unexpectedly.")


def embed_texts_with_cache(
    texts: list[str],
    *,
    cache_path: Path,
    client: EmbeddingClient | None,
    model: str,
    task_type: str,
    output_dimensionality: int,
    batch_size: int,
    progress_callback: Callable[[int], None] | None = None,
    retry_attempts: int = DEFAULT_EMBEDDING_RETRY_ATTEMPTS,
    retry_initial_delay_seconds: float = DEFAULT_EMBEDDING_RETRY_INITIAL_DELAY_SECONDS,
    retry_max_delay_seconds: float = DEFAULT_EMBEDDING_RETRY_MAX_DELAY_SECONDS,
    retry_sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[list[float]], CacheStats]:
    cache = load_embedding_cache(cache_path)
    keys = [
        embedding_cache_key(
            text,
            model=model,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )
        for text in texts
    ]
    cache_hits = sum(1 for key in keys if key in cache)
    if cache_hits and progress_callback is not None:
        progress_callback(cache_hits)
    missing: list[tuple[str, str]] = []
    seen_missing: set[str] = set()
    for key, text in zip(keys, texts):
        if key in cache or key in seen_missing:
            continue
        seen_missing.add(key)
        missing.append((key, text))

    if missing and client is None:
        raise ValueError(
            f"{len(missing)} embeddings are missing from {cache_path}; provide a Gemini API key "
            "or a complete cache."
        )

    api_batches = 0
    for batch in batched(missing, batch_size):
        batch_keys = [key for key, _ in batch]
        batch_texts = [text for _, text in batch]
        assert client is not None
        vectors = embed_text_batch_with_retries(
            client,
            batch_texts,
            retry_attempts=retry_attempts,
            retry_initial_delay_seconds=retry_initial_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
            retry_sleep=retry_sleep,
        )
        if len(vectors) != len(batch_texts):
            raise RuntimeError("Embedding client returned the wrong number of vectors.")
        api_batches += 1
        cache_entries: list[dict[str, Any]] = []
        for key, text, vector in zip(batch_keys, batch_texts, vectors):
            embedding = [float(value) for value in vector]
            cache[key] = embedding
            cache_entries.append(
                {
                    "cache_key": key,
                    "model": model_resource_name(model),
                    "task_type": task_type,
                    "output_dimensionality": output_dimensionality,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text,
                    "embedding": embedding,
                }
            )
        append_embedding_cache_entries(cache_path, cache_entries)
        if progress_callback is not None:
            batch_key_set = set(batch_keys)
            progress_callback(sum(1 for key in keys if key in batch_key_set))

    return [cache[key] for key in keys], CacheStats(
        cache_hits=cache_hits,
        api_misses=len(missing),
        api_batches=api_batches,
    )


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, size)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def gemini_api_key(cli_value: str | None) -> str | None:
    return cli_value or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "features"


def positive_int(value: str) -> int:
    parsed = safe_int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = safe_int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class FeatureItem:
    row_index: int
    feature_key: str
    label: str
    category: str
    paper_count: int
    mention_count: int
    embedding_text: str
    raw_row: dict[str, Any]


def parse_jsonish_list(value: Any) -> list[str]:
    output: list[str] = []

    def add(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                add(nested)
            return
        text = re.sub(r"\s+", " ", str(item)).strip()
        if text:
            output.append(text)

    if value is None:
        return output
    if isinstance(value, (list, tuple, set)):
        add(value)
        return output
    if not isinstance(value, str):
        add(value)
        return output

    text = value.strip()
    if not text:
        return output
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            add(parsed)
            return output
        if isinstance(parsed, dict):
            add(parsed.values())
            return output

    for separator in ("|", ";"):
        if separator in text:
            for part in text.split(separator):
                add(part)
            return output
    add(text)
    return output


def row_field_value(row: dict[str, Any], field: str) -> Any:
    if field in row:
        return row.get(field)
    if "." not in field:
        return None

    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            value = None
            break
        value = value.get(part)
    if value not in (None, ""):
        return value

    # Annotated CSV exports flatten dotted annotation fields with underscores.
    return row.get(field.replace(".", "_"))


def first_text_value(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        values = parse_jsonish_list(row_field_value(row, field))
        if values:
            return values[0]
    return ""
