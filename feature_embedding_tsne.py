#!/usr/bin/env python3
"""Visualize feature-set Gemini embeddings with t-SNE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TASK_TYPE = "CLUSTERING"
DEFAULT_OUTPUT_DIMENSIONALITY = 768
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
DEFAULT_RANDOM_STATE = 42
DEFAULT_ANNOTATE_TOP = 80


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


def parse_text_fields(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        fields = [field.strip() for field in value.split(",")]
    else:
        fields = [str(field).strip() for field in value]
    parsed = [field for field in fields if field]
    if not parsed:
        raise ValueError("At least one embedding text field is required.")
    return parsed


def build_embedding_text(row: dict[str, Any], text_fields: Iterable[str]) -> str:
    sections: list[str] = []
    seen_by_field: set[tuple[str, str]] = set()
    for field in text_fields:
        values = []
        for value in parse_jsonish_list(row_field_value(row, field)):
            normalized = value.casefold()
            key = (field, normalized)
            if key in seen_by_field:
                continue
            seen_by_field.add(key)
            values.append(value)
        if values:
            sections.append(f"{human_field_name(field)}: {'; '.join(values)}")
    return "\n".join(sections).strip()


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


def human_field_name(field: str) -> str:
    return field.replace(".", " ").replace("_", " ").strip().title()


def first_field_value(row: dict[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        value = row_field_value(row, field)
        if value not in (None, ""):
            return value
    return None


def first_text_value(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        values = parse_jsonish_list(row_field_value(row, field))
        if values:
            return values[0]
    return ""


def load_feature_items(path: Path, text_fields: Iterable[str]) -> list[FeatureItem]:
    rows = read_feature_rows(path)
    fields = parse_text_fields(text_fields)
    items: list[FeatureItem] = []
    for index, row in enumerate(rows, start=1):
        embedding_text = build_embedding_text(row, fields)
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


def embed_texts_with_cache(
    texts: list[str],
    *,
    cache_path: Path,
    client: EmbeddingClient | None,
    model: str,
    task_type: str,
    output_dimensionality: int,
    batch_size: int,
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
    cache_entries: list[dict[str, Any]] = []
    for batch in batched(missing, batch_size):
        batch_keys = [key for key, _ in batch]
        batch_texts = [text for _, text in batch]
        assert client is not None
        vectors = client.embed_texts(batch_texts)
        if len(vectors) != len(batch_texts):
            raise RuntimeError("Embedding client returned the wrong number of vectors.")
        api_batches += 1
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
    return [cache[key] for key in keys], CacheStats(
        cache_hits=cache_hits,
        api_misses=len(missing),
        api_batches=api_batches,
    )


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, size)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def choose_perplexity(n_samples: int, requested: float | None = None) -> float:
    if n_samples < 3:
        raise ValueError("t-SNE requires at least 3 usable features.")
    if requested is not None:
        if requested <= 0:
            raise ValueError("--perplexity must be greater than 0.")
        if requested >= n_samples:
            raise ValueError("--perplexity must be less than the number of usable features.")
        return requested
    return min(30.0, max(2.0, (n_samples - 1) / 3.0))


def tsne_coordinates(
    embeddings: list[list[float]],
    *,
    perplexity: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[tuple[float, float]]:
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError(
            "NumPy and scikit-learn are required for t-SNE. Install project dependencies with `uv sync`."
        ) from exc

    if len(embeddings) < 3:
        raise ValueError("t-SNE requires at least 3 usable features.")
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Embeddings must be a non-empty 2D matrix.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)

    max_pca_components = min(matrix.shape[0], matrix.shape[1])
    if max_pca_components > 50:
        matrix = PCA(n_components=50, random_state=random_state).fit_transform(matrix)

    selected_perplexity = choose_perplexity(matrix.shape[0], perplexity)
    coordinates = TSNE(
        n_components=2,
        perplexity=selected_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(matrix)
    return [(float(x), float(y)) for x, y in coordinates]


def write_coordinates_csv(
    path: Path,
    items: list[FeatureItem],
    coordinates: list[tuple[float, float]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index",
                "feature_key",
                "feature_name",
                "category",
                "paper_count",
                "mention_count",
                "x",
                "y",
                "embedding_text",
            ],
        )
        writer.writeheader()
        for item, (x, y) in zip(items, coordinates):
            writer.writerow(
                {
                    "row_index": item.row_index,
                    "feature_key": item.feature_key,
                    "feature_name": item.label,
                    "category": item.category,
                    "paper_count": item.paper_count,
                    "mention_count": item.mention_count,
                    "x": f"{x:.8f}",
                    "y": f"{y:.8f}",
                    "embedding_text": item.embedding_text,
                }
            )
    return path


def write_tsne_plot(
    path: Path,
    items: list[FeatureItem],
    coordinates: list[tuple[float, float]],
    *,
    title: str,
    annotate_top: int,
    dpi: int,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required to write the t-SNE plot.") from exc

    setup_plot_style(plt)
    categories = sorted({item.category for item in items}, key=lambda value: value.casefold())
    color_map = plt.get_cmap("tab20")
    colors_by_category = {
        category: color_map(index % color_map.N)
        for index, category in enumerate(categories)
    }

    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    sizes = point_sizes([item.paper_count for item in items])

    fig, ax = plt.subplots(figsize=(11.0, 8.2))
    for category in categories:
        indices = [index for index, item in enumerate(items) if item.category == category]
        ax.scatter(
            [xs[index] for index in indices],
            [ys[index] for index in indices],
            s=[sizes[index] for index in indices],
            c=[colors_by_category[category]],
            label=category,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.6,
        )

    for index in annotation_indices(items, annotate_top):
        ax.annotate(
            shorten_label(items[index].label),
            (xs[index], ys[index]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.2,
            color="#1f2933",
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.22, linewidth=0.8)
    if len(categories) <= 16:
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def write_interactive_html(
    path: Path,
    items: list[FeatureItem],
    coordinates: list[tuple[float, float]],
    *,
    title: str,
    annotate_top: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_interactive_html(items, coordinates, title=title, annotate_top=annotate_top),
        encoding="utf-8",
    )
    return path


def build_interactive_html(
    items: list[FeatureItem],
    coordinates: list[tuple[float, float]],
    *,
    title: str,
    annotate_top: int,
) -> str:
    categories = sorted({item.category for item in items}, key=lambda value: value.casefold())
    colors_by_category = category_colors(categories)
    payload = {
        "title": title,
        "annotateTop": annotate_top,
        "categories": categories,
        "colors": colors_by_category,
        "points": [
            {
                "row_index": item.row_index,
                "feature_key": item.feature_key,
                "label": item.label,
                "category": item.category,
                "paper_count": item.paper_count,
                "mention_count": item.mention_count,
                "short_definition": feature_short_definition(item.raw_row),
                "x": x,
                "y": y,
                "embedding_text": item.embedding_text,
            }
            for item, (x, y) in zip(items, coordinates)
        ],
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #64748b;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-2: #7c3aed;
      --shadow: 0 12px 32px rgba(15, 23, 42, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(260px, 320px) minmax(0, 1fr);
      min-height: 100vh;
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow: auto;
    }}
    main {{
      min-width: 0;
      padding: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 19px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .control-group {{
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }}
    label {{
      display: block;
      margin-bottom: 7px;
      font-size: 12px;
      font-weight: 700;
      color: #334155;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    input[type="search"],
    select {{
      width: 100%;
      min-height: 36px;
      border: 1px solid #cbd5e1;
      border-radius: 7px;
      padding: 7px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    button {{
      min-height: 34px;
      border: 1px solid #b7c2d1;
      border-radius: 7px;
      padding: 6px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); }}
    .category-list {{
      display: grid;
      gap: 7px;
      margin-top: 8px;
    }}
    .category-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      border: 1px solid rgba(15, 23, 42, 0.2);
      flex: 0 0 auto;
    }}
    .chart-wrap {{
      position: relative;
      height: calc(100vh - 36px);
      min-height: 560px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .chart-head {{
      position: absolute;
      top: 12px;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      pointer-events: none;
      z-index: 2;
    }}
    .metric {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid rgba(216, 222, 232, 0.9);
      border-radius: 7px;
      padding: 7px 9px;
      color: var(--muted);
      font-size: 12px;
      backdrop-filter: blur(8px);
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
      touch-action: none;
    }}
    svg.dragging {{ cursor: grabbing; }}
    .axis-label {{
      fill: #94a3b8;
      font-size: 12px;
    }}
    .point {{
      stroke: #fff;
      stroke-width: 1.3;
      cursor: pointer;
      opacity: 0.82;
      transition: opacity 120ms ease, stroke-width 120ms ease;
    }}
    .point:hover,
    .point.selected {{
      opacity: 1;
      stroke: #111827;
      stroke-width: 2.2;
    }}
    .label {{
      fill: #243447;
      font-size: 11px;
      paint-order: stroke;
      stroke: rgba(255, 255, 255, 0.92);
      stroke-width: 4px;
      stroke-linejoin: round;
      pointer-events: none;
    }}
    .tooltip {{
      position: absolute;
      max-width: 340px;
      display: none;
      padding: 10px 12px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: var(--shadow);
      font-size: 13px;
      line-height: 1.35;
      pointer-events: none;
      z-index: 5;
    }}
    .tooltip strong {{ display: block; margin-bottom: 4px; }}
    .detail {{
      margin-top: 10px;
      min-height: 98px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
      font-size: 13px;
      line-height: 1.42;
    }}
    .detail-title {{
      margin-bottom: 5px;
      font-weight: 700;
    }}
    .detail-text {{
      display: -webkit-box;
      -webkit-line-clamp: 8;
      -webkit-box-orient: vertical;
      overflow: hidden;
      color: #475569;
      white-space: pre-wrap;
    }}
    @media (max-width: 860px) {{
      .app {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .chart-wrap {{ height: 72vh; min-height: 480px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>{escaped_title}</h1>
      <div class="subtle" id="summary"></div>
      <div class="control-group">
        <label for="feature-search">Search</label>
        <input id="feature-search" type="search" placeholder="Feature, category, definition">
      </div>
      <div class="control-group">
        <label for="min-paper">Minimum Papers <span id="min-paper-value">0</span></label>
        <input id="min-paper" type="range" min="0" value="0" step="1">
      </div>
      <div class="control-group">
        <label for="label-mode">Labels</label>
        <select id="label-mode">
          <option value="top">Top supported</option>
          <option value="hover">Hovered or selected</option>
          <option value="all">All visible</option>
          <option value="none">None</option>
        </select>
      </div>
      <div class="control-group">
        <label>Categories</label>
        <div class="category-list" id="category-list"></div>
      </div>
      <div class="control-group">
        <button type="button" id="reset-view">Reset View</button>
      </div>
      <div class="detail" id="detail">
        <div class="detail-title">No feature selected</div>
        <div class="detail-text">Click a point to pin details here.</div>
      </div>
    </aside>
    <main>
      <div class="chart-wrap">
        <div class="chart-head">
          <div class="metric" id="visible-count"></div>
          <div class="metric">Wheel zoom, drag pan</div>
        </div>
        <svg id="plot" viewBox="0 0 1100 720" role="img" aria-label="{html.escape(title)} interactive t-SNE plot">
          <g id="grid-layer"></g>
          <g id="point-layer"></g>
          <g id="label-layer"></g>
        </svg>
        <div class="tooltip" id="tooltip"></div>
      </div>
    </main>
  </div>
  <script>
    const payload = {data_json};
    const svg = document.getElementById("plot");
    const gridLayer = document.getElementById("grid-layer");
    const pointLayer = document.getElementById("point-layer");
    const labelLayer = document.getElementById("label-layer");
    const tooltip = document.getElementById("tooltip");
    const searchInput = document.getElementById("feature-search");
    const minPaper = document.getElementById("min-paper");
    const minPaperValue = document.getElementById("min-paper-value");
    const labelMode = document.getElementById("label-mode");
    const categoryList = document.getElementById("category-list");
    const summary = document.getElementById("summary");
    const visibleCount = document.getElementById("visible-count");
    const detail = document.getElementById("detail");
    const resetView = document.getElementById("reset-view");
    const ns = "http://www.w3.org/2000/svg";
    const width = 1100;
    const height = 720;
    const pad = 72;
    const state = {{
      enabled: new Set(payload.categories),
      query: "",
      minPaper: 0,
      labelMode: "top",
      zoom: 1,
      panX: 0,
      panY: 0,
      hoverIndex: null,
      selectedIndex: null,
      dragging: false,
      dragX: 0,
      dragY: 0
    }};
    const maxPaper = Math.max(0, ...payload.points.map((d) => d.paper_count || 0));
    const bounds = computeBounds(payload.points);
    minPaper.max = String(maxPaper);
    summary.textContent = `${{payload.points.length}} features, ${{payload.categories.length}} categories`;

    function computeBounds(points) {{
      const xs = points.map((d) => d.x);
      const ys = points.map((d) => d.y);
      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);
      const yMin = Math.min(...ys);
      const yMax = Math.max(...ys);
      return {{
        xMin,
        xMax: xMax === xMin ? xMin + 1 : xMax,
        yMin,
        yMax: yMax === yMin ? yMin + 1 : yMax
      }};
    }}

    function project(d) {{
      const baseX = pad + ((d.x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * (width - pad * 2);
      const baseY = height - pad - ((d.y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * (height - pad * 2);
      return {{
        x: (baseX - width / 2) * state.zoom + width / 2 + state.panX,
        y: (baseY - height / 2) * state.zoom + height / 2 + state.panY
      }};
    }}

    function visiblePoints() {{
      const q = state.query.trim().toLowerCase();
      return payload.points.filter((d) => {{
        if (!state.enabled.has(d.category)) return false;
        if ((d.paper_count || 0) < state.minPaper) return false;
        if (!q) return true;
        return [d.label, d.category, d.feature_key, d.embedding_text].join(" ").toLowerCase().includes(q);
      }});
    }}

    function pointRadius(d) {{
      if (maxPaper <= 0) return 5;
      return 4 + Math.sqrt(Math.max(0, d.paper_count || 0)) / Math.sqrt(maxPaper) * 8;
    }}

    function topIndexSet(points) {{
      return new Set([...points]
        .sort((a, b) => (b.paper_count - a.paper_count) || (b.mention_count - a.mention_count) || a.label.localeCompare(b.label))
        .slice(0, payload.annotateTop)
        .map((d) => d.row_index));
    }}

    function renderCategories() {{
      categoryList.innerHTML = "";
      payload.categories.forEach((category) => {{
        const id = `cat-${{category.replace(/[^a-z0-9]+/gi, "-")}}`;
        const row = document.createElement("div");
        row.className = "category-item";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.id = id;
        input.checked = state.enabled.has(category);
        input.addEventListener("change", () => {{
          if (input.checked) state.enabled.add(category);
          else state.enabled.delete(category);
          render();
        }});
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = payload.colors[category];
        const label = document.createElement("label");
        label.htmlFor = id;
        label.style.display = "inline";
        label.style.margin = "0";
        label.style.textTransform = "none";
        label.style.letterSpacing = "0";
        label.style.fontWeight = "500";
        label.textContent = category;
        row.append(input, swatch, label);
        categoryList.append(row);
      }});
    }}

    function renderGrid() {{
      gridLayer.innerHTML = "";
      for (let i = 0; i <= 4; i += 1) {{
        const x = pad + i * (width - pad * 2) / 4;
        const y = pad + i * (height - pad * 2) / 4;
        addLine(gridLayer, x, pad, x, height - pad);
        addLine(gridLayer, pad, y, width - pad, y);
      }}
      addText(gridLayer, width / 2, height - 22, "t-SNE 1", "axis-label", "middle");
      addText(gridLayer, 22, height / 2, "t-SNE 2", "axis-label", "middle", -90);
    }}

    function addLine(parent, x1, y1, x2, y2) {{
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
      line.setAttribute("stroke", "#e5eaf1");
      line.setAttribute("stroke-width", "1");
      parent.append(line);
    }}

    function addText(parent, x, y, text, className, anchor = "start", rotate = 0) {{
      const node = document.createElementNS(ns, "text");
      node.setAttribute("x", x);
      node.setAttribute("y", y);
      node.setAttribute("class", className);
      node.setAttribute("text-anchor", anchor);
      if (rotate) node.setAttribute("transform", `rotate(${{rotate}} ${{x}} ${{y}})`);
      node.textContent = text;
      parent.append(node);
      return node;
    }}

    function render() {{
      const points = visiblePoints();
      const top = topIndexSet(points);
      pointLayer.innerHTML = "";
      labelLayer.innerHTML = "";
      visibleCount.textContent = `${{points.length}} visible`;
      minPaperValue.textContent = String(state.minPaper);
      points.forEach((d) => {{
        const p = project(d);
        const circle = document.createElementNS(ns, "circle");
        circle.setAttribute("class", `point${{state.selectedIndex === d.row_index ? " selected" : ""}}`);
        circle.setAttribute("cx", p.x);
        circle.setAttribute("cy", p.y);
        circle.setAttribute("r", pointRadius(d));
        circle.setAttribute("fill", payload.colors[d.category] || "#64748b");
        circle.addEventListener("mouseenter", (event) => {{
          state.hoverIndex = d.row_index;
          showTooltip(event, d);
          renderLabels(points, top);
        }});
        circle.addEventListener("mousemove", (event) => showTooltip(event, d));
        circle.addEventListener("mouseleave", () => {{
          state.hoverIndex = null;
          tooltip.style.display = "none";
          renderLabels(points, top);
        }});
        circle.addEventListener("click", () => {{
          state.selectedIndex = d.row_index;
          updateDetail(d);
          render();
        }});
        pointLayer.append(circle);
      }});
      renderLabels(points, top);
    }}

    function renderLabels(points, top) {{
      labelLayer.innerHTML = "";
      points.forEach((d) => {{
        const show = state.labelMode === "all"
          || (state.labelMode === "top" && top.has(d.row_index))
          || (state.labelMode === "hover" && (state.hoverIndex === d.row_index || state.selectedIndex === d.row_index));
        if (!show || state.labelMode === "none") return;
        const p = project(d);
        addText(labelLayer, p.x + pointRadius(d) + 4, p.y - 5, shorten(d.label, 36), "label");
      }});
    }}

    function showTooltip(event, d) {{
      tooltip.style.display = "block";
      const shortDefinition = d.short_definition
        ? `<div style="margin-top:6px">${{escapeHtml(d.short_definition)}}</div>`
        : "";
      tooltip.innerHTML = `<strong>${{escapeHtml(d.label)}}</strong>${{escapeHtml(d.category)}}<br>${{d.paper_count}} papers / ${{d.mention_count}} mentions${{shortDefinition}}`;
      const rect = event.currentTarget.ownerSVGElement.getBoundingClientRect();
      const wrap = event.currentTarget.ownerSVGElement.parentElement.getBoundingClientRect();
      tooltip.style.left = `${{event.clientX - wrap.left + 14}}px`;
      tooltip.style.top = `${{event.clientY - wrap.top + 14}}px`;
    }}

    function updateDetail(d) {{
      detail.innerHTML = `<div class="detail-title">${{escapeHtml(d.label)}}</div>
        <div class="subtle">${{escapeHtml(d.category)}} | ${{d.paper_count}} papers / ${{d.mention_count}} mentions</div>
        <div class="detail-text">${{escapeHtml(d.embedding_text)}}</div>`;
    }}

    function shorten(value, maxChars) {{
      return value.length <= maxChars ? value : `${{value.slice(0, maxChars - 3)}}...`;
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, (ch) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\\"": "&quot;",
        "'": "&#39;"
      }}[ch]));
    }}

    searchInput.addEventListener("input", () => {{
      state.query = searchInput.value;
      render();
    }});
    minPaper.addEventListener("input", () => {{
      state.minPaper = Number(minPaper.value || 0);
      render();
    }});
    labelMode.addEventListener("change", () => {{
      state.labelMode = labelMode.value;
      render();
    }});
    resetView.addEventListener("click", () => {{
      state.enabled = new Set(payload.categories);
      state.query = "";
      state.minPaper = 0;
      state.labelMode = "top";
      state.zoom = 1;
      state.panX = 0;
      state.panY = 0;
      state.hoverIndex = null;
      state.selectedIndex = null;
      searchInput.value = "";
      minPaper.value = "0";
      labelMode.value = "top";
      renderCategories();
      render();
    }});
    svg.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.12 : 0.89;
      state.zoom = Math.min(10, Math.max(0.35, state.zoom * factor));
      render();
    }}, {{ passive: false }});
    svg.addEventListener("pointerdown", (event) => {{
      state.dragging = true;
      state.dragX = event.clientX;
      state.dragY = event.clientY;
      svg.classList.add("dragging");
      svg.setPointerCapture(event.pointerId);
    }});
    svg.addEventListener("pointermove", (event) => {{
      if (!state.dragging) return;
      state.panX += event.clientX - state.dragX;
      state.panY += event.clientY - state.dragY;
      state.dragX = event.clientX;
      state.dragY = event.clientY;
      render();
    }});
    svg.addEventListener("pointerup", (event) => {{
      state.dragging = false;
      svg.classList.remove("dragging");
      svg.releasePointerCapture(event.pointerId);
    }});
    svg.addEventListener("pointerleave", () => {{
      state.dragging = false;
      svg.classList.remove("dragging");
    }});

    renderCategories();
    renderGrid();
    render();
  </script>
</body>
</html>
"""


def category_colors(categories: list[str]) -> dict[str, str]:
    palette = [
        "#0f766e",
        "#7c3aed",
        "#dc2626",
        "#2563eb",
        "#ca8a04",
        "#16a34a",
        "#c2410c",
        "#0891b2",
        "#be185d",
        "#4f46e5",
        "#65a30d",
        "#9333ea",
        "#b45309",
        "#0284c7",
        "#475569",
        "#db2777",
    ]
    return {
        category: palette[index % len(palette)]
        for index, category in enumerate(categories)
    }


def feature_short_definition(row: dict[str, Any]) -> str:
    return first_text_value(
        row,
        (
            "annotation.short_definition",
            "short_definition",
            "annotation.formal_definition",
            "definition",
            "definitions",
        ),
    )


def setup_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
        }
    )


def point_sizes(counts: list[int]) -> list[float]:
    roots = [math.sqrt(max(0, count)) for count in counts]
    max_root = max(roots or [0.0])
    if max_root <= 0:
        return [42.0 for _ in roots]
    return [36.0 + (root / max_root) * 150.0 for root in roots]


def annotation_indices(items: list[FeatureItem], annotate_top: int) -> list[int]:
    if annotate_top <= 0:
        return []
    ordered = sorted(
        range(len(items)),
        key=lambda index: (
            -items[index].paper_count,
            -items[index].mention_count,
            items[index].label.casefold(),
        ),
    )
    return ordered[:annotate_top]


def shorten_label(value: str, max_chars: int = 34) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 1].rstrip()}..."


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed a feature set with Gemini and visualize it with t-SNE.",
    )
    parser.add_argument("features", type=Path, help="Feature CSV, JSONL, or JSON file.")
    parser.add_argument("--output-dir", type=Path, help="Directory for PNG/CSV outputs.")
    parser.add_argument("--prefix", help="Output filename prefix. Defaults to the input stem.")
    parser.add_argument("--cache", type=Path, help="Embedding cache JSONL path.")
    parser.add_argument("--html", action="store_true", help="Also write a self-contained interactive HTML plot.")
    parser.add_argument("--html-path", type=Path, help="Interactive HTML output path. Implies --html.")
    parser.add_argument(
        "--text-fields",
        default=",".join(DEFAULT_TEXT_FIELDS),
        help="Comma-separated row fields used to build embedding text.",
    )
    parser.add_argument("--gemini-api-key", help="Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY.")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini embedding model.")
    parser.add_argument("--base-url", default=DEFAULT_GEMINI_BASE_URL, help="Gemini API base URL.")
    parser.add_argument("--task-type", default=DEFAULT_TASK_TYPE, help="Gemini embedding task type.")
    parser.add_argument(
        "--output-dimensionality",
        type=positive_int,
        default=DEFAULT_OUTPUT_DIMENSIONALITY,
        help="Gemini embedding output dimensionality.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=50, help="Gemini embedding batch size.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Gemini request timeout in seconds.")
    parser.add_argument("--perplexity", type=float, help="t-SNE perplexity. Defaults to an automatic valid value.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="t-SNE random seed.")
    parser.add_argument(
        "--annotate-top",
        type=nonnegative_int,
        default=DEFAULT_ANNOTATE_TOP,
        help="Number of highest-support features to label on the plot.",
    )
    parser.add_argument("--dpi", type=positive_int, default=220, help="PNG output DPI.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.features.exists():
            raise FileNotFoundError(f"Feature file does not exist: {args.features}")
        if args.timeout <= 0:
            raise ValueError("--timeout must be greater than 0.")

        text_fields = parse_text_fields(args.text_fields)
        items = load_feature_items(args.features, text_fields)
        if len(items) < 3:
            raise ValueError(f"Found {len(items)} usable features; t-SNE requires at least 3.")

        output_dir = args.output_dir or args.features.parent / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = sanitize_filename_part(args.prefix or args.features.stem)
        cache_path = args.cache or output_dir / f"{prefix}_gemini_embedding_cache.jsonl"

        api_key = gemini_api_key(args.gemini_api_key)
        client: EmbeddingClient | None = None
        if api_key:
            client = GeminiEmbeddingClient(
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                task_type=args.task_type,
                output_dimensionality=args.output_dimensionality,
                timeout=args.timeout,
            )

        embeddings, stats = embed_texts_with_cache(
            [item.embedding_text for item in items],
            cache_path=cache_path,
            client=client,
            model=args.model,
            task_type=args.task_type,
            output_dimensionality=args.output_dimensionality,
            batch_size=args.batch_size,
        )
        coordinates = tsne_coordinates(
            embeddings,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )

        csv_path = output_dir / f"{prefix}_gemini_embedding_tsne.csv"
        png_path = output_dir / f"{prefix}_gemini_embedding_tsne.png"
        html_path = args.html_path or output_dir / f"{prefix}_gemini_embedding_tsne.html"
        write_coordinates_csv(csv_path, items, coordinates)
        write_tsne_plot(
            png_path,
            items,
            coordinates,
            title=f"{args.features.stem} Gemini embedding t-SNE",
            annotate_top=args.annotate_top,
            dpi=args.dpi,
        )
        if args.html or args.html_path:
            write_interactive_html(
                html_path,
                items,
                coordinates,
                title=f"{args.features.stem} Gemini embedding t-SNE",
                annotate_top=args.annotate_top,
            )
    except Exception as exc:
        print(f"feature_embedding_tsne: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(items)} usable features from {args.features}")
    print(f"Embedding cache: {stats.cache_hits} hits, {stats.api_misses} misses, {stats.api_batches} API batches")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    if args.html or args.html_path:
        print(f"Wrote {html_path}")
    print(f"Cache path: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
