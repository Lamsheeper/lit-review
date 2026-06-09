#!/usr/bin/env python3
"""Merge feature inventories using Gemini embeddings and cosine similarity."""

from __future__ import annotations

import argparse
import csv
import copy
import json
from decimal import Decimal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from feature_embedding_tsne import (
    DEFAULT_ANNOTATE_TOP,
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TASK_TYPE,
    DEFAULT_TEXT_FIELDS,
    EmbeddingClient,
    FeatureItem,
    GeminiEmbeddingClient,
    embed_texts_with_cache,
    gemini_api_key,
    load_feature_items,
    nonnegative_int,
    parse_jsonish_list,
    parse_text_fields,
    positive_int,
    row_field_value,
    safe_int,
    sanitize_filename_part,
    tsne_coordinates,
    write_interactive_html,
)


PAIR_FIELDNAMES = [
    "similarity",
    "feature_key_1",
    "feature_name_1",
    "category_1",
    "row_index_1",
    "paper_count_1",
    "mention_count_1",
    "feature_key_2",
    "feature_name_2",
    "category_2",
    "row_index_2",
    "paper_count_2",
    "mention_count_2",
]

GROUP_FIELDNAMES = [
    "group_id",
    "member_count",
    "canonical_feature_key",
    "canonical_feature_name",
    "categories",
    "member_feature_keys",
    "member_feature_names",
    "edge_count",
    "max_similarity",
    "min_similarity",
    "avg_similarity",
]

SWEEP_FIELDNAMES = [
    "threshold",
    "input_features",
    "similarity_pairs",
    "merge_groups",
    "merged_features",
    "merged_away_features",
]

CATEGORY_SWEEP_FIELDNAMES = [
    "threshold",
    "category",
    "input_features",
    "merge_groups",
    "resulting_features",
    "merged_away_features",
]

QUALITATIVE_FIELDNAMES = [
    "category",
    "rank",
    "merged_feature_key",
    "merged_feature_name",
    "merged_feature_category",
    "merged_feature_paper_count",
    "merged_feature_mention_count",
    "merged_into_feature_key",
    "merged_into_feature_name",
    "merged_into_category",
    "group_id",
    "group_member_count",
    "direct_similarity_to_canonical",
    "max_group_similarity",
    "min_group_similarity",
    "avg_group_similarity",
]

MERGED_FIELD_PRIORITY = [
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
    "source_feature_names",
    "source_normalized_feature_names",
    "evidence_types",
    "annotation",
    "embedding_merge",
]

AGGREGATE_LIST_FIELDS = [
    "parent_categories",
    "definitions",
    "synonyms",
    "examples",
    "source_paper_ids",
    "source_titles",
    "source_feature_names",
    "source_normalized_feature_names",
    "evidence_types",
]

AGGREGATE_ANNOTATION_LIST_FIELDS = [
    "annotation.synonyms",
    "annotation.examples",
]


@dataclass(frozen=True)
class SimilarityPair:
    left_index: int
    right_index: int
    similarity: float


@dataclass(frozen=True)
class MergeGroup:
    group_id: str
    member_indices: list[int]
    canonical_index: int
    pairs: list[SimilarityPair]


@dataclass(frozen=True)
class PreparedEmbeddings:
    text_fields: list[str]
    items: list[FeatureItem]
    embeddings: list[list[float]]
    cache_stats: Any


def threshold_value(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def threshold_range(start: float, end: float, step: float = 0.01) -> list[float]:
    validate_threshold_number(start, "threshold start")
    validate_threshold_number(end, "threshold end")
    validate_threshold_number(step, "threshold step")
    if start > end:
        raise ValueError("threshold start must be less than or equal to threshold end.")

    current = Decimal(str(start))
    end_value = Decimal(str(end))
    step_value = Decimal(str(step))
    thresholds: list[float] = []
    while current <= end_value:
        rounded = round(float(current), 2)
        if not thresholds or thresholds[-1] != rounded:
            thresholds.append(rounded)
        current += step_value

    rounded_end = round(end, 2)
    if not thresholds or thresholds[-1] != rounded_end:
        thresholds.append(rounded_end)
    return thresholds


def validate_threshold_number(value: float, label: str = "threshold") -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{label} must be greater than 0 and at most 1.")


def normalized_matrix(embeddings: list[list[float]]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("NumPy is required for embedding merge similarity search.") from exc

    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Embeddings must be a non-empty 2D matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def find_similarity_pairs(
    items: list[FeatureItem],
    embeddings: list[list[float]],
    *,
    threshold: float,
    scope: str = "all",
    block_size: int = 512,
) -> list[SimilarityPair]:
    matrix = normalized_matrix(embeddings)
    return find_similarity_pairs_from_normalized_matrix(
        items,
        matrix,
        threshold=threshold,
        scope=scope,
        block_size=block_size,
    )


def find_similarity_pairs_from_normalized_matrix(
    items: list[FeatureItem],
    matrix: Any,
    *,
    threshold: float,
    scope: str = "all",
    block_size: int = 512,
) -> list[SimilarityPair]:
    validate_threshold_number(threshold)
    if scope not in {"all", "category"}:
        raise ValueError("scope must be 'all' or 'category'.")
    if len(items) != len(matrix):
        raise ValueError("Item and embedding counts must match.")
    if not items:
        return []

    pairs: list[SimilarityPair] = []
    n_items = len(items)

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for embedding merge similarity search.") from exc

    for start in range(0, n_items, max(1, block_size)):
        end = min(n_items, start + max(1, block_size))
        similarities = matrix[start:end] @ matrix.T
        for local_index, row in enumerate(similarities):
            left_index = start + local_index
            right_start = left_index + 1
            if right_start >= n_items:
                continue
            candidate_offsets = np.where(row[right_start:] >= threshold)[0]
            for offset in candidate_offsets:
                right_index = right_start + int(offset)
                if scope == "category" and items[left_index].category != items[right_index].category:
                    continue
                pairs.append(
                    SimilarityPair(
                        left_index=left_index,
                        right_index=right_index,
                        similarity=round(float(row[right_index]), 8),
                    )
                )

    return sorted(
        pairs,
        key=lambda pair: (
            -pair.similarity,
            items[pair.left_index].label.casefold(),
            items[pair.right_index].label.casefold(),
            items[pair.left_index].row_index,
            items[pair.right_index].row_index,
        ),
    )


def pairwise_similarity_histogram(
    items: list[FeatureItem],
    matrix: Any,
    *,
    scope: str,
    bins: int = 80,
    block_size: int = 512,
) -> dict[str, Any]:
    if scope not in {"all", "category"}:
        raise ValueError("scope must be 'all' or 'category'.")
    if len(items) != len(matrix):
        raise ValueError("Item and embedding counts must match.")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for embedding similarity distribution plots.") from exc

    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    counts = np.zeros(bins, dtype=int)
    all_values: list[float] = []
    pair_count = 0
    similarity_sum = 0.0
    min_similarity: float | None = None
    max_similarity: float | None = None
    n_items = len(items)

    for start in range(0, n_items, max(1, block_size)):
        end = min(n_items, start + max(1, block_size))
        similarities = matrix[start:end] @ matrix.T
        for local_index, row in enumerate(similarities):
            left_index = start + local_index
            right_start = left_index + 1
            if right_start >= n_items:
                continue
            values = row[right_start:]
            if scope == "category":
                mask = np.array(
                    [items[left_index].category == items[right_index].category for right_index in range(right_start, n_items)],
                    dtype=bool,
                )
                values = values[mask]
            if values.size == 0:
                continue
            values = np.clip(values.astype(float), -1.0, 1.0)
            hist, _ = np.histogram(values, bins=bin_edges)
            counts += hist
            all_values.extend(float(value) for value in values)
            pair_count += int(values.size)
            similarity_sum += float(values.sum())
            current_min = float(values.min())
            current_max = float(values.max())
            min_similarity = current_min if min_similarity is None else min(min_similarity, current_min)
            max_similarity = current_max if max_similarity is None else max(max_similarity, current_max)

    return {
        "bin_edges": [float(value) for value in bin_edges],
        "counts": [int(value) for value in counts],
        "values": all_values,
        "pair_count": pair_count,
        "min_similarity": round(min_similarity, 8) if min_similarity is not None else None,
        "max_similarity": round(max_similarity, 8) if max_similarity is not None else None,
        "mean_similarity": round(similarity_sum / pair_count, 8) if pair_count else None,
    }


def similarity_histogram_from_values(values: Any, bin_edges: Any) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for embedding similarity distribution plots.") from exc

    clipped = np.clip(np.asarray(values, dtype=float), -1.0, 1.0)
    counts, _ = np.histogram(clipped, bins=bin_edges)
    pair_count = int(clipped.size)
    return {
        "bin_edges": [float(value) for value in bin_edges],
        "counts": [int(value) for value in counts],
        "values": [float(value) for value in clipped],
        "pair_count": pair_count,
        "min_similarity": round(float(clipped.min()), 8) if pair_count else None,
        "max_similarity": round(float(clipped.max()), 8) if pair_count else None,
        "mean_similarity": round(float(clipped.mean()), 8) if pair_count else None,
    }


def pairwise_similarity_histograms_by_category(
    items: list[FeatureItem],
    matrix: Any,
    *,
    bins: int = 80,
) -> dict[str, dict[str, Any]]:
    if len(items) != len(matrix):
        raise ValueError("Item and embedding counts must match.")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for embedding similarity distribution plots.") from exc

    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    categories = sorted(
        {item.category for item in items},
        key=lambda category: (
            -sum(1 for item in items if item.category == category),
            category.casefold(),
        ),
    )
    histograms: dict[str, dict[str, Any]] = {}
    for category in categories:
        indices = [index for index, item in enumerate(items) if item.category == category]
        if len(indices) < 2:
            histogram = similarity_histogram_from_values([], bin_edges)
        else:
            category_matrix = matrix[indices]
            similarities = category_matrix @ category_matrix.T
            upper_indices = np.triu_indices(len(indices), k=1)
            histogram = similarity_histogram_from_values(similarities[upper_indices], bin_edges)
        histogram["feature_count"] = len(indices)
        histograms[category] = histogram
    return histograms


def similarity_distribution_summary(histogram: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_count": histogram["pair_count"],
        "min_similarity": histogram["min_similarity"],
        "max_similarity": histogram["max_similarity"],
        "mean_similarity": histogram["mean_similarity"],
        "bin_count": len(histogram["counts"]),
    }


def similarity_distribution_category_summary(
    histograms: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        category: {
            **similarity_distribution_summary(histogram),
            "feature_count": histogram.get("feature_count", 0),
        }
        for category, histogram in histograms.items()
    }


def category_histogram_order(category_histograms: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        category_histograms,
        key=lambda category: (
            -int(category_histograms[category].get("feature_count") or 0),
            category.casefold(),
        ),
    )


def write_similarity_distribution_plot(
    path: Path,
    histogram: dict[str, Any],
    *,
    title: str,
    thresholds: Iterable[float] = (),
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("Matplotlib is required for embedding similarity distribution plots.") from exc

    edges = histogram["bin_edges"]
    counts = histogram["counts"]
    centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:])]
    widths = [right - left for left, right in zip(edges[:-1], edges[1:])]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(centers, counts, width=widths, align="center", color="#2563eb", alpha=0.78, edgecolor="#ffffff")
    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Pair count")
    ax.set_xlim(0.5, 1.0)
    ax.grid(True, axis="y", alpha=0.28)
    subtitle = (
        f"{histogram['pair_count']} pairs"
        if histogram.get("mean_similarity") is None
        else f"{histogram['pair_count']} pairs | mean {histogram['mean_similarity']:.3f}"
    )
    ax.text(0.01, 0.97, subtitle, transform=ax.transAxes, va="top", color="#475569")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_similarity_boxplot(
    path: Path,
    histogram: dict[str, Any],
    *,
    title: str,
    thresholds: Iterable[float] = (),
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("Matplotlib is required for embedding similarity box plots.") from exc

    values = [float(value) for value in histogram.get("values", [])]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if values:
        ax.boxplot(
            values,
            orientation="horizontal",
            showmeans=True,
            widths=0.42,
            patch_artist=True,
            boxprops={"facecolor": "#bfdbfe", "edgecolor": "#1d4ed8", "linewidth": 1.5},
            medianprops={"color": "#dc2626", "linewidth": 1.8},
            meanprops={
                "marker": "D",
                "markerfacecolor": "#111827",
                "markeredgecolor": "#111827",
                "markersize": 5,
            },
            whiskerprops={"color": "#1d4ed8", "linewidth": 1.2},
            capprops={"color": "#1d4ed8", "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#93c5fd",
                "markeredgecolor": "#2563eb",
                "markersize": 3,
                "alpha": 0.45,
            },
        )
    else:
        ax.text(0.5, 0.5, "No feature pairs available", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.set_yticks([])
    ax.set_xlim(0.5, 1.0)
    ax.grid(True, axis="x", alpha=0.28)
    subtitle = (
        f"{histogram['pair_count']} pairs"
        if histogram.get("mean_similarity") is None
        else f"{histogram['pair_count']} pairs | mean {histogram['mean_similarity']:.3f}"
    )
    ax.text(0.01, 0.95, subtitle, transform=ax.transAxes, va="top", color="#475569")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_similarity_percentile_plot(
    path: Path,
    histogram: dict[str, Any],
    *,
    title: str,
    thresholds: Iterable[float] = (),
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("Matplotlib is required for embedding similarity percentile plots.") from exc

    values = sorted(float(value) for value in histogram.get("values", []))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted_values: list[float] = []
    if values:
        if len(values) == 1:
            percentiles = [100.0]
        else:
            percentiles = [index / (len(values) - 1) * 100.0 for index in range(len(values))]
        plotted = [(percentile, value) for percentile, value in zip(percentiles, values) if percentile >= 50.0]
        if plotted:
            plot_percentiles, plotted_values = zip(*plotted)
            plotted_values = list(plotted_values)
            ax.plot(plot_percentiles, plotted_values, color="#2563eb", linewidth=2.0)
    else:
        ax.text(0.5, 0.5, "No feature pairs available", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Cosine similarity")
    ax.set_xlim(50.0, 100.0)
    if plotted_values:
        y_min = min(plotted_values)
        y_max = max(plotted_values)
        padding = max(0.02, (y_max - y_min) * 0.08)
        ax.set_ylim(max(-1.0, y_min - padding), min(1.0, y_max + padding))
    else:
        ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.28)
    subtitle = (
        f"{histogram['pair_count']} pairs"
        if histogram.get("mean_similarity") is None
        else f"{histogram['pair_count']} pairs | mean {histogram['mean_similarity']:.3f}"
    )
    ax.text(0.01, 0.97, subtitle, transform=ax.transAxes, va="top", color="#475569")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_similarity_distribution_by_category_plot(
    path: Path,
    category_histograms: dict[str, dict[str, Any]],
    *,
    title: str,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("Matplotlib is required for embedding similarity distribution plots.") from exc

    fig, ax = plt.subplots(figsize=(11, 6.5))
    color_map = plt.get_cmap("tab20")
    categories = category_histogram_order(category_histograms)
    plotted = False
    for index, category in enumerate(categories):
        histogram = category_histograms[category]
        if not histogram.get("pair_count"):
            continue
        edges = histogram["bin_edges"]
        counts = histogram["counts"]
        centers = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:])]
        ax.step(
            centers,
            counts,
            where="mid",
            color=color_map(index % color_map.N),
            linewidth=1.8,
            alpha=0.9,
            label=f"{category} ({histogram['pair_count']} pairs)",
        )
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No within-category feature pairs available", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Pair count")
    ax.set_xlim(0.5, 1.0)
    ax.grid(True, axis="y", alpha=0.28)
    if plotted:
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_similarity_boxplot_by_category(
    path: Path,
    category_histograms: dict[str, dict[str, Any]],
    *,
    title: str,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("Matplotlib is required for embedding similarity box plots.") from exc

    categories = [
        category
        for category in category_histogram_order(category_histograms)
        if category_histograms[category].get("values")
    ]
    values = [[float(value) for value in category_histograms[category]["values"]] for category in categories]
    labels = [
        f"{category} ({category_histograms[category]['pair_count']})"
        for category in categories
    ]
    fig_height = max(4.8, min(14.0, 2.0 + len(categories) * 0.48))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    if values:
        boxplot_kwargs = {
            "showmeans": True,
            "widths": 0.52,
            "patch_artist": True,
            "boxprops": {"facecolor": "#bfdbfe", "edgecolor": "#1d4ed8", "linewidth": 1.2},
            "medianprops": {"color": "#dc2626", "linewidth": 1.6},
            "meanprops": {
                "marker": "D",
                "markerfacecolor": "#111827",
                "markeredgecolor": "#111827",
                "markersize": 4,
            },
            "whiskerprops": {"color": "#1d4ed8", "linewidth": 1.0},
            "capprops": {"color": "#1d4ed8", "linewidth": 1.0},
            "flierprops": {
                "marker": "o",
                "markerfacecolor": "#93c5fd",
                "markeredgecolor": "#2563eb",
                "markersize": 2.5,
                "alpha": 0.35,
            },
        }
        try:
            plot = ax.boxplot(
                values,
                orientation="horizontal",
                tick_labels=labels,
                **boxplot_kwargs,
            )
        except TypeError:  # Matplotlib < 3.9 used "labels" here.
            plot = ax.boxplot(
                values,
                orientation="horizontal",
                labels=labels,
                **boxplot_kwargs,
            )
        color_map = plt.get_cmap("tab20")
        for index, patch in enumerate(plot["boxes"]):
            patch.set_facecolor(color_map(index % color_map.N))
            patch.set_alpha(0.35)
    else:
        ax.text(0.5, 0.5, "No within-category feature pairs available", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.set_xlim(0.5, 1.0)
    ax.grid(True, axis="x", alpha=0.28)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def percentile_series(histogram: dict[str, Any], min_percentile: float = 50.0) -> tuple[list[float], list[float]]:
    values = sorted(float(value) for value in histogram.get("values", []))
    if not values:
        return [], []
    if len(values) == 1:
        percentiles = [100.0]
    else:
        percentiles = [index / (len(values) - 1) * 100.0 for index in range(len(values))]
    plotted = [
        (percentile, value)
        for percentile, value in zip(percentiles, values)
        if percentile >= min_percentile
    ]
    if not plotted:
        return [], []
    plot_percentiles, plot_values = zip(*plotted)
    return list(plot_percentiles), list(plot_values)


def write_similarity_percentile_by_category_plot(
    path: Path,
    histogram: dict[str, Any],
    category_histograms: dict[str, dict[str, Any]],
    *,
    title: str,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("Matplotlib is required for embedding similarity percentile plots.") from exc

    fig, ax = plt.subplots(figsize=(11, 6.5))
    all_plotted_values: list[float] = []
    overall_percentiles, overall_values = percentile_series(histogram)
    if overall_percentiles:
        ax.plot(
            overall_percentiles,
            overall_values,
            color="#111827",
            linewidth=2.1,
            label=f"overall ({histogram['pair_count']} pairs)",
        )
        all_plotted_values.extend(overall_values)

    color_map = plt.get_cmap("tab20")
    for index, category in enumerate(category_histogram_order(category_histograms)):
        category_histogram = category_histograms[category]
        percentiles, values = percentile_series(category_histogram)
        if not percentiles:
            continue
        ax.plot(
            percentiles,
            values,
            color=color_map(index % color_map.N),
            linewidth=1.6,
            linestyle=":",
            label=f"{category} ({category_histogram['pair_count']} pairs)",
        )
        all_plotted_values.extend(values)

    if not all_plotted_values:
        ax.text(0.5, 0.5, "No feature pairs available", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Cosine similarity")
    ax.set_xlim(50.0, 100.0)
    if all_plotted_values:
        y_min = min(all_plotted_values)
        y_max = max(all_plotted_values)
        padding = max(0.02, (y_max - y_min) * 0.08)
        ax.set_ylim(max(-1.0, y_min - padding), min(1.0, y_max + padding))
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def build_merge_groups(items: list[FeatureItem], pairs: list[SimilarityPair]) -> list[MergeGroup]:
    dsu = DisjointSet(len(items))
    for pair in pairs:
        dsu.union(pair.left_index, pair.right_index)

    components: dict[int, list[int]] = {}
    for index in range(len(items)):
        components.setdefault(dsu.find(index), []).append(index)

    pair_lookup: dict[frozenset[int], SimilarityPair] = {
        frozenset((pair.left_index, pair.right_index)): pair
        for pair in pairs
    }
    raw_groups: list[tuple[int, list[int], list[SimilarityPair]]] = []
    for member_indices in components.values():
        if len(member_indices) < 2:
            continue
        group_pairs = [
            pair
            for pair in pair_lookup.values()
            if pair.left_index in member_indices and pair.right_index in member_indices
        ]
        canonical_index = choose_canonical_index(items, member_indices)
        ordered_members = sorted(
            member_indices,
            key=lambda index: (0 if index == canonical_index else 1, member_sort_key(items[index])),
        )
        raw_groups.append((canonical_index, ordered_members, sorted_group_pairs(items, group_pairs)))

    raw_groups.sort(key=lambda item: member_sort_key(items[item[0]]))
    return [
        MergeGroup(
            group_id=f"merge_group_{index:04d}",
            canonical_index=canonical_index,
            member_indices=member_indices,
            pairs=group_pairs,
        )
        for index, (canonical_index, member_indices, group_pairs) in enumerate(raw_groups, start=1)
    ]


def choose_canonical_index(items: list[FeatureItem], member_indices: Iterable[int]) -> int:
    return sorted(
        member_indices,
        key=lambda index: (
            -items[index].paper_count,
            -items[index].mention_count,
            len(items[index].label),
            items[index].row_index,
        ),
    )[0]


def member_sort_key(item: FeatureItem) -> tuple[int, int, int, str, int]:
    return (
        -item.paper_count,
        -item.mention_count,
        len(item.label),
        item.label.casefold(),
        item.row_index,
    )


def sorted_group_pairs(items: list[FeatureItem], pairs: list[SimilarityPair]) -> list[SimilarityPair]:
    return sorted(
        pairs,
        key=lambda pair: (
            -pair.similarity,
            items[pair.left_index].label.casefold(),
            items[pair.right_index].label.casefold(),
        ),
    )


def pair_json(pair: SimilarityPair, items: list[FeatureItem]) -> dict[str, Any]:
    left = items[pair.left_index]
    right = items[pair.right_index]
    return {
        "similarity": pair.similarity,
        "feature_key_1": left.feature_key,
        "feature_name_1": left.label,
        "category_1": left.category,
        "row_index_1": left.row_index,
        "paper_count_1": left.paper_count,
        "mention_count_1": left.mention_count,
        "feature_key_2": right.feature_key,
        "feature_name_2": right.label,
        "category_2": right.category,
        "row_index_2": right.row_index,
        "paper_count_2": right.paper_count,
        "mention_count_2": right.mention_count,
    }


def group_json(group: MergeGroup, items: list[FeatureItem], threshold: float, scope: str) -> dict[str, Any]:
    canonical = items[group.canonical_index]
    members = [items[index] for index in group.member_indices]
    pair_rows = [pair_json(pair, items) for pair in group.pairs]
    similarities = [pair.similarity for pair in group.pairs]
    return {
        "group_id": group.group_id,
        "threshold": threshold,
        "scope": scope,
        "canonical_feature_key": canonical.feature_key,
        "canonical_feature_name": canonical.label,
        "canonical_row_index": canonical.row_index,
        "member_count": len(members),
        "member_row_indices": [member.row_index for member in members],
        "member_feature_keys": [member.feature_key for member in members],
        "member_feature_names": [member.label for member in members],
        "member_categories": sorted({member.category for member in members}),
        "similarity_summary": similarity_summary(similarities),
        "pairs": pair_rows,
        "members": [
            {
                "row_index": member.row_index,
                "feature_key": member.feature_key,
                "feature_name": member.label,
                "category": member.category,
                "paper_count": member.paper_count,
                "mention_count": member.mention_count,
            }
            for member in members
        ],
    }


def similarity_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"edge_count": 0, "max": 0.0, "min": 0.0, "avg": 0.0}
    return {
        "edge_count": len(values),
        "max": round(max(values), 8),
        "min": round(min(values), 8),
        "avg": round(sum(values) / len(values), 8),
    }


def group_csv_row(group: MergeGroup, items: list[FeatureItem]) -> dict[str, Any]:
    payload = group_json(group, items, threshold=0.0, scope="")
    summary = payload["similarity_summary"]
    return {
        "group_id": group.group_id,
        "member_count": payload["member_count"],
        "canonical_feature_key": payload["canonical_feature_key"],
        "canonical_feature_name": payload["canonical_feature_name"],
        "categories": " | ".join(payload["member_categories"]),
        "member_feature_keys": " | ".join(payload["member_feature_keys"]),
        "member_feature_names": " | ".join(payload["member_feature_names"]),
        "edge_count": summary["edge_count"],
        "max_similarity": summary["max"],
        "min_similarity": summary["min"],
        "avg_similarity": summary["avg"],
    }


def qualitative_merge_rows(
    items: list[FeatureItem],
    groups: list[MergeGroup],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    if top_n <= 0:
        return []

    rows_by_category: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        canonical = items[group.canonical_index]
        similarity_by_pair = {
            frozenset((pair.left_index, pair.right_index)): pair.similarity
            for pair in group.pairs
        }
        summary = similarity_summary([pair.similarity for pair in group.pairs])
        for member_index in group.member_indices:
            if member_index == group.canonical_index:
                continue
            member = items[member_index]
            direct_similarity = similarity_by_pair.get(frozenset((member_index, group.canonical_index)))
            rows_by_category.setdefault(member.category, []).append(
                {
                    "category": member.category,
                    "rank": 0,
                    "merged_feature_key": member.feature_key,
                    "merged_feature_name": member.label,
                    "merged_feature_category": member.category,
                    "merged_feature_paper_count": member.paper_count,
                    "merged_feature_mention_count": member.mention_count,
                    "merged_into_feature_key": canonical.feature_key,
                    "merged_into_feature_name": canonical.label,
                    "merged_into_category": canonical.category,
                    "group_id": group.group_id,
                    "group_member_count": len(group.member_indices),
                    "direct_similarity_to_canonical": direct_similarity,
                    "max_group_similarity": summary["max"],
                    "min_group_similarity": summary["min"],
                    "avg_group_similarity": summary["avg"],
                }
            )

    output: list[dict[str, Any]] = []
    categories = sorted(
        rows_by_category,
        key=lambda category: (
            -sum(row["merged_feature_mention_count"] for row in rows_by_category[category]),
            category.casefold(),
        ),
    )
    for category in categories:
        ranked = sorted(
            rows_by_category[category],
            key=lambda row: (
                -row["merged_feature_mention_count"],
                -row["merged_feature_paper_count"],
                str(row["merged_feature_name"]).casefold(),
                str(row["group_id"]),
            ),
        )[:top_n]
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            output.append(row)
    return output


def build_merged_features(
    items: list[FeatureItem],
    groups: list[MergeGroup],
    *,
    threshold: float,
    scope: str,
) -> list[dict[str, Any]]:
    group_by_member: dict[int, MergeGroup] = {
        member_index: group
        for group in groups
        for member_index in group.member_indices
    }
    output: list[dict[str, Any]] = []
    emitted_groups: set[str] = set()
    for index, item in enumerate(items):
        group = group_by_member.get(index)
        if group is None:
            row = copy.deepcopy(item.raw_row)
            row.setdefault("feature_key", item.feature_key)
            output.append(row)
            continue
        if group.group_id in emitted_groups or index != group.canonical_index:
            continue
        output.append(merged_group_row(items, group, threshold=threshold, scope=scope))
        emitted_groups.add(group.group_id)
    return output


def merged_group_row(
    items: list[FeatureItem],
    group: MergeGroup,
    *,
    threshold: float,
    scope: str,
) -> dict[str, Any]:
    canonical = items[group.canonical_index]
    member_items = [items[index] for index in group.member_indices]
    member_rows = [item.raw_row for item in member_items]
    row = copy.deepcopy(canonical.raw_row)
    row.setdefault("feature_key", canonical.feature_key)

    for field in AGGREGATE_LIST_FIELDS:
        values = unique_text_values(row_field_value(member_row, field) for member_row in member_rows)
        if values:
            row[field] = values

    for field in AGGREGATE_ANNOTATION_LIST_FIELDS:
        values = unique_text_values(row_field_value(member_row, field) for member_row in member_rows)
        if values:
            set_nested_or_flat_annotation_value(row, field, values)

    source_paper_ids = unique_text_values(row_field_value(member_row, "source_paper_ids") for member_row in member_rows)
    row["paper_count"] = len(source_paper_ids) if source_paper_ids else max(item.paper_count for item in member_items)
    row["mention_count"] = sum(item.mention_count for item in member_items)
    if isinstance(row.get("annotation"), dict):
        row["annotation"]["paper_count"] = row["paper_count"]
        row["annotation"]["mention_count"] = row["mention_count"]

    group_payload = group_json(group, items, threshold=threshold, scope=scope)
    row["embedding_merge"] = {
        "group_id": group.group_id,
        "threshold": threshold,
        "scope": scope,
        "canonical_feature_key": canonical.feature_key,
        "canonical_feature_name": canonical.label,
        "member_count": group_payload["member_count"],
        "member_row_indices": group_payload["member_row_indices"],
        "member_feature_keys": group_payload["member_feature_keys"],
        "member_feature_names": group_payload["member_feature_names"],
        "member_categories": group_payload["member_categories"],
        "similarity_summary": group_payload["similarity_summary"],
        "pairs": group_payload["pairs"],
    }
    return row


def unique_text_values(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in parse_jsonish_list(value):
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
    return output


def set_nested_or_flat_annotation_value(row: dict[str, Any], field: str, value: list[str]) -> None:
    flat_field = field.replace(".", "_")
    if isinstance(row.get("annotation"), dict) or flat_field not in row:
        annotation = row.get("annotation")
        if not isinstance(annotation, dict):
            annotation = {}
            row["annotation"] = annotation
        annotation[field.split(".", 1)[1]] = value
    if flat_field in row:
        row[flat_field] = value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})
    return path


def write_qualitative_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    features_path: Path,
    threshold: float,
    scope: str,
    top_n: int,
) -> Path:
    lines = [
        "# Top merged features",
        "",
        f"Input: `{features_path}`",
        f"Threshold: `{threshold}` | Scope: `{scope}` | Top per category: `{top_n}` | Rows: `{len(rows)}`",
        "",
    ]
    if not rows:
        lines.append("_No merged features were selected for the qualitative report._")
    else:
        headers = [
            "Category",
            "Rank",
            "Merged feature",
            "Merged into",
            "Into category",
            "Mentions",
            "Papers",
            "Similarity",
            "Group",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_table_value(row.get("category")),
                        markdown_table_value(row.get("rank")),
                        markdown_table_value(row.get("merged_feature_name")),
                        markdown_table_value(row.get("merged_into_feature_name")),
                        markdown_table_value(row.get("merged_into_category")),
                        markdown_table_value(row.get("merged_feature_mention_count")),
                        markdown_table_value(row.get("merged_feature_paper_count")),
                        markdown_table_value(row.get("direct_similarity_to_canonical")),
                        markdown_table_value(row.get("group_id")),
                    ]
                )
                + " |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def markdown_table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.3f}"
    else:
        text = str(value)
    return text.replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def merged_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for row in rows:
        fields.update(row)
    ordered = [field for field in MERGED_FIELD_PRIORITY if field in fields]
    ordered.extend(sorted(field for field in fields if field not in set(ordered)))
    return ordered


def output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "pairs_jsonl": output_dir / f"{prefix}_embedding_merge_pairs.jsonl",
        "pairs_csv": output_dir / f"{prefix}_embedding_merge_pairs.csv",
        "groups_json": output_dir / f"{prefix}_embedding_merge_groups.json",
        "groups_csv": output_dir / f"{prefix}_embedding_merge_groups.csv",
        "qualitative_json": output_dir / f"{prefix}_embedding_merge_qualitative.json",
        "qualitative_csv": output_dir / f"{prefix}_embedding_merge_qualitative.csv",
        "qualitative_md": output_dir / f"{prefix}_embedding_merge_qualitative.md",
        "merged_json": output_dir / f"{prefix}_embedding_merged_features.json",
        "merged_jsonl": output_dir / f"{prefix}_embedding_merged_features.jsonl",
        "merged_csv": output_dir / f"{prefix}_embedding_merged_features.csv",
        "similarity_distribution_png": output_dir / f"{prefix}_embedding_similarity_distribution.png",
        "similarity_boxplot_png": output_dir / f"{prefix}_embedding_similarity_boxplot.png",
        "similarity_percentiles_png": output_dir / f"{prefix}_embedding_similarity_percentiles.png",
        "similarity_distribution_by_category_png": output_dir / f"{prefix}_embedding_similarity_distribution_by_category.png",
        "similarity_boxplot_by_category_png": output_dir / f"{prefix}_embedding_similarity_boxplot_by_category.png",
        "similarity_percentiles_by_category_png": output_dir / f"{prefix}_embedding_similarity_percentiles_by_category.png",
        "trace_json": output_dir / f"{prefix}_embedding_merge_trace.json",
    }


def merged_tsne_html_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f"{prefix}_embedding_merged_features_gemini_embedding_tsne.html"


def sweep_output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "sweep_csv": output_dir / f"{prefix}_embedding_merge_sweep.csv",
        "sweep_json": output_dir / f"{prefix}_embedding_merge_sweep.json",
        "sweep_trace_json": output_dir / f"{prefix}_embedding_merge_sweep_trace.json",
        "sweep_plot_png": output_dir / f"{prefix}_embedding_merge_sweep_metrics.png",
        "category_sweep_csv": output_dir / f"{prefix}_embedding_merge_sweep_by_category.csv",
        "category_sweep_json": output_dir / f"{prefix}_embedding_merge_sweep_by_category.json",
        "category_sweep_plot_png": output_dir / f"{prefix}_embedding_merge_sweep_by_category.png",
        "similarity_distribution_png": output_dir / f"{prefix}_embedding_similarity_distribution.png",
        "similarity_boxplot_png": output_dir / f"{prefix}_embedding_similarity_boxplot.png",
        "similarity_percentiles_png": output_dir / f"{prefix}_embedding_similarity_percentiles.png",
        "similarity_distribution_by_category_png": output_dir / f"{prefix}_embedding_similarity_distribution_by_category.png",
        "similarity_boxplot_by_category_png": output_dir / f"{prefix}_embedding_similarity_boxplot_by_category.png",
        "similarity_percentiles_by_category_png": output_dir / f"{prefix}_embedding_similarity_percentiles_by_category.png",
    }


def write_merge_artifacts(
    *,
    paths: dict[str, Path],
    features_path: Path,
    items: list[FeatureItem],
    pairs: list[SimilarityPair],
    groups: list[MergeGroup],
    qualitative_rows: list[dict[str, Any]],
    qualitative_top: int,
    merged_features: list[dict[str, Any]],
    threshold: float,
    scope: str,
    text_fields: list[str],
    model: str,
    task_type: str,
    output_dimensionality: int,
    cache_path: Path,
    cache_stats: Any,
    similarity_distribution: dict[str, Any],
    similarity_distributions_by_category: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pair_rows = [pair_json(pair, items) for pair in pairs]
    group_rows = [group_json(group, items, threshold=threshold, scope=scope) for group in groups]
    output_dir = paths["trace_json"].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(paths["pairs_jsonl"], pair_rows)
    write_csv(paths["pairs_csv"], pair_rows, PAIR_FIELDNAMES)
    paths["groups_json"].write_text(
        json.dumps(
            {
                "input": str(features_path),
                "threshold": threshold,
                "scope": scope,
                "group_count": len(groups),
                "groups": group_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(paths["groups_csv"], [group_csv_row(group, items) for group in groups], GROUP_FIELDNAMES)
    write_csv(paths["qualitative_csv"], qualitative_rows, QUALITATIVE_FIELDNAMES)
    paths["qualitative_json"].write_text(
        json.dumps(
            {
                "input": str(features_path),
                "threshold": threshold,
                "scope": scope,
                "top_n_per_category": qualitative_top,
                "row_count": len(qualitative_rows),
                "results": qualitative_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_qualitative_markdown(
        paths["qualitative_md"],
        qualitative_rows,
        features_path=features_path,
        threshold=threshold,
        scope=scope,
        top_n=qualitative_top,
    )
    paths["merged_json"].write_text(
        json.dumps(
            {
                "input": str(features_path),
                "threshold": threshold,
                "scope": scope,
                "input_feature_count": len(items),
                "merged_feature_count": len(merged_features),
                "merge_group_count": len(groups),
                "features": merged_features,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(paths["merged_jsonl"], merged_features)
    write_csv(paths["merged_csv"], merged_features, merged_fieldnames(merged_features))

    trace = {
        "project": "EmbeddingMerge",
        "input": str(features_path),
        "threshold": threshold,
        "scope": scope,
        "text_fields": text_fields,
        "model": model,
        "task_type": task_type,
        "output_dimensionality": output_dimensionality,
        "cache_path": str(cache_path),
        "cache": {
            "hits": cache_stats.cache_hits,
            "misses": cache_stats.api_misses,
            "api_batches": cache_stats.api_batches,
        },
        "counts": {
            "input_features": len(items),
            "similarity_pairs": len(pairs),
            "merge_groups": len(groups),
            "merged_features": len(merged_features),
            "merged_away_features": len(items) - len(merged_features),
            "qualitative_rows": len(qualitative_rows),
        },
        "qualitative_top": qualitative_top,
        "similarity_distribution": similarity_distribution_summary(similarity_distribution),
        "similarity_distributions_by_category": similarity_distribution_category_summary(
            similarity_distributions_by_category
        ),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["trace_json"].write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return trace


def prepare_feature_embeddings(
    *,
    features_path: Path,
    text_fields: Iterable[str],
    cache_path: Path,
    client: EmbeddingClient | None,
    model: str,
    task_type: str,
    output_dimensionality: int,
    batch_size: int,
) -> PreparedEmbeddings:
    fields = parse_text_fields(text_fields)
    items = load_feature_items(features_path, fields)
    if not items:
        raise ValueError("No usable features found for embedding merge.")
    embeddings, cache_stats = embed_texts_with_cache(
        [item.embedding_text for item in items],
        cache_path=cache_path,
        client=client,
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        batch_size=batch_size,
    )
    return PreparedEmbeddings(
        text_fields=fields,
        items=items,
        embeddings=embeddings,
        cache_stats=cache_stats,
    )


def evaluate_threshold_sweep(
    items: list[FeatureItem],
    base_pairs: list[SimilarityPair],
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    rows, _ = evaluate_threshold_sweep_details(items, base_pairs, thresholds)
    return rows


def evaluate_threshold_sweep_details(
    items: list[FeatureItem],
    base_pairs: list[SimilarityPair],
    thresholds: Iterable[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        validate_threshold_number(threshold)
        pairs = [pair for pair in base_pairs if pair.similarity >= threshold]
        groups = build_merge_groups(items, pairs)
        merged_away_features = sum(len(group.member_indices) - 1 for group in groups)
        rows.append(
            {
                "threshold": threshold,
                "input_features": len(items),
                "similarity_pairs": len(pairs),
                "merge_groups": len(groups),
                "merged_features": len(items) - merged_away_features,
                "merged_away_features": merged_away_features,
            }
        )
        category_rows.extend(category_sweep_rows(items, groups, threshold))
    return rows, category_rows


def category_sweep_rows(
    items: list[FeatureItem],
    groups: list[MergeGroup],
    threshold: float,
) -> list[dict[str, Any]]:
    input_counts: dict[str, int] = {}
    for item in items:
        input_counts[item.category] = input_counts.get(item.category, 0) + 1

    merge_group_counts = {category: 0 for category in input_counts}
    merged_away_counts = {category: 0 for category in input_counts}
    for group in groups:
        group_categories = {items[index].category for index in group.member_indices}
        for category in group_categories:
            merge_group_counts[category] = merge_group_counts.get(category, 0) + 1
        for index in group.member_indices:
            if index == group.canonical_index:
                continue
            category = items[index].category
            merged_away_counts[category] = merged_away_counts.get(category, 0) + 1

    return [
        {
            "threshold": threshold,
            "category": category,
            "input_features": input_counts[category],
            "merge_groups": merge_group_counts.get(category, 0),
            "resulting_features": input_counts[category] - merged_away_counts.get(category, 0),
            "merged_away_features": merged_away_counts.get(category, 0),
        }
        for category in sorted(input_counts, key=lambda value: (-input_counts[value], value.casefold()))
    ]


def write_sweep_plot(path: Path, rows: list[dict[str, Any]]) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("Matplotlib is required for threshold sweep plots.") from exc

    thresholds = [row["threshold"] for row in rows]
    metrics = [
        ("merged_away_features", "Features Merged Away"),
        ("merged_features", "Resulting Feature Count"),
        ("similarity_pairs", "Similar Pair Count"),
        ("merge_groups", "Merge Group Count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (field, title) in zip(axes.ravel(), metrics):
        axis.plot(thresholds, [row[field] for row in rows], marker="o", linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Threshold")
        axis.set_ylabel("Count")
        axis.grid(True, alpha=0.3)
    fig.suptitle("Embedding Merge Threshold Sweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_category_sweep_plot(path: Path, rows: list[dict[str, Any]]) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("Matplotlib is required for threshold sweep plots.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    categories = sorted(
        {str(row["category"]) for row in rows},
        key=lambda category: (
            -max(int(row["input_features"]) for row in rows if row["category"] == category),
            category.casefold(),
        ),
    )
    thresholds = sorted({float(row["threshold"]) for row in rows})
    by_category_threshold = {
        (str(row["category"]), float(row["threshold"])): row
        for row in rows
    }

    fig_width = max(12.0, min(20.0, 10.0 + len(categories) * 0.25))
    fig_height = max(7.0, min(14.0, 6.0 + len(categories) * 0.18))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    color_map = plt.get_cmap("tab20")
    line_styles = ["-", "--", "-.", ":"]
    metrics = [
        ("resulting_features", "Resulting Feature Count by Category"),
        ("merge_groups", "Merge Group Count by Category"),
    ]
    for axis, (field, title) in zip(axes, metrics):
        for index, category in enumerate(categories):
            values = [
                by_category_threshold[(category, threshold)][field]
                for threshold in thresholds
            ]
            axis.plot(
                thresholds,
                values,
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=color_map(index % color_map.N),
                linestyle=line_styles[(index // color_map.N) % len(line_styles)],
                label=category,
            )
        axis.set_title(title)
        axis.set_xlabel("Threshold")
        axis.set_ylabel("Count")
        axis.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(1, min(4, len(categories))),
            fontsize=7,
        )
    fig.suptitle("Embedding Merge Threshold Sweep by Category")
    fig.tight_layout(rect=(0, 0.16 if handles else 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_sweep_artifacts(
    *,
    paths: dict[str, Path],
    features_path: Path,
    rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    thresholds: list[float],
    items: list[FeatureItem],
    scope: str,
    text_fields: list[str],
    model: str,
    task_type: str,
    output_dimensionality: int,
    cache_path: Path,
    cache_stats: Any,
    similarity_distribution: dict[str, Any],
    similarity_distributions_by_category: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output_dir = paths["sweep_trace_json"].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(paths["sweep_csv"], rows, SWEEP_FIELDNAMES)
    write_csv(paths["category_sweep_csv"], category_rows, CATEGORY_SWEEP_FIELDNAMES)
    paths["sweep_json"].write_text(
        json.dumps(
            {
                "input": str(features_path),
                "threshold_start": thresholds[0],
                "threshold_end": thresholds[-1],
                "threshold_count": len(thresholds),
                "scope": scope,
                "input_feature_count": len(items),
                "results": rows,
                "category_results": category_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["category_sweep_json"].write_text(
        json.dumps(
            {
                "input": str(features_path),
                "threshold_start": thresholds[0],
                "threshold_end": thresholds[-1],
                "threshold_count": len(thresholds),
                "scope": scope,
                "results": category_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_sweep_plot(paths["sweep_plot_png"], rows)
    write_category_sweep_plot(paths["category_sweep_plot_png"], category_rows)
    distribution_thresholds = [thresholds[0]]
    if thresholds[-1] != thresholds[0]:
        distribution_thresholds.append(thresholds[-1])
    write_similarity_distribution_plot(
        paths["similarity_distribution_png"],
        similarity_distribution,
        title="Embedding Similarity Distribution",
        thresholds=distribution_thresholds,
    )
    write_similarity_boxplot(
        paths["similarity_boxplot_png"],
        similarity_distribution,
        title="Embedding Similarity Box Plot",
        thresholds=distribution_thresholds,
    )
    write_similarity_percentile_plot(
        paths["similarity_percentiles_png"],
        similarity_distribution,
        title="Embedding Similarity Percentile Plot",
        thresholds=distribution_thresholds,
    )
    write_similarity_distribution_by_category_plot(
        paths["similarity_distribution_by_category_png"],
        similarity_distributions_by_category,
        title="Embedding Similarity Distribution by Category",
    )
    write_similarity_boxplot_by_category(
        paths["similarity_boxplot_by_category_png"],
        similarity_distributions_by_category,
        title="Embedding Similarity Box Plot by Category",
    )
    write_similarity_percentile_by_category_plot(
        paths["similarity_percentiles_by_category_png"],
        similarity_distribution,
        similarity_distributions_by_category,
        title="Embedding Similarity Percentile Plot by Category",
    )

    trace = {
        "project": "EmbeddingMergeThresholdSweep",
        "input": str(features_path),
        "threshold_start": thresholds[0],
        "threshold_end": thresholds[-1],
        "threshold_count": len(thresholds),
        "thresholds": thresholds,
        "scope": scope,
        "text_fields": text_fields,
        "model": model,
        "task_type": task_type,
        "output_dimensionality": output_dimensionality,
        "cache_path": str(cache_path),
        "cache": {
            "hits": cache_stats.cache_hits,
            "misses": cache_stats.api_misses,
            "api_batches": cache_stats.api_batches,
        },
        "counts": {
            "input_features": len(items),
            "thresholds": len(thresholds),
            "category_result_rows": len(category_rows),
        },
        "similarity_distribution": similarity_distribution_summary(similarity_distribution),
        "similarity_distributions_by_category": similarity_distribution_category_summary(
            similarity_distributions_by_category
        ),
        "results": rows,
        "category_results": category_rows,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["sweep_trace_json"].write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return trace


def maybe_write_merged_tsne_html(
    *,
    merged_json_path: Path,
    html_path: Path,
    text_fields: Iterable[str],
    cache_path: Path,
    client: EmbeddingClient | None,
    model: str,
    task_type: str,
    output_dimensionality: int,
    batch_size: int,
    perplexity: float | None,
    random_state: int,
    annotate_top: int,
) -> dict[str, Any]:
    try:
        items = load_feature_items(merged_json_path, text_fields)
        if len(items) < 3:
            return {
                "status": "skipped",
                "reason": f"Found {len(items)} usable merged features; t-SNE requires at least 3.",
                "usable_features": len(items),
            }
        embeddings, stats = embed_texts_with_cache(
            [item.embedding_text for item in items],
            cache_path=cache_path,
            client=client,
            model=model,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            batch_size=batch_size,
        )
        coordinates = tsne_coordinates(
            embeddings,
            perplexity=perplexity,
            random_state=random_state,
        )
        write_interactive_html(
            html_path,
            items,
            coordinates,
            title=f"{merged_json_path.stem} Gemini embedding t-SNE",
            annotate_top=annotate_top,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
        }

    return {
        "status": "written",
        "html_path": str(html_path),
        "usable_features": len(items),
        "cache": {
            "hits": stats.cache_hits,
            "misses": stats.api_misses,
            "api_batches": stats.api_batches,
        },
    }


def run_embedding_merge(
    *,
    features_path: Path,
    threshold: float,
    text_fields: Iterable[str],
    scope: str,
    output_dir: Path,
    prefix: str,
    cache_path: Path,
    client: EmbeddingClient | None,
    model: str = DEFAULT_GEMINI_MODEL,
    task_type: str = DEFAULT_TASK_TYPE,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
    batch_size: int = 50,
    qualitative_top: int = 10,
    write_tsne_html: bool = True,
    tsne_perplexity: float | None = None,
    tsne_random_state: int = DEFAULT_RANDOM_STATE,
    tsne_annotate_top: int = DEFAULT_ANNOTATE_TOP,
) -> dict[str, Any]:
    prepared = prepare_feature_embeddings(
        features_path=features_path,
        text_fields=text_fields,
        cache_path=cache_path,
        client=client,
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        batch_size=batch_size,
    )
    matrix = normalized_matrix(prepared.embeddings)
    pairs = find_similarity_pairs_from_normalized_matrix(prepared.items, matrix, threshold=threshold, scope=scope)
    groups = build_merge_groups(prepared.items, pairs)
    merged_features = build_merged_features(prepared.items, groups, threshold=threshold, scope=scope)
    qualitative_rows = qualitative_merge_rows(prepared.items, groups, top_n=qualitative_top)
    paths = output_paths(output_dir, prefix)
    similarity_distribution = pairwise_similarity_histogram(prepared.items, matrix, scope=scope)
    similarity_distributions_by_category = pairwise_similarity_histograms_by_category(prepared.items, matrix)
    write_similarity_distribution_plot(
        paths["similarity_distribution_png"],
        similarity_distribution,
        title="Embedding Similarity Distribution",
        thresholds=[threshold],
    )
    write_similarity_boxplot(
        paths["similarity_boxplot_png"],
        similarity_distribution,
        title="Embedding Similarity Box Plot",
        thresholds=[threshold],
    )
    write_similarity_percentile_plot(
        paths["similarity_percentiles_png"],
        similarity_distribution,
        title="Embedding Similarity Percentile Plot",
        thresholds=[threshold],
    )
    write_similarity_distribution_by_category_plot(
        paths["similarity_distribution_by_category_png"],
        similarity_distributions_by_category,
        title="Embedding Similarity Distribution by Category",
    )
    write_similarity_boxplot_by_category(
        paths["similarity_boxplot_by_category_png"],
        similarity_distributions_by_category,
        title="Embedding Similarity Box Plot by Category",
    )
    write_similarity_percentile_by_category_plot(
        paths["similarity_percentiles_by_category_png"],
        similarity_distribution,
        similarity_distributions_by_category,
        title="Embedding Similarity Percentile Plot by Category",
    )
    trace = write_merge_artifacts(
        paths=paths,
        features_path=features_path,
        items=prepared.items,
        pairs=pairs,
        groups=groups,
        qualitative_rows=qualitative_rows,
        qualitative_top=qualitative_top,
        merged_features=merged_features,
        threshold=threshold,
        scope=scope,
        text_fields=prepared.text_fields,
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        cache_path=cache_path,
        cache_stats=prepared.cache_stats,
        similarity_distribution=similarity_distribution,
        similarity_distributions_by_category=similarity_distributions_by_category,
    )
    if write_tsne_html:
        tsne_result = maybe_write_merged_tsne_html(
            merged_json_path=paths["merged_json"],
            html_path=merged_tsne_html_path(output_dir, prefix),
            text_fields=prepared.text_fields,
            cache_path=cache_path,
            client=client,
            model=model,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            batch_size=batch_size,
            perplexity=tsne_perplexity,
            random_state=tsne_random_state,
            annotate_top=tsne_annotate_top,
        )
    else:
        tsne_result = {"status": "disabled"}
    trace["tsne"] = tsne_result
    if tsne_result.get("status") == "written":
        trace["outputs"]["merged_tsne_html"] = tsne_result["html_path"]
    paths["trace_json"].write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return trace


def run_embedding_merge_sweep(
    *,
    features_path: Path,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
    text_fields: Iterable[str],
    scope: str,
    output_dir: Path,
    prefix: str,
    cache_path: Path,
    client: EmbeddingClient | None,
    model: str = DEFAULT_GEMINI_MODEL,
    task_type: str = DEFAULT_TASK_TYPE,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
    batch_size: int = 50,
) -> dict[str, Any]:
    thresholds = threshold_range(threshold_start, threshold_end, threshold_step)
    prepared = prepare_feature_embeddings(
        features_path=features_path,
        text_fields=text_fields,
        cache_path=cache_path,
        client=client,
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        batch_size=batch_size,
    )
    matrix = normalized_matrix(prepared.embeddings)
    base_pairs = find_similarity_pairs_from_normalized_matrix(
        prepared.items,
        matrix,
        threshold=thresholds[0],
        scope=scope,
    )
    rows, category_rows = evaluate_threshold_sweep_details(prepared.items, base_pairs, thresholds)
    paths = sweep_output_paths(output_dir, prefix)
    similarity_distribution = pairwise_similarity_histogram(prepared.items, matrix, scope=scope)
    similarity_distributions_by_category = pairwise_similarity_histograms_by_category(prepared.items, matrix)
    return write_sweep_artifacts(
        paths=paths,
        features_path=features_path,
        rows=rows,
        category_rows=category_rows,
        thresholds=thresholds,
        items=prepared.items,
        scope=scope,
        text_fields=prepared.text_fields,
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        cache_path=cache_path,
        cache_stats=prepared.cache_stats,
        similarity_distribution=similarity_distribution,
        similarity_distributions_by_category=similarity_distributions_by_category,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge feature inventories using Gemini embeddings and cosine similarity.",
    )
    parser.add_argument("features", type=Path, help="Feature CSV, JSONL, or JSON file.")
    parser.add_argument(
        "--threshold",
        type=threshold_value,
        help="Run one full merge at this cosine similarity threshold.",
    )
    parser.add_argument(
        "--threshold-start",
        type=threshold_value,
        help="First threshold for diagnostic sweep mode.",
    )
    parser.add_argument(
        "--threshold-end",
        type=threshold_value,
        help="Last threshold for diagnostic sweep mode.",
    )
    parser.add_argument(
        "--threshold-step",
        type=threshold_value,
        default=0.01,
        help="Threshold increment for sweep mode.",
    )
    parser.add_argument(
        "--text-fields",
        default=",".join(DEFAULT_TEXT_FIELDS),
        help="Comma-separated row fields used to build embedding text.",
    )
    parser.add_argument(
        "--scope",
        default="all",
        choices=["all", "category"],
        help="Compare all features or only features within the same category.",
    )
    parser.add_argument(
        "--qualitative-top",
        type=positive_int,
        default=10,
        help="Number of most-mentioned absorbed features to report per category in single-threshold mode.",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip automatic t-SNE HTML generation for single-threshold merged outputs.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        help="t-SNE perplexity for the automatic merged-set HTML. Defaults to an automatic valid value.",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="t-SNE random seed for the automatic merged-set HTML.",
    )
    parser.add_argument(
        "--tsne-annotate-top",
        type=nonnegative_int,
        default=DEFAULT_ANNOTATE_TOP,
        help="Number of highest-support merged features to label in the automatic t-SNE HTML.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for merge outputs.")
    parser.add_argument("--prefix", help="Output filename prefix. Defaults to the input stem.")
    parser.add_argument("--cache", type=Path, help="Embedding cache JSONL path.")
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
    return parser


def validate_cli_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    using_single_threshold = args.threshold is not None
    using_sweep_bounds = args.threshold_start is not None or args.threshold_end is not None
    if using_single_threshold and using_sweep_bounds:
        parser.error("--threshold cannot be combined with --threshold-start or --threshold-end")
    if not using_single_threshold:
        if args.threshold_start is None or args.threshold_end is None:
            parser.error("sweep mode requires --threshold-start and --threshold-end unless --threshold is provided")
        try:
            threshold_range(args.threshold_start, args.threshold_end, args.threshold_step)
        except ValueError as exc:
            parser.error(str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_cli_args(args, parser)
    try:
        if not args.features.exists():
            raise FileNotFoundError(f"Feature file does not exist: {args.features}")
        if args.timeout <= 0:
            raise ValueError("--timeout must be greater than 0.")

        output_dir = args.output_dir or args.features.parent / "embedding_merge"
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

        if args.threshold is not None:
            trace = run_embedding_merge(
                features_path=args.features,
                threshold=args.threshold,
                text_fields=parse_text_fields(args.text_fields),
                scope=args.scope,
                output_dir=output_dir,
                prefix=prefix,
                cache_path=cache_path,
                client=client,
                model=args.model,
                task_type=args.task_type,
                output_dimensionality=args.output_dimensionality,
                batch_size=args.batch_size,
                qualitative_top=args.qualitative_top,
                write_tsne_html=not args.skip_tsne,
                tsne_perplexity=args.tsne_perplexity,
                tsne_random_state=args.tsne_random_state,
                tsne_annotate_top=args.tsne_annotate_top,
            )
        else:
            trace = run_embedding_merge_sweep(
                features_path=args.features,
                threshold_start=args.threshold_start,
                threshold_end=args.threshold_end,
                threshold_step=args.threshold_step,
                text_fields=parse_text_fields(args.text_fields),
                scope=args.scope,
                output_dir=output_dir,
                prefix=prefix,
                cache_path=cache_path,
                client=client,
                model=args.model,
                task_type=args.task_type,
                output_dimensionality=args.output_dimensionality,
                batch_size=args.batch_size,
            )
    except Exception as exc:
        print(f"embedding_merge: {exc}", file=sys.stderr)
        return 1

    counts = trace["counts"]
    print(f"Loaded {counts['input_features']} usable features from {args.features}")
    if args.threshold is not None:
        print(f"Found {counts['similarity_pairs']} similar pairs in {counts['merge_groups']} merge groups")
        print(f"Wrote merged inventory with {counts['merged_features']} features")
        tsne = trace.get("tsne", {})
        if tsne.get("status") in {"skipped", "failed", "disabled"}:
            reason = tsne.get("reason")
            suffix = f": {reason}" if reason else ""
            print(f"t-SNE HTML {tsne.get('status')}{suffix}")
    else:
        print(
            f"Evaluated {counts['thresholds']} thresholds from "
            f"{trace['threshold_start']:.2f} to {trace['threshold_end']:.2f}"
        )
    for label, path in trace["outputs"].items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
