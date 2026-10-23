"""Features scoring_artifacts."""

from __future__ import annotations


from pathlib import Path

from typing import Any

from lit_review.core.io import utc_now, write_json


from lit_review.features.records import feature_key, truncate_text

from lit_review.features.prompts import (
    annotation_int,
    literature_support_score,
    scoring_mention_count,
    scoring_paper_count,
)


from lit_review.features.filtering import FEATURE_FIELDNAMES, write_csv
from lit_review.core.io import write_jsonl

from lit_review.features.annotation_artifacts import (
    ANNOTATED_FEATURE_FIELDNAMES,
    CURATED_FEATURE_FIELDNAMES,
    flatten_annotated_feature,
)


SCORED_FEATURE_FIELDNAMES = list(
    dict.fromkeys(
        [
            "feature_key",
            *CURATED_FEATURE_FIELDNAMES,
            *FEATURE_FIELDNAMES,
            "annotation",
            "embedding_merge",
            *ANNOTATED_FEATURE_FIELDNAMES,
            "score",
            "score_literature_support",
            "score_literature_support_rationale",
            "score_annotation_reliability",
            "score_annotation_reliability_rationale",
            "score_distinctiveness",
            "score_distinctiveness_rationale",
            "score_total",
            "score_confidence",
            "score_notes",
            "score_model",
            "score_run_id",
            "score_scored_at",
        ]
    )
)


SCORING_PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


def literature_support_rationale(paper_count: Any) -> str:
    count = annotation_int(paper_count)
    return f"paper_count={count} maps to literature_support={literature_support_score(count)}."


def scoring_criterion_score(value: Any, default: int = 0) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = default
    return max(0, min(5, score))


def scoring_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.5
    return round(max(0.0, min(1.0, confidence)), 4)


def total_feature_score(
    literature_support: Any,
    annotation_reliability: Any,
    distinctiveness: Any,
) -> float:
    return round(
        0.4 * scoring_criterion_score(literature_support)
        + 0.3 * scoring_criterion_score(annotation_reliability)
        + 0.3 * scoring_criterion_score(distinctiveness),
        2,
    )


def normalize_scoring_decision(
    raw: dict[str, Any],
    feature: dict[str, Any],
    *,
    model: str,
    run_id: str,
    scored_at: str,
) -> dict[str, Any]:
    paper_count = scoring_paper_count(feature)
    literature = literature_support_score(paper_count)
    annotation_reliability = scoring_criterion_score(raw.get("annotation_reliability"))
    distinctiveness = scoring_criterion_score(raw.get("distinctiveness"))
    return {
        "feature_key": str(raw.get("feature_key") or feature.get("feature_key") or feature_key(feature)),
        "category": str(feature.get("category") or ""),
        "feature_name": str(feature.get("feature_name") or ""),
        "normalized_feature_name": str(feature.get("normalized_feature_name") or ""),
        "paper_count": paper_count,
        "mention_count": scoring_mention_count(feature),
        "literature_support": literature,
        "literature_support_rationale": literature_support_rationale(paper_count),
        "annotation_reliability": annotation_reliability,
        "annotation_reliability_rationale": truncate_text(
            str(raw.get("annotation_reliability_rationale") or ""),
            500,
        ),
        "distinctiveness": distinctiveness,
        "distinctiveness_rationale": truncate_text(
            str(raw.get("distinctiveness_rationale") or ""),
            500,
        ),
        "total": total_feature_score(literature, annotation_reliability, distinctiveness),
        "confidence": scoring_confidence(raw.get("confidence")),
        "notes": truncate_text(str(raw.get("notes") or ""), 500),
        "model": model,
        "run_id": run_id,
        "scored_at": scored_at,
    }


def fallback_scoring_decision(
    feature: dict[str, Any],
    *,
    model: str = "",
    run_id: str = "",
    scored_at: str = "",
) -> dict[str, Any]:
    return normalize_scoring_decision(
        {
            "feature_key": feature.get("feature_key") or feature_key(feature),
            "annotation_reliability": 0,
            "annotation_reliability_rationale": "No model score returned; needs human review.",
            "distinctiveness": 0,
            "distinctiveness_rationale": "No model score returned; needs human review.",
            "confidence": 0.0,
            "notes": "No model score returned; needs human review.",
        },
        feature,
        model=model,
        run_id=run_id,
        scored_at=scored_at,
    )


def latest_scores_by_feature(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        key = str(decision.get("feature_key") or "")
        if key:
            latest[key] = decision
    return latest


def score_payload(
    decision: dict[str, Any],
    feature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = feature if feature is not None else decision
    paper_count = scoring_paper_count(source)
    literature_support = literature_support_score(paper_count)
    annotation_reliability = scoring_criterion_score(decision.get("annotation_reliability"))
    distinctiveness = scoring_criterion_score(decision.get("distinctiveness"))
    return {
        "literature_support": literature_support,
        "literature_support_rationale": literature_support_rationale(paper_count),
        "annotation_reliability": annotation_reliability,
        "annotation_reliability_rationale": decision.get("annotation_reliability_rationale") or "",
        "distinctiveness": distinctiveness,
        "distinctiveness_rationale": decision.get("distinctiveness_rationale") or "",
        "total": total_feature_score(
            literature_support,
            annotation_reliability,
            distinctiveness,
        ),
        "confidence": scoring_confidence(decision.get("confidence")),
        "notes": decision.get("notes") or "",
        "model": decision.get("model") or "",
        "run_id": decision.get("run_id") or "",
        "scored_at": decision.get("scored_at") or "",
    }


def build_scored_features(
    features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decision_map = latest_scores_by_feature(decisions)
    scored: list[dict[str, Any]] = []
    for feature in features:
        key = str(feature.get("feature_key") or feature_key(feature))
        decision = decision_map.get(key) or fallback_scoring_decision(feature)
        scored.append({**feature, "score": score_payload(decision, feature)})
    return scored


def flatten_scored_feature(row: dict[str, Any]) -> dict[str, Any]:
    output = flatten_annotated_feature(row)
    score = output.get("score") if isinstance(output.get("score"), dict) else {}
    output.update(
        {
            "score_literature_support": score.get("literature_support") or 0,
            "score_literature_support_rationale": score.get("literature_support_rationale") or "",
            "score_annotation_reliability": score.get("annotation_reliability") or 0,
            "score_annotation_reliability_rationale": score.get("annotation_reliability_rationale") or "",
            "score_distinctiveness": score.get("distinctiveness") or 0,
            "score_distinctiveness_rationale": score.get("distinctiveness_rationale") or "",
            "score_total": score.get("total") or 0.0,
            "score_confidence": score.get("confidence") or 0.0,
            "score_notes": score.get("notes") or "",
            "score_model": score.get("model") or "",
            "score_run_id": score.get("run_id") or "",
            "score_scored_at": score.get("scored_at") or "",
        }
    )
    return output


def scored_feature_sort_key(row: dict[str, Any]) -> tuple[float, int, int, str]:
    score = row.get("score") if isinstance(row.get("score"), dict) else {}
    return (
        -float(score.get("total") or 0.0),
        -scoring_criterion_score(score.get("literature_support")),
        -scoring_paper_count(row),
        str(row.get("feature_name") or "").casefold(),
    )


def scoring_summary_by_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category") or "uncategorized"), []).append(row)

    categories: dict[str, dict[str, Any]] = {}
    for category, category_rows in sorted(by_category.items()):
        scores = [row.get("score") if isinstance(row.get("score"), dict) else {} for row in category_rows]
        count = len(category_rows)
        categories[category] = {
            "feature_count": count,
            "mean_literature_support": round(
                sum(scoring_criterion_score(score.get("literature_support")) for score in scores)
                / max(count, 1),
                4,
            ),
            "mean_annotation_reliability": round(
                sum(scoring_criterion_score(score.get("annotation_reliability")) for score in scores)
                / max(count, 1),
                4,
            ),
            "mean_distinctiveness": round(
                sum(scoring_criterion_score(score.get("distinctiveness")) for score in scores)
                / max(count, 1),
                4,
            ),
            "mean_total": round(
                sum(float(score.get("total") or 0.0) for score in scores) / max(count, 1),
                4,
            ),
        }
    return {
        "generated_at": utc_now(),
        "feature_count": len(rows),
        "category_count": len(categories),
        "categories": categories,
    }


def score_metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        score = row.get("score") if isinstance(row.get("score"), dict) else {}
        try:
            value = float(score.get(metric))
        except (TypeError, ValueError):
            continue
        values.append(value)
    return values


def percentile_value(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 4)
    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    value = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    return round(value, 4)


def scoring_percentile_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        ("total", "total_score"),
        ("literature_support", "literature_support"),
        ("annotation_reliability", "annotation_reliability"),
        ("distinctiveness", "distinctiveness"),
    ]
    output: list[dict[str, Any]] = []
    for metric, label in metrics:
        values = score_metric_values(rows, metric)
        for percentile in SCORING_PERCENTILES:
            output.append(
                {
                    "scope": "all",
                    "category": "",
                    "metric": label,
                    "percentile": percentile,
                    "value": percentile_value(values, percentile),
                    "feature_count": len(values),
                }
            )

    categories = sorted({str(row.get("category") or "uncategorized") for row in rows})
    for category in categories:
        category_rows = [row for row in rows if str(row.get("category") or "uncategorized") == category]
        values = score_metric_values(category_rows, "total")
        for percentile in SCORING_PERCENTILES:
            output.append(
                {
                    "scope": "category",
                    "category": category,
                    "metric": "total_score",
                    "percentile": percentile,
                    "value": percentile_value(values, percentile),
                    "feature_count": len(values),
                }
            )
    return output


def write_scoring_percentile_artifacts(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    percentile_rows = scoring_percentile_rows(rows)
    write_json(
        output_dir / "scoring_percentiles.json",
        {
            "generated_at": utc_now(),
            "percentiles": SCORING_PERCENTILES,
            "rows": percentile_rows,
        },
    )
    write_csv(
        output_dir / "scoring_percentiles.csv",
        percentile_rows,
        ["scope", "category", "metric", "percentile", "value", "feature_count"],
    )
    return percentile_rows


def set_scoring_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def no_score_data_text(ax: Any) -> None:
    ax.text(0.5, 0.5, "No scored features", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def write_scoring_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
    percentile_rows: list[dict[str, Any]],
) -> list[Path]:
    from lit_review.core.plotting import optional_pyplot

    plt = optional_pyplot()
    if plt is None:
        return []

    set_scoring_plot_style()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    total_values = score_metric_values(rows, "total")
    total_path = figure_dir / "scoring_total_distribution.png"
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    if total_values:
        bins = [index / 2 for index in range(0, 11)]
        ax.hist(total_values, bins=bins, color="#2f6f8f", edgecolor="white")
        ax.set_xlim(0, 5)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xlabel("Weighted total score")
        ax.set_ylabel("Feature count")
    else:
        no_score_data_text(ax)
    ax.set_title("Total Score Distribution")
    fig.tight_layout()
    fig.savefig(total_path, bbox_inches="tight")
    plt.close(fig)
    written.append(total_path)

    criteria_path = figure_dir / "scoring_criteria_distributions.png"
    criteria = [
        ("literature_support", "Literature support", "#4c78a8"),
        ("annotation_reliability", "Annotation reliability", "#59a14f"),
        ("distinctiveness", "Distinctiveness", "#e15759"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), sharey=True)
    for ax, (metric, label, color) in zip(axes, criteria):
        values = score_metric_values(rows, metric)
        if values:
            bins = [value - 0.5 for value in range(0, 7)]
            ax.hist(values, bins=bins, color=color, edgecolor="white")
            ax.set_xlim(-0.5, 5.5)
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_xlabel("Score")
        else:
            no_score_data_text(ax)
        ax.set_title(label)
    axes[0].set_ylabel("Feature count")
    fig.suptitle("Criterion Score Distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(criteria_path, bbox_inches="tight")
    plt.close(fig)
    written.append(criteria_path)

    percentile_path = figure_dir / "scoring_percentiles.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    all_rows = [row for row in percentile_rows if row.get("scope") == "all"]
    metric_labels = ["total_score", "literature_support", "annotation_reliability", "distinctiveness"]
    display_labels = {
        "total_score": "Total",
        "literature_support": "Literature",
        "annotation_reliability": "Reliability",
        "distinctiveness": "Distinctiveness",
    }
    if all_rows:
        for metric in metric_labels:
            metric_rows = [row for row in all_rows if row.get("metric") == metric]
            xs = [int(row["percentile"]) for row in metric_rows]
            ys = [float(row["value"]) for row in metric_rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=display_labels[metric])
        ax.set_ylim(0, 5)
        ax.set_xticks(SCORING_PERCENTILES)
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Score")
        ax.legend(frameon=False)
    else:
        no_score_data_text(ax)
    ax.set_title("Score Percentiles")
    fig.tight_layout()
    fig.savefig(percentile_path, bbox_inches="tight")
    plt.close(fig)
    written.append(percentile_path)

    by_category_path = figure_dir / "scoring_total_percentiles_by_category.png"
    category_rows = [row for row in percentile_rows if row.get("scope") == "category"]
    categories = sorted({str(row.get("category") or "") for row in category_rows if row.get("category")})
    fig, ax = plt.subplots(figsize=(9.6, max(4.8, 0.8 * max(len(categories), 1))))
    if categories:
        p25 = {
            str(row.get("category")): float(row.get("value") or 0.0)
            for row in category_rows
            if row.get("percentile") == 25
        }
        p50 = {
            str(row.get("category")): float(row.get("value") or 0.0)
            for row in category_rows
            if row.get("percentile") == 50
        }
        p75 = {
            str(row.get("category")): float(row.get("value") or 0.0)
            for row in category_rows
            if row.get("percentile") == 75
        }
        y_positions = list(range(len(categories)))
        left_errors = [max(0.0, p50.get(category, 0.0) - p25.get(category, 0.0)) for category in categories]
        right_errors = [max(0.0, p75.get(category, 0.0) - p50.get(category, 0.0)) for category in categories]
        ax.errorbar(
            [p50.get(category, 0.0) for category in categories],
            y_positions,
            xerr=[left_errors, right_errors],
            fmt="o",
            color="#2f6f8f",
            ecolor="#8ab6cc",
            elinewidth=3,
            capsize=4,
        )
        ax.set_yticks(y_positions, labels=[category.replace("_", " ") for category in categories])
        ax.invert_yaxis()
        ax.set_xlim(0, 5)
        ax.set_xlabel("Total score percentile")
        ax.set_title("Total Score Median and IQR by Category")
    else:
        no_score_data_text(ax)
        ax.set_title("Total Score Median and IQR by Category")
    fig.tight_layout()
    fig.savefig(by_category_path, bbox_inches="tight")
    plt.close(fig)
    written.append(by_category_path)

    return written


def write_scoring_artifacts(
    output_dir: Path,
    features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, int]:
    scored = build_scored_features(features, decisions)
    ranked = sorted(scored, key=scored_feature_sort_key)
    write_jsonl(output_dir / "scored_features.jsonl", scored)
    write_json(
        output_dir / "scored_features.json",
        {
            "feature_count": len(scored),
            "features": scored,
        },
    )
    write_csv(
        output_dir / "scored_features.csv",
        [flatten_scored_feature(row) for row in scored],
        SCORED_FEATURE_FIELDNAMES,
    )
    write_csv(
        output_dir / "scored_features_ranked.csv",
        [flatten_scored_feature(row) for row in ranked],
        SCORED_FEATURE_FIELDNAMES,
    )
    summary = scoring_summary_by_category(scored)
    write_json(output_dir / "scoring_summary_by_category.json", summary)
    percentile_rows = write_scoring_percentile_artifacts(output_dir, scored)
    figure_paths = write_scoring_figures(output_dir, scored, percentile_rows)
    return {
        "scored_features": len(scored),
        "scored_features_ranked": len(ranked),
        "scoring_categories": int(summary.get("category_count") or 0),
        "scoring_percentile_rows": len(percentile_rows),
        "scoring_figures": len(figure_paths),
    }
