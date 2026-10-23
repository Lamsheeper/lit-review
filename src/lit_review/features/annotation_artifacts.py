"""Features annotation_artifacts."""

from __future__ import annotations

from lit_review.core.context import PROFILE, family_order


import re


from pathlib import Path

from typing import Any

from lit_review.core.io import utc_now, write_json


from lit_review.features.records import coerce_str_list, compact_list, feature_key, truncate_text
from lit_review.core.text import normalize_feature_name

from lit_review.features.prompts import annotation_int


from lit_review.features.filtering import FEATURE_FIELDNAMES, write_csv
from lit_review.core.io import write_jsonl

from lit_review.profiles.media_attributes import (
    add_moral_frame_alias_synonyms,
    annotation_exclusion_reason,
    annotation_hard_exclusion_reason,
    annotation_synonym_blocked_norms,
    is_definition_like_annotation_synonym,
)


EXCLUDED_ANNOTATION_REPORT_LIMIT_PER_CATEGORY = 25


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


ANNOTATED_FEATURE_FIELDNAMES = list(
    dict.fromkeys(
        [
            "feature_key",
            *CURATED_FEATURE_FIELDNAMES,
            *FEATURE_FIELDNAMES,
            "annotation_status",
            "annotation_formal_definition",
            "annotation_short_definition",
            "annotation_synonyms",
            "annotation_examples",
            "annotation_paper_count",
            "annotation_mention_count",
            "annotation_confidence",
            "annotation_definition_basis",
            "annotation_notes",
            "annotation_model",
            "annotation_run_id",
            "annotation_annotated_at",
        ]
    )
)


def annotation_confidence(value: Any, status: str) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.2 if status == "needs_review" else 0.5
    return round(max(0.0, min(1.0, confidence)), 4)


def canonical_annotation_status(value: Any, formal_definition: str) -> str:
    normalized = normalize_feature_name(str(value or ""))
    if normalized in {
        "exclude",
        "excluded",
        "filter",
        "filtered",
        "reject",
        "rejected",
        "too broad",
        "generic descriptor",
        "not a feature",
        "not annotatable",
        "descriptor",
    }:
        return "excluded"
    if normalized in {"needs review", "review", "insufficient evidence", "unclear"}:
        return "needs_review"
    if normalized in {"annotated", "complete", "ok", "ready"}:
        return "annotated"
    return "annotated" if formal_definition else "needs_review"


def append_annotation_note(notes: str, addition: str) -> str:
    notes = truncate_text(str(notes or ""), 500)
    addition = truncate_text(str(addition or ""), 300)
    if not addition:
        return notes
    if not notes:
        return addition
    if addition.casefold() in notes.casefold():
        return notes
    return truncate_text(f"{notes}; {addition}", 500)


def normalize_annotation_synonyms(raw_value: Any, feature: dict[str, Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    blocked = (
        annotation_synonym_blocked_norms(feature)
        if PROFILE.get() == "media_attributes"
        else {normalize_feature_name(str(feature.get("category") or ""))}
    )
    for raw in coerce_str_list(raw_value):
        text = re.sub(r"\s+", " ", str(raw or "")).strip().strip("\"'“”‘’")
        if not text or is_definition_like_annotation_synonym(text):
            continue
        normalized = normalize_feature_name(text)
        if not normalized or normalized in blocked or normalized in seen:
            continue
        seen.add(normalized)
        output.append(text)
        if len(output) >= 12:
            break
    return output


def normalize_annotation_decision(
    raw: dict[str, Any],
    feature: dict[str, Any],
    *,
    model: str,
    run_id: str,
    annotated_at: str,
) -> dict[str, Any]:
    formal_definition = re.sub(r"\s+", " ", str(raw.get("formal_definition") or "")).strip()
    short_definition = re.sub(r"\s+", " ", str(raw.get("short_definition") or "")).strip()
    status = canonical_annotation_status(raw.get("status"), formal_definition)
    hard_exclusion_reason = (
        annotation_hard_exclusion_reason(feature, formal_definition, short_definition)
        if PROFILE.get() == "media_attributes"
        else ""
    )
    exclusion_reason = ""
    if status == "excluded":
        exclusion_reason = (
            annotation_exclusion_reason(feature, formal_definition, short_definition)
            if PROFILE.get() == "media_attributes"
            else ""
        ) or hard_exclusion_reason
    elif hard_exclusion_reason:
        status = "excluded"
        exclusion_reason = hard_exclusion_reason

    if status == "excluded":
        definition_basis = "excluded_descriptor"
    elif status == "needs_review" and not raw.get("definition_basis"):
        definition_basis = "needs_review"
    else:
        definition_basis = str(raw.get("definition_basis") or "evidence_synthesized").strip()
    if status == "needs_review" and not formal_definition:
        short_definition = short_definition or ""
    notes = append_annotation_note(str(raw.get("notes") or ""), exclusion_reason)
    if status == "excluded" and not notes:
        notes = "Excluded because the item is too broad or descriptor-like for the target feature inventory."
    synonyms = normalize_annotation_synonyms(raw.get("synonyms"), feature)
    if PROFILE.get() == "media_attributes":
        synonyms = add_moral_frame_alias_synonyms(synonyms, feature)

    return {
        "feature_key": str(raw.get("feature_key") or feature.get("feature_key") or feature_key(feature)),
        "category": str(feature.get("category") or ""),
        "feature_name": str(feature.get("feature_name") or ""),
        "normalized_feature_name": str(feature.get("normalized_feature_name") or ""),
        "status": status,
        "formal_definition": formal_definition,
        "short_definition": short_definition,
        "synonyms": synonyms,
        "examples": compact_list(raw.get("examples"), 8, max_chars=180),
        "paper_count": annotation_int(feature.get("paper_count")),
        "mention_count": annotation_int(feature.get("mention_count")),
        "confidence": annotation_confidence(raw.get("confidence"), status),
        "definition_basis": definition_basis,
        "notes": notes,
        "model": model,
        "run_id": run_id,
        "annotated_at": annotated_at,
    }


def fallback_annotation_decision(
    feature: dict[str, Any],
    *,
    model: str = "",
    run_id: str = "",
    annotated_at: str = "",
) -> dict[str, Any]:
    return normalize_annotation_decision(
        {
            "feature_key": feature.get("feature_key") or feature_key(feature),
            "status": "needs_review",
            "formal_definition": "",
            "short_definition": "",
            "synonyms": [],
            "examples": [],
            "confidence": 0.0,
            "definition_basis": "needs_review",
            "notes": "No model annotation returned; needs human review.",
        },
        feature,
        model=model,
        run_id=run_id,
        annotated_at=annotated_at,
    )


def latest_annotations_by_feature(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        key = str(decision.get("feature_key") or "")
        if key:
            latest[key] = decision
    return latest


def annotation_payload(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": decision.get("status") or "needs_review",
        "formal_definition": decision.get("formal_definition") or "",
        "short_definition": decision.get("short_definition") or "",
        "synonyms": decision.get("synonyms") or [],
        "examples": decision.get("examples") or [],
        "paper_count": annotation_int(decision.get("paper_count")),
        "mention_count": annotation_int(decision.get("mention_count")),
        "confidence": annotation_confidence(decision.get("confidence"), str(decision.get("status") or "")),
        "definition_basis": decision.get("definition_basis") or "needs_review",
        "notes": decision.get("notes") or "",
        "model": decision.get("model") or "",
        "run_id": decision.get("run_id") or "",
        "annotated_at": decision.get("annotated_at") or "",
    }


def build_annotated_features(
    features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    include_excluded: bool = False,
) -> list[dict[str, Any]]:
    decision_map = latest_annotations_by_feature(decisions)
    annotated: list[dict[str, Any]] = []
    for feature in features:
        key = str(feature.get("feature_key") or feature_key(feature))
        decision = decision_map.get(key) or fallback_annotation_decision(feature)
        if not include_excluded and decision.get("status") == "excluded":
            continue
        annotated.append({**feature, "annotation": annotation_payload(decision)})
    return annotated


def flatten_annotated_feature(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    annotation = output.get("annotation") if isinstance(output.get("annotation"), dict) else {}
    output.update(
        {
            "annotation_status": annotation.get("status") or "",
            "annotation_formal_definition": annotation.get("formal_definition") or "",
            "annotation_short_definition": annotation.get("short_definition") or "",
            "annotation_synonyms": annotation.get("synonyms") or [],
            "annotation_examples": annotation.get("examples") or [],
            "annotation_paper_count": annotation.get("paper_count") or 0,
            "annotation_mention_count": annotation.get("mention_count") or 0,
            "annotation_confidence": annotation.get("confidence") or 0.0,
            "annotation_definition_basis": annotation.get("definition_basis") or "",
            "annotation_notes": annotation.get("notes") or "",
            "annotation_model": annotation.get("model") or "",
            "annotation_run_id": annotation.get("run_id") or "",
            "annotation_annotated_at": annotation.get("annotated_at") or "",
        }
    )
    return output


def annotation_row_int(row: dict[str, Any], field: str) -> int:
    annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
    return annotation_int(annotation.get(field) or row.get(field))


def markdown_table_cell(value: Any, max_chars: int = 180) -> str:
    text = truncate_text(str(value or ""), max_chars)
    text = text.replace("\n", " ")
    text = text.replace("|", "\\|")
    return text


def build_excluded_annotations_markdown(
    excluded: list[dict[str, Any]],
    *,
    limit_per_category: int = EXCLUDED_ANNOTATION_REPORT_LIMIT_PER_CATEGORY,
) -> str:
    limit = max(1, int(limit_per_category))
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in excluded:
        by_category.setdefault(str(row.get("category") or "uncategorized"), []).append(row)

    lines = [
        "# Excluded Annotation Summary",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Top {limit} most-mentioned excluded features per category.",
        "",
    ]
    if not excluded:
        lines.append("No excluded features.")
        lines.append("")
        return "\n".join(lines)

    ordered_categories = [category for category in family_order() if category in by_category]
    ordered_categories.extend(
        category for category in sorted(by_category) if category not in set(ordered_categories)
    )

    for category in ordered_categories:
        rows = sorted(
            by_category[category],
            key=lambda row: (
                -annotation_row_int(row, "mention_count"),
                -annotation_row_int(row, "paper_count"),
                str(row.get("feature_name") or "").casefold(),
            ),
        )
        lines.extend(
            [
                f"## {category} ({len(rows)} excluded)",
                "",
                "| Rank | Feature | Mentions | Papers | Exclusion reason | Definition |",
                "| ---: | --- | ---: | ---: | --- | --- |",
            ]
        )
        for rank, row in enumerate(rows[:limit], start=1):
            annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        markdown_table_cell(row.get("feature_name"), 80),
                        str(annotation_row_int(row, "mention_count")),
                        str(annotation_row_int(row, "paper_count")),
                        markdown_table_cell(annotation.get("notes"), 180),
                        markdown_table_cell(
                            annotation.get("short_definition") or annotation.get("formal_definition"), 180
                        ),
                    ]
                )
                + " |"
            )
        if len(rows) > limit:
            lines.append(f"|  | _{len(rows) - limit} more omitted_ |  |  |  |  |")
        lines.append("")
    return "\n".join(lines)


def annotation_status_label(status: str) -> str:
    normalized = str(status or "needs_review")
    labels = {
        "annotated": "Kept",
        "needs_review": "Needs review",
        "excluded": "Cut",
    }
    return labels.get(normalized, normalized.replace("_", " ").title())


def ordered_annotation_categories(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row.get("category") or "uncategorized") for row in rows}
    ordered = [category for category in family_order() if category in present]
    ordered.extend(category for category in sorted(present) if category not in set(ordered))
    return ordered


def annotation_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"annotated": 0, "needs_review": 0, "excluded": 0}
    for row in rows:
        annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
        status = str(annotation.get("status") or "needs_review")
        counts[status] = counts.get(status, 0) + 1
    return counts


def annotation_status_counts_by_category(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row.get("category") or "uncategorized")
        annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
        status = str(annotation.get("status") or "needs_review")
        category_counts = counts.setdefault(
            category,
            {"annotated": 0, "needs_review": 0, "excluded": 0},
        )
        category_counts[status] = category_counts.get(status, 0) + 1
    return counts


def annotate_bar_values(ax: Any, values: list[int], *, horizontal: bool = False) -> None:
    for index, value in enumerate(values):
        if value <= 0:
            continue
        if horizontal:
            ax.text(value + max(values) * 0.015, index, str(value), va="center", fontsize=9)
        else:
            ax.text(index, value + max(values) * 0.02, str(value), ha="center", fontsize=9)


def write_annotation_cut_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> list[Path]:
    from lit_review.core.plotting import optional_pyplot

    plt = optional_pyplot()
    if plt is None:
        return []

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
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

    written: list[Path] = []
    statuses = ["annotated", "needs_review", "excluded"]
    status_colors = {
        "annotated": "#2f855a",
        "needs_review": "#b7791f",
        "excluded": "#c53030",
    }
    total_counts = annotation_status_counts(rows)

    status_path = figure_dir / "annotation_status_counts.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = [annotation_status_label(status) for status in statuses]
    values = [total_counts.get(status, 0) for status in statuses]
    ax.bar(labels, values, color=[status_colors[status] for status in statuses])
    ax.set_title("Annotation Outcomes")
    ax.set_ylabel("Feature count")
    ax.set_ylim(0, max(values + [1]) * 1.18)
    annotate_bar_values(ax, values)
    fig.tight_layout()
    fig.savefig(status_path, bbox_inches="tight")
    plt.close(fig)
    written.append(status_path)

    category_counts = annotation_status_counts_by_category(rows)
    categories = ordered_annotation_categories(rows)
    excluded_values = [category_counts.get(category, {}).get("excluded", 0) for category in categories]

    excluded_path = figure_dir / "annotation_excluded_by_category.png"
    fig, ax = plt.subplots(figsize=(8.8, max(4.4, 0.6 * max(len(categories), 1))))
    display_categories = [category.replace("_", " ") for category in categories]
    y_positions = list(range(len(categories)))
    ax.barh(y_positions, excluded_values, color=status_colors["excluded"])
    ax.set_yticks(y_positions, labels=display_categories)
    ax.invert_yaxis()
    ax.set_title("Features Cut by Category")
    ax.set_xlabel("Cut feature count")
    ax.set_xlim(0, max(excluded_values + [1]) * 1.18)
    annotate_bar_values(ax, excluded_values, horizontal=True)
    fig.tight_layout()
    fig.savefig(excluded_path, bbox_inches="tight")
    plt.close(fig)
    written.append(excluded_path)

    by_category_path = figure_dir / "annotation_status_by_category.png"
    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.65 * max(len(categories), 1))))
    left = [0] * len(categories)
    for status in statuses:
        values = [category_counts.get(category, {}).get(status, 0) for category in categories]
        ax.barh(
            y_positions,
            values,
            left=left,
            label=annotation_status_label(status),
            color=status_colors[status],
        )
        left = [current + value for current, value in zip(left, values)]
    ax.set_yticks(y_positions, labels=display_categories)
    ax.invert_yaxis()
    ax.set_title("Annotation Outcomes by Category")
    ax.set_xlabel("Feature count")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(by_category_path, bbox_inches="tight")
    plt.close(fig)
    written.append(by_category_path)

    return written


def write_annotation_artifacts(
    output_dir: Path,
    features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, int]:
    all_rows = build_annotated_features(features, decisions, include_excluded=True)
    excluded = [
        row
        for row in all_rows
        if isinstance(row.get("annotation"), dict) and row["annotation"].get("status") == "excluded"
    ]
    annotated = [
        row
        for row in all_rows
        if not (isinstance(row.get("annotation"), dict) and row["annotation"].get("status") == "excluded")
    ]
    write_jsonl(output_dir / "annotated_features.jsonl", annotated)
    write_json(
        output_dir / "annotated_features.json",
        {
            "feature_count": len(annotated),
            "features": annotated,
        },
    )
    write_csv(
        output_dir / "annotated_features.csv",
        [flatten_annotated_feature(row) for row in annotated],
        ANNOTATED_FEATURE_FIELDNAMES,
    )
    write_jsonl(output_dir / "excluded_annotations.jsonl", excluded)
    write_json(
        output_dir / "excluded_annotations.json",
        {
            "feature_count": len(excluded),
            "features": excluded,
        },
    )
    write_csv(
        output_dir / "excluded_annotations.csv",
        [flatten_annotated_feature(row) for row in excluded],
        ANNOTATED_FEATURE_FIELDNAMES,
    )
    (output_dir / "excluded_annotations_by_category.md").write_text(
        build_excluded_annotations_markdown(excluded),
        encoding="utf-8",
    )
    write_annotation_cut_figures(output_dir, all_rows)
    needs_review = sum(
        1
        for row in annotated
        if isinstance(row.get("annotation"), dict) and row["annotation"].get("status") == "needs_review"
    )
    return {
        "annotated_features": len(annotated),
        "needs_review_annotations": needs_review,
        "excluded_annotations": len(excluded),
    }
