"""Generate matplotlib figures for the SemEval-2020 Task 11 sanity check."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from compare_expected_features import (
    DEFAULT_EXPECTED,
    load_expected,
    load_feature_rows,
    normalize,
    row_label,
    row_terms,
)


DEFAULT_EXTRACT_DIR = Path("review_runs/semeval_2020_task11_sanity/feature_extract_pdf")
DEFAULT_OUTPUT_DIR = DEFAULT_EXTRACT_DIR / "figures"
STAGE_FILES = [
    ("Raw extraction", "features.csv"),
    ("Curated", "curated_features.csv"),
    ("Rule final", "final_curated_features_top100_by_category.csv"),
    ("LLM final", "final_curated_features_llm_top100_by_category.jsonl"),
]


def parse_count(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def matched_expected_rows(
    expected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed_rows = [(row, row_terms(row)) for row in rows]
    matches: dict[str, dict[str, Any]] = {}
    for item in expected:
        expected_aliases = set(item["aliases"])
        for row, terms in indexed_rows:
            if expected_aliases & terms:
                matches[item["canonical"]] = row
                break
    return matches


def stage_metrics(
    expected: list[dict[str, Any]],
    extract_dir: Path,
) -> list[dict[str, Any]]:
    metrics = []
    for label, filename in STAGE_FILES:
        path = extract_dir / filename
        if not path.exists():
            metrics.append(
                {
                    "label": label,
                    "path": path,
                    "row_count": 0,
                    "matched_count": 0,
                    "missing_count": len(expected),
                }
            )
            continue
        rows = load_feature_rows(path)
        matches = matched_expected_rows(expected, rows)
        metrics.append(
            {
                "label": label,
                "path": path,
                "row_count": len(rows),
                "matched_count": len(matches),
                "missing_count": len(expected) - len(matches),
            }
        )
    return metrics


def final_feature_rows(extract_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    candidates = [
        extract_dir / "final_curated_features_llm_top100_by_category.jsonl",
        extract_dir / "final_curated_features_llm_top100_by_category.csv",
        extract_dir / "final_curated_features_top100_by_category.csv",
    ]
    for path in candidates:
        if path.exists():
            return path, load_feature_rows(path)
    raise FileNotFoundError(f"No final feature artifact found in {extract_dir}")


def short_label(value: str, max_chars: int = 34) -> str:
    value = value.replace(", ", ",\n")
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "."


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
        }
    )


def plot_stage_coverage(
    expected: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    labels = [item["label"] for item in metrics]
    matched = [item["matched_count"] for item in metrics]
    row_counts = [item["row_count"] for item in metrics]
    total = len(expected)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bars = ax.bar(labels, matched, color="#2c7a7b", width=0.58)
    ax.axhline(total, color="#444444", linestyle="--", linewidth=1.2, label=f"Expected labels ({total})")
    ax.set_ylim(0, total + 1.2)
    ax.set_ylabel("Expected labels recovered")
    ax.set_title("SemEval-2020 Task 11 Expected-Label Coverage by Pipeline Stage")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    for bar, count, rows in zip(bars, matched, row_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{count}/{total}\n{rows} rows",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    path = output_dir / "stage_expected_coverage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_expected_support(
    expected: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    matches = matched_expected_rows(expected, final_rows)
    labels = [item["canonical"] for item in expected]
    paper_counts = [parse_count(matches.get(label, {}).get("paper_count")) for label in labels]
    mention_counts = [parse_count(matches.get(label, {}).get("mention_count")) for label in labels]

    y_positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    ax.barh(y_positions, mention_counts, color="#b7d7d8", label="Mentions", height=0.72)
    ax.barh(y_positions, paper_counts, color="#285e61", label="Papers", height=0.42)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([short_label(label) for label in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Support count in final features")
    ax.set_title("Expected SemEval Labels Recovered in Final PDF Extraction")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    for y, papers, mentions in zip(y_positions, paper_counts, mention_counts):
        value = max(papers, mentions)
        ax.text(value + 0.6, y, f"{papers} papers / {mentions} mentions", va="center", fontsize=8.5)

    fig.tight_layout()
    path = output_dir / "expected_label_support.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_category_mix(final_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    counts: dict[str, int] = {}
    for row in final_rows:
        category = str(row.get("category") or "uncategorized").strip() or "uncategorized"
        counts[category] = counts.get(category, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    labels = [label for label, _ in ordered]
    values = [value for _, value in ordered]

    fig_height = max(3.8, 0.45 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    bars = ax.barh(labels, values, color="#805ad5", height=0.6)
    ax.set_xlabel("Final feature rows")
    ax.set_title("Final Feature Category Mix")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    for bar, value in zip(bars, values):
        ax.text(value + 0.2, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)

    fig.tight_layout()
    path = output_dir / "final_category_mix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_unmatched_final_features(
    expected: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    matches = matched_expected_rows(expected, final_rows)
    matched_labels = {normalize(row_label(row)) for row in matches.values()}
    unmatched = [
        row
        for row in final_rows
        if normalize(row_label(row)) not in matched_labels
    ]
    if not unmatched:
        return None

    top_rows = sorted(
        unmatched,
        key=lambda row: (parse_count(row.get("paper_count")), parse_count(row.get("mention_count"))),
        reverse=True,
    )[:15]
    labels = [short_label(row_label(row), max_chars=42) for row in top_rows]
    values = [parse_count(row.get("paper_count")) for row in top_rows]

    fig_height = max(4.0, 0.46 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    bars = ax.barh(list(range(len(labels))), values, color="#d69e2e", height=0.62)
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paper count")
    ax.set_title("Top Final Features Outside the Official SemEval Label List")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    for bar, row in zip(bars, top_rows):
        mentions = parse_count(row.get("mention_count"))
        ax.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{mentions} mentions",
            va="center",
            fontsize=8.5,
        )

    fig.tight_layout()
    path = output_dir / "top_unmatched_final_features.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures for the SemEval-2020 Task 11 sanity-check extraction."
    )
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    expected = load_expected(args.expected)
    metrics = stage_metrics(expected, args.extract_dir)
    final_path, final_rows = final_feature_rows(args.extract_dir)

    paths = [
        plot_stage_coverage(expected, metrics, args.output_dir),
        plot_expected_support(expected, final_rows, args.output_dir),
        plot_category_mix(final_rows, args.output_dir),
    ]
    unmatched_path = plot_unmatched_final_features(expected, final_rows, args.output_dir)
    if unmatched_path is not None:
        paths.append(unmatched_path)

    print(f"Final feature source: {final_path}")
    print("Wrote figures:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
