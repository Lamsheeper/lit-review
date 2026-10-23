"""Optional figures must not prevent inventories from being exported."""

import logging
from pathlib import Path
from typing import Any


def optional_pyplot():
    try:
        import matplotlib
    except ImportError:
        logging.getLogger(__name__).info("Skipping figures; install lit-review[analysis] to enable plotting.")
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_category_retention_plot(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    plt = optional_pyplot()
    if plt is None:
        return None

    categories = [str(row["category"]) for row in rows]
    retained = [int(row["retained_features"]) for row in rows]
    merged_away = [int(row["merged_away_features"]) for row in rows]
    fig_width = max(8.0, min(18.0, 5.0 + len(categories) * 1.4))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    positions = list(range(len(categories)))
    green_bars = ax.bar(
        positions,
        retained,
        color="#16a34a",
        edgecolor="#ffffff",
        label="Retained after merge",
    )
    ax.bar(
        positions,
        merged_away,
        bottom=retained,
        color="#dc2626",
        edgecolor="#ffffff",
        label="Merged away",
    )
    ax.bar_label(green_bars, labels=[str(value) for value in retained], label_type="center", color="#ffffff")
    max_input = max((int(row["input_features"]) for row in rows), default=0)
    for position, retained_count, merged_away_count in zip(positions, retained, merged_away):
        if not merged_away_count:
            continue
        if merged_away_count >= max(1, max_input * 0.035):
            ax.text(
                position,
                retained_count + merged_away_count / 2,
                str(merged_away_count),
                ha="center",
                va="center",
                color="#ffffff",
            )
        else:
            ax.text(
                position,
                retained_count + merged_away_count + max(2, max_input * 0.008),
                str(merged_away_count),
                ha="center",
                va="bottom",
                color="#b91c1c",
                fontweight="bold",
            )
    ax.set_title("Feature Retention by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Input feature count")
    ax.set_xticks(
        positions, labels=[category.replace("_", " ") for category in categories], rotation=20, ha="right"
    )
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
