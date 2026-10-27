"""Analysis visualization."""

from __future__ import annotations

from lit_review.embeddings.core import add_google_auth_arguments, make_embedding_client

import argparse

import csv


import html

import json

import math


import re


import sys


from pathlib import Path

from typing import Any

from lit_review.embeddings.core import (
    FeatureItem,
    first_text_value,
    DEFAULT_RANDOM_STATE,
    DEFAULT_ANNOTATE_TOP,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_TASK_TYPE,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    DEFAULT_TEXT_FIELDS,
    gemini_api_key,
    parse_text_fields,
    load_feature_items,
    embed_texts_with_cache,
    sanitize_filename_part,
    positive_int,
    nonnegative_int,
)


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
        category: color_map(index % color_map.N) for index, category in enumerate(categories)
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
    return {category: palette[index % len(palette)] for index, category in enumerate(categories)}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed a feature set with Gemini and visualize it with t-SNE.",
    )
    parser.add_argument("features", type=Path, help="Feature CSV, JSONL, or JSON file.")
    parser.add_argument("--output-dir", type=Path, help="Directory for PNG/CSV outputs.")
    parser.add_argument("--prefix", help="Output filename prefix. Defaults to the input stem.")
    parser.add_argument("--cache", type=Path, help="Embedding cache JSONL path.")
    parser.add_argument(
        "--html", action="store_true", help="Also write a self-contained interactive HTML plot."
    )
    parser.add_argument("--html-path", type=Path, help="Interactive HTML output path. Implies --html.")
    parser.add_argument(
        "--text-fields",
        default=",".join(DEFAULT_TEXT_FIELDS),
        help="Comma-separated row fields used to build embedding text.",
    )
    parser.add_argument(
        "--gemini-api-key", help="Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY."
    )
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
    parser.add_argument(
        "--perplexity", type=float, help="t-SNE perplexity. Defaults to an automatic valid value."
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="t-SNE random seed.")
    parser.add_argument(
        "--annotate-top",
        type=nonnegative_int,
        default=DEFAULT_ANNOTATE_TOP,
        help="Number of highest-support features to label on the plot.",
    )
    parser.add_argument("--dpi", type=positive_int, default=220, help="PNG output DPI.")
    add_google_auth_arguments(parser)
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
        client = make_embedding_client(
            api_key=api_key, auth_mode=args.auth_mode,
            vertex_project=args.vertex_project, vertex_location=args.vertex_location,
            model=args.model, base_url=args.base_url, task_type=args.task_type,
            output_dimensionality=args.output_dimensionality, timeout=args.timeout,
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
    print(
        f"Embedding cache: {stats.cache_hits} hits, {stats.api_misses} misses, {stats.api_batches} API batches"
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    if args.html or args.html_path:
        print(f"Wrote {html_path}")
    print(f"Cache path: {cache_path}")
    return 0
