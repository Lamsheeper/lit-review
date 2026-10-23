"""Features merging."""

from __future__ import annotations


import dataclasses

import hashlib


import time


from dataclasses import dataclass

from pathlib import Path

from typing import Any, Iterable

from lit_review.core.io import utc_now, write_json


from lit_review.embeddings.core import (
    DEFAULT_GEMINI_MODEL as DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    DEFAULT_TASK_TYPE,
    DEFAULT_TEXT_FIELDS,
    EmbeddingClient,
    CacheStats,
    EmbeddingTextCleanup,
    FeatureItem,
    build_embedding_text,
    embed_texts_with_cache,
    parse_text_fields,
)

from lit_review.embeddings.similarity import CategoryMergeSpec, SimilarityPair, normalized_matrix, pair_json


from lit_review.core.llm import make_llm_client

from lit_review.features.records import feature_key, load_goal_text, read_feature_inventory, truncate_text
from lit_review.core.text import normalize_feature_name

from lit_review.features.prompts import build_llm_merge_prompt, scoring_mention_count, scoring_paper_count

from lit_review.features.responses import LOGGER, parse_llm_merge_json_with_repair

from lit_review.features.filtering import VERSION
from lit_review.core.io import append_jsonl, read_jsonl
from lit_review.core.progress import progress_bar


from lit_review.features.merge_artifacts import latest_llm_merge_decisions_by_pair, write_llm_merge_artifacts


LLM_MERGE_PROMPT_VERSION = "lit_extract_llm_merge_v1"


@dataclass
class LlmMergeStats:
    feature_count: int = 0
    embedded_feature_count: int = 0
    unembedded_feature_count: int = 0
    candidate_pair_count: int = 0
    batch_count: int = 0
    completed_batch_count: int = 0
    skipped_batch_count: int = 0
    failed_batch_count: int = 0
    decision_count: int = 0
    accepted_pair_count: int = 0
    merge_group_count: int = 0
    merged_feature_count: int = 0
    repaired_json_count: int = 0


def llm_merge_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Reviewing merge pairs", unit="pair", show=show)


def llm_merge_embedding_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Embedding merge features", unit="feature", show=show)


def llm_merge_candidate_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Finding merge candidates", unit="feature", show=show)


def llm_merge_feature_label(row: dict[str, Any]) -> str:
    for field in ("feature_name", "canonical_feature_name", "normalized_feature_name", "feature_key"):
        value = str(row.get(field) or "").strip()
        if value:
            if field == "feature_key" and "::" in value:
                return value.split("::", 1)[1].strip() or value
            return value
    return "feature"


def llm_merge_specs_for_rows(
    rows: list[dict[str, Any]],
    category_config: Path | None,
) -> tuple[dict[str, CategoryMergeSpec], dict[str, dict[str, Any]]]:
    categories = {str(row.get("category") or "").strip() or "uncategorized" for row in rows}
    if category_config is None:
        specs = {
            category: CategoryMergeSpec(
                text_fields=parse_text_fields(DEFAULT_TEXT_FIELDS),
                threshold=1.0,
            )
            for category in categories
        }
        ignored: dict[str, dict[str, Any]] = {}
        return specs, ignored

    from lit_review.core.io import read_json
    from lit_review.embeddings.core import EmbeddingTextCleanup

    payload = read_json(category_config)
    specs = {}
    for category, values in payload.items():
        if set(values) - {"text_fields", "text_cleanup"}:
            raise ValueError("Threshold and linkage apply only to analysis clustering")
        cleanup = EmbeddingTextCleanup(**values["text_cleanup"]) if values.get("text_cleanup") else None
        specs[category] = CategoryMergeSpec(
            text_fields=parse_text_fields(values.get("text_fields", DEFAULT_TEXT_FIELDS)),
            threshold=1.0,
            embedding_text_cleanup=cleanup,
        )
    missing = categories - set(specs)
    if missing:
        raise ValueError(f"Category config is missing input categories: {sorted(missing)}")
    return specs, {}


def llm_merge_text_cleanup_fields(cleanup: EmbeddingTextCleanup | None) -> Iterable[str] | None:
    if cleanup is None:
        return None
    return None


def build_llm_merge_items(
    rows: list[dict[str, Any]],
    specs: dict[str, CategoryMergeSpec],
) -> list[FeatureItem]:
    items: list[FeatureItem] = []
    for index, row in enumerate(rows, start=1):
        category = str(row.get("category") or "").strip() or "uncategorized"
        spec = specs.get(category)
        if spec is None:
            continue
        embedding_text = build_embedding_text(
            row,
            spec.text_fields,
            cleanup=spec.embedding_text_cleanup,
            cleanup_fields=llm_merge_text_cleanup_fields(spec.embedding_text_cleanup),
        )
        if not embedding_text:
            continue
        label = llm_merge_feature_label(row)
        items.append(
            FeatureItem(
                row_index=index,
                feature_key=str(row.get("feature_key") or feature_key(row)),
                label=label,
                category=category,
                paper_count=scoring_paper_count(row),
                mention_count=scoring_mention_count(row),
                embedding_text=embedding_text,
                raw_row=row,
            )
        )
    return items


def prepare_llm_merge_embeddings(
    *,
    input_file: Path,
    category_config: Path | None,
    cache_path: Path,
    embedding_client: EmbeddingClient | None,
    embedding_model: str,
    embedding_task_type: str,
    embedding_output_dimensionality: int,
    embedding_batch_size: int,
    show_progress: bool = True,
) -> tuple[
    list[dict[str, Any]],
    dict[str, CategoryMergeSpec],
    dict[str, dict[str, Any]],
    list[FeatureItem],
    list[list[float]],
    CacheStats,
]:
    rows = read_feature_inventory(input_file)
    specs, ignored = llm_merge_specs_for_rows(rows, category_config)
    items = build_llm_merge_items(rows, specs)
    if not items:
        return rows, specs, ignored, items, [], CacheStats(cache_hits=0, api_misses=0, api_batches=0)
    with llm_merge_embedding_progress(total=len(items), show=show_progress) as progress:
        embeddings, cache_stats = embed_texts_with_cache(
            [item.embedding_text for item in items],
            cache_path=cache_path,
            client=embedding_client,
            model=embedding_model,
            task_type=embedding_task_type,
            output_dimensionality=embedding_output_dimensionality,
            batch_size=embedding_batch_size,
            progress_callback=progress.update,
        )
    return rows, specs, ignored, items, embeddings, cache_stats


def top_k_llm_merge_candidates(
    items: list[FeatureItem],
    embeddings: list[list[float]],
    *,
    top_k: int,
    candidate_min_similarity: float | None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    if not items or not embeddings:
        return []
    matrix = normalized_matrix(embeddings)
    candidates_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    with llm_merge_candidate_progress(total=len(items), show=show_progress) as progress:
        for index, item in enumerate(items):
            similarities = matrix[index] @ matrix.T
            ranked: list[tuple[float, int]] = []
            for other_index, other in enumerate(items):
                if other_index == index or other.category != item.category:
                    continue
                similarity = round(float(similarities[other_index]), 8)
                if candidate_min_similarity is not None and similarity < candidate_min_similarity:
                    continue
                ranked.append((similarity, other_index))
            ranked.sort(
                key=lambda value: (
                    -value[0],
                    items[value[1]].label.casefold(),
                    items[value[1]].row_index,
                )
            )
            for rank, (similarity, other_index) in enumerate(ranked[:top_k], start=1):
                left_index, right_index = sorted((index, other_index))
                pair_key = (left_index, right_index)
                row = candidates_by_pair.get(pair_key)
                if row is None:
                    pair = SimilarityPair(
                        left_index=left_index, right_index=right_index, similarity=similarity
                    )
                    pair_payload = pair_json(pair, items)
                    row = {
                        "pair_id": "",
                        "similarity": similarity,
                        "category": item.category,
                        "selected_by_feature_keys": [],
                        "selected_by_ranks": [],
                        **pair_payload,
                    }
                    candidates_by_pair[pair_key] = row
                row["selected_by_feature_keys"].append(item.feature_key)
                row["selected_by_ranks"].append(rank)
            progress.update(1)

    candidates = sorted(
        candidates_by_pair.values(),
        key=lambda row: (
            -float(row.get("similarity") or 0.0),
            str(row.get("category") or "").casefold(),
            str(row.get("feature_name_1") or "").casefold(),
            str(row.get("feature_name_2") or "").casefold(),
        ),
    )
    for index, row in enumerate(candidates, start=1):
        row["pair_id"] = f"merge_pair_{index:06d}"
    return candidates


def llm_merge_batch_id(batch_index: int, rows: list[dict[str, Any]]) -> str:
    digest_source = "\n".join(str(row.get("pair_id") or "") for row in rows)
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"llm_merge:{batch_index}:{digest}"


def prepare_llm_merge_batches(
    candidates: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    size = max(1, batch_size)
    for batch_index, start in enumerate(range(0, len(candidates), size), start=1):
        rows = candidates[start : start + size]
        batches.append(
            {
                "batch_id": llm_merge_batch_id(batch_index, rows),
                "batch_index": batch_index,
                "candidates": rows,
            }
        )
    return batches


def canonical_llm_merge_decision(value: Any) -> str:
    normalized = normalize_feature_name(str(value or ""))
    if normalized in {"merge", "merged", "same", "same concept", "equivalent", "duplicate", "alias"}:
        return "merge"
    if normalized in {"separate", "separated", "different", "distinct", "no merge", "not merge"}:
        return "separate"
    return "uncertain"


def llm_merge_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.5
    return round(max(0.0, min(1.0, confidence)), 4)


def fallback_llm_merge_decision(
    candidate: dict[str, Any],
    *,
    model: str,
    run_id: str,
    decided_at: str,
) -> dict[str, Any]:
    return normalize_llm_merge_decision(
        {
            "pair_id": candidate.get("pair_id"),
            "decision": "uncertain",
            "confidence": 0.0,
            "rationale": "No model decision returned; left unmerged.",
        },
        candidate,
        model=model,
        run_id=run_id,
        decided_at=decided_at,
    )


def normalize_llm_merge_decision(
    raw: dict[str, Any],
    candidate: dict[str, Any],
    *,
    model: str,
    run_id: str,
    decided_at: str,
) -> dict[str, Any]:
    return {
        "pair_id": str(raw.get("pair_id") or candidate.get("pair_id") or ""),
        "similarity": candidate.get("similarity"),
        "category": candidate.get("category"),
        "feature_key_1": candidate.get("feature_key_1"),
        "feature_name_1": candidate.get("feature_name_1"),
        "feature_key_2": candidate.get("feature_key_2"),
        "feature_name_2": candidate.get("feature_name_2"),
        "decision": canonical_llm_merge_decision(raw.get("decision")),
        "confidence": llm_merge_confidence(raw.get("confidence")),
        "rationale": truncate_text(str(raw.get("rationale") or raw.get("notes") or ""), 500),
        "model": model,
        "run_id": run_id,
        "decided_at": decided_at,
    }


def normalize_llm_merge_decisions(
    raw_decisions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    model: str,
    run_id: str,
    decided_at: str,
) -> list[dict[str, Any]]:
    candidates_by_id = {str(row.get("pair_id") or ""): row for row in candidates}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_decisions:
        pair_id = str(raw.get("pair_id") or "")
        candidate = candidates_by_id.get(pair_id)
        if candidate is None or pair_id in seen:
            continue
        output.append(
            normalize_llm_merge_decision(
                raw,
                candidate,
                model=model,
                run_id=run_id,
                decided_at=decided_at,
            )
        )
        seen.add(pair_id)
    for pair_id, candidate in candidates_by_id.items():
        if pair_id not in seen:
            output.append(
                fallback_llm_merge_decision(
                    candidate,
                    model=model,
                    run_id=run_id,
                    decided_at=decided_at,
                )
            )
    return output


def completed_llm_merge_batch_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    for row in read_jsonl(output_dir / "completed_llm_merge_batches.jsonl"):
        if str(row.get("prompt_version") or "") != LLM_MERGE_PROMPT_VERSION:
            continue
        batch_id = str(row.get("batch_id") or "")
        if batch_id:
            completed.add(batch_id)
    return completed


def load_existing_llm_merge_decisions(output_dir: Path) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in read_jsonl(output_dir / "llm_merge_decisions.jsonl"):
        if str(row.get("prompt_version") or "") not in {"", LLM_MERGE_PROMPT_VERSION}:
            continue
        decisions.append(row)
    return decisions


def remove_llm_merge_outputs(output_dir: Path) -> None:
    for name in [
        "llm_merge_candidates.jsonl",
        "llm_merge_candidates.csv",
        "llm_merge_decisions.jsonl",
        "llm_merge_decisions.csv",
        "llm_merged_features.json",
        "llm_merged_features.jsonl",
        "llm_merged_features.csv",
        "llm_merge_summary_by_category.json",
        "llm_merge_retention_by_category.png",
        "llm_merge_qualitative_examples.md",
        "llm_merge_trace.json",
        "llm_merge_errors.jsonl",
        "completed_llm_merge_batches.jsonl",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def dry_run_llm_merge_trace(
    *,
    input_file: Path,
    output_dir: Path,
    features: list[dict[str, Any]],
    specs: dict[str, CategoryMergeSpec],
    ignored_config: dict[str, dict[str, Any]],
    model: str,
    run_id: str,
    goal_path: str | None,
    goal_text: str,
    category_config: Path | None,
    top_k: int,
    candidate_min_similarity: float | None,
    batch_pairs: int,
    max_tokens: int,
    timeout: float | None,
) -> dict[str, Any]:
    stats = LlmMergeStats(feature_count=len(features))
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": LLM_MERGE_PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": True,
        "model": model,
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "goal_path": goal_path,
        "goal_used": bool(goal_text.strip()),
        "category_config": str(category_config) if category_config else None,
        "category_text_fields": {category: spec.text_fields for category, spec in specs.items()},
        "ignored_category_config_fields": ignored_config,
        "config": {
            "top_k": top_k,
            "candidate_min_similarity": candidate_min_similarity,
            "batch_pairs": batch_pairs,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "merge_mode": "embedding_top_k_llm_review",
        },
        "counts": dataclasses.asdict(stats),
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "llm_merge_trace.json", trace)
    return trace


def run_llm_merge(
    *,
    input_file: Path,
    output_dir: Path | None,
    api_key: str | None,
    model: str,
    base_url: str,
    api_provider: str = "auto",
    reasoning_effort: str | None,
    category_config: Path | None = None,
    top_k: int = 10,
    candidate_min_similarity: float | None = None,
    batch_pairs: int = 25,
    goal_path: Path | None = None,
    force: bool,
    dry_run: bool,
    temperature: float,
    max_tokens: int,
    timeout: float | None = None,
    show_progress: bool = True,
    cache_path: Path | None = None,
    embedding_client: EmbeddingClient | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_task_type: str = DEFAULT_TASK_TYPE,
    embedding_output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
    embedding_batch_size: int = 50,
    client: Any | None = None,
) -> dict[str, Any]:
    input_file = input_file.expanduser()
    output_dir = output_dir.expanduser() if output_dir is not None else input_file.parent / "llm_merge"
    category_config = category_config.expanduser() if category_config is not None else None
    if api_provider not in {"auto", "openai", "gemini"}:
        raise ValueError("api_provider must be 'auto', 'openai', or 'gemini'.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if batch_pairs <= 0:
        raise ValueError("batch_pairs must be greater than 0.")
    if candidate_min_similarity is not None and not -1.0 <= candidate_min_similarity <= 1.0:
        raise ValueError("candidate_min_similarity must be between -1 and 1.")
    if embedding_output_dimensionality <= 0:
        raise ValueError("embedding_output_dimensionality must be greater than 0.")
    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be greater than 0.")
    max_tokens = max(1, int(max_tokens))
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        remove_llm_merge_outputs(output_dir)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    features = read_feature_inventory(input_file)
    specs, ignored_config = llm_merge_specs_for_rows(features, category_config)
    goal_text, resolved_goal_path = load_goal_text(goal_path)
    if dry_run:
        return dry_run_llm_merge_trace(
            input_file=input_file,
            output_dir=output_dir,
            features=features,
            specs=specs,
            ignored_config=ignored_config,
            model=model,
            run_id=run_id,
            goal_path=resolved_goal_path,
            goal_text=goal_text,
            category_config=category_config,
            top_k=top_k,
            candidate_min_similarity=candidate_min_similarity,
            batch_pairs=batch_pairs,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    resolved_cache_path = (
        cache_path.expanduser()
        if cache_path is not None
        else output_dir / f"{input_file.stem}_llm_merge_embedding_cache.jsonl"
    )
    features, specs, ignored_config, items, embeddings, cache_stats = prepare_llm_merge_embeddings(
        input_file=input_file,
        category_config=category_config,
        cache_path=resolved_cache_path,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        embedding_task_type=embedding_task_type,
        embedding_output_dimensionality=embedding_output_dimensionality,
        embedding_batch_size=embedding_batch_size,
        show_progress=show_progress,
    )
    candidates = top_k_llm_merge_candidates(
        items,
        embeddings,
        top_k=top_k,
        candidate_min_similarity=candidate_min_similarity,
        show_progress=show_progress,
    )
    batches = prepare_llm_merge_batches(candidates, batch_size=batch_pairs)
    completed = completed_llm_merge_batch_ids(output_dir)
    decisions = load_existing_llm_merge_decisions(output_dir)
    output_counts = write_llm_merge_artifacts(
        output_dir=output_dir,
        input_file=input_file,
        features=features,
        items=items,
        candidates=candidates,
        decisions=decisions,
    )
    stats = LlmMergeStats(
        feature_count=len(features),
        embedded_feature_count=len(items),
        unembedded_feature_count=len(features) - len(items),
        candidate_pair_count=len(candidates),
        batch_count=len(batches),
        accepted_pair_count=int(output_counts.get("accepted_pair_count") or 0),
        merge_group_count=int(output_counts.get("merge_group_count") or 0),
        merged_feature_count=int(output_counts.get("llm_merged_features") or len(features)),
    )
    failures: list[dict[str, Any]] = []

    pending_pair_count = sum(
        len(batch["candidates"]) for batch in batches if str(batch["batch_id"]) not in completed
    )
    if pending_pair_count and client is None:
        if not api_key:
            raise ValueError("An API key is required to review merge candidates unless --dry-run is used.")
        client = make_llm_client(
            api_key=api_key,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            api_provider=api_provider,
            input_mode="text",
            timeout=timeout,
        )

    rows_by_key = {str(row.get("feature_key") or feature_key(row)): row for row in features}
    with llm_merge_progress(total=pending_pair_count, show=show_progress) as progress:
        for batch in batches:
            batch_id = str(batch["batch_id"])
            batch_candidates = list(batch["candidates"])
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
                                "You are an evidence-bound taxonomy merge-review agent. "
                                "Return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_llm_merge_prompt(
                                candidates=batch_candidates,
                                rows_by_key=rows_by_key,
                                goal_text=goal_text,
                            ),
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_decisions, repaired_json = parse_llm_merge_json_with_repair(
                    raw_response,
                    client=client,
                )
                if repaired_json:
                    stats.repaired_json_count += 1
                batch_decisions = normalize_llm_merge_decisions(
                    raw_decisions,
                    batch_candidates,
                    model=model,
                    run_id=run_id,
                    decided_at=utc_now(),
                )
                enriched_decisions = [
                    {
                        **decision,
                        "batch_id": batch_id,
                        "batch_index": batch.get("batch_index"),
                        "prompt_version": LLM_MERGE_PROMPT_VERSION,
                    }
                    for decision in batch_decisions
                ]
                for decision in enriched_decisions:
                    append_jsonl(output_dir / "llm_merge_decisions.jsonl", decision)
                append_jsonl(
                    output_dir / "completed_llm_merge_batches.jsonl",
                    {
                        "batch_id": batch_id,
                        "batch_index": batch.get("batch_index"),
                        "candidate_pair_count": len(batch_candidates),
                        "decision_count": len(batch_decisions),
                        "prompt_version": LLM_MERGE_PROMPT_VERSION,
                        "run_id": run_id,
                        "completed_at": utc_now(),
                    },
                )
                completed.add(batch_id)
                decisions.extend(enriched_decisions)
                stats.completed_batch_count += 1
                stats.decision_count += len(enriched_decisions)
                stats.accepted_pair_count = sum(
                    1
                    for row in latest_llm_merge_decisions_by_pair(decisions).values()
                    if row.get("decision") == "merge"
                )
                output_counts = write_llm_merge_artifacts(
                    output_dir=output_dir,
                    input_file=input_file,
                    features=features,
                    items=items,
                    candidates=candidates,
                    decisions=decisions,
                )
                stats.merge_group_count = int(output_counts.get("merge_group_count") or 0)
                stats.merged_feature_count = int(output_counts.get("llm_merged_features") or 0)
            except Exception as exc:
                stats.failed_batch_count += 1
                error_row = {
                    "batch_id": batch_id,
                    "batch_index": batch.get("batch_index"),
                    "pair_ids": [str(row.get("pair_id") or "") for row in batch_candidates],
                    "error": str(exc),
                    "raw_response_excerpt": raw_response[:2000] if raw_response else "",
                    "run_id": run_id,
                    "created_at": utc_now(),
                }
                failures.append(error_row)
                append_jsonl(output_dir / "llm_merge_errors.jsonl", error_row)
                LOGGER.warning("LLM merge failed for %s: %s", batch_id, exc)
            finally:
                progress.update(len(batch_candidates))

    errors_path = output_dir / "llm_merge_errors.jsonl"
    if not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")
    final_outputs = write_llm_merge_artifacts(
        output_dir=output_dir,
        input_file=input_file,
        features=features,
        items=items,
        candidates=candidates,
        decisions=decisions,
    )
    stats.accepted_pair_count = int(final_outputs.get("accepted_pair_count") or 0)
    stats.merge_group_count = int(final_outputs.get("merge_group_count") or 0)
    stats.merged_feature_count = int(final_outputs.get("llm_merged_features") or 0)
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": LLM_MERGE_PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": False,
        "model": model,
        "base_url": base_url,
        "api_provider": api_provider,
        "reasoning_effort": reasoning_effort,
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "goal_path": resolved_goal_path,
        "goal_used": bool(goal_text.strip()),
        "category_config": str(category_config) if category_config else None,
        "category_text_fields": {category: spec.text_fields for category, spec in specs.items()},
        "ignored_category_config_fields": ignored_config,
        "embedding": {
            "model": embedding_model,
            "task_type": embedding_task_type,
            "output_dimensionality": embedding_output_dimensionality,
            "cache_path": str(resolved_cache_path),
            "cache_stats": dataclasses.asdict(cache_stats),
        },
        "config": {
            "top_k": top_k,
            "candidate_min_similarity": candidate_min_similarity,
            "batch_pairs": batch_pairs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "force": force,
            "progress": show_progress,
            "merge_mode": "embedding_top_k_llm_review",
        },
        "counts": dataclasses.asdict(stats),
        "outputs": final_outputs,
        "failures": failures,
    }
    write_json(output_dir / "llm_merge_trace.json", trace)
    return trace
