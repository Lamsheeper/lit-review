"""Features filtering."""

from __future__ import annotations

from lit_review.core.context import family_order
from lit_review.core.llm import make_llm_client

from lit_review.core.io import append_jsonl, read_jsonl, write_jsonl


import csv

import dataclasses

import hashlib

import json


import time


from dataclasses import dataclass

from pathlib import Path

from typing import Any, Iterable

from lit_review.core.io import utc_now, write_json

from lit_review.core.chat import ChatCompletionClient


from lit_review.core.chat import clean_api_key

from lit_review.features.records import ensure_feature_identity, feature_key, load_goal_text
from lit_review.core.text import normalize_feature_name

from lit_review.features.prompts import build_curation_prompt

from lit_review.features.responses import LOGGER, parse_curation_json_with_repair

from lit_review.core.progress import progress_bar


VERSION = "0.2.0"


PROMPT_VERSION = "lit_extract_v1"


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


FILTERED_FEATURE_FIELDNAMES = list(
    dict.fromkeys(
        [
            "feature_key",
            *FEATURE_FIELDNAMES,
            "curation_confidence",
            "curation_notes",
            "curated_at",
        ]
    )
)


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


@dataclass
class CurationStats:
    feature_count: int = 0
    batch_count: int = 0
    completed_batch_count: int = 0
    skipped_batch_count: int = 0
    failed_batch_count: int = 0
    decision_count: int = 0
    repaired_json_count: int = 0


def curation_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Curating features", unit="feature", show=show)


def json_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_csv_value(row.get(field)) for field in fieldnames})
    tmp_path.replace(path)


def load_feature_rows(input_dir: Path) -> list[dict[str, Any]]:
    features_path = input_dir / "features.jsonl"
    if not features_path.exists():
        raise FileNotFoundError(f"features.jsonl does not exist: {features_path}")
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

    ordered_categories = [category for category in family_order() if category in by_category]
    ordered_categories.extend(
        category for category in sorted(by_category) if category not in set(ordered_categories)
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

    feature_key_value = str(raw.get("feature_key") or feature.get("feature_key") or feature_key(feature))
    category = str(feature.get("category") or "")
    original_name = str(feature.get("feature_name") or "")
    original_normalized = str(feature.get("normalized_feature_name") or normalize_feature_name(original_name))

    action = "reject" if action == "reject" else "keep"
    canonical_name = "" if action == "reject" else original_name
    canonical_normalized = "" if action == "reject" else original_normalized

    if action == "reject":
        canonical_name = ""
        canonical_normalized = ""

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
            "action": "keep",
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
    features_by_key = {
        str(feature.get("feature_key") or feature_key(feature)): feature for feature in features
    }
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


def build_filter_curation_outputs(
    features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features_by_key = {
        str(feature.get("feature_key") or feature_key(feature)): feature for feature in features
    }
    decision_map = latest_decisions_by_feature(decisions)
    filtered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for key, feature in features_by_key.items():
        decision = decision_map.get(key)
        if decision is None:
            continue
        if decision.get("action") == "reject":
            rejected.append(
                {
                    **feature,
                    "rejection_reason": decision.get("rejection_reason"),
                    "curation_confidence": decision.get("curation_confidence"),
                    "curation_notes": decision.get("curation_notes"),
                    "model": decision.get("model"),
                    "run_id": decision.get("run_id"),
                    "curated_at": decision.get("curated_at"),
                }
            )
            continue
        filtered.append(
            {
                **feature,
                "feature_key": key,
                "curation_confidence": decision.get("curation_confidence"),
                "curation_notes": decision.get("curation_notes"),
                "curated_at": decision.get("curated_at"),
            }
        )
    return filtered, rejected


def write_curation_artifacts(
    output_dir: Path,
    features: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, int]:
    filtered, rejected = build_filter_curation_outputs(features, decisions)
    write_jsonl(output_dir / "filtered_features.jsonl", filtered)
    write_csv(output_dir / "filtered_features.csv", filtered, FILTERED_FEATURE_FIELDNAMES)
    write_jsonl(output_dir / "rejected_features.jsonl", rejected)
    write_csv(output_dir / "rejected_features.csv", rejected, REJECTED_FEATURE_FIELDNAMES)
    return {
        "filtered_features": len(filtered),
        "curation_decisions": len(decisions),
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
    decisions = read_jsonl(output_dir / "curation_decisions.jsonl")
    if decisions:
        return decisions
    return read_jsonl(output_dir / "curation_batch_decisions.jsonl")


def remove_curation_outputs(output_dir: Path) -> None:
    for name in [
        "filtered_features.jsonl",
        "filtered_features.csv",
        "curated_feature_members.jsonl",
        "curated_feature_members.csv",
        "rejected_features.jsonl",
        "rejected_features.csv",
        "curation_trace.json",
        "curation_errors.jsonl",
        "completed_curation_batches.jsonl",
        "curation_decisions.jsonl",
        "curation_batch_decisions.jsonl",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


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
    api_provider: str = "auto",
    reasoning_effort: str | None,
    batch_features: int,
    goal_path: Path | None = None,
    force: bool,
    dry_run: bool,
    temperature: float,
    max_tokens: int = 6000,
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
        )

    if client is None:
        if not api_key:
            raise ValueError("An API key is required unless --dry-run is used.")
        client = make_llm_client(
            api_key=clean_api_key(api_key),
            api_provider=api_provider,
            input_mode="chunks",
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
        len(batch["features"]) for batch in batches if str(batch["batch_id"]) not in completed
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
                                "You are an evidence-bound taxonomy curation agent. Return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_curation_prompt(
                                features=batch_rows,
                                category=str(batch["category"]),
                                goal_text=goal_text,
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
                batch_decisions = normalize_curation_decisions(
                    raw_decisions,
                    batch_rows,
                    model=model,
                    run_id=run_id,
                    curated_at=utc_now(),
                )
                for decision in batch_decisions:
                    append_jsonl(
                        output_dir / "curation_decisions.jsonl",
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
                    "feature_keys": [str(row.get("feature_key") or feature_key(row)) for row in batch_rows],
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
