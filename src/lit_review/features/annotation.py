"""Features annotation."""

from __future__ import annotations


import dataclasses

import hashlib


import time


from dataclasses import dataclass

from pathlib import Path

from typing import Any

from lit_review.core.io import utc_now, write_json


from lit_review.core.llm import make_llm_client

from lit_review.features.records import feature_key, load_goal_text, read_feature_inventory

from lit_review.features.prompts import build_annotation_prompt

from lit_review.features.responses import LOGGER, parse_annotation_json_with_repair

from lit_review.features.filtering import VERSION
from lit_review.core.io import append_jsonl, read_jsonl
from lit_review.core.progress import progress_bar


from lit_review.features.annotation_artifacts import (
    fallback_annotation_decision,
    normalize_annotation_decision,
    write_annotation_artifacts,
)


ANNOTATION_PROMPT_VERSION = "lit_extract_annotation_v5"


ANNOTATION_FIGURE_FILENAMES = [
    "annotation_status_counts.png",
    "annotation_excluded_by_category.png",
    "annotation_status_by_category.png",
]


@dataclass
class AnnotationStats:
    feature_count: int = 0
    batch_count: int = 0
    completed_batch_count: int = 0
    skipped_batch_count: int = 0
    failed_batch_count: int = 0
    annotation_count: int = 0
    needs_review_count: int = 0
    excluded_count: int = 0
    repaired_json_count: int = 0


def annotation_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Annotating features", unit="feature", show=show)


def annotation_batch_id(batch_index: int, rows: list[dict[str, Any]]) -> str:
    digest_source = "\n".join(str(row.get("feature_key") or feature_key(row)) for row in rows)
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"annotation:{batch_index}:{digest}"


def prepare_annotation_batches(
    features: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    size = max(1, batch_size)
    for batch_index, start in enumerate(range(0, len(features), size), start=1):
        rows = features[start : start + size]
        batches.append(
            {
                "batch_id": annotation_batch_id(batch_index, rows),
                "batch_index": batch_index,
                "features": rows,
            }
        )
    return batches


def normalize_annotation_decisions(
    raw_annotations: list[dict[str, Any]],
    features: list[dict[str, Any]],
    *,
    model: str,
    run_id: str,
    annotated_at: str,
) -> list[dict[str, Any]]:
    features_by_key = {
        str(feature.get("feature_key") or feature_key(feature)): feature for feature in features
    }
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_annotations:
        key = str(raw.get("feature_key") or "")
        if key in seen:
            continue
        feature = features_by_key.get(key)
        if feature is None:
            continue
        output.append(
            normalize_annotation_decision(
                raw,
                feature,
                model=model,
                run_id=run_id,
                annotated_at=annotated_at,
            )
        )
        seen.add(key)

    for key, feature in features_by_key.items():
        if key not in seen:
            output.append(
                fallback_annotation_decision(
                    feature,
                    model=model,
                    run_id=run_id,
                    annotated_at=annotated_at,
                )
            )
    return output


def completed_annotation_batch_ids(
    output_dir: Path,
    prompt_version: str = ANNOTATION_PROMPT_VERSION,
) -> set[str]:
    completed: set[str] = set()
    for row in read_jsonl(output_dir / "completed_annotation_batches.jsonl"):
        if str(row.get("prompt_version") or "") != prompt_version:
            continue
        batch_id = str(row.get("batch_id") or "")
        if batch_id:
            completed.add(batch_id)
    return completed


def load_existing_annotation_decisions(
    output_dir: Path,
    prompt_version: str = ANNOTATION_PROMPT_VERSION,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in read_jsonl(output_dir / "annotation_decisions.jsonl"):
        if str(row.get("prompt_version") or "") != prompt_version:
            continue
        decisions.append(row)
    return decisions


def remove_annotation_outputs(output_dir: Path) -> None:
    for name in [
        "annotated_features.json",
        "annotated_features.jsonl",
        "annotated_features.csv",
        "excluded_annotations.json",
        "excluded_annotations.jsonl",
        "excluded_annotations.csv",
        "excluded_annotations_by_category.md",
        "annotation_trace.json",
        "annotation_errors.jsonl",
        "annotation_decisions.jsonl",
        "completed_annotation_batches.jsonl",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for name in ANNOTATION_FIGURE_FILENAMES:
        path = output_dir / "figures" / name
        if path.exists():
            path.unlink()


def dry_run_annotation_trace(
    *,
    input_file: Path,
    output_dir: Path,
    features: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    model: str,
    run_id: str,
    goal_path: str | None,
    goal_text: str,
    batch_features: int,
    max_tokens: int,
    timeout: float | None,
) -> dict[str, Any]:
    stats = AnnotationStats(
        feature_count=len(features),
        batch_count=len(batches),
    )
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": ANNOTATION_PROMPT_VERSION,
        "generated_at": utc_now(),
        "run_id": run_id,
        "dry_run": True,
        "model": model,
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "goal_path": goal_path,
        "goal_used": bool(goal_text.strip()),
        "config": {
            "batch_features": batch_features,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "annotation_mode": "hybrid_evidence_grounded",
        },
        "counts": dataclasses.asdict(stats),
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "annotation_trace.json", trace)
    return trace


def run_annotation(
    *,
    input_file: Path,
    output_dir: Path | None,
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
    max_tokens: int,
    timeout: float | None = None,
    show_progress: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    input_file = input_file.expanduser()
    output_dir = output_dir.expanduser() if output_dir is not None else input_file.parent
    if api_provider not in {"auto", "openai", "gemini"}:
        raise ValueError("api_provider must be 'auto', 'openai', or 'gemini'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        remove_annotation_outputs(output_dir)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    features = read_feature_inventory(input_file)
    batches = prepare_annotation_batches(features, batch_size=batch_features)
    goal_text, resolved_goal_path = load_goal_text(goal_path)
    max_tokens = max(1, int(max_tokens))
    if dry_run:
        return dry_run_annotation_trace(
            input_file=input_file,
            output_dir=output_dir,
            features=features,
            batches=batches,
            model=model,
            run_id=run_id,
            goal_path=resolved_goal_path,
            goal_text=goal_text,
            batch_features=batch_features,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if client is None:
        if not api_key:
            raise ValueError("An API key is required unless --dry-run is used.")
        client = make_llm_client(
            api_key=api_key,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            api_provider=api_provider,
            input_mode="text",
            timeout=timeout,
        )

    completed = completed_annotation_batch_ids(output_dir, prompt_version=ANNOTATION_PROMPT_VERSION)
    decisions = load_existing_annotation_decisions(output_dir, prompt_version=ANNOTATION_PROMPT_VERSION)
    stats = AnnotationStats(feature_count=len(features), batch_count=len(batches))
    output_counts = write_annotation_artifacts(output_dir, features, decisions)
    failures: list[dict[str, Any]] = []
    pending_feature_count = sum(
        len(batch["features"]) for batch in batches if str(batch["batch_id"]) not in completed
    )

    with annotation_progress(total=pending_feature_count, show=show_progress) as progress:
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
                                "You are an evidence-grounded taxonomy annotation agent. "
                                "Return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_annotation_prompt(
                                features=batch_rows,
                                goal_text=goal_text,
                            ),
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_annotations, repaired_json = parse_annotation_json_with_repair(
                    raw_response,
                    client=client,
                )
                if repaired_json:
                    stats.repaired_json_count += 1
                batch_decisions = normalize_annotation_decisions(
                    raw_annotations,
                    batch_rows,
                    model=model,
                    run_id=run_id,
                    annotated_at=utc_now(),
                )
                for decision in batch_decisions:
                    append_jsonl(
                        output_dir / "annotation_decisions.jsonl",
                        {
                            **decision,
                            "batch_id": batch_id,
                            "batch_index": batch.get("batch_index"),
                            "prompt_version": ANNOTATION_PROMPT_VERSION,
                        },
                    )
                append_jsonl(
                    output_dir / "completed_annotation_batches.jsonl",
                    {
                        "batch_id": batch_id,
                        "batch_index": batch.get("batch_index"),
                        "feature_count": len(batch_rows),
                        "annotation_count": len(batch_decisions),
                        "prompt_version": ANNOTATION_PROMPT_VERSION,
                        "run_id": run_id,
                        "completed_at": utc_now(),
                    },
                )
                completed.add(batch_id)
                decisions.extend(batch_decisions)
                stats.completed_batch_count += 1
                stats.annotation_count += len(batch_decisions)
                stats.needs_review_count += sum(
                    1 for decision in batch_decisions if decision.get("status") == "needs_review"
                )
                stats.excluded_count += sum(
                    1 for decision in batch_decisions if decision.get("status") == "excluded"
                )
                output_counts = write_annotation_artifacts(output_dir, features, decisions)
            except Exception as exc:
                stats.failed_batch_count += 1
                error_row = {
                    "batch_id": batch_id,
                    "batch_index": batch.get("batch_index"),
                    "feature_keys": [str(row.get("feature_key") or feature_key(row)) for row in batch_rows],
                    "error": str(exc),
                    "raw_response_excerpt": raw_response[:2000] if raw_response else "",
                    "run_id": run_id,
                    "created_at": utc_now(),
                }
                failures.append(error_row)
                append_jsonl(output_dir / "annotation_errors.jsonl", error_row)
                LOGGER.warning("Annotation failed for %s: %s", batch_id, exc)
            finally:
                progress.update(len(batch_rows))

    errors_path = output_dir / "annotation_errors.jsonl"
    if not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": ANNOTATION_PROMPT_VERSION,
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
        "config": {
            "batch_features": batch_features,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "force": force,
            "progress": show_progress,
            "annotation_mode": "hybrid_evidence_grounded",
            "output_mode": "new_files_only",
        },
        "counts": dataclasses.asdict(stats),
        "outputs": output_counts,
        "failures": failures,
    }
    write_json(output_dir / "annotation_trace.json", trace)
    return trace
