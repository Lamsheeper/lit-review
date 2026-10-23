"""Features scoring."""

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

from lit_review.features.prompts import build_scoring_prompt

from lit_review.features.responses import LOGGER, parse_scoring_json_with_repair

from lit_review.features.filtering import VERSION
from lit_review.core.io import append_jsonl, read_jsonl
from lit_review.core.progress import progress_bar


from lit_review.features.scoring_artifacts import (
    fallback_scoring_decision,
    normalize_scoring_decision,
    write_scoring_artifacts,
)


SCORING_PROMPT_VERSION = "lit_extract_scoring_v1"


SCORING_FIGURE_FILENAMES = [
    "scoring_total_distribution.png",
    "scoring_criteria_distributions.png",
    "scoring_percentiles.png",
    "scoring_total_percentiles_by_category.png",
]


@dataclass
class ScoringStats:
    feature_count: int = 0
    batch_count: int = 0
    completed_batch_count: int = 0
    skipped_batch_count: int = 0
    failed_batch_count: int = 0
    scoring_count: int = 0
    repaired_json_count: int = 0


def scoring_progress(total: int, show: bool) -> Any:
    return progress_bar(total=total, desc="Scoring features", unit="feature", show=show)


def scoring_batch_id(batch_index: int, rows: list[dict[str, Any]]) -> str:
    digest_source = "\n".join(str(row.get("feature_key") or feature_key(row)) for row in rows)
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"scoring:{batch_index}:{digest}"


def prepare_scoring_batches(
    features: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    size = max(1, batch_size)
    for batch_index, start in enumerate(range(0, len(features), size), start=1):
        rows = features[start : start + size]
        batches.append(
            {
                "batch_id": scoring_batch_id(batch_index, rows),
                "batch_index": batch_index,
                "features": rows,
            }
        )
    return batches


def normalize_scoring_decisions(
    raw_scores: list[dict[str, Any]],
    features: list[dict[str, Any]],
    *,
    model: str,
    run_id: str,
    scored_at: str,
) -> list[dict[str, Any]]:
    features_by_key = {
        str(feature.get("feature_key") or feature_key(feature)): feature for feature in features
    }
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_scores:
        key = str(raw.get("feature_key") or "")
        if key in seen:
            continue
        feature = features_by_key.get(key)
        if feature is None:
            continue
        output.append(
            normalize_scoring_decision(
                raw,
                feature,
                model=model,
                run_id=run_id,
                scored_at=scored_at,
            )
        )
        seen.add(key)

    for key, feature in features_by_key.items():
        if key not in seen:
            output.append(
                fallback_scoring_decision(
                    feature,
                    model=model,
                    run_id=run_id,
                    scored_at=scored_at,
                )
            )
    return output


def completed_scoring_batch_ids(
    output_dir: Path,
    prompt_version: str = SCORING_PROMPT_VERSION,
) -> set[str]:
    completed: set[str] = set()
    for row in read_jsonl(output_dir / "completed_scoring_batches.jsonl"):
        if str(row.get("prompt_version") or "") != prompt_version:
            continue
        batch_id = str(row.get("batch_id") or "")
        if batch_id:
            completed.add(batch_id)
    return completed


def load_existing_scoring_decisions(
    output_dir: Path,
    prompt_version: str = SCORING_PROMPT_VERSION,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in read_jsonl(output_dir / "scoring_decisions.jsonl"):
        if str(row.get("prompt_version") or "") != prompt_version:
            continue
        decisions.append(row)
    return decisions


def remove_scoring_outputs(output_dir: Path) -> None:
    for name in [
        "scored_features.json",
        "scored_features.jsonl",
        "scored_features.csv",
        "scored_features_ranked.csv",
        "scoring_summary_by_category.json",
        "scoring_percentiles.json",
        "scoring_percentiles.csv",
        "scoring_trace.json",
        "scoring_errors.jsonl",
        "scoring_decisions.jsonl",
        "completed_scoring_batches.jsonl",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for name in SCORING_FIGURE_FILENAMES:
        path = output_dir / "figures" / name
        if path.exists():
            path.unlink()


def dry_run_scoring_trace(
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
    stats = ScoringStats(
        feature_count=len(features),
        batch_count=len(batches),
    )
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": SCORING_PROMPT_VERSION,
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
            "scoring_mode": "feature_quality_scores_v1",
        },
        "counts": dataclasses.asdict(stats),
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "scoring_trace.json", trace)
    return trace


def run_scoring(
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
        remove_scoring_outputs(output_dir)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    features = read_feature_inventory(input_file)
    batches = prepare_scoring_batches(features, batch_size=batch_features)
    goal_text, resolved_goal_path = load_goal_text(goal_path)
    max_tokens = max(1, int(max_tokens))
    if dry_run:
        return dry_run_scoring_trace(
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

    completed = completed_scoring_batch_ids(output_dir, prompt_version=SCORING_PROMPT_VERSION)
    decisions = load_existing_scoring_decisions(output_dir, prompt_version=SCORING_PROMPT_VERSION)
    stats = ScoringStats(feature_count=len(features), batch_count=len(batches))
    output_counts = write_scoring_artifacts(output_dir, features, decisions)
    failures: list[dict[str, Any]] = []
    pending_feature_count = sum(
        len(batch["features"]) for batch in batches if str(batch["batch_id"]) not in completed
    )

    with scoring_progress(total=pending_feature_count, show=show_progress) as progress:
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
                                "You are an evidence-grounded taxonomy scoring agent. Return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_scoring_prompt(
                                features=batch_rows,
                                all_features=features,
                                goal_text=goal_text,
                            ),
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_scores, repaired_json = parse_scoring_json_with_repair(
                    raw_response,
                    client=client,
                )
                if repaired_json:
                    stats.repaired_json_count += 1
                batch_decisions = normalize_scoring_decisions(
                    raw_scores,
                    batch_rows,
                    model=model,
                    run_id=run_id,
                    scored_at=utc_now(),
                )
                for decision in batch_decisions:
                    append_jsonl(
                        output_dir / "scoring_decisions.jsonl",
                        {
                            **decision,
                            "batch_id": batch_id,
                            "batch_index": batch.get("batch_index"),
                            "prompt_version": SCORING_PROMPT_VERSION,
                        },
                    )
                append_jsonl(
                    output_dir / "completed_scoring_batches.jsonl",
                    {
                        "batch_id": batch_id,
                        "batch_index": batch.get("batch_index"),
                        "feature_count": len(batch_rows),
                        "scoring_count": len(batch_decisions),
                        "prompt_version": SCORING_PROMPT_VERSION,
                        "run_id": run_id,
                        "completed_at": utc_now(),
                    },
                )
                completed.add(batch_id)
                decisions.extend(batch_decisions)
                stats.completed_batch_count += 1
                stats.scoring_count += len(batch_decisions)
                output_counts = write_scoring_artifacts(output_dir, features, decisions)
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
                append_jsonl(output_dir / "scoring_errors.jsonl", error_row)
                LOGGER.warning("Scoring failed for %s: %s", batch_id, exc)
            finally:
                progress.update(len(batch_rows))

    errors_path = output_dir / "scoring_errors.jsonl"
    if not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")
    trace = {
        "project": "LitExtract",
        "version": VERSION,
        "prompt_version": SCORING_PROMPT_VERSION,
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
            "scoring_mode": "feature_quality_scores_v1",
            "output_mode": "new_files_only",
        },
        "counts": dataclasses.asdict(stats),
        "outputs": output_counts,
        "failures": failures,
    }
    write_json(output_dir / "scoring_trace.json", trace)
    return trace
