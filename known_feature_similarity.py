#!/usr/bin/env python3
"""Measure embedding similarities within an authoritative known feature set."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from embedding_merge import SimilarityPair, build_merge_groups, normalized_matrix
from feature_embedding_tsne import (
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    DEFAULT_TASK_TYPE,
    EMBEDDING_TEXT_CLEANUP_BUILTINS,
    EmbeddingClient,
    EmbeddingTextCleanup,
    GeminiEmbeddingClient,
    embed_texts_with_cache,
    gemini_api_key,
    load_feature_items,
    normalize_key,
    parse_jsonish_list,
    parse_text_fields,
    positive_int,
    read_feature_rows,
    row_field_value,
    sanitize_filename_part,
    setup_plot_style,
)
from lit_extract import make_llm_client, normalize_annotation_synonyms, parse_annotation_json_with_repair
from lit_pdf_service import utc_now


DEFAULT_TEXT_FIELDS = ("feature_name", "definitions", "synonyms")
ATTRIBUTE_COMPARISON_FIELDS = ("feature_name", "synonyms", "definitions", "examples")
ATTRIBUTE_FIELD_SLUGS = {
    "feature_name": "name",
    "synonyms": "synonyms",
    "definitions": "definitions",
    "examples": "examples",
}
ATTRIBUTE_GENERATION_MODEL = "gemini-3.5-flash"
ATTRIBUTE_GENERATION_PROMPT_VERSION = "known_feature_generated_attributes_v1"
ATTRIBUTE_GENERATION_MAX_RETRIES = 2
SYNONYM_REVIEW_SCHEMA_VERSION = 1
SYNONYM_REVIEW_PROMPT_VERSION = "known_feature_synonym_review_v1"
SYNONYM_GENERATION_MAX_TOKENS = 1200
HEATMAP_MIN_SIMILARITY = 0.8
HEATMAP_MAX_SIMILARITY = 1.0
HEATMAP_COLORMAP = "Greens"
MOST_SIMILAR_DISTINCT_TOP = 20
FEATURE_SET_EVALUATION_LINKAGE = "complete"
FEATURE_SET_THRESHOLD_MAX_POINTS = 401
DEFAULT_ANNOTATED_FEATURES = Path(
    "review_runs/media_attributes_v3/feature_extract_pdf/annotated/annotated_features.json"
)
KNOWN_NAME_FIELDS = (
    "canonical_feature_name",
    "feature_name",
    "canonical_name",
    "name",
    "label",
)
CANDIDATE_NAME_FIELDS = (
    "feature_name",
    "canonical_feature_name",
    "normalized_feature_name",
)


def embedding_text_cleanup_from_args(
    *,
    strip_builtins: Iterable[str] | None,
    min_chars: int,
    fallback: str,
) -> EmbeddingTextCleanup | None:
    builtins = tuple(str(item).strip() for item in (strip_builtins or []) if str(item).strip())
    if not builtins:
        return None
    unknown = sorted(set(builtins) - set(EMBEDDING_TEXT_CLEANUP_BUILTINS))
    if unknown:
        raise ValueError("Unknown embedding text cleanup builtin(s): " + ", ".join(unknown))
    if min_chars < 0:
        raise ValueError("--embedding-text-cleanup-min-chars must be non-negative.")
    fallback = fallback.strip().lower()
    if fallback not in {"original", "empty"}:
        raise ValueError("--embedding-text-cleanup-fallback must be 'original' or 'empty'.")
    return EmbeddingTextCleanup(
        strip_builtins=builtins,
        min_chars=min_chars,
        fallback=fallback,
    )


def embedding_text_cleanup_trace_payload(
    cleanup: EmbeddingTextCleanup | None,
    cleanup_fields: Iterable[str] | None,
    selected_fields: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    if cleanup is None:
        return None
    cleanup_field_list = list(parse_text_fields(cleanup_fields)) if cleanup_fields is not None else None
    payload: dict[str, Any] = {
        "strip_builtins": list(cleanup.strip_builtins),
        "min_chars": cleanup.min_chars,
        "fallback": cleanup.fallback,
        "fields": cleanup_field_list,
    }
    if selected_fields is not None:
        selected = list(parse_text_fields(selected_fields))
        cleanup_field_set = set(cleanup_field_list) if cleanup_field_list is not None else set(selected)
        payload["applied_to_selected_fields"] = [
            field for field in selected if field in cleanup_field_set
        ]
    return payload


def unicode_normalize_key(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    ascii_like = "".join(character for character in decomposed if not unicodedata.combining(character))
    return normalize_key(ascii_like)


def unique_text_values(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        for text in parse_jsonish_list(value):
            key = unicode_normalize_key(text)
            if key and key not in seen:
                seen.add(key)
                output.append(text)
    return output


def first_text(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        values = parse_jsonish_list(row_field_value(row, field))
        if values:
            return values[0]
    return ""


def known_label(row: dict[str, Any], index: int) -> str:
    label = first_text(row, KNOWN_NAME_FIELDS)
    if not label:
        raise ValueError(f"Known-set row {index} has no recognized feature name.")
    return label


def known_matching_aliases(row: dict[str, Any], label: str) -> list[str]:
    return unique_text_values(
        [
            label,
            row.get("matching_aliases"),
            # Legacy inventories used aliases for both matching and controls.
            # They remain matching-only for compatibility.
            row.get("aliases"),
            row.get("alternate_names"),
            row.get("alternative_names"),
        ]
    )


def candidate_name_keys(row: dict[str, Any]) -> set[str]:
    return {
        key
        for field in CANDIDATE_NAME_FIELDS
        for value in parse_jsonish_list(row_field_value(row, field))
        if (key := unicode_normalize_key(value))
    }


def categories_match(known_row: dict[str, Any], candidate_row: dict[str, Any]) -> bool:
    known_category = unicode_normalize_key(known_row.get("category"))
    if not known_category:
        return True
    return known_category == unicode_normalize_key(candidate_row.get("category"))


def matched_candidate_rows(
    known_row: dict[str, Any],
    label: str,
    aliases: list[str],
    candidates: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    canonical_key = unicode_normalize_key(label)
    eligible = [row for row in candidates if categories_match(known_row, row)]
    exact = [row for row in eligible if canonical_key in candidate_name_keys(row)]
    if exact:
        return "exact_canonical", exact

    alias_keys = {unicode_normalize_key(alias) for alias in aliases}
    alias_keys.discard(canonical_key)
    matched = [row for row in eligible if alias_keys & candidate_name_keys(row)]
    if matched:
        return "alias_aggregate", matched
    return "known_metadata_fallback", []


def matched_row_reference(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_key": str(row.get("feature_key") or ""),
        "feature_name": first_text(row, CANDIDATE_NAME_FIELDS),
        "category": str(row.get("category") or ""),
        "paper_count": row_field_value(row, "annotation.paper_count") or row.get("paper_count") or 0,
        "mention_count": row_field_value(row, "annotation.mention_count") or row.get("mention_count") or 0,
    }


def aggregate_known_row(
    known_row: dict[str, Any],
    *,
    index: int,
    label: str,
    aliases: list[str],
    status: str,
    matched_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    category = str(known_row.get("category") or "uncategorized").strip() or "uncategorized"
    definitions = unique_text_values(
        [
            known_row.get("definitions"),
            known_row.get("definition"),
            *(row_field_value(row, field) for row in matched_rows for field in (
                "annotation.formal_definition",
                "annotation.short_definition",
                "definitions",
                "definition",
            )),
        ]
    )
    synonyms = unique_text_values(
        [
            known_row.get("synonyms"),
            *(row_field_value(row, field) for row in matched_rows for field in (
                "annotation.synonyms",
                "synonyms",
            )),
            *(first_text(row, CANDIDATE_NAME_FIELDS) for row in matched_rows),
        ]
    )
    label_key = unicode_normalize_key(label)
    synonyms = [value for value in synonyms if unicode_normalize_key(value) != label_key]
    examples = unique_text_values(
        [
            known_row.get("examples"),
            *(row_field_value(row, field) for row in matched_rows for field in (
                "annotation.examples",
                "examples",
            )),
        ]
    )
    feature_id = str(known_row.get("feature_id") or "").strip()
    feature_key = feature_id or f"{unicode_normalize_key(category)}::{label_key}"
    return {
        "known_set_index": index,
        "feature_id": feature_id,
        "feature_key": feature_key,
        "category": category,
        "feature_name": label,
        "normalized_feature_name": label_key,
        "definitions": definitions,
        "synonyms": synonyms,
        "examples": examples,
        "known_set_metadata": dict(known_row),
        "resolution": {
            "status": status,
            "matched_row_count": len(matched_rows),
            "matched_rows": [matched_row_reference(row) for row in matched_rows],
        },
    }


def resolve_known_rows(
    known_rows: list[dict[str, Any]],
    annotated_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates = annotated_rows or []
    resolved: list[dict[str, Any]] = []
    for index, known_row in enumerate(known_rows, start=1):
        label = known_label(known_row, index)
        aliases = known_matching_aliases(known_row, label)
        status, matched_rows = matched_candidate_rows(known_row, label, aliases, candidates)
        resolved.append(
            aggregate_known_row(
                known_row,
                index=index,
                label=label,
                aliases=aliases,
                status=status,
                matched_rows=matched_rows,
            )
        )
    return resolved


def feature_identity(feature_name: Any, category: Any) -> tuple[str, str]:
    return (
        unicode_normalize_key(feature_name),
        unicode_normalize_key(category or "uncategorized"),
    )


def feature_identity_key(feature_name: Any, category: Any) -> str:
    payload = {
        "feature_name": unicode_normalize_key(feature_name),
        "category": unicode_normalize_key(category or "uncategorized"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def inventory_fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "feature_id": str(row.get("feature_id") or ""),
            "feature_name": known_label(row, index),
            "category": str(row.get("category") or "uncategorized"),
        }
        for index, row in enumerate(rows, start=1)
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def synonym_review_entry(row: dict[str, Any], index: int) -> dict[str, Any]:
    label = known_label(row, index)
    category = str(row.get("category") or "uncategorized").strip() or "uncategorized"
    return {
        "identity_key": feature_identity_key(label, category),
        "feature_id": str(row.get("feature_id") or ""),
        "feature_name": label,
        "category": category,
        "status": "pending",
        "approved_synonyms": [],
        "current_proposal": [],
        "proposal_history": [],
        "updated_at": utc_now(),
    }


def reconcile_synonym_review(
    known_rows: list[dict[str, Any]],
    existing: dict[str, Any] | None,
    *,
    known_set_path: Path,
    model: str,
) -> dict[str, Any]:
    old_entries = {
        str(entry.get("identity_key") or ""): entry
        for entry in (existing or {}).get("entries", [])
        if isinstance(entry, dict)
    }
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(known_rows, start=1):
        fresh = synonym_review_entry(row, index)
        old = old_entries.get(fresh["identity_key"])
        if old:
            for field in (
                "status",
                "approved_synonyms",
                "current_proposal",
                "proposal_history",
                "updated_at",
            ):
                if field in old:
                    fresh[field] = copy.deepcopy(old[field])
        entries.append(fresh)
    for entry in entries:
        if entry.get("status") != "approved":
            continue
        try:
            entry["approved_synonyms"] = validate_review_synonyms(
                entry.get("approved_synonyms"),
                entry=entry,
                entries=entries,
            )
        except ValueError as exc:
            entry["status"] = "pending"
            entry["approved_synonyms"] = []
            append_review_history(entry, {"action": "approval_reopened", "reason": str(exc)})
    return {
        "schema_version": SYNONYM_REVIEW_SCHEMA_VERSION,
        "known_set": str(known_set_path),
        "inventory_fingerprint": inventory_fingerprint(known_rows),
        "generation_model": model,
        "prompt_version": SYNONYM_REVIEW_PROMPT_VERSION,
        "updated_at": utc_now(),
        "entries": entries,
    }


def load_synonym_review(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Synonym review must contain a JSON object: {path}")
    return payload


def write_synonym_review(path: Path, review: dict[str, Any]) -> Path:
    review["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def cleaned_review_synonyms(values: Any, feature_name: str, category: str) -> list[str]:
    feature = {
        "feature_name": feature_name,
        "normalized_feature_name": unicode_normalize_key(feature_name),
        "category": category,
    }
    return normalize_annotation_synonyms(values, feature)


def validate_review_synonyms(
    values: Any,
    *,
    entry: dict[str, Any],
    entries: list[dict[str, Any]],
) -> list[str]:
    feature_name = str(entry.get("feature_name") or "")
    category = str(entry.get("category") or "uncategorized")
    synonyms = cleaned_review_synonyms(values, feature_name, category)
    if len(synonyms) != 2:
        raise ValueError("Exactly two valid synonym terms are required.")
    keys = [unicode_normalize_key(value) for value in synonyms]
    if len(set(keys)) != 2:
        raise ValueError("The two approved synonyms must be distinct.")

    current_identity = str(entry.get("identity_key") or "")
    occupied: dict[str, str] = {}
    for other in entries:
        other_identity = str(other.get("identity_key") or "")
        if other_identity != current_identity:
            canonical_key = unicode_normalize_key(other.get("feature_name"))
            if canonical_key:
                occupied[canonical_key] = str(other.get("feature_name") or "")
            if other.get("status") == "approved":
                for value in parse_jsonish_list(other.get("approved_synonyms")):
                    occupied[unicode_normalize_key(value)] = value
    collisions = [value for value, key in zip(synonyms, keys) if key in occupied]
    if collisions:
        raise ValueError(
            "Synonym controls collide with another canonical or approved control: "
            + ", ".join(collisions)
        )
    return synonyms


def approved_synonym_map(
    review: dict[str, Any],
    known_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    expected_fingerprint = inventory_fingerprint(known_rows)
    if review.get("inventory_fingerprint") != expected_fingerprint:
        raise ValueError("Synonym review is stale; run --review-synonyms to reconcile it.")
    entries = review.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Synonym review has no entries.")
    expected_identities = {
        feature_identity(known_label(row, index), row.get("category"))
        for index, row in enumerate(known_rows, start=1)
    }
    approved: dict[tuple[str, str], list[str]] = {}
    incomplete: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        identity = feature_identity(entry.get("feature_name"), entry.get("category"))
        if identity not in expected_identities:
            continue
        if entry.get("status") != "approved":
            incomplete.append(str(entry.get("feature_name") or ""))
            continue
        approved[identity] = validate_review_synonyms(
            entry.get("approved_synonyms"),
            entry=entry,
            entries=entries,
        )
    missing = expected_identities - set(approved)
    if missing:
        labels = incomplete or sorted(name for name, _ in missing)
        raise ValueError(
            "Synonym review is incomplete; every feature needs exactly two approved synonyms. "
            f"Pending: {', '.join(labels[:10])}"
        )
    return approved


def build_synonym_review_prompt(feature_name: str, category: str, feedback: str = "") -> str:
    payload = {"feature_name": feature_name, "category": category}
    if feedback.strip():
        payload["user_feedback"] = feedback.strip()
    return f"""Propose exactly two close synonyms for one taxonomy feature.

Return strict JSON only:
{{"synonyms": ["first synonym", "second synonym"]}}

Rules:
- Use only the supplied feature name, category, and optional user feedback.
- Return exactly two concise, established, substitutable alternate labels.
- Preserve the feature's meaning, scope, intensity, and granularity.
- Do not return the original label, definitions, sentences, examples, broader or narrower
  concepts, associated concepts, or category terms.

INPUT
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def parse_synonym_proposal(response: str, feature_name: str, category: str) -> list[str]:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    payload: Any = None
    parse_error: Exception | None = None
    candidates = [text]
    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1])
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end > array_start:
        candidates.append(text[array_start : array_end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = exc

    if isinstance(payload, dict):
        raw_synonyms = payload.get("synonyms")
    elif isinstance(payload, list):
        raw_synonyms = payload
    else:
        # Recover a truncated response once two quoted array values are present.
        match = re.search(r'["\']synonyms["\']\s*:\s*\[(.*)', text, flags=re.IGNORECASE | re.DOTALL)
        recovered: list[str] = []
        if match:
            for quoted in re.findall(r'"(?:\\.|[^"\\])*"', match.group(1)):
                try:
                    recovered.append(json.loads(quoted))
                except json.JSONDecodeError:
                    continue
        raw_synonyms = recovered

    synonyms = cleaned_review_synonyms(raw_synonyms, feature_name, category)
    if len(synonyms) != 2:
        preview = " ".join(text[:300].split())
        detail = f": {parse_error}" if parse_error else ""
        raise ValueError(
            "Synonym proposal must contain exactly two valid terms"
            f"{detail}; response preview={preview!r}"
        )
    return synonyms


def generate_synonym_proposal(
    entry: dict[str, Any],
    *,
    client: Any,
    feedback: str = "",
) -> list[str]:
    if client is None:
        raise ValueError("A Gemini API key is required to generate synonym proposals.")
    feature_name = str(entry.get("feature_name") or "")
    category = str(entry.get("category") or "uncategorized")
    response = client.chat(
        [
            {
                "role": "system",
                "content": "You propose exactly two high-precision synonyms. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": build_synonym_review_prompt(feature_name, category, feedback),
            },
        ],
        temperature=0.0,
        max_tokens=SYNONYM_GENERATION_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return parse_synonym_proposal(response, feature_name, category)


def append_review_history(entry: dict[str, Any], event: dict[str, Any]) -> None:
    entry.setdefault("proposal_history", []).append({"at": utc_now(), **event})
    entry["updated_at"] = utc_now()


def run_synonym_review(
    *,
    known_set_path: Path,
    review_path: Path,
    client: Any | None,
    model: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    known_rows = read_feature_rows(known_set_path)
    review = reconcile_synonym_review(
        known_rows,
        load_synonym_review(review_path),
        known_set_path=known_set_path,
        model=model,
    )
    write_synonym_review(review_path, review)
    entries = review["entries"]
    quit_early = False
    for index, entry in enumerate(entries, start=1):
        if entry.get("status") == "approved":
            continue
        output_fn(f"\n[{index}/{len(entries)}] {entry['feature_name']} ({entry['category']})")
        if not entry.get("current_proposal"):
            try:
                proposal = generate_synonym_proposal(entry, client=client)
                entry["current_proposal"] = proposal
                append_review_history(
                    entry,
                    {
                        "action": "generated",
                        "synonyms": proposal,
                        "model": model,
                        "prompt_version": SYNONYM_REVIEW_PROMPT_VERSION,
                    },
                )
            except Exception as exc:
                append_review_history(
                    entry,
                    {
                        "action": "generation_failed",
                        "error": str(exc),
                        "model": model,
                        "prompt_version": SYNONYM_REVIEW_PROMPT_VERSION,
                    },
                )
                output_fn(f"Generation failed: {exc}")
            write_synonym_review(review_path, review)

        while True:
            proposal = parse_jsonish_list(entry.get("current_proposal"))
            output_fn("Current proposal: " + (", ".join(proposal) if proposal else "(none)"))
            action = input_fn("[a]pprove, [e]dit, [r]egenerate with feedback, [s]kip, [q]uit: ").strip().casefold()
            if action in {"a", "approve"}:
                try:
                    approved = validate_review_synonyms(proposal, entry=entry, entries=entries)
                except ValueError as exc:
                    output_fn(f"Cannot approve: {exc}")
                    continue
                entry["approved_synonyms"] = approved
                entry["status"] = "approved"
                append_review_history(entry, {"action": "approved", "synonyms": approved})
                write_synonym_review(review_path, review)
                break
            if action in {"e", "edit"}:
                edited = [
                    input_fn("Synonym 1: ").strip(),
                    input_fn("Synonym 2: ").strip(),
                ]
                try:
                    entry["current_proposal"] = validate_review_synonyms(edited, entry=entry, entries=entries)
                except ValueError as exc:
                    output_fn(f"Invalid terms: {exc}")
                    continue
                append_review_history(entry, {"action": "edited", "synonyms": entry["current_proposal"]})
                write_synonym_review(review_path, review)
                continue
            if action in {"r", "regenerate"}:
                feedback = input_fn("Feedback or suggested terms (optional): ").strip()
                try:
                    proposal = generate_synonym_proposal(entry, client=client, feedback=feedback)
                    entry["current_proposal"] = proposal
                    append_review_history(
                        entry,
                        {
                            "action": "regenerated",
                            "feedback": feedback,
                            "synonyms": proposal,
                            "model": model,
                            "prompt_version": SYNONYM_REVIEW_PROMPT_VERSION,
                        },
                    )
                except Exception as exc:
                    append_review_history(
                        entry,
                        {
                            "action": "generation_failed",
                            "feedback": feedback,
                            "error": str(exc),
                            "model": model,
                            "prompt_version": SYNONYM_REVIEW_PROMPT_VERSION,
                        },
                    )
                    output_fn(f"Generation failed: {exc}")
                write_synonym_review(review_path, review)
                continue
            if action in {"s", "skip"}:
                entry["status"] = "skipped"
                append_review_history(entry, {"action": "skipped"})
                write_synonym_review(review_path, review)
                break
            if action in {"q", "quit"}:
                append_review_history(entry, {"action": "saved_and_quit"})
                write_synonym_review(review_path, review)
                quit_early = True
                break
            output_fn("Unknown action.")
        if quit_early:
            break

    approved_count = sum(entry.get("status") == "approved" for entry in entries)
    return {
        "review_path": str(review_path),
        "feature_count": len(entries),
        "approved_count": approved_count,
        "complete": approved_count == len(entries),
        "quit_early": quit_early,
    }


def synonym_review_is_complete(known_set_path: Path, review_path: Path) -> bool:
    review = load_synonym_review(review_path)
    if review is None:
        return False
    try:
        approved_synonym_map(review, read_feature_rows(known_set_path))
    except ValueError:
        return False
    return True


def ensure_synonym_review_for_run(
    *,
    known_set_path: Path,
    review_path: Path,
    client: Any | None,
    model: str,
    interactive: bool,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any] | None:
    if synonym_review_is_complete(known_set_path, review_path):
        return None
    if not interactive:
        raise ValueError(
            "Synonym review is missing or incomplete and cannot be completed non-interactively. "
            "Run the same command in an interactive terminal or use --review-synonyms."
        )
    output_fn("Synonym review is missing or incomplete; starting interactive review.")
    trace = run_synonym_review(
        known_set_path=known_set_path,
        review_path=review_path,
        client=client,
        model=model,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    if not trace["complete"]:
        raise ValueError(
            "Synonym review is incomplete. Rerun the same command to resume and continue the experiment."
        )
    output_fn("Synonym review complete; continuing the experiment.")
    return trace


def build_synonym_control_rows(
    resolved_rows: list[dict[str, Any]],
    approved_synonyms: dict[tuple[str, str], list[str]],
    annotated_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    control_index = 0
    for row in resolved_rows:
        identity = feature_identity(row.get("feature_name"), row.get("category"))
        synonyms = approved_synonyms.get(identity)
        if not synonyms or len(synonyms) != 2:
            raise ValueError(f"No complete approved synonym review for {row.get('feature_name')!r}.")
        group_feature_key = str(row.get("feature_key") or "")
        variants = [("canonical", str(row.get("feature_name") or "")), *[("synonym", value) for value in synonyms]]
        for variant_index, (variant_type, label) in enumerate(variants):
            control_index += 1
            if variant_type == "canonical":
                control = copy.deepcopy(row)
            else:
                metadata = row.get("known_set_metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                synonym_known_row = {
                    "feature_id": f"{group_feature_key}::approved_synonym::{variant_index}",
                    "canonical_feature_name": label,
                    "category": row.get("category") or "uncategorized",
                    "source_feature_id": row.get("feature_id") or group_feature_key,
                    "source_feature_name": row.get("feature_name") or "",
                    "source_note": "Human-approved synonym control",
                }
                control = resolve_known_rows([synonym_known_row], annotated_rows)[0]
                control["known_set_metadata"]["source_known_set_metadata"] = copy.deepcopy(metadata)
            control["known_set_index"] = control_index
            control["feature_key"] = (
                f"{group_feature_key}::synonym_control::{variant_index:03d}::{unicode_normalize_key(label)}"
            )
            control["feature_name"] = label
            control["normalized_feature_name"] = unicode_normalize_key(label)
            control["synonym_control"] = {
                "group_feature_key": group_feature_key,
                "group_feature_name": str(row.get("feature_name") or ""),
                "variant_type": variant_type,
                "variant_index": variant_index,
                "source": "human_approved_synonym",
            }
            controls.append(control)
    return controls


def attribute_combinations() -> list[tuple[str, ...]]:
    return [
        combination
        for size in range(1, len(ATTRIBUTE_COMPARISON_FIELDS) + 1)
        for combination in itertools.combinations(ATTRIBUTE_COMPARISON_FIELDS, size)
    ]


def attribute_combination_slug(fields: Iterable[str]) -> str:
    return "_".join(ATTRIBUTE_FIELD_SLUGS[field] for field in fields)


def generated_attribute_cache_key(label: str, category: str, model: str) -> str:
    payload = {
        "prompt_version": ATTRIBUTE_GENERATION_PROMPT_VERSION,
        "model": model,
        "feature_name": label,
        "category": category,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_generated_attribute_prompt(label: str, category: str) -> str:
    payload = {"feature_name": label, "category": category}
    return f"""Generate independent embedding metadata for exactly one taxonomy feature.

Return strict JSON only with this shape:
{{
  "annotations": [
    {{
      "feature_name": "the exact input feature_name",
      "definition": "one concise standalone definition",
      "synonyms": ["1 to 5 extremely close substitutable alternate labels"],
      "examples": ["2 to 3 brief representative examples"]
    }}
  ]
}}

Rules:
- Use only the supplied feature name and category. Do not assume or invent a parent family.
- Preserve the exact feature identity and sense implied by its category.
- Return one to five concise synonyms that are extremely close, substitutable alternate
  labels for the feature in this category.
- Prefer established alternate names, spelling variants, and close lexical equivalents.
- Do not return the original label, category labels, definitions, sentences, examples,
  broad category terms, or merely associated concepts.
- Keep the definition to one sentence.
- Return two or three short, concrete examples.

INPUT
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def build_generated_attribute_retry_prompt(label: str, category: str, rejection_reason: str) -> str:
    return (
        build_generated_attribute_prompt(label, category)
        + f"""

RETRY CORRECTION
The previous response was rejected after normalization: {rejection_reason}
- Return every required field with usable content.
- If the feature is a paired contrast such as "X vs. Y", synonyms must name the
  complete contrast, dichotomy, or dimension. Do not return only X or only Y.
- Return at least two distinct concrete examples illustrating the full feature.
"""
    )


def normalize_generated_attributes(raw: dict[str, Any], label: str, category: str) -> dict[str, Any]:
    definition_values = unique_text_values(
        [
            raw.get("definition"),
            raw.get("definitions"),
            raw.get("formal_definition"),
            raw.get("short_definition"),
        ]
    )
    definition = definition_values[0] if definition_values else ""
    if len(definition) > 320:
        definition = ""
    feature = {
        "feature_name": label,
        "normalized_feature_name": unicode_normalize_key(label),
        "category": category,
    }
    synonyms = normalize_annotation_synonyms(raw.get("synonyms"), feature)[:5]
    examples = unique_text_values([raw.get("examples")])[:3]
    return {
        "feature_name": label,
        "normalized_feature_name": unicode_normalize_key(label),
        "category": category,
        "definitions": [definition] if definition else [],
        "synonyms": synonyms,
        "examples": examples,
    }


def generated_attributes_are_complete(row: dict[str, Any]) -> bool:
    return bool(row.get("definitions") and row.get("synonyms") and len(row.get("examples") or []) >= 2)


def load_generated_attribute_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            key = str(row.get("cache_key") or "")
            attributes = row.get("attributes")
            if key and isinstance(attributes, dict) and generated_attributes_are_complete(attributes):
                cache[key] = row
    return cache


def append_generated_attribute_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def generate_attribute_metadata(
    rows: list[dict[str, Any]],
    *,
    cache_path: Path,
    client: Any | None,
    model: str = ATTRIBUTE_GENERATION_MODEL,
    refresh: bool = False,
    max_retries: int = ATTRIBUTE_GENERATION_MAX_RETRIES,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    unique_labels: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        label = str(row.get("feature_name") or "").strip()
        category = str(row.get("category") or "uncategorized").strip() or "uncategorized"
        identity = (unicode_normalize_key(label), unicode_normalize_key(category))
        if not label or identity in seen:
            continue
        seen.add(identity)
        unique_labels.append((label, category))

    cache = load_generated_attribute_cache(cache_path)
    generated: dict[tuple[str, str], dict[str, Any]] = {}
    cache_hits = 0
    generated_count = 0
    attempts = 0
    failures: list[str] = []
    for label, category in unique_labels:
        identity = (unicode_normalize_key(label), unicode_normalize_key(category))
        cache_key = generated_attribute_cache_key(label, category, model)
        cached = cache.get(cache_key)
        if cached and not refresh:
            generated[identity] = copy.deepcopy(cached["attributes"])
            cache_hits += 1
            continue
        if client is None:
            failures.append(f"{label}: no matching cache entry and no attribute-generation client")
            continue

        attributes: dict[str, Any] | None = None
        last_error = ""
        for attempt in range(max_retries + 1):
            attempts += 1
            try:
                prompt = (
                    build_generated_attribute_prompt(label, category)
                    if attempt == 0
                    else build_generated_attribute_retry_prompt(label, category, last_error)
                )
                response = client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You generate high-precision taxonomy metadata for one feature. "
                                "Return only valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=1200,
                    response_format={"type": "json_object"},
                )
                annotations, _ = parse_annotation_json_with_repair(response, client=client)
                if len(annotations) != 1:
                    raise ValueError("Expected exactly one generated annotation.")
                candidate = normalize_generated_attributes(annotations[0], label, category)
                if not generated_attributes_are_complete(candidate):
                    raise ValueError(
                        "Generated metadata requires one definition, at least one synonym, "
                        "and at least two examples."
                    )
                attributes = candidate
                break
            except Exception as exc:
                last_error = str(exc)
        if attributes is None:
            failures.append(f"{label}: {last_error}")
            continue

        cache_row = {
            "cache_key": cache_key,
            "prompt_version": ATTRIBUTE_GENERATION_PROMPT_VERSION,
            "model": model,
            "feature_name": label,
            "category": category,
            "attributes": attributes,
            "generated_at": utc_now(),
        }
        append_generated_attribute_cache(cache_path, cache_row)
        generated[identity] = copy.deepcopy(attributes)
        generated_count += 1

    if failures:
        raise ValueError(
            "Generated attributes are incomplete for "
            f"{len(failures)} rows: "
            + "; ".join(failures[:10])
        )
    return generated, {
        "prompt_version": ATTRIBUTE_GENERATION_PROMPT_VERSION,
        "model": model,
        "cache_path": str(cache_path),
        "refresh": refresh,
        "unique_rows": len(unique_labels),
        "cache_hits": cache_hits,
        "generated_rows": generated_count,
        "generation_attempts": attempts,
        "max_retries": max_retries,
    }


def apply_generated_attributes(
    rows: list[dict[str, Any]],
    generated: dict[tuple[str, str], dict[str, Any]],
    *,
    model: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        result = copy.deepcopy(row)
        identity = (
            unicode_normalize_key(result.get("feature_name")),
            unicode_normalize_key(result.get("category") or "uncategorized"),
        )
        attributes = generated.get(identity)
        if not attributes:
            raise ValueError(f"No generated attributes found for {result.get('feature_name')!r}.")
        for field in ("definitions", "synonyms", "examples"):
            result[field] = copy.deepcopy(attributes[field])
        result["generated_attributes"] = {
            "prompt_version": ATTRIBUTE_GENERATION_PROMPT_VERSION,
            "model": model,
            "input_fields": ["feature_name", "category"],
        }
        output.append(result)
    return output


def pairwise_similarity_data(
    labels: list[str],
    matrix: Any,
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("NumPy is required for pairwise similarity analysis.") from exc

    matrix = np.asarray(matrix, dtype=float)
    if len(labels) != len(matrix):
        raise ValueError("Label and embedding counts must match.")
    similarities = matrix @ matrix.T
    similarity_matrix = [
        [round(float(value), 8) for value in row]
        for row in similarities
    ]
    pairs = [
        {
            "similarity": similarity_matrix[left][right],
            "left_index": left + 1,
            "left_feature_name": labels[left],
            "right_index": right + 1,
            "right_feature_name": labels[right],
        }
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    ]
    pairs.sort(
        key=lambda row: (
            -row["similarity"],
            str(row["left_feature_name"]).casefold(),
            str(row["right_feature_name"]).casefold(),
        )
    )
    return pairs, similarity_matrix


def synonym_control_pair_data(items: list[Any], matrix: Any) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("NumPy is required for synonym-control similarity analysis.") from exc

    matrix = np.asarray(matrix, dtype=float)
    if len(items) != len(matrix):
        raise ValueError("Synonym-control item and embedding counts must match.")
    similarities = matrix @ matrix.T
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(items):
        left_control = left.raw_row.get("synonym_control")
        if not isinstance(left_control, dict):
            continue
        for right_index in range(left_index + 1, len(items)):
            right = items[right_index]
            right_control = right.raw_row.get("synonym_control")
            if not isinstance(right_control, dict):
                continue
            if left_control.get("group_feature_key") != right_control.get("group_feature_key"):
                continue
            left_type = str(left_control.get("variant_type") or "")
            right_type = str(right_control.get("variant_type") or "")
            pairs.append(
                {
                    "similarity": round(float(similarities[left_index][right_index]), 8),
                    "group_feature_key": left_control.get("group_feature_key") or "",
                    "group_feature_name": left_control.get("group_feature_name") or "",
                    "pair_type": (
                        "canonical_synonym"
                        if "canonical" in {left_type, right_type}
                        else "synonym_synonym"
                    ),
                    "left_index": left.row_index,
                    "left_feature_name": left.label,
                    "left_variant_type": left_type,
                    "right_index": right.row_index,
                    "right_feature_name": right.label,
                    "right_variant_type": right_type,
                }
            )
    return sorted(
        pairs,
        key=lambda row: (
            -row["similarity"],
            str(row["group_feature_name"]).casefold(),
            str(row["left_feature_name"]).casefold(),
            str(row["right_feature_name"]).casefold(),
        ),
    )


def similarity_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("NumPy is required for similarity summaries.") from exc

    values = np.asarray([row["similarity"] for row in pairs], dtype=float)
    if not values.size:
        return {
            "pair_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "percentiles": {},
        }
    return {
        "pair_count": int(values.size),
        "min": round(float(values.min()), 8),
        "max": round(float(values.max()), 8),
        "mean": round(float(values.mean()), 8),
        "median": round(float(np.median(values)), 8),
        "standard_deviation": round(float(values.std()), 8),
        "percentiles": {
            str(percentile): round(float(np.percentile(values, percentile)), 8)
            for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
    }


def control_truth_key(item: Any) -> str:
    control = item.raw_row.get("synonym_control")
    if not isinstance(control, dict) or not control.get("group_feature_key"):
        raise ValueError(f"Synonym-control row {item.row_index} has no group feature key.")
    return str(control["group_feature_key"])


def control_similarity_pairs(items: list[Any], matrix: Any) -> list[SimilarityPair]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("NumPy is required for feature-set threshold metrics.") from exc

    matrix = np.asarray(matrix, dtype=float)
    if len(items) != len(matrix):
        raise ValueError("Synonym-control item and embedding counts must match.")
    similarities = matrix @ matrix.T
    return [
        SimilarityPair(left, right, round(float(similarities[left][right]), 8))
        for left in range(len(items))
        for right in range(left + 1, len(items))
        if items[left].category == items[right].category
    ]


def produced_feature_clusters(
    items: list[Any],
    pairs: list[SimilarityPair],
    *,
    threshold: float,
    linkage: str = FEATURE_SET_EVALUATION_LINKAGE,
) -> list[frozenset[int]]:
    active_pairs = [pair for pair in pairs if pair.similarity >= threshold]
    groups = build_merge_groups(items, active_pairs, linkage=linkage)
    clusters = [frozenset(group.member_indices) for group in groups]
    grouped_indices = {index for cluster in clusters for index in cluster}
    clusters.extend(frozenset((index,)) for index in range(len(items)) if index not in grouped_indices)
    return clusters


def maximum_feature_set_match_count(
    predicted_clusters: list[frozenset[int]],
    truth_keys: list[str],
) -> int:
    candidates = [
        sorted({truth_keys[index] for index in cluster})
        for cluster in predicted_clusters
    ]
    truth_to_prediction: dict[str, int] = {}

    def assign(prediction_index: int, seen_truths: set[str]) -> bool:
        for truth_key in candidates[prediction_index]:
            if truth_key in seen_truths:
                continue
            seen_truths.add(truth_key)
            previous = truth_to_prediction.get(truth_key)
            if previous is None or assign(previous, seen_truths):
                truth_to_prediction[truth_key] = prediction_index
                return True
        return False

    return sum(assign(index, set()) for index in range(len(predicted_clusters)))


def threshold_feature_set_metrics_sweep(
    items: list[Any],
    matrix: Any,
    *,
    linkage: str = FEATURE_SET_EVALUATION_LINKAGE,
    max_threshold_points: int = FEATURE_SET_THRESHOLD_MAX_POINTS,
) -> list[dict[str, Any]]:
    if not items:
        raise ValueError("Synonym-control rows are required for feature-set threshold metrics.")

    truth_keys = [control_truth_key(item) for item in items]
    actual_feature_count = len(set(truth_keys))
    pairs = control_similarity_pairs(items, matrix)
    if not pairs:
        raise ValueError("At least one same-category synonym-control pair is required.")
    strict_no_merge_threshold = math.nextafter(max(pair.similarity for pair in pairs), math.inf)
    observed_thresholds = sorted({pair.similarity for pair in pairs})
    if max_threshold_points < 3:
        raise ValueError("Feature-set threshold sweep requires at least three threshold points.")
    observed_limit = max_threshold_points - 1
    if len(observed_thresholds) > observed_limit:
        indices = {
            round(index * (len(observed_thresholds) - 1) / (observed_limit - 1))
            for index in range(observed_limit)
        }
        observed_thresholds = [observed_thresholds[index] for index in sorted(indices)]
    thresholds = [*observed_thresholds, strict_no_merge_threshold]
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        clusters = produced_feature_clusters(items, pairs, threshold=threshold, linkage=linkage)
        produced_feature_count = len(clusters)
        matched_feature_count = maximum_feature_set_match_count(clusters, truth_keys)
        precision = matched_feature_count / produced_feature_count
        recall = matched_feature_count / actual_feature_count
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "precision": round(precision, 8),
                "recall": round(recall, 8),
                "f1": round(f1, 8),
                "actual_feature_count": actual_feature_count,
                "produced_feature_count": produced_feature_count,
                "matched_feature_count": matched_feature_count,
                "extra_produced_features": produced_feature_count - matched_feature_count,
                "missing_actual_features": actual_feature_count - matched_feature_count,
            }
        )
    return rows


def best_threshold_f1_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one threshold feature-set metrics row is required.")
    return copy.deepcopy(
        max(
            rows,
            key=lambda row: (
                float(row["f1"]),
                float(row["threshold"]),
            ),
        )
    )


def output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "resolved_json": output_dir / f"{prefix}_resolved_features.json",
        "resolved_jsonl": output_dir / f"{prefix}_resolved_features.jsonl",
        "pairs_csv": output_dir / f"{prefix}_pairwise_similarities.csv",
        "pairs_jsonl": output_dir / f"{prefix}_pairwise_similarities.jsonl",
        "most_similar_distinct_features_markdown": output_dir / f"{prefix}_most_similar_distinct_features.md",
        "matrix_csv": output_dir / f"{prefix}_similarity_matrix.csv",
        "heatmap_png": output_dir / f"{prefix}_similarity_heatmap.png",
        "histogram_png": output_dir / f"{prefix}_similarity_histogram.png",
        "boxplot_png": output_dir / f"{prefix}_similarity_boxplot.png",
        "synonym_control_rows_json": output_dir / f"{prefix}_synonym_control_rows.json",
        "synonym_control_rows_jsonl": output_dir / f"{prefix}_synonym_control_rows.jsonl",
        "synonym_pairs_csv": output_dir / f"{prefix}_synonym_pairwise_similarities.csv",
        "synonym_pairs_jsonl": output_dir / f"{prefix}_synonym_pairwise_similarities.jsonl",
        "synonym_histogram_png": output_dir / f"{prefix}_synonym_similarity_histogram.png",
        "synonym_boxplot_png": output_dir / f"{prefix}_synonym_similarity_boxplot.png",
        "distribution_comparison_png": output_dir / f"{prefix}_similarity_distribution_comparison.png",
        "feature_set_metrics_png": output_dir / f"{prefix}_feature_set_precision_recall_f1.png",
        "trace_json": output_dir / f"{prefix}_similarity_trace.json",
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_resolved_rows(paths: dict[str, Path], known_set: Path, rows: list[dict[str, Any]]) -> None:
    paths["resolved_json"].write_text(
        json.dumps(
            {
                "known_set": str(known_set),
                "feature_count": len(rows),
                "features": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(paths["resolved_jsonl"], rows)


def write_pairs(paths: dict[str, Path], pairs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "similarity",
        "left_index",
        "left_feature_name",
        "right_index",
        "right_feature_name",
    ]
    with paths["pairs_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    write_jsonl(paths["pairs_jsonl"], pairs)


def write_most_similar_distinct_features_markdown(
    path: Path,
    *,
    known_set_path: Path,
    text_fields: list[str],
    pairs: list[dict[str, Any]],
    categories: list[str],
    top_n: int = MOST_SIMILAR_DISTINCT_TOP,
) -> Path:
    ordered = sorted(
        pairs,
        key=lambda row: (
            -float(row["similarity"]),
            str(row["left_feature_name"]).casefold(),
            str(row["right_feature_name"]).casefold(),
        ),
    )[:top_n]
    lines = [
        "# Most Similar Distinct Features",
        "",
        f"Input: `{known_set_path}`",
        f"Embedding text fields: `{', '.join(text_fields)}`",
        "",
        (
            f"These are the top {len(ordered)} distinct-feature pairs that become merge-eligible first "
            "as the threshold is lowered. Pairwise eligibility uses `similarity >= threshold`; "
            "it does not model final linkage or merge-group behavior."
        ),
        "",
    ]
    if not ordered:
        lines.append("_No distinct-feature pairs are available._")
    else:
        headers = [
            "Rank",
            "Left feature",
            "Left category",
            "Right feature",
            "Right category",
            "Cosine similarity",
            "First merge-eligible threshold",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for rank, row in enumerate(ordered, start=1):
            left_index = int(row["left_index"]) - 1
            right_index = int(row["right_index"]) - 1
            left_category = categories[left_index] if 0 <= left_index < len(categories) else ""
            right_category = categories[right_index] if 0 <= right_index < len(categories) else ""
            similarity = float(row["similarity"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        markdown_table_text(row.get("left_feature_name")),
                        markdown_table_text(left_category),
                        markdown_table_text(row.get("right_feature_name")),
                        markdown_table_text(right_category),
                        f"{similarity:.8f}",
                        f"{similarity:.8f}",
                    ]
                )
                + " |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_synonym_controls(
    paths: dict[str, Path],
    known_set: Path,
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    paths["synonym_control_rows_json"].write_text(
        json.dumps(
            {
                "known_set": str(known_set),
                "row_count": len(rows),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(paths["synonym_control_rows_jsonl"], rows)
    fieldnames = [
        "similarity",
        "group_feature_key",
        "group_feature_name",
        "pair_type",
        "left_index",
        "left_feature_name",
        "left_variant_type",
        "right_index",
        "right_feature_name",
        "right_variant_type",
    ]
    with paths["synonym_pairs_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    write_jsonl(paths["synonym_pairs_jsonl"], pairs)


def write_matrix_csv(path: Path, labels: list[str], matrix: list[list[float]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_name", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])
    return path


def observed_limits(values: list[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        padding = 0.02
    else:
        padding = max(0.01, (high - low) * 0.06)
    return max(-1.0, low - padding), min(1.0, high + padding)


def write_heatmap(path: Path, labels: list[str], matrix: list[list[float]], *, title: str) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("Matplotlib and NumPy are required for the similarity heatmap.") from exc

    setup_plot_style(plt)
    size = max(7.0, min(20.0, 4.5 + len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(
        np.asarray(matrix, dtype=float),
        cmap=HEATMAP_COLORMAP,
        vmin=HEATMAP_MIN_SIMILARITY,
        vmax=HEATMAP_MAX_SIMILARITY,
    )
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(labelsize=max(5.5, 9.5 - len(labels) * 0.08))
    if len(labels) <= 20:
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                color = "white" if value >= 0.95 else "#102a1b"
                ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    colorbar = fig.colorbar(image, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)
    colorbar.set_ticks([0.8, 0.85, 0.9, 0.95, 1.0])
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_histogram(path: Path, values: list[float], *, title: str) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("Matplotlib is required for the similarity histogram.") from exc

    setup_plot_style(plt)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    if values:
        ax.hist(values, bins=min(30, max(5, round(len(values) ** 0.5))), color="#2563eb", alpha=0.8, edgecolor="white")
        ax.set_xlim(*observed_limits(values))
    else:
        ax.text(0.5, 0.5, "No feature pairs available", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Pair count")
    ax.grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_boxplot(path: Path, values: list[float], *, title: str) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("Matplotlib is required for the similarity box plot.") from exc

    setup_plot_style(plt)
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    if values:
        ax.boxplot(
            values,
            orientation="horizontal",
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "#bfdbfe", "edgecolor": "#1d4ed8"},
            medianprops={"color": "#dc2626", "linewidth": 1.8},
            meanprops={"marker": "D", "markerfacecolor": "#111827", "markeredgecolor": "#111827"},
        )
        ax.set_xlim(*observed_limits(values))
    else:
        ax.text(0.5, 0.5, "No feature pairs available", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.28)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_distribution_comparison(
    path: Path,
    unique_values: list[float],
    synonym_values: list[float],
    *,
    title: str,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("Matplotlib is required for the distribution comparison plot.") from exc

    setup_plot_style(plt)
    combined = [*unique_values, *synonym_values]
    limits = observed_limits(combined)
    labels = [
        f"Distinct known features\nshould not merge (n={len(unique_values)})",
        f"Within-feature synonyms\nshould merge (n={len(synonym_values)})",
    ]
    values = [unique_values, synonym_values]
    colors = ["#60a5fa", "#4ade80"]
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    if all(values):
        boxplot = ax.boxplot(
            values,
            orientation="horizontal",
            tick_labels=labels,
            showmeans=True,
            widths=0.5,
            patch_artist=True,
            medianprops={"color": "#dc2626", "linewidth": 2.0},
            meanprops={
                "marker": "D",
                "markerfacecolor": "#111827",
                "markeredgecolor": "#111827",
                "markersize": 5,
            },
            whiskerprops={"color": "#475569", "linewidth": 1.2},
            capprops={"color": "#475569", "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#94a3b8",
                "markeredgecolor": "#475569",
                "markersize": 4,
                "alpha": 0.6,
            },
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#334155")
            patch.set_alpha(0.78)
    else:
        ax.text(0.5, 0.5, "Both pair distributions are required", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Cosine similarity")
    ax.set_xlim(*limits)
    ax.grid(True, axis="x", alpha=0.28)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_feature_set_metrics_plot(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("Matplotlib is required for the feature-set metrics plot.") from exc

    if not rows:
        raise ValueError("At least one threshold feature-set metrics row is required.")
    setup_plot_style(plt)
    best = best_threshold_f1_row(rows)
    thresholds = [float(row["threshold"]) for row in rows]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.plot(
        thresholds,
        [float(row["f1"]) for row in rows],
        color="#16a34a",
        linewidth=2.2,
        label="F1",
    )
    ax.plot(
        thresholds,
        [float(row["precision"]) for row in rows],
        color="#2563eb",
        linewidth=1.8,
        linestyle="--",
        label="Precision",
    )
    ax.plot(
        thresholds,
        [float(row["recall"]) for row in rows],
        color="#9333ea",
        linewidth=1.8,
        linestyle="-.",
        label="Recall",
    )
    best_threshold = float(best["threshold"])
    best_f1 = float(best["f1"])
    ax.scatter(
        [best_threshold],
        [best_f1],
        color="#dc2626",
        s=48,
        zorder=3,
        label=f"Best F1 ({best_f1:.3f} at {best_threshold:.4f})",
    )
    ax.set_title(title)
    ax.set_xlabel("Merge threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(*observed_limits(thresholds))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_known_feature_similarity(
    *,
    known_set_path: Path,
    annotated_features_path: Path | None,
    text_fields: Iterable[str],
    output_dir: Path,
    prefix: str,
    cache_path: Path,
    client: EmbeddingClient | None,
    model: str = DEFAULT_GEMINI_MODEL,
    task_type: str = DEFAULT_TASK_TYPE,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
    batch_size: int = 50,
    synonym_review_path: Path | None = None,
    distinct_only: bool = False,
    resolved_rows_override: list[dict[str, Any]] | None = None,
    synonym_control_rows_override: list[dict[str, Any]] | None = None,
    annotated_candidate_count_override: int | None = None,
    trace_context: dict[str, Any] | None = None,
    embedding_text_cleanup: EmbeddingTextCleanup | None = None,
    embedding_text_cleanup_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    known_rows = read_feature_rows(known_set_path)
    if len(known_rows) < 2:
        raise ValueError("Known feature similarity requires at least two known-set rows.")
    annotated_rows = read_feature_rows(annotated_features_path) if annotated_features_path else []
    if resolved_rows_override is None:
        resolved_rows = resolve_known_rows(known_rows, annotated_rows)
        annotated_candidate_count = len(annotated_rows)
    else:
        resolved_rows = copy.deepcopy(resolved_rows_override)
        annotated_candidate_count = int(annotated_candidate_count_override or 0)
    review_provenance: dict[str, Any] = {}
    if distinct_only:
        synonym_control_rows = []
    elif synonym_control_rows_override is not None:
        synonym_control_rows = copy.deepcopy(synonym_control_rows_override)
    else:
        if synonym_review_path is None:
            raise ValueError(
                "Synonym analysis requires --synonym-review with two approved synonyms "
                "per feature, or use --distinct-only."
            )
        review = load_synonym_review(synonym_review_path)
        if review is None:
            raise ValueError(
                f"Synonym review does not exist: {synonym_review_path}. Run --review-synonyms first."
            )
        approved = approved_synonym_map(review, known_rows)
        synonym_control_rows = build_synonym_control_rows(resolved_rows, approved, annotated_rows)
        review_provenance = {
            "path": str(synonym_review_path),
            "inventory_fingerprint": review.get("inventory_fingerprint"),
            "prompt_version": review.get("prompt_version"),
            "generation_model": review.get("generation_model"),
        }
    paths = output_paths(output_dir, prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_rows(paths, known_set_path, resolved_rows)

    fields = parse_text_fields(text_fields)
    cleanup_fields = (
        parse_text_fields(embedding_text_cleanup_fields)
        if embedding_text_cleanup is not None and embedding_text_cleanup_fields is not None
        else None
    )
    items = load_feature_items(
        paths["resolved_json"],
        fields,
        cleanup=embedding_text_cleanup,
        cleanup_fields=cleanup_fields,
    )
    if len(items) != len(resolved_rows):
        raise ValueError("One or more resolved known-set rows produced empty embedding text.")
    embeddings, cache_stats = embed_texts_with_cache(
        [item.embedding_text for item in items],
        cache_path=cache_path,
        client=client,
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        batch_size=batch_size,
    )
    matrix = normalized_matrix(embeddings)
    labels = [item.label for item in items]
    pairs, similarity_matrix = pairwise_similarity_data(labels, matrix)
    values = [float(row["similarity"]) for row in pairs]
    write_pairs(paths, pairs)
    write_most_similar_distinct_features_markdown(
        paths["most_similar_distinct_features_markdown"],
        known_set_path=known_set_path,
        text_fields=fields,
        pairs=pairs,
        categories=[str(item.category or "uncategorized") for item in items],
    )
    write_matrix_csv(paths["matrix_csv"], labels, similarity_matrix)
    write_heatmap(paths["heatmap_png"], labels, similarity_matrix, title=f"{known_set_path.stem} Pairwise Similarity")
    write_histogram(paths["histogram_png"], values, title=f"{known_set_path.stem} Similarity Distribution")
    write_boxplot(paths["boxplot_png"], values, title=f"{known_set_path.stem} Similarity Box Plot")

    synonym_items: list[Any] = []
    synonym_pairs: list[dict[str, Any]] = []
    synonym_matrix: Any | None = None
    synonym_cache_stats = None
    if synonym_control_rows:
        paths["synonym_control_rows_json"].write_text(
            json.dumps({"rows": synonym_control_rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        synonym_items = load_feature_items(
            paths["synonym_control_rows_json"],
            fields,
            cleanup=embedding_text_cleanup,
            cleanup_fields=cleanup_fields,
        )
        if len(synonym_items) != len(synonym_control_rows):
            raise ValueError("One or more synonym-control rows produced empty embedding text.")
        synonym_embeddings, synonym_cache_stats = embed_texts_with_cache(
            [item.embedding_text for item in synonym_items],
            cache_path=cache_path,
            client=client,
            model=model,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            batch_size=batch_size,
        )
        synonym_matrix = normalized_matrix(synonym_embeddings)
        synonym_pairs = synonym_control_pair_data(synonym_items, synonym_matrix)
    if not distinct_only:
        write_synonym_controls(paths, known_set_path, synonym_control_rows, synonym_pairs)
    synonym_values = [float(row["similarity"]) for row in synonym_pairs]
    threshold_set_metrics: dict[str, Any] | None = None
    if not distinct_only:
        write_histogram(
            paths["synonym_histogram_png"],
            synonym_values,
            title=f"{known_set_path.stem} Within-Feature Synonym Similarity Distribution",
        )
        write_boxplot(
            paths["synonym_boxplot_png"],
            synonym_values,
            title=f"{known_set_path.stem} Within-Feature Synonym Similarity Box Plot",
        )
        write_distribution_comparison(
            paths["distribution_comparison_png"],
            values,
            synonym_values,
            title=f"{known_set_path.stem} Similarity Distributions",
        )
        if synonym_matrix is None:
            raise ValueError("Synonym-control embeddings are required for feature-set threshold metrics.")
        threshold_rows = threshold_feature_set_metrics_sweep(synonym_items, synonym_matrix)
        best_threshold = best_threshold_f1_row(threshold_rows)
        write_feature_set_metrics_plot(
            paths["feature_set_metrics_png"],
            threshold_rows,
            title=f"{known_set_path.stem} Produced Feature Set Precision, Recall, and F1",
        )
        threshold_set_metrics = {
            "evaluation_unit": "produced feature inventory",
            "actual_features": "known feature groups represented by canonical and synonym-control rows",
            "produced_features": "complete-link clusters and singleton rows within category",
            "matching": "maximum one-to-one matching between produced features and actual features by control provenance",
            "linkage": FEATURE_SET_EVALUATION_LINKAGE,
            "scope": "within category",
            "threshold_sampling": f"up to {FEATURE_SET_THRESHOLD_MAX_POINTS - 1} observed pair thresholds plus strict no-merge",
            "metric_definitions": {
                "precision": "matched produced features divided by all produced features",
                "recall": "matched actual features divided by all actual features",
                "f1": "harmonic mean of feature-set precision and recall",
            },
            "best_selection": "maximum F1; ties prefer the higher threshold",
            "threshold_count": len(threshold_rows),
            "best": best_threshold,
        }

    resolution_counts = Counter(row["resolution"]["status"] for row in resolved_rows)
    synonym_pair_types = sorted({str(row["pair_type"]) for row in synonym_pairs})
    synonym_group_count = len(
        {
            row["synonym_control"]["group_feature_key"]
            for row in synonym_control_rows
        }
    )
    total_cache_hits = cache_stats.cache_hits + (synonym_cache_stats.cache_hits if synonym_cache_stats else 0)
    total_cache_misses = cache_stats.api_misses + (synonym_cache_stats.api_misses if synonym_cache_stats else 0)
    total_api_batches = cache_stats.api_batches + (synonym_cache_stats.api_batches if synonym_cache_stats else 0)
    trace = {
        "project": "KnownFeatureSimilarity",
        "known_set": str(known_set_path),
        "annotated_features": str(annotated_features_path) if annotated_features_path else None,
        "text_fields": fields,
        "embedding_text_cleanup": embedding_text_cleanup_trace_payload(
            embedding_text_cleanup,
            cleanup_fields,
            selected_fields=fields,
        ),
        "model": model,
        "task_type": task_type,
        "output_dimensionality": output_dimensionality,
        "cache_path": str(cache_path),
        "cache": {
            "hits": total_cache_hits,
            "misses": total_cache_misses,
            "api_batches": total_api_batches,
            "known_features": {
                "hits": cache_stats.cache_hits,
                "misses": cache_stats.api_misses,
                "api_batches": cache_stats.api_batches,
            },
            "synonym_controls": {
                "hits": synonym_cache_stats.cache_hits if synonym_cache_stats else 0,
                "misses": synonym_cache_stats.api_misses if synonym_cache_stats else 0,
                "api_batches": synonym_cache_stats.api_batches if synonym_cache_stats else 0,
            },
        },
        "counts": {
            "known_features": len(resolved_rows),
            "annotated_candidates": annotated_candidate_count,
            "pairwise_similarities": len(pairs),
            "synonym_control_groups": synonym_group_count,
            "synonym_control_rows": len(synonym_control_rows),
            "synonym_pairwise_similarities": len(synonym_pairs),
        },
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "similarity_summary": similarity_summary(pairs),
        "synonym_similarity_summary": similarity_summary(synonym_pairs),
        "synonym_similarity_summaries_by_pair_type": {
            pair_type: similarity_summary(
                [row for row in synonym_pairs if row["pair_type"] == pair_type]
            )
            for pair_type in synonym_pair_types
        },
        "synonym_control_method": {
            "source": "human_approved_synonym" if synonym_control_rows else None,
            "row_generation": "canonical anchor plus exactly two approved synonym labels",
            "metadata_resolution": "each control label resolved independently",
            "pair_scope": "within approved feature group only",
            "review": review_provenance,
        },
        "distinct_only": distinct_only,
        "experiment_context": trace_context or {},
        "outputs": {
            key: str(path)
            for key, path in paths.items()
            if (
                not distinct_only
                or (
                    not key.startswith("synonym_")
                    and key not in {"distribution_comparison_png", "feature_set_metrics_png"}
                )
            )
        },
    }
    if threshold_set_metrics is not None:
        trace["threshold_set_metrics"] = threshold_set_metrics
    paths["trace_json"].write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return trace


def separation_metrics(
    distinct_values: list[float],
    synonym_values: list[float],
) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover - dependencies are required
        raise RuntimeError("NumPy and scikit-learn are required for separation metrics.") from exc

    if not distinct_values or not synonym_values:
        raise ValueError("Both distinct-feature and synonym-control similarities are required.")
    distinct = np.asarray(distinct_values, dtype=float)
    synonyms = np.asarray(synonym_values, dtype=float)
    labels = np.asarray([0] * len(distinct) + [1] * len(synonyms), dtype=int)
    scores = np.concatenate([distinct, synonyms])
    roc_auc = float(roc_auc_score(labels, scores))
    mean_gap = float(synonyms.mean() - distinct.mean())
    median_gap = float(np.median(synonyms) - np.median(distinct))
    pooled_standard_deviation = float(math.sqrt((float(distinct.var()) + float(synonyms.var())) / 2.0))
    standardized_mean_gap = mean_gap / pooled_standard_deviation if pooled_standard_deviation else None

    def rounded(value: float | None) -> float | None:
        return None if value is None else round(float(value), 8)

    return {
        "roc_auc": rounded(roc_auc),
        "cliffs_delta": rounded(2.0 * roc_auc - 1.0),
        "mean_gap": rounded(mean_gap),
        "median_gap": rounded(median_gap),
        "standardized_mean_gap": rounded(standardized_mean_gap),
        "distinct_pair_count": int(len(distinct)),
        "synonym_pair_count": int(len(synonyms)),
    }


def attribute_comparison_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "generated_attributes_json": output_dir / f"{prefix}_generated_attributes.json",
        "generated_attributes_jsonl": output_dir / f"{prefix}_generated_attributes.jsonl",
        "attribute_generation_trace_json": output_dir / f"{prefix}_attribute_generation_trace.json",
        "synonym_controls_markdown": output_dir / f"{prefix}_synonym_controls.md",
        "rankings_csv": output_dir / f"{prefix}_attribute_comparison_rankings.csv",
        "rankings_json": output_dir / f"{prefix}_attribute_comparison_rankings.json",
        "ranking_png": output_dir / f"{prefix}_attribute_comparison_ranking.png",
        "trace_json": output_dir / f"{prefix}_attribute_comparison_trace.json",
    }


def write_generated_attributes(
    paths: dict[str, Path],
    generated: dict[tuple[str, str], dict[str, Any]],
    generation_trace: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = sorted(
        (copy.deepcopy(row) for row in generated.values()),
        key=lambda row: (
            str(row.get("category") or "").casefold(),
            str(row.get("feature_name") or "").casefold(),
        ),
    )
    paths["generated_attributes_json"].write_text(
        json.dumps({"row_count": len(rows), "rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_jsonl(paths["generated_attributes_jsonl"], rows)
    paths["attribute_generation_trace_json"].write_text(
        json.dumps(generation_trace, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return rows


def markdown_table_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text.replace("|", "\\|")


def known_set_display_name(path: Path) -> str:
    slug = path.parent.name or path.stem
    semeval = slug.split("_")
    if len(semeval) >= 3 and semeval[0].casefold() == "semeval" and semeval[1].isdigit():
        task = semeval[2].casefold().removeprefix("task")
        if task.isdigit():
            return f"SemEval-{semeval[1]} Task {task}"
    if len(semeval) >= 2 and semeval[-2].isdigit() and semeval[-1].isdigit():
        version = f"{semeval[-2]}.{semeval[-1]}"
        semeval = semeval[:-2]
    else:
        version = ""
    name = " ".join(word.capitalize() for word in semeval)
    return f"{name} {version}".strip()


def write_synonym_controls_markdown(
    path: Path,
    *,
    known_set_path: Path,
    rows: list[dict[str, Any]],
    known_rows: list[dict[str, Any]] | None = None,
) -> Path:
    groups: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for row in rows:
        control = row.get("synonym_control")
        if not isinstance(control, dict):
            continue
        group_key = str(control.get("group_feature_key") or "")
        if group_key not in groups:
            groups[group_key] = []
            group_order.append(group_key)
        groups[group_key].append(row)

    controlled_keys = set(groups)
    without_controls = [
        row
        for row in (known_rows or [])
        if str(row.get("feature_key") or "") not in controlled_keys
    ]
    prompt_versions = sorted(
        {
            str(metadata.get("prompt_version") or "")
            for row in rows
            if isinstance((metadata := row.get("generated_attributes")), dict)
            and metadata.get("prompt_version")
        }
    )
    generation_models = sorted(
        {
            str(metadata.get("model") or "")
            for row in rows
            if isinstance((metadata := row.get("generated_attributes")), dict)
            and metadata.get("model")
        }
    )
    known_feature_count = len(known_rows) if known_rows is not None else len(groups)
    lines = [
        f"# {known_set_display_name(known_set_path)} Synonym Controls",
        "",
        (
            f"This report lists {len(rows)} synonym-control rows across {len(groups)} "
            f"of {known_feature_count} atomic features."
        ),
        "",
        (
            "Each feature has exactly two **human-approved synonym control labels**. "
            "Definitions, generated embedding-metadata synonyms, and examples were "
            "independently generated for each canonical and control label using only that "
            "label and category, so the approved relationship is not leaked into its metadata."
        ),
    ]
    if prompt_versions or generation_models:
        lines.extend(
            [
                "",
                (
                    f"Generated metadata model: `{', '.join(generation_models) or 'unknown'}`; "
                    f"prompt version: `{', '.join(prompt_versions) or 'unknown'}`."
                ),
            ]
        )
    lines.extend(
        [
            "",
        "## Approved Control Summary",
        "",
        "| # | Atomic feature | Control labels |",
        "|---:|---|---|",
        ]
    )
    for index, group_key in enumerate(group_order, start=1):
        group_rows = groups[group_key]
        family_name = str(group_rows[0]["synonym_control"].get("group_feature_name") or group_key)
        labels = ", ".join(f"`{markdown_table_text(row.get('feature_name'))}`" for row in group_rows)
        lines.append(f"| {index} | {markdown_table_text(family_name)} | {labels} |")

    if known_rows is not None:
        lines.extend(["", "## Features Without Synonym Controls", ""])
        if without_controls:
            lines.append(
                "These atomic features do not have complete approved synonym controls:"
            )
            lines.append("")
            lines.extend(f"- `{markdown_table_text(row.get('feature_name'))}`" for row in without_controls)
        else:
            lines.append("Every atomic feature has exactly two approved synonym controls.")

    lines.extend(["", "## Control Details", ""])
    for index, group_key in enumerate(group_order, start=1):
        group_rows = groups[group_key]
        family_name = str(group_rows[0]["synonym_control"].get("group_feature_name") or group_key)
        lines.extend(
            [
                f"### {index}. {family_name}",
                "",
                "| Type | Control label | Generated close synonyms | Generated definition | Example |",
                "|---|---|---|---|---|",
            ]
        )
        for row in group_rows:
            control = row["synonym_control"]
            variant_type = str(control.get("variant_type") or "")
            synonyms = "; ".join(parse_jsonish_list(row.get("synonyms"))) or "_None_"
            definitions = parse_jsonish_list(row.get("definitions"))
            examples = parse_jsonish_list(row.get("examples"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_table_text(variant_type),
                        f"`{markdown_table_text(row.get('feature_name'))}`",
                        markdown_table_text(synonyms),
                        markdown_table_text(definitions[0] if definitions else ""),
                        markdown_table_text(examples[0] if examples else ""),
                    ]
                )
                + " |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_attribute_rankings(paths: dict[str, Path], rankings: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "combination",
        "field_count",
        "text_fields",
        "roc_auc",
        "mean_gap",
        "median_gap",
        "standardized_mean_gap",
        "cliffs_delta",
        "best_set_threshold",
        "best_set_precision",
        "best_set_recall",
        "best_set_f1",
        "distinct_pair_count",
        "synonym_pair_count",
        "output_dir",
        "trace_path",
    ]
    with paths["rankings_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "text_fields": ",".join(row["text_fields"]),
            }
            for row in rankings
        )
    paths["rankings_json"].write_text(
        json.dumps({"combination_count": len(rankings), "rankings": rankings}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_attribute_ranking_plot(path: Path, rankings: list[dict[str, Any]], *, title: str) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise RuntimeError("Matplotlib is required for the attribute ranking plot.") from exc

    setup_plot_style(plt)
    ordered = list(reversed(rankings))
    labels = [str(row["combination"]) for row in ordered]
    values = [float(row["roc_auc"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(12.5, max(7.5, len(rankings) * 0.52)))
    bars = ax.barh(labels, values, color="#16a34a", alpha=0.82)
    ax.axvline(0.5, color="#dc2626", linestyle="--", linewidth=1.5, label="Chance ROC AUC")
    for bar, row in zip(bars, ordered):
        ax.text(
            min(1.005, float(row["roc_auc"]) + 0.008),
            bar.get_y() + bar.get_height() / 2,
            f"ROC AUC {float(row['roc_auc']):.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0.45, 1.08)
    ax.set_xlabel("ROC AUC (synonym-control pair ranks above distinct-feature pair)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.28)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def rank_attribute_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rankings = [copy.deepcopy(row) for row in rows]
    rankings.sort(
        key=lambda row: (
            -float(row["roc_auc"]),
            -float(row["mean_gap"]),
            int(row["field_count"]),
            str(row["combination"]),
        )
    )
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank
    return rankings


def run_attribute_comparison(
    *,
    known_set_path: Path,
    annotated_features_path: Path | None,
    output_dir: Path,
    prefix: str,
    embedding_cache_path: Path,
    attribute_cache_path: Path,
    synonym_review_path: Path,
    embedding_client: EmbeddingClient | None,
    attribute_client: Any | None,
    model: str = DEFAULT_GEMINI_MODEL,
    attribute_generation_model: str = ATTRIBUTE_GENERATION_MODEL,
    task_type: str = DEFAULT_TASK_TYPE,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
    batch_size: int = 50,
    refresh_generated_attributes: bool = False,
    embedding_text_cleanup: EmbeddingTextCleanup | None = None,
    embedding_text_cleanup_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    known_rows = read_feature_rows(known_set_path)
    if len(known_rows) < 2:
        raise ValueError("Known feature similarity requires at least two known-set rows.")
    annotated_rows = read_feature_rows(annotated_features_path) if annotated_features_path else []
    resolved_rows = resolve_known_rows(known_rows, annotated_rows)
    review = load_synonym_review(synonym_review_path)
    if review is None:
        raise ValueError(
            f"Synonym review does not exist: {synonym_review_path}. Run --review-synonyms first."
        )
    approved = approved_synonym_map(review, known_rows)
    control_rows = build_synonym_control_rows(resolved_rows, approved, annotated_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_paths = attribute_comparison_paths(output_dir, prefix)
    generated, generation_trace = generate_attribute_metadata(
        [*resolved_rows, *control_rows],
        cache_path=attribute_cache_path,
        client=attribute_client,
        model=attribute_generation_model,
        refresh=refresh_generated_attributes,
    )
    generated_known_rows = apply_generated_attributes(
        resolved_rows,
        generated,
        model=attribute_generation_model,
    )
    generated_control_rows = apply_generated_attributes(
        control_rows,
        generated,
        model=attribute_generation_model,
    )
    generated_rows = write_generated_attributes(comparison_paths, generated, generation_trace)
    write_synonym_controls_markdown(
        comparison_paths["synonym_controls_markdown"],
        known_set_path=known_set_path,
        rows=generated_control_rows,
        known_rows=generated_known_rows,
    )

    ranking_rows: list[dict[str, Any]] = []
    expected_counts: tuple[int, int, int, int] | None = None
    cleanup_fields = (
        parse_text_fields(embedding_text_cleanup_fields)
        if embedding_text_cleanup is not None and embedding_text_cleanup_fields is not None
        else None
    )
    for fields in attribute_combinations():
        slug = attribute_combination_slug(fields)
        combination_dir = output_dir / "combinations" / slug
        run_trace = run_known_feature_similarity(
            known_set_path=known_set_path,
            annotated_features_path=annotated_features_path,
            text_fields=fields,
            output_dir=combination_dir,
            prefix=prefix,
            cache_path=embedding_cache_path,
            client=embedding_client,
            model=model,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            batch_size=batch_size,
            resolved_rows_override=generated_known_rows,
            synonym_control_rows_override=generated_control_rows,
            annotated_candidate_count_override=len(annotated_rows),
            embedding_text_cleanup=embedding_text_cleanup,
            embedding_text_cleanup_fields=cleanup_fields,
            trace_context={
                "mode": "embed_attribute_comparison",
                "combination": slug,
                "generated_attribute_prompt_version": ATTRIBUTE_GENERATION_PROMPT_VERSION,
                "generated_attribute_model": attribute_generation_model,
                "generated_attribute_cache": str(attribute_cache_path),
                "generated_attribute_input_fields": ["feature_name", "category"],
                "embedding_text_cleanup": embedding_text_cleanup_trace_payload(
                    embedding_text_cleanup,
                    cleanup_fields,
                    selected_fields=fields,
                ),
                "synonym_review": {
                    "path": str(synonym_review_path),
                    "inventory_fingerprint": review.get("inventory_fingerprint"),
                    "prompt_version": review.get("prompt_version"),
                    "generation_model": review.get("generation_model"),
                },
            },
        )
        counts = run_trace["counts"]
        population = (
            int(counts["known_features"]),
            int(counts["pairwise_similarities"]),
            int(counts["synonym_control_rows"]),
            int(counts["synonym_pairwise_similarities"]),
        )
        if expected_counts is None:
            expected_counts = population
        elif population != expected_counts:
            raise ValueError(
                f"Attribute combination {slug} evaluated a different row population: "
                f"{population} != {expected_counts}."
            )
        distinct_pairs = read_feature_rows(Path(run_trace["outputs"]["pairs_csv"]))
        synonym_pairs = read_feature_rows(Path(run_trace["outputs"]["synonym_pairs_csv"]))
        metrics = separation_metrics(
            [float(row["similarity"]) for row in distinct_pairs],
            [float(row["similarity"]) for row in synonym_pairs],
        )
        best_set = run_trace["threshold_set_metrics"]["best"]
        ranking_rows.append(
            {
                "combination": slug,
                "field_count": len(fields),
                "text_fields": list(fields),
                **metrics,
                "best_set_threshold": best_set["threshold"],
                "best_set_precision": best_set["precision"],
                "best_set_recall": best_set["recall"],
                "best_set_f1": best_set["f1"],
                "output_dir": str(combination_dir),
                "trace_path": run_trace["outputs"]["trace_json"],
            }
        )

    ranking_rows = rank_attribute_comparisons(ranking_rows)
    write_attribute_rankings(comparison_paths, ranking_rows)
    write_attribute_ranking_plot(
        comparison_paths["ranking_png"],
        ranking_rows,
        title=f"{known_set_path.stem} Embedding Attribute Separation Ranking",
    )
    trace = {
        "project": "KnownFeatureAttributeComparison",
        "known_set": str(known_set_path),
        "annotated_features": str(annotated_features_path) if annotated_features_path else None,
        "model": model,
        "attribute_generation": generation_trace,
        "embedding_cache_path": str(embedding_cache_path),
        "attribute_cache_path": str(attribute_cache_path),
        "synonym_review": {
            "path": str(synonym_review_path),
            "inventory_fingerprint": review.get("inventory_fingerprint"),
            "prompt_version": review.get("prompt_version"),
            "generation_model": review.get("generation_model"),
        },
        "comparison_fields": list(ATTRIBUTE_COMPARISON_FIELDS),
        "embedding_text_cleanup": embedding_text_cleanup_trace_payload(
            embedding_text_cleanup,
            cleanup_fields,
        ),
        "combination_count": len(ranking_rows),
        "generated_attribute_row_count": len(generated_rows),
        "population_counts": {
            "known_features": expected_counts[0] if expected_counts else 0,
            "distinct_pairs": expected_counts[1] if expected_counts else 0,
            "synonym_control_rows": expected_counts[2] if expected_counts else 0,
            "synonym_pairs": expected_counts[3] if expected_counts else 0,
        },
        "ranking_method": {
            "primary": "roc_auc_descending",
            "tie_breakers": ["mean_gap_descending", "field_count_ascending", "combination_ascending"],
            "positive_class": "within-feature synonym-control pair",
            "negative_class": "distinct known-feature pair",
        },
        "rankings": ranking_rows,
        "outputs": {key: str(path) for key, path in comparison_paths.items()},
    }
    comparison_paths["trace_json"].write_text(
        json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure embedding similarities within an authoritative known feature inventory.",
    )
    parser.add_argument("known_set", type=Path, help="Known-set CSV, JSONL, or JSON file.")
    parser.add_argument(
        "--annotated-features",
        type=Path,
        help=f"Optional annotated feature inventory used to enrich known rows. Suggested: {DEFAULT_ANNOTATED_FEATURES}",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for experiment outputs.")
    parser.add_argument("--prefix", help="Output filename prefix. Defaults to the known-set stem.")
    parser.add_argument("--cache", type=Path, help="Embedding cache JSONL path.")
    parser.add_argument(
        "--text-fields",
        default=None,
        help="Comma-separated resolved-row fields used to build embedding text.",
    )
    parser.add_argument(
        "--embedding-text-cleanup-builtin",
        action="append",
        choices=sorted(EMBEDDING_TEXT_CLEANUP_BUILTINS),
        help=(
            "Named embedding-text cleanup builtin to apply before embedding. "
            "May be repeated. By default no cleanup is applied."
        ),
    )
    parser.add_argument(
        "--embedding-text-cleanup-fields",
        default="definitions",
        help=(
            "Comma-separated row fields to clean when cleanup is enabled. "
            "Defaults to definitions so non-definition attribute combinations remain controls."
        ),
    )
    parser.add_argument(
        "--embedding-text-cleanup-min-chars",
        type=int,
        default=24,
        help="Use the original field value if cleanup leaves fewer than this many characters.",
    )
    parser.add_argument(
        "--embedding-text-cleanup-fallback",
        choices=["original", "empty"],
        default="original",
        help="Fallback when cleanup removes too much text.",
    )
    parser.add_argument(
        "--embed-attribute-comparison",
        action="store_true",
        help=(
            "Ensure synonym review is complete, then generate independent row metadata "
            "and run all 15 non-empty text-field combinations."
        ),
    )
    parser.add_argument(
        "--review-synonyms",
        action="store_true",
        help="Interactively propose and approve exactly two synonym controls per feature, then exit without analysis.",
    )
    parser.add_argument(
        "--synonym-review",
        type=Path,
        help="Human synonym-review JSON. Defaults to <output-dir>/<prefix>_synonym_review.json.",
    )
    parser.add_argument(
        "--synonym-generation-model",
        default=ATTRIBUTE_GENERATION_MODEL,
        help="Gemini model used to propose synonym controls during --review-synonyms.",
    )
    parser.add_argument(
        "--distinct-only",
        action="store_true",
        help="Run only distinct-feature analysis without synonym controls or synonym artifacts.",
    )
    parser.add_argument(
        "--attribute-generation-model",
        default=ATTRIBUTE_GENERATION_MODEL,
        help="Gemini model used to independently generate definitions, synonyms, and examples.",
    )
    parser.add_argument(
        "--attribute-generation-cache",
        type=Path,
        help="Generated-attribute cache JSONL path.",
    )
    parser.add_argument(
        "--refresh-generated-attributes",
        action="store_true",
        help="Regenerate attribute metadata even when a matching cache entry exists.",
    )
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.known_set.exists():
            raise FileNotFoundError(f"Known-set file does not exist: {args.known_set}")
        if args.annotated_features and not args.annotated_features.exists():
            raise FileNotFoundError(f"Annotated feature file does not exist: {args.annotated_features}")
        if args.timeout <= 0:
            raise ValueError("--timeout must be greater than 0.")

        output_dir = args.output_dir or args.known_set.parent / "embedding_similarity"
        prefix = sanitize_filename_part(args.prefix or args.known_set.stem)
        cache_path = args.cache or output_dir / f"{prefix}_gemini_embedding_cache.jsonl"
        attribute_cache_path = (
            args.attribute_generation_cache
            or output_dir / f"{prefix}_generated_attribute_cache.jsonl"
        )
        synonym_review_path = (
            args.synonym_review
            or output_dir / f"{prefix}_synonym_review.json"
        )
        if args.review_synonyms and args.embed_attribute_comparison:
            raise ValueError("--review-synonyms cannot be used with --embed-attribute-comparison.")
        if args.review_synonyms and args.distinct_only:
            raise ValueError("--review-synonyms cannot be used with --distinct-only.")
        if args.embed_attribute_comparison and args.distinct_only:
            raise ValueError("--embed-attribute-comparison cannot be used with --distinct-only.")
        if args.embed_attribute_comparison and args.text_fields is not None:
            raise ValueError("--text-fields cannot be used with --embed-attribute-comparison.")
        embedding_text_cleanup = embedding_text_cleanup_from_args(
            strip_builtins=args.embedding_text_cleanup_builtin,
            min_chars=args.embedding_text_cleanup_min_chars,
            fallback=args.embedding_text_cleanup_fallback,
        )
        embedding_text_cleanup_fields = (
            parse_text_fields(args.embedding_text_cleanup_fields)
            if embedding_text_cleanup is not None
            else None
        )
        api_key = gemini_api_key(args.gemini_api_key)
        client: EmbeddingClient | None = None
        attribute_client: Any | None = None
        synonym_client: Any | None = None
        if api_key:
            client = GeminiEmbeddingClient(
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                task_type=args.task_type,
                output_dimensionality=args.output_dimensionality,
                timeout=args.timeout,
            )
            attribute_client = make_llm_client(
                api_key=api_key,
                model=args.attribute_generation_model,
                base_url=args.base_url,
                reasoning_effort="low",
                api_provider="gemini",
                input_mode="text",
                timeout=args.timeout,
            )
            synonym_client = make_llm_client(
                api_key=api_key,
                model=args.synonym_generation_model,
                base_url=args.base_url,
                reasoning_effort="minimal",
                api_provider="gemini",
                input_mode="text",
                timeout=args.timeout,
            )
        if args.review_synonyms:
            review_trace = run_synonym_review(
                known_set_path=args.known_set,
                review_path=synonym_review_path,
                client=synonym_client,
                model=args.synonym_generation_model,
            )
            print(
                f"Synonym review: {review_trace['approved_count']}/"
                f"{review_trace['feature_count']} features approved"
            )
            print(f"Review file: {review_trace['review_path']}")
            return 0
        if not args.distinct_only:
            ensure_synonym_review_for_run(
                known_set_path=args.known_set,
                review_path=synonym_review_path,
                client=synonym_client,
                model=args.synonym_generation_model,
                interactive=sys.stdin.isatty(),
            )
        if args.embed_attribute_comparison:
            trace = run_attribute_comparison(
                known_set_path=args.known_set,
                annotated_features_path=args.annotated_features,
                output_dir=output_dir,
                prefix=prefix,
                embedding_cache_path=cache_path,
                attribute_cache_path=attribute_cache_path,
                synonym_review_path=synonym_review_path,
                embedding_client=client,
                attribute_client=attribute_client,
                model=args.model,
                attribute_generation_model=args.attribute_generation_model,
                task_type=args.task_type,
                output_dimensionality=args.output_dimensionality,
                batch_size=args.batch_size,
                refresh_generated_attributes=args.refresh_generated_attributes,
                embedding_text_cleanup=embedding_text_cleanup,
                embedding_text_cleanup_fields=embedding_text_cleanup_fields,
            )
        else:
            trace = run_known_feature_similarity(
                known_set_path=args.known_set,
                annotated_features_path=args.annotated_features,
                text_fields=parse_text_fields(args.text_fields or DEFAULT_TEXT_FIELDS),
                output_dir=output_dir,
                prefix=prefix,
                cache_path=cache_path,
                client=client,
                synonym_review_path=synonym_review_path,
                distinct_only=args.distinct_only,
                model=args.model,
                task_type=args.task_type,
                output_dimensionality=args.output_dimensionality,
                batch_size=args.batch_size,
                embedding_text_cleanup=embedding_text_cleanup,
                embedding_text_cleanup_fields=embedding_text_cleanup_fields,
            )
    except Exception as exc:
        print(f"known_feature_similarity: {exc}", file=sys.stderr)
        return 1

    if args.embed_attribute_comparison:
        print(f"Completed {trace['combination_count']} generated-attribute combinations")
        print(f"Best combination: {trace['rankings'][0]['combination']} (ROC AUC {trace['rankings'][0]['roc_auc']})")
    else:
        print(f"Resolved {trace['counts']['known_features']} known features from {args.known_set}")
        print(f"Computed {trace['counts']['pairwise_similarities']} unique pairwise similarities")
        print(
            f"Embedding cache: {trace['cache']['hits']} hits, {trace['cache']['misses']} misses, "
            f"{trace['cache']['api_batches']} API batches"
        )
    for label, path in trace["outputs"].items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
