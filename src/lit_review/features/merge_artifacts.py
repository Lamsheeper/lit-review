"""Features merge_artifacts."""

from __future__ import annotations


import hashlib


from collections import Counter


from pathlib import Path

from typing import Any

from lit_review.core.io import utc_now, write_json

from lit_review.embeddings.core import FeatureItem

from lit_review.embeddings.similarity import (
    MergeGroup,
    SimilarityPair,
    build_merge_groups,
    build_merged_features,
    category_merge_summary,
    merged_fieldnames,
)

from lit_review.core.plotting import write_category_retention_plot


from lit_review.features.records import compact_list, feature_key, truncate_text

from lit_review.features.prompts import annotation_int


from lit_review.features.filtering import write_csv
from lit_review.core.io import write_jsonl

from lit_review.features.annotation_artifacts import markdown_table_cell


LLM_MERGE_QUALITATIVE_TOP_SIMILAR = 20


LLM_MERGE_QUALITATIVE_TOP_ACCEPTED = 20


LLM_MERGE_QUALITATIVE_RANDOM = 10


LLM_MERGE_CANDIDATE_FIELDNAMES = [
    "pair_id",
    "similarity",
    "category",
    "feature_key_1",
    "feature_name_1",
    "row_index_1",
    "paper_count_1",
    "mention_count_1",
    "feature_key_2",
    "feature_name_2",
    "row_index_2",
    "paper_count_2",
    "mention_count_2",
    "selected_by_feature_keys",
    "selected_by_ranks",
]


LLM_MERGE_DECISION_FIELDNAMES = [
    "pair_id",
    "similarity",
    "category",
    "feature_key_1",
    "feature_name_1",
    "feature_key_2",
    "feature_name_2",
    "decision",
    "confidence",
    "rationale",
    "model",
    "run_id",
    "decided_at",
    "batch_id",
    "batch_index",
]


def latest_llm_merge_decisions_by_pair(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        pair_id = str(decision.get("pair_id") or "")
        if pair_id:
            latest[pair_id] = decision
    return latest


def accepted_llm_merge_pairs(
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    item_index_by_feature_key: dict[str, int],
) -> list[SimilarityPair]:
    decision_map = latest_llm_merge_decisions_by_pair(decisions)
    pairs: list[SimilarityPair] = []
    for candidate in candidates:
        pair_id = str(candidate.get("pair_id") or "")
        decision = decision_map.get(pair_id)
        if not decision or decision.get("decision") != "merge":
            continue
        left_index = item_index_by_feature_key.get(str(candidate.get("feature_key_1") or ""))
        right_index = item_index_by_feature_key.get(str(candidate.get("feature_key_2") or ""))
        if left_index is None or right_index is None:
            continue
        pairs.append(
            SimilarityPair(
                left_index=left_index,
                right_index=right_index,
                similarity=round(float(candidate.get("similarity") or 0.0), 8),
            )
        )
    return pairs


def enrich_llm_merged_features(
    rows: list[dict[str, Any]],
    groups: list[MergeGroup],
    items: list[FeatureItem],
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decision_map = latest_llm_merge_decisions_by_pair(decisions)
    group_by_canonical_key = {items[group.canonical_index].feature_key: group for group in groups}
    output: list[dict[str, Any]] = []
    for row in rows:
        merge = row.get("embedding_merge")
        if not isinstance(merge, dict):
            output.append(row)
            continue
        group = group_by_canonical_key.get(str(merge.get("canonical_feature_key") or ""))
        if group is None:
            output.append(row)
            continue
        member_keys = {items[index].feature_key for index in group.member_indices}
        group_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("feature_key_1") or "") in member_keys
            and str(candidate.get("feature_key_2") or "") in member_keys
        ]
        group_decisions = [
            decision
            for candidate in group_candidates
            for decision in [decision_map.get(str(candidate.get("pair_id") or ""))]
            if decision is not None
        ]
        accepted_decisions = [decision for decision in group_decisions if decision.get("decision") == "merge"]
        merge["mode"] = "llm_top_k"
        merge["top_k_candidates"] = group_candidates
        merge["llm_decisions"] = group_decisions
        merge["accepted_pair_ids"] = [decision.get("pair_id") for decision in accepted_decisions]
        merge["decision_summary"] = {
            "candidate_pairs": len(group_candidates),
            "accepted_pairs": len(accepted_decisions),
        }
        output.append(row)
    return output


def llm_merge_candidate_pair_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("pair_id") or "")


def llm_merge_candidate_similarity(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("similarity") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def llm_merge_candidate_support(candidate: dict[str, Any]) -> tuple[int, int]:
    mention_count = annotation_int(candidate.get("mention_count_1")) + annotation_int(
        candidate.get("mention_count_2")
    )
    paper_count = annotation_int(candidate.get("paper_count_1")) + annotation_int(
        candidate.get("paper_count_2")
    )
    return mention_count, paper_count


def sort_llm_merge_candidates_for_report(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            -llm_merge_candidate_similarity(row),
            str(row.get("category") or "").casefold(),
            str(row.get("feature_name_1") or "").casefold(),
            str(row.get("feature_name_2") or "").casefold(),
            llm_merge_candidate_pair_id(row),
        ),
    )


def deterministic_llm_merge_candidate_sample(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    excluded_pair_ids: set[str],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    pool = [
        candidate
        for candidate in candidates
        if llm_merge_candidate_pair_id(candidate) not in excluded_pair_ids
    ]
    return sorted(
        pool,
        key=lambda row: (
            hashlib.sha1(
                f"llm_merge_qualitative:{llm_merge_candidate_pair_id(row)}".encode("utf-8")
            ).hexdigest(),
            llm_merge_candidate_pair_id(row),
        ),
    )[:limit]


def llm_merge_report_decision(
    candidate: dict[str, Any],
    decision_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decision = decision_map.get(llm_merge_candidate_pair_id(candidate))
    if decision is None:
        return {
            "decision": "pending",
            "confidence": "",
            "rationale": "No LLM decision recorded yet.",
        }
    return {
        "decision": decision.get("decision") or "uncertain",
        "confidence": decision.get("confidence"),
        "rationale": decision.get("rationale") or "",
    }


def llm_merge_report_feature_evidence(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
    definitions = compact_list(row.get("definitions"), 1)
    definition = (
        annotation.get("short_definition")
        or annotation.get("formal_definition")
        or (definitions[0] if definitions else "")
        or ""
    )
    synonyms = compact_list(
        annotation.get("synonyms") or row.get("annotation_synonyms") or row.get("synonyms"), 4
    )
    examples = compact_list(
        annotation.get("examples") or row.get("annotation_examples") or row.get("examples"), 3
    )
    parts: list[str] = []
    if definition:
        parts.append(truncate_text(str(definition), 150))
    if synonyms:
        parts.append("Synonyms: " + ", ".join(str(value) for value in synonyms))
    if examples:
        parts.append("Examples: " + "; ".join(str(value) for value in examples))
    return truncate_text(" | ".join(parts), 260)


def llm_merge_report_row(
    candidate: dict[str, Any],
    *,
    decision_map: dict[str, dict[str, Any]],
    rows_by_key: dict[str, dict[str, Any]],
) -> list[str]:
    decision = llm_merge_report_decision(candidate, decision_map)
    row_1 = rows_by_key.get(str(candidate.get("feature_key_1") or ""))
    row_2 = rows_by_key.get(str(candidate.get("feature_key_2") or ""))
    return [
        markdown_table_cell(candidate.get("pair_id"), 40),
        markdown_table_cell(candidate.get("category"), 40),
        markdown_table_cell(f"{llm_merge_candidate_similarity(candidate):.4f}", 20),
        markdown_table_cell(decision.get("decision"), 20),
        markdown_table_cell(decision.get("confidence"), 20),
        markdown_table_cell(candidate.get("feature_name_1"), 80),
        markdown_table_cell(candidate.get("feature_name_2"), 80),
        markdown_table_cell(decision.get("rationale"), 180),
        markdown_table_cell(llm_merge_report_feature_evidence(row_1), 260),
        markdown_table_cell(llm_merge_report_feature_evidence(row_2), 260),
    ]


def append_llm_merge_report_section(
    lines: list[str],
    title: str,
    description: str,
    candidates: list[dict[str, Any]],
    *,
    decision_map: dict[str, dict[str, Any]],
    rows_by_key: dict[str, dict[str, Any]],
) -> None:
    lines.extend([f"## {title}", "", description, ""])
    if not candidates:
        lines.extend(["_No examples in this section._", ""])
        return
    headers = [
        "Pair",
        "Category",
        "Similarity",
        "LLM",
        "Confidence",
        "Feature 1",
        "Feature 2",
        "Rationale",
        "Feature 1 evidence",
        "Feature 2 evidence",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for candidate in candidates:
        lines.append(
            "| "
            + " | ".join(
                llm_merge_report_row(
                    candidate,
                    decision_map=decision_map,
                    rows_by_key=rows_by_key,
                )
            )
            + " |"
        )
    lines.append("")


def build_llm_merge_qualitative_markdown(
    *,
    input_file: Path | None,
    features: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    accepted_pair_count: int,
    merge_group_count: int,
) -> str:
    decision_map = latest_llm_merge_decisions_by_pair(decisions)
    rows_by_key = {str(row.get("feature_key") or feature_key(row)): row for row in features}
    decision_counts = Counter(
        llm_merge_report_decision(candidate, decision_map).get("decision") or "pending"
        for candidate in candidates
    )
    top_similar = sort_llm_merge_candidates_for_report(candidates)[:LLM_MERGE_QUALITATIVE_TOP_SIMILAR]
    accepted = [
        candidate
        for candidate in candidates
        if decision_map.get(llm_merge_candidate_pair_id(candidate), {}).get("decision") == "merge"
    ]
    top_accepted = sorted(
        accepted,
        key=lambda row: (
            -llm_merge_candidate_similarity(row),
            -llm_merge_candidate_support(row)[0],
            -llm_merge_candidate_support(row)[1],
            str(row.get("category") or "").casefold(),
            str(row.get("feature_name_1") or "").casefold(),
            str(row.get("feature_name_2") or "").casefold(),
        ),
    )[:LLM_MERGE_QUALITATIVE_TOP_ACCEPTED]
    excluded_for_random = {
        llm_merge_candidate_pair_id(candidate) for candidate in [*top_similar, *top_accepted]
    }
    random_examples = deterministic_llm_merge_candidate_sample(
        candidates,
        limit=LLM_MERGE_QUALITATIVE_RANDOM,
        excluded_pair_ids=excluded_for_random,
    )
    counts_text = (
        ", ".join(f"{decision}: {count}" for decision, count in sorted(decision_counts.items())) or "none"
    )
    lines = [
        "# LLM Top-K Merge Qualitative Examples",
        "",
        f"Generated: {utc_now()}",
        f"Input: `{input_file}`" if input_file is not None else "Input: unknown",
        (
            f"Features: {len(features)} | Candidate pairs: {len(candidates)} | "
            f"Decisions: {len(decisions)} | Accepted pairs: {accepted_pair_count} | "
            f"Merge groups: {merge_group_count}"
        ),
        f"Decision counts across candidate pairs: {counts_text}",
        "",
        (
            "This report samples the embedding-neighbor pairs reviewed by the LLM. "
            "The random section is deterministic, based on pair IDs, so reruns are comparable."
        ),
        "",
    ]
    append_llm_merge_report_section(
        lines,
        "Most Similar Candidate Pairs",
        f"Top {LLM_MERGE_QUALITATIVE_TOP_SIMILAR} pairs by embedding cosine similarity.",
        top_similar,
        decision_map=decision_map,
        rows_by_key=rows_by_key,
    )
    append_llm_merge_report_section(
        lines,
        "Accepted Merge Examples",
        f"Top {LLM_MERGE_QUALITATIVE_TOP_ACCEPTED} LLM-accepted merges by embedding similarity.",
        top_accepted,
        decision_map=decision_map,
        rows_by_key=rows_by_key,
    )
    append_llm_merge_report_section(
        lines,
        "Deterministic Random Candidate Pairs",
        f"{LLM_MERGE_QUALITATIVE_RANDOM} non-top candidate pairs for spot-checking.",
        random_examples,
        decision_map=decision_map,
        rows_by_key=rows_by_key,
    )
    return "\n".join(lines).rstrip() + "\n"


def write_llm_merge_artifacts(
    *,
    output_dir: Path,
    features: list[dict[str, Any]],
    items: list[FeatureItem],
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    input_file: Path | None = None,
) -> dict[str, Any]:
    item_index_by_feature_key = {item.feature_key: index for index, item in enumerate(items)}
    accepted_pairs = accepted_llm_merge_pairs(candidates, decisions, item_index_by_feature_key)
    groups = build_merge_groups(items, accepted_pairs, linkage="single") if accepted_pairs else []
    merged = build_merged_features(
        items,
        groups,
        threshold=0.0,
        scope="category_llm_top_k",
        all_rows=features,
    )
    merged = enrich_llm_merged_features(merged, groups, items, candidates, decisions)
    write_jsonl(output_dir / "llm_merge_candidates.jsonl", candidates)
    write_csv(output_dir / "llm_merge_candidates.csv", candidates, LLM_MERGE_CANDIDATE_FIELDNAMES)
    write_jsonl(output_dir / "llm_merge_decisions.jsonl", decisions)
    write_csv(output_dir / "llm_merge_decisions.csv", decisions, LLM_MERGE_DECISION_FIELDNAMES)
    write_jsonl(output_dir / "llm_merged_features.jsonl", merged)
    write_json(
        output_dir / "llm_merged_features.json",
        {
            "feature_count": len(merged),
            "merge_group_count": len(groups),
            "accepted_pair_count": len(accepted_pairs),
            "features": merged,
        },
    )
    write_csv(output_dir / "llm_merged_features.csv", merged, merged_fieldnames(merged))
    summary = category_merge_summary(items, groups, all_rows=features)
    write_json(output_dir / "llm_merge_summary_by_category.json", {"categories": summary})
    figure = write_category_retention_plot(output_dir / "llm_merge_retention_by_category.png", summary)
    (output_dir / "llm_merge_qualitative_examples.md").write_text(
        build_llm_merge_qualitative_markdown(
            input_file=input_file,
            features=features,
            candidates=candidates,
            decisions=decisions,
            accepted_pair_count=len(accepted_pairs),
            merge_group_count=len(groups),
        ),
        encoding="utf-8",
    )
    return {
        "llm_merge_candidates": len(candidates),
        "llm_merge_decisions": len(decisions),
        "llm_merged_features": len(merged),
        "accepted_pair_count": len(accepted_pairs),
        "merge_group_count": len(groups),
        "llm_merge_retention_by_category": int(figure is not None),
        "llm_merge_qualitative_examples": len(candidates),
        "category_merge_summary": summary,
    }
