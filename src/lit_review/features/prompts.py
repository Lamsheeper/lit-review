"""Features prompts."""

from __future__ import annotations

from lit_review.core.context import PROFILE, TAXONOMY
from lit_review.profiles.annotation_instructions import MEDIA_ANNOTATION_RULES, MEDIA_SYNONYM_RULES


import json


from typing import Any, Iterable


from lit_review.core.taxonomy import taxonomy_prompt_payload


from lit_review.features.records import (
    coerce_str_list,
    compact_feature_for_prompt,
    compact_list,
    feature_key,
    truncate_text,
)
from lit_review.core.text import normalize_feature_name


def paper_prompt_metadata(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "authors": paper.get("authors") or [],
        "year": paper.get("year"),
        "doi": paper.get("doi"),
        "venue": paper.get("venue"),
    }


def chunk_prompt_block(chunk: dict[str, Any], max_chunk_chars: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "pages": chunk.get("pages") or [],
        "text": truncate_text(str(chunk.get("text") or ""), max_chunk_chars),
    }


def build_extraction_prompt(
    paper: dict[str, Any],
    chunks: list[dict[str, Any]],
    max_chunk_chars: int,
    goal_text: str = "",
    max_features_per_chunk: int = 12,
    taxonomy: dict[str, Any] | None = None,
) -> str:
    taxonomy = taxonomy or TAXONOMY.get()
    if taxonomy is None:
        raise ValueError("An explicit taxonomy is required")
    prompt_payload = {
        "paper": paper_prompt_metadata(paper),
        "chunks": [chunk_prompt_block(chunk, max_chunk_chars) for chunk in chunks],
    }
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    category_shape = (
        "one allowed family id from ALLOWED FEATURE FAMILIES" if taxonomy else "one allowed family id"
    )
    taxonomy_section = (
        "\nALLOWED FEATURE FAMILIES\n"
        + json.dumps(taxonomy_prompt_payload(taxonomy), ensure_ascii=False, indent=2)
        + "\n"
        if taxonomy
        else ""
    )
    relevance_rules = (
        "- Extract only feature concepts that fit one of the allowed feature families and the goal context.\n"
        "- Set category to the exact allowed family id, not its label or alias."
        if taxonomy
        else "- Use only the supplied taxonomy and goal."
    )
    return f"""Extract taxonomy feature mentions from the provided literature chunks.
{goal_section}
{taxonomy_section}

Return strict JSON only with this shape:
{{
  "features": [
    {{
      "category": "{category_shape}",
      "feature_name": "named feature exactly as supported by the chunk",
      "parent_category": "optional parent/group label or empty string",
      "definition": "definition from the chunk, or empty string",
      "synonyms": ["optional related terms"],
      "examples": ["optional example phrases from the chunk"],
      "evidence_type": "definition | taxonomy_row | example | named_mention",
      "chunk_id": "chunk id from input",
      "source_quote": "short quote/paraphrase from the chunk; escape internal quotes",
      "confidence": 0.0,
      "notes": "brief ambiguity note if needed"
    }}
  ]
}}

Extraction rules:
{relevance_rules}
- Named mentions are valid, but use evidence_type="named_mention" when there is
  no definition, taxonomy row, or example.
- Extract only named concepts fitting the supplied taxonomy and goal; reject unrelated uses of the word feature.
- Do not infer feature names that are absent from the chunks.
- Use only the provided chunks and include the source chunk_id for every item.
- Prefer source_quote paraphrases without internal quotation marks when possible.
- Return at most {max_features_per_chunk} features per chunk. If a chunk contains
  more candidates, prioritize explicitly named taxonomy labels, definitions,
  annotation labels, and examples aligned with the goal context.
- If there are no supported features, return {{"features": []}}.

INPUT
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}
"""


def pdf_chunk_id(paper: dict[str, Any]) -> str:
    return f"{paper.get('paper_id') or 'paper'}:pdf"


def build_pdf_extraction_prompt(
    paper: dict[str, Any],
    goal_text: str = "",
    max_features_per_pdf: int = 40,
    taxonomy: dict[str, Any] | None = None,
) -> str:
    taxonomy = taxonomy or TAXONOMY.get()
    if taxonomy is None:
        raise ValueError("An explicit taxonomy is required")
    chunk_id = pdf_chunk_id(paper)
    prompt_payload = {
        "paper": paper_prompt_metadata(paper),
        "pdf_input": {
            "chunk_id": chunk_id,
            "pages": "Use 1-based PDF page numbers when evidence is page-local.",
        },
    }
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    category_shape = (
        "one allowed family id from ALLOWED FEATURE FAMILIES" if taxonomy else "one allowed family id"
    )
    taxonomy_section = (
        "\nALLOWED FEATURE FAMILIES\n"
        + json.dumps(taxonomy_prompt_payload(taxonomy), ensure_ascii=False, indent=2)
        + "\n"
        if taxonomy
        else ""
    )
    relevance_rules = (
        "- Extract only feature concepts that fit one of the allowed feature families and the goal context.\n"
        "- Set category to the exact allowed family id, not its label or alias."
        if taxonomy
        else "- Use only the supplied taxonomy and goal."
    )
    return f"""Extract taxonomy feature mentions from the attached academic PDF.
{goal_section}
{taxonomy_section}

Return strict JSON only with this shape:
{{
  "features": [
    {{
      "category": "{category_shape}",
      "feature_name": "named feature exactly as supported by the PDF",
      "parent_category": "optional parent/group label or empty string",
      "definition": "definition from the PDF, or empty string",
      "synonyms": ["optional related terms"],
      "examples": ["optional example phrases from the PDF"],
      "evidence_type": "definition | taxonomy_row | example | named_mention",
      "chunk_id": "{chunk_id}",
      "pages": [1],
      "source_quote": "short quote/paraphrase from the PDF; escape internal quotes",
      "snippet": "short local evidence snippet if helpful",
      "confidence": 0.0,
      "notes": "brief ambiguity note if needed"
    }}
  ]
}}

Extraction rules:
{relevance_rules}
- Named mentions are valid, but use evidence_type="named_mention" when there is
  no definition, taxonomy row, or example.
- Extract only named concepts fitting the supplied taxonomy and goal; reject unrelated uses of the word feature.
- Do not infer feature names that are absent from the PDF.
- Use only the attached PDF and set chunk_id="{chunk_id}" for every item.
- Include 1-based page numbers when the PDF provides enough evidence to locate
  the mention.
- Prefer source_quote paraphrases without internal quotation marks when possible.
- Return at most {max_features_per_pdf} features from this PDF. If the PDF
  contains more candidates, prioritize explicitly named taxonomy labels,
  definitions, annotation labels, and examples aligned with the goal context.
- If there are no supported features, return {{"features": []}}.

INPUT
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}
"""


def build_curation_prompt(
    *,
    features: list[dict[str, Any]],
    category: str,
    goal_text: str,
) -> str:
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    payload = {
        "category": category,
        "features": [compact_feature_for_prompt(row) for row in features],
    }
    return f"""Filter extracted taxonomy feature candidates.
{goal_section}

Return strict JSON only with this shape:
{{
  "decisions": [
    {{
      "feature_key": "category::normalized feature name from input",
      "action": "keep | reject",
      "rejection_reason": "reason if rejected, otherwise empty",
      "curation_confidence": 0.0,
      "notes": "short explanation"
    }}
  ]
}}

Filtering rules:
- Work only within the input category: {category}.
- This pass is filter-only. Do not merge, rename, canonicalize, or cap features.
- Keep every useful feature under its original feature_key, feature_name, and
  normalized_feature_name even when it seems duplicative of another row.
- Reject features that are off-goal, overly broad, non-taxonomic, or artifacts of
  extraction noise.
- Keep useful features even when their evidence is imperfect.
- Every input feature_key must appear exactly once in decisions.

INPUT
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def compact_annotation_feature_for_prompt(row: dict[str, Any], rank: int) -> dict[str, Any]:
    payload = compact_feature_for_prompt(row)
    payload["rank"] = rank
    payload["source_feature_names"] = compact_list(row.get("source_feature_names"), 16)
    payload["source_normalized_feature_names"] = compact_list(
        row.get("source_normalized_feature_names"),
        16,
    )
    payload["paper_count"] = annotation_int(row.get("paper_count"))
    payload["mention_count"] = annotation_int(row.get("mention_count"))
    payload["curation_confidence"] = row.get("curation_confidence")
    payload["curation_notes"] = truncate_text(str(row.get("curation_notes") or ""), 350)
    return payload


def build_annotation_prompt(
    *,
    features: list[dict[str, Any]],
    goal_text: str,
) -> str:
    domain_rules = (
        MEDIA_ANNOTATION_RULES
        if PROFILE.get() == "media_attributes"
        else "- Apply the supplied taxonomy and goal criteria; retain concrete, evidence-backed features."
    )
    domain_synonym_rules = MEDIA_SYNONYM_RULES if PROFILE.get() == "media_attributes" else ""
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    payload = {
        "annotation_mode": "hybrid_evidence_grounded",
        "features": [
            compact_annotation_feature_for_prompt(row, rank=idx) for idx, row in enumerate(features, start=1)
        ],
    }
    return f"""Write polished feature annotations for a taxonomy inventory.
{goal_section}

Return strict JSON only with this shape:
{{
  "annotations": [
    {{
      "feature_key": "category::normalized feature name from input",
      "status": "annotated | needs_review | excluded",
      "formal_definition": "one formal, publication-ready definition",
      "short_definition": "brief glossary definition",
      "synonyms": ["zero or more close, substitutable aliases; [] is valid"],
      "examples": ["brief representative examples"],
      "paper_count": 0,
      "mention_count": 0,
      "confidence": 0.0,
      "definition_basis": "evidence_synthesized | evidence_limited | model_suggested | needs_review | excluded_descriptor",
      "notes": "short caveat, evidence limit, or empty string"
    }}
  ]
}}

Annotation rules:
- Every input feature_key must appear exactly once in annotations.
- Preserve the input category and feature identity; do not merge or rename features.
- Ground definitions in the supplied feature evidence: names, categories, existing definitions,
  examples, synonyms, source feature names, source titles, paper_count, mention_count, and goal context.
- Hybrid mode allows concise model-polished wording and limited standard taxonomy phrasing,
  but unsupported additions must be reflected with definition_basis="model_suggested".
- Crack down on candidates that are not concrete, category-fitting annotatable features.
  If the feature name and generated definition show that the item is an umbrella construct,
  measurement dimension, generic descriptor, analytic task, parent category, or category label
  rather than a specific feature, set status="excluded" and
  definition_basis="excluded_descriptor".
- Exclude feature names that enumerate multiple distinct labels in one row, such as
  comma- or semicolon-separated bundles of techniques, frames, emotions, or examples.
  Do not split them during annotation; mark the bundled row excluded so it stays audited.
{domain_rules}
- For excluded features, keep a concise formal_definition if it helps audit the decision,
  leave synonyms/examples empty unless directly evidenced, use confidence for the exclusion
  decision, and explain the reason in notes.
- Generate zero or more synonyms for each feature. Return [] when there are no truly close
  aliases.
- Synonyms must be definitionally close and substitutable for the original feature label in
  taxonomy use.
{domain_synonym_rules}
- Do not include parent categories, sibling feature labels, examples, explanations, domains,
  broad category names, or loosely related concepts as synonyms.
- If the supplied evidence is too thin or internally conflicting, set status="needs_review",
  use low confidence, and explain the limitation in notes instead of inventing a definition.
- Keep formal_definition to one sentence and short_definition to roughly 25 words or fewer.
- Return paper_count and mention_count from the input feature.

INPUT
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def annotation_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def row_path_value(row: dict[str, Any], field: str) -> Any:
    if field in row:
        return row.get(field)
    if "." not in field:
        return None
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value.get(part)
    return value


def first_row_path_value(row: dict[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        value = row_path_value(row, field)
        if value not in (None, ""):
            return value
    return None


def scoring_paper_count(row: dict[str, Any]) -> int:
    return annotation_int(
        first_row_path_value(
            row,
            ("paper_count", "annotation.paper_count", "annotation_paper_count"),
        )
    )


def scoring_mention_count(row: dict[str, Any]) -> int:
    return annotation_int(
        first_row_path_value(
            row,
            ("mention_count", "annotation.mention_count", "annotation_mention_count"),
        )
    )


def literature_support_score(paper_count: Any) -> int:
    count = annotation_int(paper_count)
    if count >= 15:
        return 5
    if count >= 10:
        return 4
    if count >= 6:
        return 3
    if count >= 3:
        return 2
    if count >= 2:
        return 1
    return 0


def scoring_feature_tokens(row: dict[str, Any]) -> set[str]:
    values: list[Any] = [
        row.get("feature_name"),
        row.get("normalized_feature_name"),
        row.get("source_feature_names"),
        row.get("source_normalized_feature_names"),
        row.get("synonyms"),
        row_path_value(row, "annotation.synonyms"),
    ]
    tokens: set[str] = set()
    for value in values:
        for item in coerce_str_list(value):
            tokens.update(normalize_feature_name(item).split())
    return {token for token in tokens if len(token) > 2}


def normalized_feature_label(row: dict[str, Any]) -> str:
    return normalize_feature_name(str(row.get("normalized_feature_name") or row.get("feature_name") or ""))


def token_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def distinctiveness_neighbors(
    feature: dict[str, Any],
    all_features: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    key = str(feature.get("feature_key") or feature_key(feature))
    category = str(feature.get("category") or "")
    label = normalized_feature_label(feature)
    tokens = scoring_feature_tokens(feature)
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for row in all_features:
        row_key = str(row.get("feature_key") or feature_key(row))
        if row_key == key or str(row.get("category") or "") != category:
            continue
        row_label = normalized_feature_label(row)
        row_tokens = scoring_feature_tokens(row)
        overlap = token_overlap_score(tokens, row_tokens)
        contains = bool(
            label and row_label and (label in row_label or row_label in label) and label != row_label
        )
        if overlap < 0.5 and not contains:
            continue
        score = overlap + (0.25 if contains else 0.0)
        candidates.append((score, scoring_paper_count(row), row))
    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            str(item[2].get("feature_name") or "").casefold(),
        )
    )
    return [
        {
            "feature_key": row.get("feature_key") or feature_key(row),
            "category": row.get("category"),
            "feature_name": row.get("feature_name"),
            "paper_count": scoring_paper_count(row),
            "short_definition": truncate_text(
                str(
                    row_path_value(row, "annotation.short_definition")
                    or row_path_value(row, "annotation_short_definition")
                    or ""
                ),
                160,
            ),
            "overlap_score": round(score, 3),
        }
        for score, _, row in candidates[:limit]
    ]


def compact_embedding_merge_context(row: dict[str, Any]) -> dict[str, Any]:
    merge = row.get("embedding_merge")
    if not isinstance(merge, dict):
        return {}
    pairs = merge.get("pairs") if isinstance(merge.get("pairs"), list) else []
    return {
        "group_id": merge.get("group_id") or "",
        "canonical_feature_key": merge.get("canonical_feature_key") or "",
        "canonical_feature_name": merge.get("canonical_feature_name") or "",
        "member_count": annotation_int(merge.get("member_count")),
        "member_feature_keys": compact_list(merge.get("member_feature_keys"), 12),
        "member_feature_names": compact_list(merge.get("member_feature_names"), 12),
        "member_categories": compact_list(merge.get("member_categories"), 8),
        "similarity_summary": merge.get("similarity_summary")
        if isinstance(merge.get("similarity_summary"), dict)
        else {},
        "pairs": [
            {
                "feature_name_1": pair.get("feature_name_1"),
                "feature_name_2": pair.get("feature_name_2"),
                "similarity": pair.get("similarity"),
            }
            for pair in pairs[:6]
            if isinstance(pair, dict)
        ],
    }


def compact_scoring_feature_for_prompt(
    row: dict[str, Any],
    *,
    rank: int,
    all_features: list[dict[str, Any]],
) -> dict[str, Any]:
    annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
    paper_count = scoring_paper_count(row)
    return {
        "rank": rank,
        "feature_key": row.get("feature_key") or feature_key(row),
        "category": row.get("category"),
        "feature_name": row.get("feature_name"),
        "normalized_feature_name": row.get("normalized_feature_name"),
        "paper_count": paper_count,
        "mention_count": scoring_mention_count(row),
        "literature_support_score": literature_support_score(paper_count),
        "parent_categories": compact_list(row.get("parent_categories"), 8),
        "definitions": compact_list(row.get("definitions"), 5, max_chars=240),
        "synonyms": compact_list(row.get("synonyms"), 12),
        "examples": compact_list(row.get("examples"), 8),
        "source_feature_names": compact_list(row.get("source_feature_names"), 16),
        "source_normalized_feature_names": compact_list(
            row.get("source_normalized_feature_names"),
            16,
        ),
        "source_titles": compact_list(row.get("source_titles"), 8, max_chars=160),
        "annotation": {
            "status": annotation.get("status") or row.get("annotation_status") or "",
            "formal_definition": truncate_text(
                str(annotation.get("formal_definition") or row.get("annotation_formal_definition") or ""),
                350,
            ),
            "short_definition": truncate_text(
                str(annotation.get("short_definition") or row.get("annotation_short_definition") or ""),
                180,
            ),
            "synonyms": compact_list(annotation.get("synonyms") or row.get("annotation_synonyms"), 12),
            "examples": compact_list(annotation.get("examples") or row.get("annotation_examples"), 8),
            "definition_basis": annotation.get("definition_basis")
            or row.get("annotation_definition_basis")
            or "",
            "notes": truncate_text(str(annotation.get("notes") or row.get("annotation_notes") or ""), 240),
        },
        "embedding_merge": compact_embedding_merge_context(row),
        "distinctiveness_neighbors": distinctiveness_neighbors(row, all_features),
    }


def build_scoring_prompt(
    *,
    features: list[dict[str, Any]],
    all_features: list[dict[str, Any]],
    goal_text: str,
) -> str:
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    payload = {
        "scoring_mode": "feature_quality_scores_v1",
        "features": [
            compact_scoring_feature_for_prompt(row, rank=idx, all_features=all_features)
            for idx, row in enumerate(features, start=1)
        ],
    }
    return f"""Score taxonomy features for downstream media annotation.
{goal_section}

Return strict JSON only with this shape:
{{
  "scores": [
    {{
      "feature_key": "category::normalized feature name from input",
      "annotation_reliability": 0,
      "annotation_reliability_rationale": "short reason",
      "distinctiveness": 0,
      "distinctiveness_rationale": "short reason",
      "confidence": 0.0,
      "notes": "short caveat or empty string"
    }}
  ]
}}

Scoring rules:
- Every input feature_key must appear exactly once in scores.
- Return integer scores from 0 to 5 only.
- Do not score literature support and do not compute total_score. The pipeline
  computes literature_support deterministically from paper_count and ignores any
  model-provided literature or total fields.
- Annotation Reliability asks whether a trained human or LLM could reliably and
  consistently identify the feature in media, and whether two annotators would
  likely agree that it is present.
  5 = concrete, observable, clear boundaries, strong examples; 3 = usable but
  somewhat contextual or fuzzy; 0 = not reliably annotatable.
- Distinctiveness asks whether this feature contributes unique information
  relative to the current inventory. If removed, would another surviving feature
  already capture most of its meaning?
  5 = clearly unique; 3 = partially overlapping but still useful; 0 = duplicate
  or fully redundant.
- Use distinctiveness_neighbors and embedding_merge metadata as evidence about
  overlap. Do not penalize a canonical merged row merely because it absorbed
  duplicate labels; penalize only when another surviving feature in the current
  input still captures most of the same meaning.
- Score every row, including rows whose annotation status is needs_review or
  excluded. Excluded or very broad rows should usually receive low annotation
  reliability and/or low distinctiveness.
- Keep rationales concise and grounded in the supplied names, definitions,
  annotations, examples, counts, neighbors, and goal context.

INPUT
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def compact_llm_merge_feature_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
    return {
        "feature_key": row.get("feature_key") or feature_key(row),
        "category": row.get("category"),
        "feature_name": row.get("feature_name"),
        "normalized_feature_name": row.get("normalized_feature_name"),
        "paper_count": scoring_paper_count(row),
        "mention_count": scoring_mention_count(row),
        "parent_categories": compact_list(row.get("parent_categories"), 6),
        "definitions": compact_list(row.get("definitions"), 3, max_chars=220),
        "synonyms": compact_list(row.get("synonyms"), 10),
        "examples": compact_list(row.get("examples"), 6),
        "source_feature_names": compact_list(row.get("source_feature_names"), 12),
        "annotation": {
            "status": annotation.get("status") or row.get("annotation_status") or "",
            "formal_definition": truncate_text(
                str(annotation.get("formal_definition") or row.get("annotation_formal_definition") or ""),
                320,
            ),
            "short_definition": truncate_text(
                str(annotation.get("short_definition") or row.get("annotation_short_definition") or ""),
                180,
            ),
            "synonyms": compact_list(annotation.get("synonyms") or row.get("annotation_synonyms"), 10),
            "examples": compact_list(annotation.get("examples") or row.get("annotation_examples"), 6),
            "notes": truncate_text(str(annotation.get("notes") or row.get("annotation_notes") or ""), 220),
        },
    }


def build_llm_merge_prompt(
    *,
    candidates: list[dict[str, Any]],
    rows_by_key: dict[str, dict[str, Any]],
    goal_text: str,
) -> str:
    goal_section = (
        f"\nGOAL / TAXONOMY CONTEXT\n{goal_text}\n"
        if goal_text.strip()
        else "\nGOAL / TAXONOMY CONTEXT\nNo external goal file was provided.\n"
    )
    payload = {
        "merge_mode": "embedding_top_k_llm_review",
        "candidate_pairs": [
            {
                "pair_id": row.get("pair_id"),
                "category": row.get("category"),
                "similarity": row.get("similarity"),
                "feature_1": compact_llm_merge_feature_for_prompt(
                    rows_by_key[str(row.get("feature_key_1") or "")]
                ),
                "feature_2": compact_llm_merge_feature_for_prompt(
                    rows_by_key[str(row.get("feature_key_2") or "")]
                ),
            }
            for row in candidates
            if str(row.get("feature_key_1") or "") in rows_by_key
            and str(row.get("feature_key_2") or "") in rows_by_key
        ],
    }
    return f"""Decide whether embedding-neighbor taxonomy feature pairs should merge.
{goal_section}

Return strict JSON only with this shape:
{{
  "decisions": [
    {{
      "pair_id": "merge_pair_000001",
      "decision": "merge | separate | uncertain",
      "confidence": 0.0,
      "rationale": "short explanation"
    }}
  ]
}}

Merge rules:
- Each pair is within one input category and was shortlisted by embedding similarity.
- Choose "merge" only when both rows are the same conceptual taxonomy feature,
  including clear aliases, spelling variants, plural/singular variants, or labels
  whose definitions and examples describe the same annotatable construct.
- Choose "separate" when the pair overlaps but has distinct taxonomy roles,
  different scopes, parent/child relation, sibling relation, or different
  annotatable meanings.
- Choose "uncertain" when the evidence is too thin or conflicting.
- Do not invent new feature names. This pass only accepts or rejects candidate
  pair merges; canonical names are chosen deterministically later.
- Every input pair_id must appear exactly once in decisions.

INPUT
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
