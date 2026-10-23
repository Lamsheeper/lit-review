"""Profiles media_ranking."""

from __future__ import annotations


import re


from lit_review.core.text import normalize_feature_name


TITLE_RANK_RULES = [
    (r"\btaxonom(?:y|ies)\b", 14),
    (r"\bclassification\b", 5),
    (r"\bannotation\b", 6),
    (r"\bdataset\b", 5),
    (r"\bbenchmark\b", 5),
    (r"\bpropaganda techniques?\b", 10),
    (r"\bpersua\w* techniques?\b", 9),
    (r"\bpropaganda\b", 16),
    (r"\bpersua\w*\b", 15),
    (r"\brhetoric\w*\b", 13),
    (r"\bfallac(?:y|ies)\b", 12),
    (r"\bframing\b", 7),
    (r"\bframe(?:s|d)?\b", 3),
    (r"\bmoral(?:ity)?\b", 7),
    (r"\bmoral foundations?\b", 9),
    (r"\bsentiment\b", 9),
    (r"\bemotion\w*\b", 9),
    (r"\baffect(?:ive|s)?\b", 8),
    (r"\bmedia bias\b", 11),
    (r"\bmedia analys(?:is|es)\b", 8),
    (r"\bnews media\b", 7),
    (r"\bjournalis\w*\b", 6),
    (r"\bdisinformation\b", 10),
    (r"\bmisinformation\b", 10),
    (r"\bnarrative(?:s)?\b", 6),
    (r"\bideolog(?:y|ical)\b", 5),
    (r"\bpolitical communication\b", 7),
    (r"\bdiscourse\b", 4),
    (r"\bstance\b", 4),
]


TITLE_PENALTY_RULES = [
    (r"\bpoint cloud\b", 8),
    (r"\bcompression\b", 5),
    (r"\bsensor\b", 4),
    (r"\bsemiconductor\b", 5),
    (r"\bcell atlas\b", 5),
    (r"\bclinical\b", 3),
]


def title_relevance_score(title: str | None) -> int:
    normalized = normalize_feature_name(title)
    if not normalized:
        return 0
    score = 0
    for pattern, weight in TITLE_RANK_RULES:
        if re.search(pattern, normalized):
            score += weight
    for pattern, penalty in TITLE_PENALTY_RULES:
        if re.search(pattern, normalized):
            score -= penalty
    return max(0, score)
