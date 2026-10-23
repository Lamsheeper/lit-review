"""Media-attribute category aliases and relevance rules."""

import re
from typing import Any
from lit_review.core.taxonomy import normalize_family_key

CATEGORIES = {"persuasion", "moral_framing", "sentiment_affect"}


RELEVANCE_PATTERNS = [
    r"\bpersuasion\b",
    r"\bpersuasive\b",
    r"\bpropaganda\b",
    r"\bpropagandistic\b",
    r"\brhetoric(?:al)?\b",
    r"\bframing\b",
    r"\bmoral(?:ity)?\b",
    r"\bmoral foundations?\b",
    r"\bsentiment\b",
    r"\bemotion(?:al|s)?\b",
    r"\baffect(?:ive|s)?\b",
    r"\bideolog(?:y|ical)\b",
    r"\bmedia bias\b",
    r"\bdisinformation\b",
    r"\bmisinformation\b",
    r"\bnarrative(?:s)?\b",
    r"\bfallac(?:y|ies)\b",
    r"\bappeal(?:s)?\b",
    r"\bloaded language\b",
    r"\bname[- ]calling\b",
    r"\bwhataboutism\b",
    r"\bfear\b",
    r"\banger\b",
    r"\bdisgust\b",
    r"\bcare[/ -]?harm\b",
    r"\bfairness[/ -]?cheating\b",
    r"\bloyalty[/ -]?betrayal\b",
    r"\bauthority[/ -]?subversion\b",
    r"\bsanctity[/ -]?degradation\b",
    r"\bliberty[/ -]?oppression\b",
]


RELEVANCE_RE = re.compile("|".join(RELEVANCE_PATTERNS), flags=re.IGNORECASE)


def canonical_category(
    value: str | None,
    taxonomy_lookup: dict[str, str] | None = None,
) -> str | None:
    if not value:
        return None
    from lit_review.features.records import normalize_feature_name

    normalized = normalize_feature_name(value)
    if taxonomy_lookup is not None:
        return taxonomy_lookup.get(normalize_family_key(value))
    category_map = {
        "persuasion": "persuasion",
        "persuasion technique": "persuasion",
        "persuasion techniques": "persuasion",
        "propaganda": "persuasion",
        "propaganda technique": "persuasion",
        "propaganda techniques": "persuasion",
        "rhetorical": "persuasion",
        "rhetorical technique": "persuasion",
        "rhetorical techniques": "persuasion",
        "moral": "moral_framing",
        "morality": "moral_framing",
        "moral framing": "moral_framing",
        "moral frame": "moral_framing",
        "moral frames": "moral_framing",
        "mft": "moral_framing",
        "moral foundation": "moral_framing",
        "moral foundations": "moral_framing",
        "sentiment": "sentiment_affect",
        "affect": "sentiment_affect",
        "affective": "sentiment_affect",
        "emotional": "sentiment_affect",
        "emotion": "sentiment_affect",
        "emotions": "sentiment_affect",
        "emotion affect": "sentiment_affect",
        "emotional affective targeting": "sentiment_affect",
        "sentiment affect": "sentiment_affect",
        "sentiment and affect": "sentiment_affect",
    }
    return category_map.get(normalized)


def is_taxonomy_relevant(item: dict[str, Any], chunk_text: str, paper: dict[str, Any]) -> bool:
    from lit_review.features.records import coerce_str_list

    haystack = "\n".join(
        [
            str(chunk_text or ""),
            str(item.get("feature_name") or ""),
            str(item.get("parent_category") or ""),
            str(item.get("definition") or ""),
            " ".join(coerce_str_list(item.get("synonyms"))),
            " ".join(coerce_str_list(item.get("examples"))),
            str(item.get("source_quote") or item.get("quote") or ""),
            str(item.get("snippet") or ""),
            str(paper.get("title") or ""),
            str(paper.get("venue") or ""),
        ]
    )
    return bool(RELEVANCE_RE.search(haystack))
