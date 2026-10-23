"""Profiles media_attributes."""

from __future__ import annotations


import re


from typing import Any


from lit_review.features.records import coerce_str_list
from lit_review.core.text import normalize_feature_name


ANNOTATION_GENERIC_DESCRIPTOR_NAMES = {
    "*": {
        "affect",
        "category",
        "descriptor",
        "emotion",
        "emotionality",
        "feature",
        "frame",
        "framing",
        "label",
        "media analysis",
        "media attribute",
        "media framing",
        "morality",
        "polarity",
        "propaganda",
        "rhetoric",
        "rhetorical device",
        "sentiment",
        "sentiment analysis",
        "sentiment intensity",
        "sentiment polarity",
        "sentiment score",
        "subjectivity",
        "taxonomy",
        "technique",
        "valence",
    },
    "persuasion": {
        "argumentation",
        "communication strategy",
        "discourse",
        "frame",
        "framing",
        "media framing",
        "narrative",
        "persuasion",
        "persuasion technique",
        "persuasive communication",
        "propaganda",
        "propaganda technique",
        "rhetoric",
        "rhetorical strategy",
    },
    "moral_framing": {
        "basic human values",
        "cad triad",
        "mft",
        "mfq",
        "moral foundation",
        "moral foundations",
        "moral foundations questionnaire",
        "moral foundations theory",
        "moral frame",
        "moral framing",
        "moral language",
        "moral value",
        "morality",
        "morality as cooperation",
        "schwartz basic human values",
    },
    "sentiment_affect": {
        "affect",
        "activation",
        "arousal",
        "affective dimension",
        "affective intensity",
        "affective polarity",
        "affective tone",
        "affective valence",
        "attitude",
        "emotion",
        "emotion intensity",
        "emotion polarity",
        "emotional intensity",
        "emotional tone",
        "emotional valence",
        "mood",
        "negative affect",
        "negative emotion",
        "opinion",
        "polarity",
        "positive affect",
        "positive emotion",
        "sentiment",
        "sentiment intensity",
        "sentiment orientation",
        "sentiment polarity",
        "sentiment valence",
        "stance",
        "subjectivity",
        "tone",
        "toxicity",
        "valence",
    },
}


MORAL_FRAME_ALIAS_GROUPS = (
    (
        "Care/Harm",
        "Harm/Care",
        "Care foundation",
    ),
    (
        "Fairness/Cheating",
        "Fairness/Reciprocity",
        "Justice/Fairness",
        "Fairness foundation",
    ),
    (
        "Loyalty/Betrayal",
        "Loyalty/Ingroup",
        "Ingroup/Loyalty",
        "Loyalty foundation",
    ),
    (
        "Authority/Subversion",
        "Authority/Respect",
        "Respect/Authority",
        "Authority foundation",
    ),
    (
        "Sanctity/Degradation",
        "Sanctity/Purity",
        "Purity/Sanctity",
        "Purity foundation",
    ),
    (
        "Liberty/Oppression",
        "Liberty foundation",
    ),
)


MORAL_FRAME_CANONICAL_NAMES = {
    normalize_feature_name(alias) for group in MORAL_FRAME_ALIAS_GROUPS for alias in group
}


MORAL_FRAME_FRAMEWORK_NAMES = {
    "basic human values",
    "cad triad",
    "mft",
    "mfq",
    "moral foundations questionnaire",
    "moral foundations theory",
    "morality as cooperation",
    "schwartz basic human values",
}


ANNOTATION_CONCRETE_ALLOWLIST_NAMES = {
    "appeal to fear",
    "anger",
    "authority subversion",
    "care harm",
    "disgust",
    "fairness cheating",
    "fear",
    "joy",
    "liberty oppression",
    "loaded language",
    "loyalty betrayal",
    "name calling",
    "sadness",
    "sanctity degradation",
} | MORAL_FRAME_CANONICAL_NAMES


SENTIMENT_SPECIFIC_EMOTION_NAMES = {
    "admiration",
    "affection",
    "amusement",
    "anger",
    "annoyance",
    "anticipation",
    "anxiety",
    "awe",
    "boredom",
    "compassion",
    "concern",
    "contempt",
    "contentment",
    "confusion",
    "delight",
    "despair",
    "disappointment",
    "doubt",
    "disgust",
    "embarrassment",
    "empathy",
    "enthusiasm",
    "envy",
    "excitement",
    "fear",
    "frustration",
    "gratitude",
    "grief",
    "guilt",
    "happiness",
    "hate",
    "hatred",
    "hope",
    "humiliation",
    "interest",
    "irritation",
    "jealousy",
    "joy",
    "loneliness",
    "love",
    "moral outrage",
    "moral disgust",
    "nostalgia",
    "outrage",
    "panic",
    "pity",
    "pleasure",
    "pride",
    "regret",
    "relief",
    "remorse",
    "resentment",
    "sadness",
    "satisfaction",
    "shame",
    "sorrow",
    "sympathy",
    "surprise",
    "trust",
    "uncertainty",
    "worry",
}


SENTIMENT_NON_EMOTION_DESCRIPTOR_TOKENS = {
    "analysis",
    "arousal",
    "attitude",
    "activation",
    "classification",
    "dimension",
    "intensity",
    "orientation",
    "polarity",
    "score",
    "stance",
    "subjectivity",
    "tone",
    "toxicity",
    "valence",
}


def sentiment_specific_emotion_reason(
    feature: dict[str, Any],
    formal_definition: str,
    short_definition: str,
) -> str:
    if str(feature.get("category") or "") != "sentiment_affect":
        return ""
    name = normalize_feature_name(str(feature.get("feature_name") or ""))
    normalized_name = normalize_feature_name(str(feature.get("normalized_feature_name") or ""))
    candidates = {name, normalized_name}
    candidates.discard("")
    if not candidates:
        return "Sentiment/affect item is not named as a specific emotion someone can feel."

    if candidates & SENTIMENT_SPECIFIC_EMOTION_NAMES:
        return ""

    for candidate in candidates:
        if candidate in ANNOTATION_GENERIC_DESCRIPTOR_NAMES["sentiment_affect"]:
            return f"Sentiment/affect item is a descriptor or dimension, not a specific emotion: {candidate}."
        for token in SENTIMENT_NON_EMOTION_DESCRIPTOR_TOKENS:
            if re.search(rf"\b{re.escape(token)}\b", candidate):
                return (
                    f"Sentiment/affect item is a {token} descriptor, not a specific emotion someone can feel."
                )

    haystack = normalize_feature_name(
        " ".join(
            [
                name,
                formal_definition,
                short_definition,
            ]
        )
    )
    if re.search(
        r"\b(positive emotion|negative emotion|positive affect|negative affect|broad emotion|generic emotion|overall affect|overall emotion)\b",
        haystack,
    ):
        return "Sentiment/affect item is an emotion bucket or broad affect label, not a specific emotion."
    for emotion in SENTIMENT_SPECIFIC_EMOTION_NAMES:
        if re.search(rf"\b{re.escape(emotion)}\b", haystack) and re.search(
            r"\b(affective state|emotion|feeling|felt|feel|feels)\b",
            haystack,
        ):
            return ""
    return "Sentiment/affect item is not a specific emotion or affective state someone can feel."


def is_canonical_moral_frame_feature(feature: dict[str, Any]) -> bool:
    if str(feature.get("category") or "") != "moral_framing":
        return False
    candidates = {
        normalize_feature_name(str(feature.get("feature_name") or "")),
        normalize_feature_name(str(feature.get("normalized_feature_name") or "")),
    }
    candidates.discard("")
    return bool(candidates & MORAL_FRAME_CANONICAL_NAMES)


def moral_frame_aliases_for_feature(feature: dict[str, Any]) -> list[str]:
    if str(feature.get("category") or "") != "moral_framing":
        return []
    candidates = {
        normalize_feature_name(str(feature.get("feature_name") or "")),
        normalize_feature_name(str(feature.get("normalized_feature_name") or "")),
    }
    candidates.discard("")
    if not candidates:
        return []
    for aliases in MORAL_FRAME_ALIAS_GROUPS:
        normalized_aliases = {normalize_feature_name(alias) for alias in aliases}
        if candidates & normalized_aliases:
            return [alias for alias in aliases if normalize_feature_name(alias) not in candidates]
    return []


def annotation_descriptor_name_reason(feature: dict[str, Any]) -> str:
    name = normalize_feature_name(str(feature.get("feature_name") or ""))
    normalized_name = normalize_feature_name(str(feature.get("normalized_feature_name") or ""))
    candidates = {name, normalized_name}
    candidates.discard("")
    if not candidates:
        return ""
    category = str(feature.get("category") or "")
    if category == "moral_framing":
        matched_frameworks = sorted(candidates & MORAL_FRAME_FRAMEWORK_NAMES)
        if matched_frameworks:
            return (
                "Feature name is a moral framework or instrument, not a high-level "
                f"moral frame: {matched_frameworks[0]}."
            )
    if candidates & ANNOTATION_CONCRETE_ALLOWLIST_NAMES:
        return ""
    blocked = set(ANNOTATION_GENERIC_DESCRIPTOR_NAMES["*"])
    blocked.update(ANNOTATION_GENERIC_DESCRIPTOR_NAMES.get(category, set()))
    matched = sorted(candidates & blocked)
    if matched:
        return f"Feature name is a broad descriptor or category label: {matched[0]}."
    return ""


def annotation_compound_feature_name_reason(feature: dict[str, Any]) -> str:
    raw_name = re.sub(r"\s+", " ", str(feature.get("feature_name") or "")).strip()
    if not raw_name:
        return ""
    normalized_name = normalize_feature_name(raw_name)
    if normalized_name in ANNOTATION_CONCRETE_ALLOWLIST_NAMES:
        return ""
    if "," not in raw_name and ";" not in raw_name:
        return ""

    parts: list[str] = []
    for raw_part in re.split(r"[,;]", raw_name):
        cleaned = re.sub(r"^\s*(and|or)\s+", "", raw_part, flags=re.IGNORECASE).strip()
        normalized = normalize_feature_name(cleaned)
        if normalized:
            parts.append(normalized)
    unique_parts = {part for part in parts if part != normalized_name}
    if len(unique_parts) < 2:
        return ""
    return "Feature name appears to bundle multiple distinct feature labels in one row."


def annotation_descriptor_definition_reason(
    feature: dict[str, Any],
    formal_definition: str,
    short_definition: str,
) -> str:
    haystack = normalize_feature_name(
        " ".join(
            [
                str(feature.get("feature_name") or ""),
                formal_definition,
                short_definition,
            ]
        )
    )
    if not haystack:
        return ""

    category = str(feature.get("category") or "")
    if category == "sentiment_affect" and re.search(
        r"\b(positive negative|negative positive|positive neutral negative|negative neutral positive|activation level|polarity|valence|sentiment score|sentiment orientation|overall sentiment)\b",
        haystack,
    ):
        return "Sentiment item is a polarity/valence dimension or overall descriptor, not a concrete affect feature."
    if category == "persuasion" and re.search(
        r"\b(broad framing|general framing|act of framing|process of framing|selecting aspects|emphasizing aspects|overall frame|umbrella)\b",
        haystack,
    ):
        return (
            "Persuasion item is a broad framing construct or umbrella descriptor, not a concrete technique."
        )
    if category == "moral_framing" and not is_canonical_moral_frame_feature(feature):
        if re.search(
            r"\b(moral foundations theory|moral foundations questionnaire|basic human values|schwartz basic human values|cad triad|morality as cooperation|mft|mfq)\b",
            haystack,
        ):
            return "Moral item is a theoretical framework or instrument, not a high-level moral frame."
        if re.search(
            r"\b(moral framing|moral foundation|moral foundations|moral values|umbrella|broad category)\b",
            haystack,
        ):
            return "Moral item is an umbrella moral-framing construct, not a concrete moral frame."
    if re.search(
        r"\b(analytic task|analysis task|broad category|broad construct|category label|general category|generic descriptor|measurement dimension|parent category|umbrella category|umbrella construct)\b",
        haystack,
    ):
        return "Generated definition describes an umbrella construct, descriptor, task, or parent category."
    return ""


def annotation_hard_exclusion_reason(
    feature: dict[str, Any],
    formal_definition: str,
    short_definition: str,
) -> str:
    return (
        annotation_compound_feature_name_reason(feature)
        or annotation_descriptor_name_reason(feature)
        or annotation_descriptor_definition_reason(feature, formal_definition, short_definition)
    )


def annotation_exclusion_reason(
    feature: dict[str, Any],
    formal_definition: str,
    short_definition: str,
) -> str:
    return annotation_hard_exclusion_reason(
        feature, formal_definition, short_definition
    ) or sentiment_specific_emotion_reason(feature, formal_definition, short_definition)


ANNOTATION_BROAD_SYNONYM_NORMALIZED = {
    "affect",
    "affective",
    "annotation label",
    "attribute",
    "category",
    "emotion",
    "emotional",
    "feature",
    "label",
    "media analysis",
    "media attribute",
    "media framing",
    "moral frame",
    "moral framing",
    "persuasion",
    "persuasion technique",
    "propaganda",
    "propaganda technique",
    "rhetoric",
    "rhetorical technique",
    "sentiment",
    "sentiment affect",
    "taxonomy label",
    "technique",
}


def is_definition_like_annotation_synonym(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return True
    words = text.split()
    if len(text) > 80 or len(words) > 8:
        return True
    if re.search(r"[.!?;:]", text):
        return True

    normalized = text.casefold()
    if len(words) > 3 and normalized.startswith(("a ", "an ", "the ")):
        return True
    if len(words) > 3 and re.search(
        r"\b(are|captures|describes|indicates|involves|is|means|refers to|that|used to|uses|using|when|where)\b",
        normalized,
    ):
        return True
    return False


def annotation_synonym_blocked_norms(feature: dict[str, Any]) -> set[str]:
    category_labels = {
        "persuasion": ["persuasion", "persuasion technique", "propaganda technique"],
        "moral_framing": ["moral framing", "moral frame"],
        "sentiment_affect": ["sentiment", "affect", "sentiment affect", "emotion"],
    }
    blocked = set(ANNOTATION_BROAD_SYNONYM_NORMALIZED)
    blocked.add(normalize_feature_name(str(feature.get("feature_name") or "")))
    blocked.add(normalize_feature_name(str(feature.get("normalized_feature_name") or "")))
    category = str(feature.get("category") or "")
    blocked.add(normalize_feature_name(category))
    for label in category_labels.get(category, []):
        blocked.add(normalize_feature_name(label))
    for field in ("parent_category", "parent_categories", "examples", "source_titles"):
        for value in coerce_str_list(feature.get(field)):
            blocked.add(normalize_feature_name(value))
    blocked.discard("")
    return blocked


def add_moral_frame_alias_synonyms(
    synonyms: list[str],
    feature: dict[str, Any],
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    blocked = annotation_synonym_blocked_norms(feature)

    def add(value: str) -> None:
        text = re.sub(r"\s+", " ", str(value or "")).strip().strip("\"'“”‘’")
        if not text or is_definition_like_annotation_synonym(text):
            return
        normalized = normalize_feature_name(text)
        if not normalized or normalized in blocked or normalized in seen:
            return
        seen.add(normalized)
        output.append(text)

    for synonym in synonyms:
        add(synonym)
    for alias in moral_frame_aliases_for_feature(feature):
        add(alias)
        if len(output) >= 12:
            break
    return output[:12]
