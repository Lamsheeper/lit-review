"""Core text."""

from __future__ import annotations
import html


import re


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")


STOPWORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "among",
    "an",
    "and",
    "another",
    "any",
    "are",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "due",
    "during",
    "each",
    "either",
    "especially",
    "etc",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "here",
    "how",
    "however",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "may",
    "might",
    "more",
    "most",
    "must",
    "near",
    "need",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "paper",
    "papers",
    "per",
    "possible",
    "present",
    "results",
    "same",
    "several",
    "should",
    "show",
    "shown",
    "since",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "those",
    "through",
    "to",
    "toward",
    "under",
    "until",
    "up",
    "use",
    "used",
    "using",
    "via",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
}


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def content_tokens(text: str | None) -> list[str]:
    return [
        token for token in tokenize(text) if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    ]


def normalize_title(value: str | None) -> str:
    if not value:
        return ""
    value = strip_markup(value) or ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return normalize_whitespace(value)


def slugify(value: str | None, max_length: int = 80) -> str:
    if not value:
        return ""
    slug = normalize_title(value).replace(" ", "_")
    slug = re.sub(r"_+", "_", slug).strip("_")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")
    return slug


def normalize_feature_name(value: str | None) -> str:
    if not value:
        return ""
    text = value.lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


TAG_RE = re.compile(r"<[^>]+>")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_markup(value: str | None) -> str | None:
    if not value:
        return None
    return normalize_whitespace(html.unescape(TAG_RE.sub(" ", value)))
