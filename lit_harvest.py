#!/usr/bin/env python3
"""LitHarvest: draft-to-PDF literature collection pipeline.

This module intentionally avoids Google Scholar scraping and browser
automation. It talks to metadata APIs, resolves open-access PDFs where those
APIs expose them, and records every candidate and download outcome.
"""

from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import gzip
import hashlib
import html
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from lit_relevance import load_relevance_profile


VERSION = "0.1.0"
USER_AGENT_NAME = f"LitHarvest/{VERSION}"
DEFAULT_SOURCES = ["openalex", "semantic_scholar", "crossref", "europe_pmc"]
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
HARVEST_MODES = ("draft", "citations")
CITATION_EXTRACTORS = ("direct_pdf", "pymupdf_markdown")
LLM_PROVIDERS = ("gemini", "openai")
DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
LOGGER = logging.getLogger("litharvest")

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
TAG_RE = re.compile(r"<[^>]+>")

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

METHOD_HINTS = {
    "algorithm",
    "analysis",
    "approach",
    "architecture",
    "assessment",
    "benchmark",
    "classification",
    "clustering",
    "detection",
    "estimation",
    "evaluation",
    "framework",
    "inference",
    "learning",
    "method",
    "model",
    "modeling",
    "network",
    "optimization",
    "pipeline",
    "prediction",
    "regression",
    "simulation",
    "survey",
    "system",
}

APPLICATION_HINTS = {
    "aerospace",
    "air",
    "autonomous",
    "biomedical",
    "clinical",
    "cyber",
    "defense",
    "energy",
    "health",
    "manufacturing",
    "materials",
    "medical",
    "military",
    "mission",
    "robotics",
    "security",
    "sensor",
    "space",
    "surveillance",
}


@dataclass
class ExtractedTerms:
    title: str | None
    headings: list[str]
    keywords: list[str]
    noun_phrases: list[str]
    domain_terms: list[str]


@dataclass
class SearchQuery:
    bucket: str
    text: str
    terms: list[str] = field(default_factory=list)


@dataclass
class WebSearchResult:
    source: str
    query: str
    rank: int
    url: str
    title: str | None = None
    snippet: str | None = None


@dataclass
class Candidate:
    title: str
    candidate_id: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    abstract: str | None = None
    venue: str | None = None
    landing_page_url: str | None = None
    source_apis: list[str] = field(default_factory=list)
    source_ids: dict[str, str] = field(default_factory=dict)
    candidate_pdf_urls: list[str] = field(default_factory=list)
    candidate_landing_page_urls: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    relevance_score_breakdown: dict[str, Any] = field(default_factory=dict)
    source_score: float | None = None
    citation_count: int | None = None
    discovered_via: list[dict[str, Any]] = field(default_factory=list)
    pdf_resolution: dict[str, Any] = field(default_factory=dict)
    download: dict[str, Any] = field(default_factory=dict)
    matched_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CitationLLMCallResult:
    parsed: dict[str, Any]
    response: dict[str, Any]
    raw_text: str
    input_tokens: int | None
    output_tokens: int | None
    schema_valid: bool
    validation_errors: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Draft file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Draft path is not a file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def content_tokens(text: str | None) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    ]


def clean_title_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^#+\s*", "", line)
    return normalize_whitespace(line)


def find_title_and_headings(text: str) -> tuple[str | None, list[str]]:
    title = None
    headings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = clean_title_line(line)
            if heading:
                headings.append(heading)
                if title is None:
                    title = heading
            continue
        if title is None and len(line) <= 180:
            title = clean_title_line(line)
    return title, headings


def extract_domain_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", text):
        value = match.group(0)
        if value not in {"PDF", "JSON", "API", "DOI", "URL", "HTTP", "HTTPS"}:
            terms.append(value)
    for match in re.finditer(r"\b[A-Za-z]+(?:-[A-Za-z0-9]+)+\b", text):
        value = match.group(0)
        if len(value) > 5:
            terms.append(value)
    return dedupe_preserve(terms)[:30]


def extract_keywords(text: str, max_terms: int = 30) -> ExtractedTerms:
    title, headings = find_title_and_headings(text)
    title_lower = title.lower() if title else ""
    heading_lowers = [heading.lower() for heading in headings]
    tokens = content_tokens(text)

    ngram_counts: Counter[str] = Counter()
    for n in range(1, 5):
        for idx in range(0, max(0, len(tokens) - n + 1)):
            phrase_tokens = tokens[idx : idx + n]
            if n > 1 and len(set(phrase_tokens)) == 1:
                continue
            ngram_counts[" ".join(phrase_tokens)] += 1

    scored: list[tuple[float, str]] = []
    for phrase, count in ngram_counts.items():
        words = phrase.split()
        if not words:
            continue
        score = float(count) * (1.0 + (len(words) - 1) * 0.35)
        if phrase in title_lower:
            score += 4.0
        if any(phrase in heading for heading in heading_lowers):
            score += 2.0
        if len(words) > 1:
            score += 0.5
        scored.append((score, phrase))

    scored.sort(key=lambda item: (-item[0], item[1]))
    keywords = [phrase for _, phrase in scored if len(phrase.split()) == 1][:max_terms]
    noun_phrases = [phrase for _, phrase in scored if len(phrase.split()) > 1][:max_terms]
    domain_terms = extract_domain_terms(text)

    return ExtractedTerms(
        title=title,
        headings=headings[:20],
        keywords=keywords,
        noun_phrases=noun_phrases,
        domain_terms=domain_terms,
    )


def compact_query_text(text: str, max_words: int = 12) -> str:
    tokens = content_tokens(text)
    if len(tokens) >= 3:
        return " ".join(tokens[:max_words])
    return normalize_whitespace(text)[:160]


def build_search_queries(terms: ExtractedTerms, max_queries: int) -> list[SearchQuery]:
    queries: list[SearchQuery] = []

    def add(bucket: str, text: str | None, query_terms: list[str] | None = None) -> None:
        if not text:
            return
        cleaned = normalize_whitespace(text)
        if len(cleaned) < 4:
            return
        norm = normalize_title(cleaned)
        if not norm:
            return
        if any(normalize_title(query.text) == norm for query in queries):
            return
        queries.append(SearchQuery(bucket=bucket, text=cleaned, terms=query_terms or []))

    if terms.title:
        add("primary", compact_query_text(terms.title), content_tokens(terms.title))

    phrases = terms.noun_phrases
    keywords = terms.keywords
    domain_terms = [term.lower() for term in terms.domain_terms if len(term) > 2]

    if phrases:
        add("primary", phrases[0], phrases[0].split())
    if len(phrases) >= 2:
        add("primary", f"{phrases[0]} {phrases[1]}", phrases[0].split() + phrases[1].split())

    for phrase in phrases[: max(8, max_queries * 2)]:
        words = set(phrase.split())
        if words & METHOD_HINTS:
            add("method", phrase, phrase.split())
        elif words & APPLICATION_HINTS:
            add("application", phrase, phrase.split())
        else:
            add("related", phrase, phrase.split())
        if len(queries) >= max_queries:
            return queries[:max_queries]

    for idx, phrase in enumerate(phrases[1:8], start=1):
        if phrases:
            add("narrow", f"{phrases[0]} {phrase}", phrases[0].split() + phrase.split())
        if idx < len(keywords):
            add("narrow", f"{phrase} {keywords[idx]}", phrase.split() + [keywords[idx]])
        if len(queries) >= max_queries:
            return queries[:max_queries]

    for term in domain_terms:
        add("domain", term, term.split())
        if len(queries) >= max_queries:
            return queries[:max_queries]

    for keyword in keywords[: max_queries]:
        add("related", keyword, [keyword])
        if len(queries) >= max_queries:
            break

    return queries[:max_queries]


def configured_search_queries(raw_queries: Any) -> list[SearchQuery]:
    if not raw_queries:
        return []
    if not isinstance(raw_queries, list):
        raise ValueError("extra_queries must be a list of strings or objects.")

    queries: list[SearchQuery] = []
    for idx, item in enumerate(raw_queries, start=1):
        bucket = "config"
        terms: list[str] = []
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            bucket = str(item.get("bucket") or bucket).strip() or bucket
            raw_terms = item.get("terms") or []
            if isinstance(raw_terms, str):
                terms = content_tokens(raw_terms)
            elif isinstance(raw_terms, list):
                terms = [str(term).strip() for term in raw_terms if str(term).strip()]
            else:
                raise ValueError(f"extra_queries[{idx}].terms must be a string or list.")
        else:
            raise ValueError(f"extra_queries[{idx}] must be a string or object.")

        text = normalize_whitespace(text)
        if not text:
            raise ValueError(f"extra_queries[{idx}] is missing text.")
        queries.append(
            SearchQuery(bucket=bucket, text=text, terms=terms or content_tokens(text))
        )
    return queries


def combine_search_queries(
    configured: list[SearchQuery],
    generated: list[SearchQuery],
    max_queries: int,
) -> list[SearchQuery]:
    queries: list[SearchQuery] = []
    seen: set[str] = set()
    for query in configured + generated:
        normalized = normalize_title(query.text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def strip_markup(value: str | None) -> str | None:
    if not value:
        return None
    return normalize_whitespace(html.unescape(TAG_RE.sub(" ", value)))


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = urllib.parse.unquote(str(value).strip())
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = doi.strip().strip(".")
    return doi.lower() or None


def doi_to_url(doi: str | None) -> str | None:
    if not doi:
        return None
    return f"https://doi.org/{doi}"


def normalize_title(value: str | None) -> str:
    if not value:
        return ""
    value = strip_markup(value) or ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return normalize_whitespace(value)


def candidate_matches_keywords(candidate: Candidate, keywords: list[str]) -> list[str]:
    if not keywords:
        return []
    text = " ".join(
        part.strip()
        for part in [candidate.title or "", candidate.abstract or "", candidate.venue or ""]
        if part
    ).lower()
    matches: list[str] = []
    for keyword in keywords:
        if not keyword:
            continue
        normalized = keyword.lower().strip()
        if not normalized:
            continue
        if re.search(r"\b" + re.escape(normalized) + r"\b", text):
            matches.append(keyword)
    return dedupe_preserve(matches)


def year_in_range(year: int | None, year_from: int | None, year_to: int | None) -> bool:
    if year is None:
        return True
    if year_from is not None and year < year_from:
        return False
    if year_to is not None and year > year_to:
        return False
    return True


def first_date_part_year(value: Any) -> int | None:
    try:
        part = value["date-parts"][0][0]
    except (KeyError, IndexError, TypeError):
        return None
    try:
        return int(part)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value:
            continue
        key = value.strip()
        norm = key.lower()
        if key and norm not in seen:
            seen.add(norm)
            output.append(key)
    return output


def clean_url(value: str | None) -> str | None:
    if not value:
        return None
    url = html.unescape(str(value)).strip().strip("\"'")
    url = url.replace("\\/", "/")
    url = re.sub(r"\\u002[fF]", "/", url)
    url = re.sub(r"\\u0026", "&", url, flags=re.IGNORECASE)
    url = re.sub(r"\\u003[dD]", "=", url)
    url = re.sub(r"\\u003[fF]", "?", url)
    url = url.strip().rstrip(".,;)")
    return url or None


def add_url(urls: list[str], value: str | None) -> None:
    url = clean_url(value)
    if not url:
        return
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        return
    if url.lower() not in {existing.lower() for existing in urls}:
        urls.append(url)


def add_absolute_url(urls: list[str], value: str | None, base_url: str) -> None:
    url = clean_url(value)
    if not url:
        return
    absolute = urllib.parse.urljoin(base_url, url)
    add_url(urls, absolute)


PDF_URL_PATH_HINTS = (
    "/pdf",
    "pdf/",
    "article-pdf",
    "chapter-pdf",
    "full/pdf",
    "content/pdf",
    "doi/pdf",
    "download",
    "viewcontent",
    "fulltext",
)


def has_pdf_file_hint(url: str | None) -> bool:
    if not url:
        return False
    parsed = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(parsed.path.lower())
    return path.endswith(".pdf") or ".pdf" in path


def looks_like_pdf_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(parsed.path.lower())
    query = urllib.parse.unquote(parsed.query.lower())
    if has_pdf_file_hint(url):
        return True
    if any(hint in path for hint in PDF_URL_PATH_HINTS):
        return True
    if "pdf" in query and any(
        key in query for key in ("download", "type", "format", "output", "mime", "file")
    ):
        return True
    return False


class NonRetryableDownloadError(RuntimeError):
    """Raised when another attempt at the same URL cannot turn it into a PDF."""


class HttpClient:
    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 2,
        backoff: float = 1.5,
        rate_limit_delay: float = 0.1,
        email: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.rate_limit_delay = rate_limit_delay
        self.email = email
        self._last_request_at = 0.0
        self._rate_lock = threading.Lock()

    @property
    def user_agent(self) -> str:
        if self.email:
            return f"{USER_AGENT_NAME} (mailto:{self.email})"
        return USER_AGENT_NAME

    def _build_url(self, url: str, params: dict[str, Any] | None = None) -> str:
        if not params:
            return url
        clean_params = {key: value for key, value in params.items() if value is not None}
        query = urllib.parse.urlencode(clean_params, doseq=True)
        separator = "&" if urllib.parse.urlparse(url).query else "?"
        return f"{url}{separator}{query}"

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json, application/pdf;q=0.8, */*;q=0.5",
            "User-Agent": self.user_agent,
        }
        if extra:
            headers.update(extra)
        return headers

    def _wait_for_rate_limit(self) -> None:
        with self._rate_lock:
            if self.rate_limit_delay <= 0:
                self._last_request_at = time.monotonic()
                return
            elapsed = time.monotonic() - self._last_request_at
            remaining = self.rate_limit_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()

    def _retry_sleep(self, attempt: int, headers: dict[str, Any] | None = None) -> None:
        retry_after = None
        if headers:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                delay = min(float(retry_after), 30.0)
            except ValueError:
                delay = min(self.backoff * (attempt + 1), 30.0)
        else:
            delay = min(self.backoff * (attempt + 1), 30.0)
        time.sleep(delay)

    def request_bytes(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[bytes, dict[str, Any], str]:
        full_url = self._build_url(url, params)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._wait_for_rate_limit()
            request = urllib.request.Request(full_url, headers=self._headers(headers))
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    self._last_request_at = time.monotonic()
                    body = response.read()
                    response_headers = dict(response.headers.items())
                    body = decode_http_body(body, response_headers)
                    return body, response_headers, response.geturl()
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    self._retry_sleep(attempt, dict(exc.headers.items()))
                    continue
                detail = exc.read(500).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} for {full_url}: {detail}") from exc
            except urllib.error.URLError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if attempt < self.retries:
                    self._retry_sleep(attempt)
                    continue
                raise RuntimeError(f"Request failed for {full_url}: {exc}") from exc
        raise RuntimeError(f"Request failed for {full_url}: {last_error}")

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body, _, final_url = self.request_bytes(url, params=params, headers=headers)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            preview = body[:200].decode("utf-8", errors="replace")
            raise RuntimeError(f"Expected JSON from {final_url}, received: {preview}") from exc

    def download_pdf(
        self,
        url: str,
        destination: Path,
        max_bytes: int,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._wait_for_rate_limit()
            request = urllib.request.Request(
                url,
                headers=self._headers({"Accept": "application/pdf, */*;q=0.2"}),
            )
            temp_path: Path | None = None
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    self._last_request_at = time.monotonic()
                    content_type = response.headers.get("Content-Type", "")
                    content_length = safe_int(response.headers.get("Content-Length"))
                    if content_length is not None and content_length > max_bytes:
                        raise NonRetryableDownloadError(
                            f"PDF is larger than limit: {content_length} bytes > {max_bytes} bytes"
                        )

                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        "wb", delete=False, dir=str(destination.parent), suffix=".tmp"
                    ) as handle:
                        temp_path = Path(handle.name)
                        first_chunk = response.read(8192)
                        if not is_pdf_response(content_type, first_chunk):
                            preview = first_chunk[:80].decode("utf-8", errors="replace")
                            raise NonRetryableDownloadError(
                                f"URL did not return a PDF; content-type={content_type!r}; "
                                f"preview={preview!r}"
                            )
                        handle.write(first_chunk)
                        total = len(first_chunk)
                        while True:
                            chunk = response.read(65536)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > max_bytes:
                                raise NonRetryableDownloadError(
                                    f"PDF exceeded limit while downloading: {total} bytes"
                                )
                            handle.write(chunk)

                    os.replace(temp_path, destination)
                    return {
                        "status": "downloaded",
                        "url": url,
                        "final_url": response.geturl(),
                        "path": str(destination),
                        "bytes": total,
                        "content_type": content_type,
                    }
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    self._retry_sleep(attempt, dict(exc.headers.items()))
                    continue
                detail = exc.read(200).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} while downloading PDF: {detail}") from exc
            except NonRetryableDownloadError as exc:
                self._last_request_at = time.monotonic()
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                raise RuntimeError(f"PDF download failed for {url}: {exc}") from exc
            except (urllib.error.URLError, RuntimeError) as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                if attempt < self.retries:
                    self._retry_sleep(attempt)
                    continue
                raise RuntimeError(f"PDF download failed for {url}: {exc}") from exc
        raise RuntimeError(f"PDF download failed for {url}: {last_error}")


def decode_http_body(body: bytes, headers: dict[str, Any]) -> bytes:
    encoding = str(
        headers.get("Content-Encoding")
        or headers.get("content-encoding")
        or ""
    ).lower()
    if not body:
        return body
    if "gzip" in encoding:
        try:
            return gzip.decompress(body)
        except OSError:
            return body
    if "deflate" in encoding:
        try:
            return zlib.decompress(body)
        except zlib.error:
            try:
                return zlib.decompress(body, -zlib.MAX_WBITS)
            except zlib.error:
                return body
    return body


def is_pdf_response(content_type: str | None, first_chunk: bytes) -> bool:
    content_type = (content_type or "").lower()
    if first_chunk.startswith(b"%PDF-"):
        return True
    if "pdf" in content_type and not looks_like_html(first_chunk):
        return True
    return False


def looks_like_html(content: bytes) -> bool:
    sample = content[:200].lstrip().lower()
    return sample.startswith(b"<!doctype html") or sample.startswith(b"<html")


PDF_META_NAMES = {
    "citation_pdf_url",
    "bepress_citation_pdf_url",
    "eprints.document_url",
    "fulltext_pdf",
    "pdf_url",
}


class PdfLinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._inspect_tag(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._inspect_tag(tag, attrs)

    def _inspect_tag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr = {key.lower(): value or "" for key, value in attrs}

        if tag == "meta":
            name = (
                attr.get("name")
                or attr.get("property")
                or attr.get("itemprop")
                or ""
            ).lower()
            content = attr.get("content")
            if content and (
                name in PDF_META_NAMES
                or "pdf" in name
                or looks_like_pdf_url(content)
            ):
                add_absolute_url(self.urls, content, self.base_url)
            return

        if tag == "link":
            href = attr.get("href")
            rel = attr.get("rel", "").lower()
            content_type = attr.get("type", "").lower()
            title = attr.get("title", "").lower()
            if href and (
                "pdf" in rel
                or "pdf" in content_type
                or "pdf" in title
                or looks_like_pdf_url(href)
            ):
                add_absolute_url(self.urls, href, self.base_url)
            return

        for key in ("href", "src", "data"):
            value = attr.get(key)
            if value and looks_like_pdf_url(value):
                add_absolute_url(self.urls, value, self.base_url)

        for key, value in attr.items():
            if not value:
                continue
            key_lower = key.lower()
            if any(hint in key_lower for hint in ("pdf", "download", "fulltext")):
                if looks_like_pdf_url(value) or re.match(r"^https?://", value):
                    add_absolute_url(self.urls, value, self.base_url)


def extract_pdf_urls_from_text(text: str, base_url: str) -> list[str]:
    urls: list[str] = []
    for raw_url in re.findall(r"https?:\\?/\\?/[^\"'<>\\\s]+", text, flags=re.IGNORECASE):
        if looks_like_pdf_url(raw_url):
            add_url(urls, raw_url)
    for match in re.findall(
        r"""(?ix)
        ["'](?:pdf[_-]?url|pdfUrl|url_for_pdf|download[_-]?url|fullTextUrl)["']
        \s*:\s*
        ["']([^"']+)["']
        """,
        text,
    ):
        if looks_like_pdf_url(match):
            add_absolute_url(urls, match, base_url)
    return urls


def extract_pdf_urls_from_html(html_text: str, base_url: str) -> list[str]:
    urls: list[str] = []
    parser = PdfLinkParser(base_url)
    try:
        parser.feed(html_text)
        urls.extend(parser.urls)
    except Exception:
        LOGGER.debug("HTML parser failed while extracting PDF links.", exc_info=True)
    urls.extend(extract_pdf_urls_from_text(html_text, base_url))
    return dedupe_preserve(urls)


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    identifier = clean_url(value) or str(value).strip()
    identifier = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", identifier, flags=re.IGNORECASE)
    identifier = re.sub(r"^arxiv:\s*", "", identifier, flags=re.IGNORECASE)
    identifier = identifier.strip().strip("/")
    identifier = re.sub(r"\.pdf$", "", identifier, flags=re.IGNORECASE)
    if not identifier:
        return None
    if re.match(r"^\d{4}\.\d{4,5}(?:v\d+)?$", identifier, flags=re.IGNORECASE):
        return identifier
    if re.match(r"^[a-z][a-z.-]+/\d{7}(?:v\d+)?$", identifier, flags=re.IGNORECASE):
        return identifier
    return None


def normalize_acl_id(value: str | None) -> str | None:
    if not value:
        return None
    identifier = clean_url(value) or str(value).strip()
    identifier = re.sub(
        r"^https?://(?:www\.)?aclanthology\.org/", "",
        identifier,
        flags=re.IGNORECASE,
    )
    identifier = identifier.strip().strip("/")
    identifier = re.sub(r"\.pdf$", "", identifier, flags=re.IGNORECASE)
    if re.match(r"^[a-z]\d{2}-\d{4}$", identifier, flags=re.IGNORECASE):
        return identifier
    if re.match(r"^\d{4}\.[a-z0-9-]+\.\d+$", identifier, flags=re.IGNORECASE):
        return identifier
    return None


def normalize_pmcid(value: str | None) -> str | None:
    if not value:
        return None
    identifier = str(value).strip()
    identifier = re.sub(r"^pmcid:\s*", "", identifier, flags=re.IGNORECASE)
    identifier = re.sub(
        r"^https?://(?:www\.)?(?:ncbi\.nlm\.nih\.gov/pmc|pmc\.ncbi\.nlm\.nih\.gov/articles)/",
        "",
        identifier,
        flags=re.IGNORECASE,
    )
    identifier = identifier.strip().strip("/")
    match = re.search(r"(PMC\d+)", identifier, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def add_identifier_pdf_urls(urls: list[str], identifier: str | None, kind: str | None = None) -> None:
    if not identifier:
        return
    kind_lower = (kind or "").lower()
    if kind_lower in {"arxiv", "arxiv_id"} or "arxiv" in str(identifier).lower():
        arxiv_id = normalize_arxiv_id(identifier)
        if arxiv_id:
            add_url(urls, f"https://arxiv.org/pdf/{urllib.parse.quote(arxiv_id, safe='/')}.pdf")
    if kind_lower in {"acl", "acl_id"}:
        acl_id = normalize_acl_id(identifier)
        if acl_id:
            add_url(urls, f"https://aclanthology.org/{acl_id}.pdf")
    if kind_lower in {"pmc", "pmcid", "pubmedcentral"} or str(identifier).upper().startswith("PMC"):
        pmcid = normalize_pmcid(identifier)
        if pmcid:
            add_url(urls, f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/")
            add_url(urls, f"https://europepmc.org/articles/{pmcid}?pdf=render")


def add_doi_pdf_urls(urls: list[str], doi: str | None) -> None:
    if not doi:
        return
    doi_lower = doi.lower()
    if doi_lower.startswith("10.48550/arxiv."):
        add_identifier_pdf_urls(urls, doi_lower.split("arxiv.", 1)[1], kind="arxiv")
    if doi_lower.startswith("10.18653/v1/"):
        add_identifier_pdf_urls(urls, doi.split("/", 2)[2], kind="acl")


def inferred_pdf_urls(candidate: Candidate) -> list[str]:
    urls: list[str] = []
    add_doi_pdf_urls(urls, candidate.doi)
    for key, value in candidate.source_ids.items():
        key_lower = key.lower()
        if "arxiv" in key_lower:
            add_identifier_pdf_urls(urls, value, kind="arxiv")
        elif "acl" in key_lower:
            add_identifier_pdf_urls(urls, value, kind="acl")
        elif "pmc" in key_lower or "pubmedcentral" in key_lower:
            add_identifier_pdf_urls(urls, value, kind="pmc")
    return dedupe_preserve(urls)


CITATION_REFERENCE_KEYS = [
    "ref_id",
    "raw_reference",
    "authors",
    "title",
    "year",
    "venue",
    "doi",
    "arxiv_id",
    "url",
]


def nullable_schema_type(type_name: str) -> dict[str, Any]:
    return {"anyOf": [{"type": type_name}, {"type": "null"}]}


CITATION_BIBLIOGRAPHY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "string"},
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ref_id": {"type": "string"},
                    "raw_reference": nullable_schema_type("string"),
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "title": nullable_schema_type("string"),
                    "year": nullable_schema_type("integer"),
                    "venue": nullable_schema_type("string"),
                    "doi": nullable_schema_type("string"),
                    "arxiv_id": nullable_schema_type("string"),
                    "url": nullable_schema_type("string"),
                },
                "required": CITATION_REFERENCE_KEYS,
            },
        },
    },
    "required": ["paper_id", "references"],
}

CITATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "citation_extraction",
    "strict": True,
    "schema": CITATION_BIBLIOGRAPHY_SCHEMA,
}

GEMINI_CITATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "propertyOrdering": ["paper_id", "references"],
    "properties": {
        "paper_id": {"type": "string"},
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "propertyOrdering": CITATION_REFERENCE_KEYS,
                "properties": {
                    "ref_id": {"type": "string"},
                    "raw_reference": {"type": ["string", "null"]},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "title": {"type": ["string", "null"]},
                    "year": {"type": ["integer", "null"]},
                    "venue": {"type": ["string", "null"]},
                    "doi": {"type": ["string", "null"]},
                    "arxiv_id": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                },
                "required": CITATION_REFERENCE_KEYS,
            },
        },
    },
    "required": ["paper_id", "references"],
}

CITATION_EXTRACTION_PROMPT = """\
You extract bibliography/reference-list entries from academic papers.

Rules:
- Return only JSON matching the requested schema.
- Extract entries from the bibliography, references, or works-cited section.
- Do not use Semantic Scholar, Crossref, OpenAlex, DBLP, search engines, or any external citation database.
- Use only the supplied PDF or extracted page text.
- Use null for unknown scalar fields and [] for unknown authors.
- Preserve the raw reference text as faithfully as possible in raw_reference.
- Use stable reference IDs R001, R002, R003, ... in document order.
- Do not include in-text citations unless they are part of the reference list.
"""


def safe_pdf_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return stem or "paper"


def make_citation_user_prompt(paper_id: str, source_description: str) -> str:
    return (
        f"paper_id: {paper_id}\n"
        f"source: {source_description}\n\n"
        "Extract the bibliography/reference list for this paper into the JSON schema."
    )


def empty_citation_bibliography(paper_id: str) -> dict[str, Any]:
    return {"paper_id": paper_id, "references": []}


def validate_citation_bibliography(data: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["top-level output must be an object"]

    extra_top_level = sorted(set(data) - {"paper_id", "references"})
    if extra_top_level:
        errors.append(f"top-level output has extra keys: {', '.join(extra_top_level)}")

    if not isinstance(data.get("paper_id"), str):
        errors.append("paper_id must be a string")

    references = data.get("references")
    if not isinstance(references, list):
        errors.append("references must be a list")
        return errors

    nullable_strings = {"raw_reference", "title", "venue", "doi", "arxiv_id", "url"}
    for index, reference in enumerate(references):
        prefix = f"references[{index}]"
        if not isinstance(reference, dict):
            errors.append(f"{prefix} must be an object")
            continue
        extra_keys = sorted(set(reference) - set(CITATION_REFERENCE_KEYS))
        if extra_keys:
            errors.append(f"{prefix} has extra keys: {', '.join(extra_keys)}")
        missing = [key for key in CITATION_REFERENCE_KEYS if key not in reference]
        if missing:
            errors.append(f"{prefix} missing keys: {', '.join(missing)}")

        if not isinstance(reference.get("ref_id"), str):
            errors.append(f"{prefix}.ref_id must be a string")

        authors = reference.get("authors")
        if not isinstance(authors, list) or not all(isinstance(author, str) for author in authors):
            errors.append(f"{prefix}.authors must be a list of strings")

        year = reference.get("year")
        if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
            errors.append(f"{prefix}.year must be an integer or null")

        for key in nullable_strings:
            value = reference.get(key)
            if value is not None and not isinstance(value, str):
                errors.append(f"{prefix}.{key} must be a string or null")

    return errors


def parse_citation_llm_json(raw_text: str, paper_id: str) -> tuple[dict[str, Any], list[str]]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return empty_citation_bibliography(paper_id), [f"model output was not valid JSON: {exc}"]

    errors = validate_citation_bibliography(parsed)
    if errors:
        return parsed if isinstance(parsed, dict) else empty_citation_bibliography(paper_id), errors
    return parsed, []


def llm_provider_default_model(provider: str) -> str:
    if provider == "gemini":
        return DEFAULT_GEMINI_MODEL
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    raise ValueError(f"Unsupported LLM provider: {provider}")


def llm_provider_env_api_key(provider: str) -> str | None:
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    raise ValueError(f"Unsupported LLM provider: {provider}")


def retry_after_delay(exc: urllib.error.HTTPError, fallback: float) -> float:
    retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), fallback)
        except ValueError:
            return fallback
    return fallback


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT_NAME,
    }
    request_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(retry_after_delay(exc, sleep_seconds * (attempt + 1)))
                continue
            raise RuntimeError(f"LLM request failed ({exc.code}): {error_body}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))
                continue
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    raise RuntimeError(f"LLM request failed: {last_error}")


def extract_openai_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    refusals: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            content_type = content.get("type")
            if content_type in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
            elif content_type == "refusal" and isinstance(content.get("refusal"), str):
                refusals.append(content["refusal"])

    if refusals:
        raise RuntimeError(f"Model refusal: {' '.join(refusals)}")
    if not parts:
        raise RuntimeError("OpenAI response did not contain output text.")
    return "\n".join(parts)


def openai_usage_tokens(response: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = response.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int):
        input_tokens = None
    if not isinstance(output_tokens, int):
        output_tokens = None
    return input_tokens, output_tokens


def gemini_usage_tokens(response: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = response.get("usageMetadata") or {}
    input_tokens = usage.get("promptTokenCount")
    output_tokens = usage.get("candidatesTokenCount")
    if not isinstance(input_tokens, int):
        input_tokens = None
    if not isinstance(output_tokens, int):
        output_tokens = None
    return input_tokens, output_tokens


def gemini_content_part(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "input_text":
        return {"text": item.get("text") or ""}
    if item_type == "input_file":
        return {
            "inline_data": {
                "mime_type": "application/pdf",
                "data": item.get("file_data") or "",
            }
        }
    raise RuntimeError(f"Unsupported Gemini citation input item: {item_type}")


def extract_gemini_output_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    if texts:
        return "\n".join(texts)
    prompt_feedback = response.get("promptFeedback")
    if prompt_feedback:
        raise RuntimeError(f"Gemini prompt feedback: {prompt_feedback}")
    raise RuntimeError("Gemini response did not contain output text.")


def call_openai_citation_extraction(
    content: list[dict[str, Any]],
    config: dict[str, Any],
    paper_id: str,
) -> CitationLLMCallResult:
    payload: dict[str, Any] = {
        "model": config["llm_model"],
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": CITATION_EXTRACTION_PROMPT}],
            },
            {"role": "user", "content": content},
        ],
        "text": {"format": CITATION_RESPONSE_FORMAT},
    }
    if config.get("max_output_tokens"):
        payload["max_output_tokens"] = int(config["max_output_tokens"])

    response = post_json(
        OPENAI_RESPONSES_URL,
        payload,
        headers={"Authorization": f"Bearer {config['llm_api_key']}"},
        timeout=float(config["timeout"]),
        retries=int(config["retries"]),
        sleep_seconds=float(config["backoff"]),
    )
    raw_text = extract_openai_output_text(response)
    parsed, validation_errors = parse_citation_llm_json(raw_text, paper_id)
    input_tokens, output_tokens = openai_usage_tokens(response)
    return CitationLLMCallResult(
        parsed=parsed,
        response=response,
        raw_text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        schema_valid=not validation_errors,
        validation_errors=validation_errors,
    )


def call_gemini_citation_extraction(
    content: list[dict[str, Any]],
    config: dict[str, Any],
    paper_id: str,
) -> CitationLLMCallResult:
    parts = [{"text": CITATION_EXTRACTION_PROMPT}]
    parts.extend(gemini_content_part(item) for item in content)
    generation_config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseJsonSchema": GEMINI_CITATION_RESPONSE_SCHEMA,
    }
    if config.get("max_output_tokens"):
        generation_config["maxOutputTokens"] = int(config["max_output_tokens"])

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }
    model_name = str(config["llm_model"]).removeprefix("models/")
    url = (
        f"{GEMINI_GENERATE_URL.format(model=urllib.parse.quote(model_name, safe=''))}"
        f"?{urllib.parse.urlencode({'key': config['llm_api_key']})}"
    )
    response = post_json(
        url,
        payload,
        headers={},
        timeout=float(config["timeout"]),
        retries=int(config["retries"]),
        sleep_seconds=float(config["backoff"]),
    )
    raw_text = extract_gemini_output_text(response)
    parsed, validation_errors = parse_citation_llm_json(raw_text, paper_id)
    input_tokens, output_tokens = gemini_usage_tokens(response)
    return CitationLLMCallResult(
        parsed=parsed,
        response=response,
        raw_text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        schema_valid=not validation_errors,
        validation_errors=validation_errors,
    )


def call_llm_citation_extraction(
    content: list[dict[str, Any]],
    config: dict[str, Any],
    paper_id: str,
) -> CitationLLMCallResult:
    provider = config["llm_provider"]
    if provider == "gemini":
        return call_gemini_citation_extraction(content, config, paper_id)
    if provider == "openai":
        return call_openai_citation_extraction(content, config, paper_id)
    raise RuntimeError(f"Unsupported LLM provider: {provider}")


def require_pymupdf():
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for citation extractor pymupdf_markdown.") from exc
    return fitz


def extract_pdf_text_for_citations(pdf_path: Path) -> tuple[str, int]:
    fitz = require_pymupdf()
    pages: list[str] = []
    with fitz.open(pdf_path) as document:
        page_count = document.page_count
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            pages.append(f"# Page {page_number}\n\n{text.strip()}\n")
    return "\n".join(pages), page_count


def maybe_truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def build_direct_pdf_citation_content(pdf_path: Path, paper_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pdf_data = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    content = [
        {"type": "input_text", "text": make_citation_user_prompt(paper_id, "the attached PDF file")},
        {"type": "input_file", "filename": pdf_path.name, "file_data": pdf_data},
    ]
    metadata = {
        "source_description": "the attached PDF file",
        "pdf_bytes": pdf_path.stat().st_size,
    }
    return content, metadata


def build_pymupdf_markdown_citation_content(
    pdf_path: Path,
    paper_id: str,
    max_text_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    markdown, page_count = extract_pdf_text_for_citations(pdf_path)
    text_for_model, truncated = maybe_truncate_text(markdown, max_text_chars)
    prompt = make_citation_user_prompt(paper_id, "PyMuPDF-extracted page text")
    if truncated:
        prompt += (
            f"\n\nNote: the extracted text was truncated to {max_text_chars} characters "
            "by LitHarvest before citation extraction."
        )
    content = [{"type": "input_text", "text": f"{prompt}\n\n{text_for_model}"}]
    metadata = {
        "source_description": "PyMuPDF-extracted page text",
        "num_pages": page_count,
        "text_chars_total": len(markdown),
        "text_chars_sent": len(text_for_model),
        "text_truncated": truncated,
        "markdown": markdown,
    }
    return content, metadata


def quote_search_phrase(value: str) -> str:
    escaped = value.replace('"', " ").strip()
    return f'"{normalize_whitespace(escaped)}"'


def build_web_pdf_queries(candidate: Candidate, max_queries: int = 3) -> list[str]:
    queries: list[str] = []
    if candidate.doi:
        queries.append(f'{quote_search_phrase(candidate.doi)} pdf')
    title = normalize_whitespace(candidate.title)
    if title:
        quoted_title = quote_search_phrase(title)
        queries.append(f"{quoted_title} filetype:pdf")
        queries.append(f"{quoted_title} pdf")
        if candidate.authors:
            queries.append(f"{quoted_title} {quote_search_phrase(candidate.authors[0])} pdf")
    return dedupe_preserve(queries)[: max(max_queries, 0)]


def web_result_matches_candidate(candidate: Candidate, result: WebSearchResult) -> bool:
    if candidate.doi:
        needle = candidate.doi.lower()
        haystack = " ".join(
            part for part in [result.url, result.title or "", result.snippet or ""] if part
        ).lower()
        if needle in haystack:
            return True

    candidate_tokens = set(content_tokens(candidate.title))
    if not candidate_tokens:
        return True
    result_text = " ".join(
        part for part in [result.title or "", result.snippet or "", result.url] if part
    )
    result_tokens = set(content_tokens(urllib.parse.unquote(result_text)))
    overlap = candidate_tokens & result_tokens
    required = max(2, min(5, math.ceil(len(candidate_tokens) * 0.4)))
    if len(overlap) >= required:
        return True
    return result.rank <= 2 and bool(overlap)


class WebSearchProvider:
    name = "base"

    def search(self, query: str, max_results: int) -> list[WebSearchResult]:
        raise NotImplementedError


class BraveSearchProvider(WebSearchProvider):
    name = "brave"

    def __init__(
        self,
        http: HttpClient,
        api_key: str,
        country: str = "us",
        language: str = "en",
    ) -> None:
        self.http = http
        self.api_key = api_key
        self.country = country
        self.language = language

    def search(self, query: str, max_results: int) -> list[WebSearchResult]:
        data = self.http.get_json(
            "https://api.search.brave.com/res/v1/web/search",
            params={
                "q": query,
                "count": min(max(max_results, 1), 20),
                "country": self.country,
                "search_lang": self.language,
            },
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
        )
        results: list[WebSearchResult] = []
        for idx, item in enumerate(((data.get("web") or {}).get("results") or []), start=1):
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not url:
                continue
            results.append(
                WebSearchResult(
                    source=self.name,
                    query=query,
                    rank=idx,
                    url=str(url),
                    title=strip_markup(item.get("title")),
                    snippet=strip_markup(item.get("description")),
                )
            )
        return results


class SerpApiSearchProvider(WebSearchProvider):
    name = "serpapi"

    def __init__(
        self,
        http: HttpClient,
        api_key: str,
        engine: str = "google",
    ) -> None:
        self.http = http
        self.api_key = api_key
        self.engine = engine

    def search(self, query: str, max_results: int) -> list[WebSearchResult]:
        data = self.http.get_json(
            "https://serpapi.com/search.json",
            params={
                "engine": self.engine,
                "q": query,
                "num": min(max(max_results, 1), 20),
                "api_key": self.api_key,
            },
            headers={"Accept": "application/json"},
        )
        results: list[WebSearchResult] = []
        for idx, item in enumerate(data.get("organic_results") or [], start=1):
            if not isinstance(item, dict):
                continue
            url = item.get("link") or item.get("redirect_link")
            if not url:
                continue
            results.append(
                WebSearchResult(
                    source=self.name,
                    query=query,
                    rank=safe_int(item.get("position")) or idx,
                    url=str(url),
                    title=strip_markup(item.get("title")),
                    snippet=strip_markup(item.get("snippet")),
                )
            )
        return results


class WebPdfSearcher:
    def __init__(
        self,
        providers: list[WebSearchProvider],
        max_results: int = 5,
        queries_per_candidate: int = 3,
    ) -> None:
        self.providers = providers
        self.max_results = max_results
        self.queries_per_candidate = queries_per_candidate
        self._cache: dict[tuple[str, str], list[WebSearchResult]] = {}
        self._cache_lock = threading.Lock()

    def resolve(self, candidate: Candidate) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        pdf_urls: list[str] = []
        landing_page_urls: list[str] = []
        attempts: list[dict[str, Any]] = []
        queries = build_web_pdf_queries(candidate, self.queries_per_candidate)
        if not queries:
            return pdf_urls, landing_page_urls, attempts

        for provider in self.providers:
            for query in queries:
                cache_key = (provider.name, query)
                try:
                    with self._cache_lock:
                        results = self._cache.get(cache_key)
                    if results is None:
                        fresh_results = provider.search(query, self.max_results)
                        with self._cache_lock:
                            results = self._cache.setdefault(cache_key, fresh_results)
                    accepted = 0
                    for result in results:
                        if not web_result_matches_candidate(candidate, result):
                            continue
                        accepted += 1
                        if looks_like_pdf_url(result.url):
                            add_url(pdf_urls, result.url)
                        else:
                            add_url(landing_page_urls, result.url)
                    attempts.append(
                        {
                            "source": f"web_search:{provider.name}",
                            "status": "resolved" if accepted else "no_matching_result",
                            "query": query,
                            "result_count": len(results),
                            "accepted_count": accepted,
                        }
                    )
                except Exception as exc:  # pragma: no cover - network defensive path
                    attempts.append(
                        {
                            "source": f"web_search:{provider.name}",
                            "status": "error",
                            "query": query,
                            "reason": str(exc),
                        }
                    )
        return pdf_urls, landing_page_urls, attempts


def build_web_pdf_searcher(config: dict[str, Any], http: HttpClient) -> WebPdfSearcher | None:
    raw_sources = config.get("web_search_sources") or []
    if isinstance(raw_sources, str):
        raw_sources = parse_sources(raw_sources)
    sources = [normalize_source_name(str(source)) for source in raw_sources]
    providers: list[WebSearchProvider] = []
    for source in sources:
        if source == "brave":
            api_key = config.get("brave_search_api_key") or os.getenv("BRAVE_SEARCH_API_KEY")
            if api_key:
                providers.append(
                    BraveSearchProvider(
                        http,
                        api_key=str(api_key),
                        country=str(config.get("web_search_country") or "us"),
                        language=str(config.get("web_search_language") or "en"),
                    )
                )
            else:
                LOGGER.warning("Skipping Brave web search; no BRAVE_SEARCH_API_KEY configured.")
        elif source == "serpapi":
            api_key = config.get("serpapi_api_key") or os.getenv("SERPAPI_API_KEY")
            if api_key:
                providers.append(
                    SerpApiSearchProvider(
                        http,
                        api_key=str(api_key),
                        engine=str(config.get("serpapi_engine") or "google"),
                    )
                )
            else:
                LOGGER.warning("Skipping SerpAPI web search; no SERPAPI_API_KEY configured.")
        else:
            raise ValueError(f"Unknown web search source: {source}")

    if not providers:
        return None
    return WebPdfSearcher(
        providers,
        max_results=int(config.get("web_search_max_results") or 5),
        queries_per_candidate=int(config.get("web_search_queries_per_candidate") or 3),
    )


class SearchClient:
    name = "base"

    def search(
        self,
        query: SearchQuery,
        top_k: int,
        year_from: int | None,
        year_to: int | None,
    ) -> list[Candidate]:
        raise NotImplementedError


class OpenAlexClient(SearchClient):
    name = "openalex"

    def __init__(self, http: HttpClient, email: str | None = None) -> None:
        self.http = http
        self.email = email

    def search(
        self,
        query: SearchQuery,
        top_k: int,
        year_from: int | None,
        year_to: int | None,
    ) -> list[Candidate]:
        params: dict[str, Any] = {
            "search": query.text,
            "per-page": min(max(top_k, 1), 200),
        }
        if self.email:
            params["mailto"] = self.email
        filters: list[str] = []
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        data = self.http.get_json("https://api.openalex.org/works", params=params)
        candidates: list[Candidate] = []
        for item in data.get("results", []):
            candidate = self._candidate_from_work(item, query)
            if candidate and year_in_range(candidate.year, year_from, year_to):
                candidates.append(candidate)
        return candidates

    def _candidate_from_work(self, item: dict[str, Any], query: SearchQuery) -> Candidate | None:
        title = strip_markup(item.get("display_name"))
        if not title:
            return None
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = source.get("display_name")
        if not venue:
            host_venue = item.get("host_venue") or {}
            venue = host_venue.get("display_name")

        pdf_urls: list[str] = []
        landing_urls: list[str] = []
        add_url(pdf_urls, primary_location.get("pdf_url"))
        open_access = item.get("open_access") or {}
        oa_url = open_access.get("oa_url")
        landing_page = primary_location.get("landing_page_url") or doi_to_url(
            normalize_doi(item.get("doi"))
        )
        if looks_like_pdf_url(oa_url):
            add_url(pdf_urls, oa_url)
        elif oa_url:
            add_url(landing_urls, oa_url)
            if not landing_page:
                landing_page = oa_url
        add_url(landing_urls, landing_page)
        for location in ensure_list(item.get("locations")):
            if isinstance(location, dict):
                add_url(pdf_urls, location.get("pdf_url"))
                add_url(landing_urls, location.get("landing_page_url"))
        doi = normalize_doi(item.get("doi"))
        add_doi_pdf_urls(pdf_urls, doi)
        ids = item.get("ids") or {}
        source_ids = {self.name: str(item.get("id"))} if item.get("id") else {}
        if isinstance(ids, dict):
            for key, value in ids.items():
                if value:
                    source_ids[f"{self.name}:{key}"] = str(value)
                    add_identifier_pdf_urls(pdf_urls, str(value), kind=str(key))

        authors = []
        for authorship in ensure_list(item.get("authorships")):
            author = authorship.get("author") if isinstance(authorship, dict) else None
            if isinstance(author, dict) and author.get("display_name"):
                authors.append(author["display_name"])

        candidate = Candidate(
            title=title,
            authors=dedupe_preserve(authors),
            year=safe_int(item.get("publication_year")),
            doi=doi,
            abstract=reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
            venue=venue,
            landing_page_url=landing_page,
            source_apis=[self.name],
            source_ids=source_ids,
            candidate_pdf_urls=dedupe_preserve(pdf_urls),
            candidate_landing_page_urls=landing_urls,
            source_score=safe_float(item.get("relevance_score")),
            citation_count=safe_int(item.get("cited_by_count")),
            discovered_via=[
                {
                    "source": self.name,
                    "query": query.text,
                    "bucket": query.bucket,
                    "source_id": item.get("id"),
                    "source_score": item.get("relevance_score"),
                }
            ],
        )
        return candidate


def reconstruct_openalex_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not isinstance(index, dict):
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in ensure_list(positions):
            int_position = safe_int(position)
            if int_position is not None:
                positioned.append((int_position, word))
    if not positioned:
        return None
    positioned.sort(key=lambda item: item[0])
    return normalize_whitespace(" ".join(word for _, word in positioned))


class SemanticScholarClient(SearchClient):
    name = "semantic_scholar"

    def __init__(self, http: HttpClient, api_key: str | None = None) -> None:
        self.http = http
        self.api_key = api_key

    def search(
        self,
        query: SearchQuery,
        top_k: int,
        year_from: int | None,
        year_to: int | None,
    ) -> list[Candidate]:
        fields = ",".join(
            [
                "paperId",
                "title",
                "abstract",
                "year",
                "authors",
                "externalIds",
                "url",
                "venue",
                "openAccessPdf",
                "citationCount",
                "publicationDate",
            ]
        )
        params = {"query": query.text, "limit": min(max(top_k, 1), 100), "fields": fields}
        headers = {"x-api-key": self.api_key} if self.api_key else None
        data = self.http.get_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=headers,
        )

        candidates: list[Candidate] = []
        for item in data.get("data", []):
            candidate = self._candidate_from_paper(item, query)
            if candidate and year_in_range(candidate.year, year_from, year_to):
                candidates.append(candidate)
        return candidates

    def _candidate_from_paper(
        self, item: dict[str, Any], query: SearchQuery
    ) -> Candidate | None:
        title = strip_markup(item.get("title"))
        if not title:
            return None
        external_ids = item.get("externalIds") or {}
        pdf_urls: list[str] = []
        landing_urls: list[str] = []
        open_pdf = item.get("openAccessPdf") or {}
        if isinstance(open_pdf, dict):
            add_url(pdf_urls, open_pdf.get("url"))
            if not has_pdf_file_hint(open_pdf.get("url")):
                add_url(landing_urls, open_pdf.get("url"))
        add_url(landing_urls, item.get("url"))
        if isinstance(external_ids, dict):
            add_doi_pdf_urls(pdf_urls, normalize_doi(external_ids.get("DOI")))
            for key, value in external_ids.items():
                add_identifier_pdf_urls(pdf_urls, str(value), kind=str(key))
        authors = [
            author["name"]
            for author in ensure_list(item.get("authors"))
            if isinstance(author, dict) and author.get("name")
        ]
        paper_id = item.get("paperId")
        source_ids = {self.name: str(paper_id)} if paper_id else {}
        if isinstance(external_ids, dict):
            for key, value in external_ids.items():
                if value:
                    source_ids[f"{self.name}:{key}"] = str(value)
        return Candidate(
            title=title,
            authors=dedupe_preserve(authors),
            year=safe_int(item.get("year")),
            doi=normalize_doi(external_ids.get("DOI")),
            abstract=strip_markup(item.get("abstract")),
            venue=item.get("venue"),
            landing_page_url=item.get("url"),
            source_apis=[self.name],
            source_ids=source_ids,
            candidate_pdf_urls=dedupe_preserve(pdf_urls),
            candidate_landing_page_urls=landing_urls,
            citation_count=safe_int(item.get("citationCount")),
            discovered_via=[
                {
                    "source": self.name,
                    "query": query.text,
                    "bucket": query.bucket,
                    "source_id": paper_id,
                }
            ],
        )


class CrossrefClient(SearchClient):
    name = "crossref"

    def __init__(self, http: HttpClient, email: str | None = None) -> None:
        self.http = http
        self.email = email

    def search(
        self,
        query: SearchQuery,
        top_k: int,
        year_from: int | None,
        year_to: int | None,
    ) -> list[Candidate]:
        params: dict[str, Any] = {
            "query.bibliographic": query.text,
            "rows": min(max(top_k, 1), 100),
            "sort": "relevance",
            "order": "desc",
        }
        if self.email:
            params["mailto"] = self.email
        filters: list[str] = []
        if year_from:
            filters.append(f"from-pub-date:{year_from}-01-01")
        if year_to:
            filters.append(f"until-pub-date:{year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        data = self.http.get_json("https://api.crossref.org/works", params=params)
        items = (data.get("message") or {}).get("items") or []
        candidates: list[Candidate] = []
        for item in items:
            candidate = self._candidate_from_work(item, query)
            if candidate and year_in_range(candidate.year, year_from, year_to):
                candidates.append(candidate)
        return candidates

    def _candidate_from_work(self, item: dict[str, Any], query: SearchQuery) -> Candidate | None:
        title = first_string(item.get("title"))
        title = strip_markup(title)
        if not title:
            return None
        pdf_urls: list[str] = []
        landing_urls: list[str] = []
        for link in ensure_list(item.get("link")):
            if not isinstance(link, dict):
                continue
            content_type = str(link.get("content-type") or "").lower()
            url = link.get("URL")
            if "pdf" in content_type or looks_like_pdf_url(url):
                add_url(pdf_urls, url)
            elif url:
                add_url(landing_urls, url)

        authors: list[str] = []
        for author in ensure_list(item.get("author")):
            if not isinstance(author, dict):
                continue
            name_parts = [author.get("given"), author.get("family")]
            name = normalize_whitespace(" ".join(part for part in name_parts if part))
            if not name:
                name = author.get("name") or ""
            if name:
                authors.append(name)

        doi = normalize_doi(item.get("DOI"))
        add_doi_pdf_urls(pdf_urls, doi)
        add_url(landing_urls, item.get("URL") or doi_to_url(doi))
        return Candidate(
            title=title,
            authors=dedupe_preserve(authors),
            year=first_date_part_year(item.get("issued")),
            doi=doi,
            abstract=strip_markup(item.get("abstract")),
            venue=first_string(item.get("container-title")),
            landing_page_url=item.get("URL") or doi_to_url(doi),
            source_apis=[self.name],
            source_ids={self.name: doi} if doi else {},
            candidate_pdf_urls=dedupe_preserve(pdf_urls),
            candidate_landing_page_urls=landing_urls,
            source_score=safe_float(item.get("score")),
            discovered_via=[
                {
                    "source": self.name,
                    "query": query.text,
                    "bucket": query.bucket,
                    "source_id": doi,
                    "source_score": item.get("score"),
                }
            ],
        )


def first_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
    return None


class EuropePmcClient(SearchClient):
    name = "europe_pmc"

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def search(
        self,
        query: SearchQuery,
        top_k: int,
        year_from: int | None,
        year_to: int | None,
    ) -> list[Candidate]:
        params = {
            "query": query.text,
            "format": "json",
            "pageSize": min(max(top_k, 1), 100),
            "resultType": "core",
        }
        data = self.http.get_json(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params=params,
        )
        results = ((data.get("resultList") or {}).get("result")) or []
        candidates: list[Candidate] = []
        for item in results:
            candidate = self._candidate_from_result(item, query)
            if candidate and year_in_range(candidate.year, year_from, year_to):
                candidates.append(candidate)
        return candidates

    def _candidate_from_result(
        self, item: dict[str, Any], query: SearchQuery
    ) -> Candidate | None:
        title = strip_markup(item.get("title"))
        if not title:
            return None
        pdf_urls: list[str] = []
        landing_urls: list[str] = []
        full_text_list = (item.get("fullTextUrlList") or {}).get("fullTextUrl")
        for full_text in ensure_list(full_text_list):
            if not isinstance(full_text, dict):
                continue
            url = full_text.get("url")
            style = str(full_text.get("documentStyle") or "").lower()
            if style == "pdf" or looks_like_pdf_url(url):
                add_url(pdf_urls, url)
            else:
                add_url(landing_urls, url)

        authors = parse_europe_pmc_authors(item.get("authorString"))
        source = item.get("source")
        source_id = item.get("id")
        landing_page = doi_to_url(normalize_doi(item.get("doi")))
        if not landing_page and source and source_id:
            landing_page = f"https://europepmc.org/article/{source}/{source_id}"
        add_url(landing_urls, landing_page)
        doi = normalize_doi(item.get("doi"))
        add_doi_pdf_urls(pdf_urls, doi)
        if source and source.upper() == "PMC":
            add_identifier_pdf_urls(pdf_urls, str(source_id), kind="pmc")

        return Candidate(
            title=title,
            authors=authors,
            year=safe_int(item.get("pubYear")),
            doi=doi,
            abstract=strip_markup(item.get("abstractText")),
            venue=item.get("journalTitle"),
            landing_page_url=landing_page,
            source_apis=[self.name],
            source_ids={self.name: f"{source}:{source_id}"}
            if source and source_id
            else {},
            candidate_pdf_urls=dedupe_preserve(pdf_urls),
            candidate_landing_page_urls=landing_urls,
            citation_count=safe_int(item.get("citedByCount")),
            discovered_via=[
                {
                    "source": self.name,
                    "query": query.text,
                    "bucket": query.bucket,
                    "source_id": f"{source}:{source_id}" if source and source_id else source_id,
                }
            ],
        )


def parse_europe_pmc_authors(value: str | None) -> list[str]:
    if not value:
        return []
    value = value.rstrip(".")
    if "; " in value:
        return dedupe_preserve([part.strip() for part in value.split(";")])
    return dedupe_preserve([part.strip() for part in value.split(",") if part.strip()])


def build_clients(config: dict[str, Any], http: HttpClient) -> list[SearchClient]:
    raw_sources = config.get("sources") or DEFAULT_SOURCES
    if isinstance(raw_sources, str):
        raw_sources = parse_sources(raw_sources)
    clients: list[SearchClient] = []
    for source in raw_sources:
        source_key = normalize_source_name(source)
        if source_key == "openalex":
            clients.append(OpenAlexClient(http, email=config.get("email")))
        elif source_key == "semantic_scholar":
            clients.append(
                SemanticScholarClient(
                    http,
                    api_key=config.get("semantic_scholar_api_key")
                    or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
                )
            )
        elif source_key == "crossref":
            clients.append(CrossrefClient(http, email=config.get("email")))
        elif source_key == "europe_pmc":
            clients.append(EuropePmcClient(http))
        else:
            raise ValueError(f"Unknown source: {source}")
    return clients


def collect_core_pdfs(config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    raw_core_pdfs = config.get("core_pdf") or []
    if isinstance(raw_core_pdfs, str):
        raw_core_pdfs = [raw_core_pdfs]
    for value in raw_core_pdfs:
        if value:
            paths.append(Path(str(value)).expanduser())

    core_pdf_dir = config.get("core_pdf_dir")
    if core_pdf_dir:
        directory = Path(str(core_pdf_dir)).expanduser()
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Core PDF directory does not exist: {directory}")
        paths.extend(sorted(directory.glob("*.pdf")))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)

    missing = [path for path in deduped if not path.exists() or not path.is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Core PDF not found: {formatted}")
    if not deduped:
        raise ValueError("Citation harvest mode requires at least one --core-pdf or --core-pdf-dir PDF.")
    return deduped


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": stat.st_size,
    }


def citation_extraction_paths(pdf_path: Path, logs_dir: Path) -> tuple[str, Path, Path, Path]:
    paper_id = safe_pdf_stem(pdf_path)
    artifact_dir = logs_dir / "citation_extractions" / paper_id
    return (
        paper_id,
        artifact_dir,
        artifact_dir / "references.json",
        artifact_dir / "metadata.json",
    )


def load_cached_citation_extraction(
    pdf_path: Path,
    logs_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    paper_id, artifact_dir, output_path, metadata_path = citation_extraction_paths(
        pdf_path,
        logs_dir,
    )
    if not output_path.exists() or not output_path.is_file():
        return None

    try:
        output = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Could not load cached citation extraction %s: %s", output_path, exc)
        return None

    validation_errors = validate_citation_bibliography(output)
    if validation_errors:
        LOGGER.warning(
            "Cached citation extraction %s is invalid; re-extracting: %s",
            output_path,
            "; ".join(validation_errors),
        )
        return None
    if output.get("paper_id") != paper_id:
        LOGGER.warning(
            "Cached citation extraction %s has paper_id %r, expected %r; re-extracting.",
            output_path,
            output.get("paper_id"),
            paper_id,
        )
        return None

    metadata: dict[str, Any] = {}
    if metadata_path.exists() and metadata_path.is_file():
        try:
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metadata, dict):
                metadata = loaded_metadata
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Could not load cached citation metadata %s: %s", metadata_path, exc)
            return None

    for key in ["extractor", "llm_provider", "llm_model"]:
        cached_value = metadata.get(key)
        current_value = config.get("citation_extractor" if key == "extractor" else key)
        if cached_value and current_value and str(cached_value) != str(current_value):
            LOGGER.info(
                "Cached citation extraction for %s used %s=%s; current config has %s. Re-extracting.",
                pdf_path,
                key,
                cached_value,
                current_value,
            )
            return None

    metadata.setdefault("core_pdf", str(pdf_path))
    metadata.setdefault("paper_id", paper_id)
    metadata.setdefault("output_path", str(output_path))
    metadata.setdefault("metadata_path", str(metadata_path))
    metadata["loaded_from_cache"] = True

    return {
        "core_pdf": str(pdf_path),
        "paper_id": paper_id,
        "references": output.get("references", []),
        "output": output,
        "metadata": metadata,
    }


def extract_citations_from_core_pdf(
    pdf_path: Path,
    logs_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    paper_id, artifact_dir, output_path, metadata_path = citation_extraction_paths(
        pdf_path,
        logs_dir,
    )
    extractor = str(config["citation_extractor"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if extractor == "direct_pdf":
        content, extraction_metadata = build_direct_pdf_citation_content(pdf_path, paper_id)
    elif extractor == "pymupdf_markdown":
        content, extraction_metadata = build_pymupdf_markdown_citation_content(
            pdf_path,
            paper_id,
            int(config.get("max_text_chars") or 0),
        )
        markdown = str(extraction_metadata.pop("markdown"))
        markdown_path = artifact_dir / "extracted_text.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        extraction_metadata["markdown_path"] = str(markdown_path)
    else:
        raise ValueError(f"Unsupported citation extractor: {extractor}")

    start = time.perf_counter()
    call = call_llm_citation_extraction(content, config, paper_id)
    elapsed = time.perf_counter() - start

    raw_response_path = artifact_dir / "response.json"
    raw_text_path = artifact_dir / "response.txt"
    write_json(output_path, call.parsed)
    write_json(raw_response_path, call.response)
    raw_text_path.write_text(call.raw_text, encoding="utf-8")

    metadata = {
        "core_pdf": str(pdf_path),
        "paper_id": paper_id,
        "extractor": extractor,
        "llm_provider": config["llm_provider"],
        "llm_model": config["llm_model"],
        "wall_clock_seconds": round(elapsed, 6),
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "schema_valid": call.schema_valid,
        "validation_errors": call.validation_errors,
        "output_path": str(output_path),
        "raw_response_path": str(raw_response_path),
        "raw_text_path": str(raw_text_path),
    }
    metadata.update(extraction_metadata)
    write_json(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path)

    return {
        "core_pdf": str(pdf_path),
        "paper_id": paper_id,
        "references": call.parsed.get("references", []),
        "output": call.parsed,
        "metadata": metadata,
    }


def extracted_reference_to_candidate(
    reference: dict[str, Any],
    *,
    core_pdf: str,
    paper_id: str,
) -> Candidate | None:
    raw_reference = reference.get("raw_reference")
    raw_reference = normalize_whitespace(str(raw_reference)) if raw_reference else None
    title = strip_markup(reference.get("title")) or raw_reference
    if not title:
        return None

    authors = [
        normalize_whitespace(str(author))
        for author in ensure_list(reference.get("authors"))
        if normalize_whitespace(str(author))
    ]
    doi = normalize_doi(reference.get("doi"))
    url = clean_url(reference.get("url"))
    arxiv_id = normalize_arxiv_id(reference.get("arxiv_id")) or normalize_arxiv_id(url)
    pdf_urls: list[str] = []
    landing_urls: list[str] = []
    if looks_like_pdf_url(url):
        add_url(pdf_urls, url)
    else:
        add_url(landing_urls, url)
    add_doi_pdf_urls(pdf_urls, doi)
    if arxiv_id:
        add_identifier_pdf_urls(pdf_urls, arxiv_id, kind="arxiv")

    source_ids: dict[str, str] = {}
    if doi:
        source_ids["citation:doi"] = doi
    if arxiv_id:
        source_ids["citation:arxiv"] = arxiv_id

    return Candidate(
        title=title,
        authors=dedupe_preserve(authors),
        year=safe_int(reference.get("year")),
        doi=doi,
        venue=strip_markup(reference.get("venue")),
        landing_page_url=doi_to_url(doi) or (url if url and not looks_like_pdf_url(url) else None),
        source_apis=["citation_extraction"],
        source_ids=source_ids,
        candidate_pdf_urls=dedupe_preserve(pdf_urls),
        candidate_landing_page_urls=dedupe_preserve(landing_urls),
        discovered_via=[
            {
                "source": "citation_extraction",
                "core_pdf": core_pdf,
                "paper_id": paper_id,
                "ref_id": reference.get("ref_id"),
                "raw_reference": raw_reference,
            }
        ],
    )


def candidates_from_citation_extractions(extractions: list[dict[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for extraction in extractions:
        core_pdf = str(extraction.get("core_pdf") or "")
        paper_id = str(extraction.get("paper_id") or "")
        for reference in extraction.get("references") or []:
            if not isinstance(reference, dict):
                continue
            candidate = extracted_reference_to_candidate(
                reference,
                core_pdf=core_pdf,
                paper_id=paper_id,
            )
            if candidate:
                candidates.append(candidate)
    return candidates


def candidate_arxiv_id(candidate: Candidate) -> str | None:
    if candidate.doi and candidate.doi.lower().startswith("10.48550/arxiv."):
        return normalize_arxiv_id(candidate.doi.split("arxiv.", 1)[1])
    for key, value in candidate.source_ids.items():
        if "arxiv" in key.lower():
            arxiv_id = normalize_arxiv_id(value)
            if arxiv_id:
                return arxiv_id
    return None


def citation_lookup_query(candidate: Candidate) -> str | None:
    if candidate.doi:
        return candidate.doi
    arxiv_id = candidate_arxiv_id(candidate)
    if arxiv_id:
        return f"arXiv:{arxiv_id}"
    title = normalize_whitespace(candidate.title)
    if not title:
        return None
    parts = [title]
    if candidate.year:
        parts.append(str(candidate.year))
    if candidate.authors:
        parts.append(candidate.authors[0])
    return " ".join(parts)


def first_author_last_name(candidate: Candidate) -> str | None:
    if not candidate.authors:
        return None
    tokens = normalize_title(candidate.authors[0]).split()
    if not tokens:
        return None
    return tokens[-1]


def title_token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(content_tokens(left))
    right_tokens = set(content_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def citation_match_reason(seed: Candidate, result: Candidate) -> str | None:
    if seed.doi and result.doi and normalize_doi(seed.doi) == normalize_doi(result.doi):
        return "doi"

    seed_arxiv = candidate_arxiv_id(seed)
    result_arxiv = candidate_arxiv_id(result)
    if seed_arxiv and result_arxiv and seed_arxiv.lower() == result_arxiv.lower():
        return "arxiv"

    seed_title = normalize_title(seed.title)
    result_title = normalize_title(result.title)
    if seed_title and result_title and seed_title == result_title:
        return "title"

    overlap = title_token_overlap_ratio(seed.title, result.title)
    seed_author = first_author_last_name(seed)
    result_author = first_author_last_name(result)
    author_matches = bool(seed_author and result_author and seed_author == result_author)
    year_matches = bool(seed.year and result.year and seed.year == result.year)
    if overlap >= 0.8 and (author_matches or year_matches):
        if author_matches and year_matches:
            return "title_overlap_author_year"
        if author_matches:
            return "title_overlap_author"
        return "title_overlap_year"
    return None


def citation_lookup_records(candidate: Candidate) -> list[dict[str, Any]]:
    return [
        item
        for item in candidate.discovered_via
        if item.get("source") == "citation_lookup" and isinstance(item.get("attempts"), list)
    ]


def citation_lookup_resume_query(candidate: Candidate) -> str | None:
    for record in reversed(citation_lookup_records(candidate)):
        if "query" in record:
            query = record.get("query")
            return str(query) if query is not None else None
    return citation_lookup_query(candidate)


def citation_lookup_record(candidate: Candidate, query_text: str | None) -> dict[str, Any]:
    for record in reversed(citation_lookup_records(candidate)):
        if record.get("query") == query_text:
            return record
    record = {
        "source": "citation_lookup",
        "query": query_text,
        "attempts": [],
    }
    candidate.discovered_via.append(record)
    return record


def citation_lookup_attempted_sources(candidate: Candidate, query_text: str | None) -> set[str]:
    sources: set[str] = set()
    for record in citation_lookup_records(candidate):
        if record.get("query") != query_text:
            continue
        for attempt in record.get("attempts") or []:
            if isinstance(attempt, dict) and attempt.get("source"):
                sources.add(str(attempt["source"]))
    return sources


def citation_lookup_has_skip(candidate: Candidate, query_text: str | None) -> bool:
    for record in citation_lookup_records(candidate):
        if record.get("query") != query_text:
            continue
        for attempt in record.get("attempts") or []:
            if (
                isinstance(attempt, dict)
                and attempt.get("source") == "citation_lookup"
                and attempt.get("status") == "skipped"
            ):
                return True
    return False


def citation_lookup_completed_attempt_count(
    candidate: Candidate,
    clients: list[SearchClient],
) -> int:
    query_text = citation_lookup_resume_query(candidate)
    if not query_text:
        return 1 if citation_lookup_has_skip(candidate, query_text) else 0
    client_names = {client.name for client in clients}
    return len(citation_lookup_attempted_sources(candidate, query_text) & client_names)


def citation_lookup_total_attempt_count(
    candidate: Candidate,
    clients: list[SearchClient],
) -> int:
    return len(clients) if citation_lookup_resume_query(candidate) else 1


def citation_lookup_accepted_count(candidates: list[Candidate]) -> int:
    total = 0
    for candidate in candidates:
        for record in citation_lookup_records(candidate):
            for attempt in record.get("attempts") or []:
                if isinstance(attempt, dict):
                    total += int(safe_int(attempt.get("accepted_count")) or 0)
    return total


def citation_lookup_skipped_count(candidates: list[Candidate]) -> int:
    skipped = 0
    for candidate in candidates:
        query_text = citation_lookup_resume_query(candidate)
        if not query_text and citation_lookup_has_skip(candidate, query_text):
            skipped += 1
    return skipped


def citation_lookup_progress_payload(
    candidates: list[Candidate],
    clients: list[SearchClient],
    lookup_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    total_attempts = sum(citation_lookup_total_attempt_count(candidate, clients) for candidate in candidates)
    completed_attempts = sum(
        citation_lookup_completed_attempt_count(candidate, clients) for candidate in candidates
    )
    completed_candidates = sum(
        1
        for candidate in candidates
        if citation_lookup_completed_attempt_count(candidate, clients)
        >= citation_lookup_total_attempt_count(candidate, clients)
    )
    return {
        "total_candidates": len(candidates),
        "completed_candidates": completed_candidates,
        "total_lookup_attempts": total_attempts,
        "completed_lookup_attempts": completed_attempts,
        "accepted_count": citation_lookup_accepted_count(candidates),
        "error_count": len(lookup_errors),
        "skipped_count": citation_lookup_skipped_count(candidates),
    }


def enrich_citation_candidates(
    candidates: list[Candidate],
    clients: list[SearchClient],
    config: dict[str, Any],
    checkpoint: Callable[[list[Candidate], list[dict[str, Any]], dict[str, Any]], None] | None = None,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    enriched: list[Candidate] = []
    lookup_errors: list[dict[str, Any]] = [
        error for error in ensure_list(config.get("_initial_citation_lookup_errors")) if isinstance(error, dict)
    ]
    total_attempts = sum(citation_lookup_total_attempt_count(candidate, clients) for candidate in candidates)
    completed_attempts = sum(
        citation_lookup_completed_attempt_count(candidate, clients) for candidate in candidates
    )
    progress = PhaseProgressReporter(
        total_attempts,
        enabled=bool(config.get("progress")),
        label="Citation lookup",
    )
    accepted_total = citation_lookup_accepted_count(candidates)
    skipped_total = citation_lookup_skipped_count(candidates)
    progress.set_current(
        completed_attempts,
        summary=f"accepted={accepted_total} errors={len(lookup_errors)} skipped={skipped_total}",
        force=True,
    )
    LOGGER.info(
        "Looking up %d citation candidates across %d sources (%d lookup attempts).",
        len(candidates),
        len(clients),
        total_attempts,
    )
    for candidate in candidates:
        query_text = citation_lookup_resume_query(candidate)
        lookup_record = citation_lookup_record(candidate, query_text)
        lookup_attempts = lookup_record["attempts"]
        if query_text:
            query = SearchQuery(bucket="citation_lookup", text=query_text, terms=content_tokens(candidate.title))
            attempted_sources = citation_lookup_attempted_sources(candidate, query_text)
            for client in clients:
                if client.name in attempted_sources:
                    continue
                try:
                    results = client.search(
                        query,
                        top_k=int(config["top_k_per_query"]),
                        year_from=config.get("year_from"),
                        year_to=config.get("year_to"),
                    )
                    accepted = 0
                    accepted_reasons: list[str] = []
                    for result in results:
                        reason = citation_match_reason(candidate, result)
                        if not reason:
                            continue
                        accepted += 1
                        accepted_reasons.append(reason)
                        result.discovered_via = result.discovered_via + [
                            {
                                "source": "citation_lookup_match",
                                "client": client.name,
                                "query": query_text,
                                "match_reason": reason,
                            }
                        ]
                        merge_candidate_into(candidate, result)
                    accepted_total += accepted
                    lookup_attempts.append(
                        {
                            "source": client.name,
                            "query": query_text,
                            "result_count": len(results),
                            "accepted_count": accepted,
                            "accepted_reasons": dedupe_preserve(accepted_reasons),
                        }
                    )
                except Exception as exc:
                    error = {
                        "source": client.name,
                        "query": query_text,
                        "title": candidate.title,
                        "reason": str(exc),
                    }
                    lookup_errors.append(error)
                    lookup_attempts.append({**error, "status": "error"})
                attempted_sources.add(client.name)
                progress.advance(
                    summary=f"accepted={accepted_total} errors={len(lookup_errors)} skipped={skipped_total}"
                )
                if checkpoint:
                    checkpoint(
                        candidates,
                        lookup_errors,
                        citation_lookup_progress_payload(candidates, clients, lookup_errors),
                    )
        elif not citation_lookup_has_skip(candidate, query_text):
            skipped_total += 1
            lookup_attempts.append(
                {
                    "source": "citation_lookup",
                    "status": "skipped",
                    "reason": "No DOI, arXiv ID, title, or raw reference available.",
                }
            )
            progress.advance(
                summary=f"accepted={accepted_total} errors={len(lookup_errors)} skipped={skipped_total}"
            )
            if checkpoint:
                checkpoint(
                    candidates,
                    lookup_errors,
                    citation_lookup_progress_payload(candidates, clients, lookup_errors),
                )
        enriched.append(candidate)

    progress.done(summary=f"accepted={accepted_total} errors={len(lookup_errors)} skipped={skipped_total}")
    return merge_candidates(enriched), lookup_errors


def normalize_source_name(source: str) -> str:
    return source.strip().lower().replace("-", "_")


def parse_sources(value: str) -> list[str]:
    return [normalize_source_name(part) for part in value.split(",") if part.strip()]


def merge_candidates(candidates: list[Candidate]) -> list[Candidate]:
    merged: list[Candidate] = []
    by_key: dict[str, Candidate] = {}
    for candidate in candidates:
        keys = candidate_keys(candidate)
        existing = next((by_key[key] for key in keys if key in by_key), None)
        if existing is None:
            merged.append(candidate)
            for key in keys:
                by_key[key] = candidate
            continue
        merge_candidate_into(existing, candidate)
        for key in candidate_keys(existing):
            by_key[key] = existing
    return merged


def dedupe_discovered_via(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            key = json.dumps(item, sort_keys=True, default=str)
        except TypeError:
            key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def candidate_keys(candidate: Candidate) -> list[str]:
    keys: list[str] = []
    if candidate.doi:
        keys.append(f"doi:{candidate.doi}")
    arxiv_id = candidate_arxiv_id(candidate)
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id.lower()}")
    title_key = normalize_title(candidate.title)
    if title_key:
        keys.append(f"title:{title_key}")
        if candidate.year and candidate.authors:
            keys.append(
                "hint:"
                + "|".join(
                    [
                        title_key,
                        str(candidate.year),
                        normalize_title(candidate.authors[0]).split(" ")[-1],
                    ]
                )
            )
    return keys


def stable_candidate_id(candidate: Candidate) -> str:
    if candidate.candidate_id:
        return candidate.candidate_id
    identity = candidate_download_identity(candidate)
    candidate.candidate_id = "paper_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return candidate.candidate_id


def ensure_candidate_ids(candidates: list[Candidate]) -> list[Candidate]:
    for candidate in candidates:
        stable_candidate_id(candidate)
    return candidates


def partition_selected_candidates(
    candidates: list[Candidate],
    selected_candidate_ids: list[str] | None,
) -> tuple[list[Candidate], list[Candidate]]:
    if selected_candidate_ids is None:
        return candidates, []
    selected_ids = {str(value) for value in selected_candidate_ids}
    selected: list[Candidate] = []
    unselected: list[Candidate] = []
    for candidate in candidates:
        target = selected if stable_candidate_id(candidate) in selected_ids else unselected
        target.append(candidate)
    return selected, unselected


def merge_candidate_into(target: Candidate, incoming: Candidate) -> None:
    target.source_apis = dedupe_preserve(target.source_apis + incoming.source_apis)
    target.source_ids.update({key: value for key, value in incoming.source_ids.items() if value})
    target.candidate_pdf_urls = dedupe_preserve(
        target.candidate_pdf_urls + incoming.candidate_pdf_urls
    )
    target.candidate_landing_page_urls = dedupe_preserve(
        target.candidate_landing_page_urls + incoming.candidate_landing_page_urls
    )
    target.matched_keywords = dedupe_preserve(
        target.matched_keywords + incoming.matched_keywords
    )
    target.discovered_via = dedupe_discovered_via(
        target.discovered_via + incoming.discovered_via
    )
    if not target.doi and incoming.doi:
        target.doi = incoming.doi
    if not target.year and incoming.year:
        target.year = incoming.year
    if not target.authors and incoming.authors:
        target.authors = incoming.authors
    if not target.venue and incoming.venue:
        target.venue = incoming.venue
    if not target.landing_page_url and incoming.landing_page_url:
        target.landing_page_url = incoming.landing_page_url
    if incoming.abstract and (
        not target.abstract or len(incoming.abstract) > len(target.abstract)
    ):
        target.abstract = incoming.abstract
    if incoming.citation_count and (
        not target.citation_count or incoming.citation_count > target.citation_count
    ):
        target.citation_count = incoming.citation_count
    if incoming.source_score and (
        target.source_score is None or incoming.source_score > target.source_score
    ):
        target.source_score = incoming.source_score


def relevance_term_matches(term: str, text: str | None) -> bool:
    term_tokens = set(content_tokens(term))
    if not term_tokens:
        return False
    return term_tokens.issubset(set(content_tokens(text)))


def metadata_relevance_signals(candidate: Candidate) -> dict[str, float]:
    current_year = datetime.now().year
    age = max(current_year - candidate.year, 0) if candidate.year else None
    return {
        "source_relevance": min(
            math.log1p(max(candidate.source_score or 0.0, 0.0)) / 5.0,
            1.0,
        ),
        "citation_impact": min(
            math.log10(max(candidate.citation_count or 0, 0) + 1) / 4.0,
            1.0,
        ),
        "recency": max(0.0, 1.0 - age / 20.0) if age is not None else 0.0,
        "open_access": float(
            bool(candidate.candidate_pdf_urls or candidate.candidate_landing_page_urls)
        ),
        "doi": float(bool(candidate.doi)),
    }


def profile_relevance_score(
    candidate: Candidate,
    profile: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    field_text = {
        "title": candidate.title,
        "abstract": candidate.abstract or "",
        "matched_keywords": " ".join(candidate.matched_keywords),
    }
    field_weights = profile["field_weights"]
    field_weight_total = sum(field_weights.values()) or 1.0
    criterion_results: list[dict[str, Any]] = []
    positive_total = 0.0
    positive_weight = 0.0

    for criterion in profile["criteria"]:
        field_scores: dict[str, float] = {}
        matched_terms: dict[str, list[str]] = {}
        for field_name, field_weight in field_weights.items():
            matches = [
                term
                for term in criterion["terms"]
                if relevance_term_matches(term, field_text[field_name])
            ]
            field_scores[field_name] = float(bool(matches))
            if matches:
                matched_terms[field_name] = matches
        criterion_score = sum(
            field_scores[field_name] * field_weights[field_name]
            for field_name in field_weights
        ) / field_weight_total
        criterion_weight = float(criterion["weight"])
        contribution = criterion_score * criterion_weight
        positive_total += contribution
        positive_weight += criterion_weight
        criterion_results.append(
            {
                "id": criterion["id"],
                "label": criterion["label"],
                "score": round(criterion_score, 5),
                "weight": criterion_weight,
                "contribution": round(contribution, 5),
                "matched_terms": matched_terms,
            }
        )

    metadata_values = metadata_relevance_signals(candidate)
    metadata_results: dict[str, dict[str, float]] = {}
    for signal, signal_weight in profile["metadata_weights"].items():
        value = metadata_values[signal]
        contribution = value * signal_weight
        positive_total += contribution
        positive_weight += signal_weight
        metadata_results[signal] = {
            "value": round(value, 5),
            "weight": signal_weight,
            "contribution": round(contribution, 5),
        }

    base_score = positive_total / positive_weight if positive_weight else 0.0
    exclusion_results: list[dict[str, Any]] = []
    exclusion_text = " ".join([candidate.title, candidate.abstract or "", candidate.venue or ""])
    penalty = 0.0
    for exclusion in profile["exclusions"]:
        if relevance_term_matches(exclusion["term"], exclusion_text):
            penalty += float(exclusion["penalty"])
            exclusion_results.append(exclusion)
    final_score = max(0.0, min(1.0, base_score - penalty))
    breakdown = {
        "mode": "generated_profile",
        "profile_title": profile["title"],
        "base_score": round(base_score, 5),
        "exclusion_penalty": round(penalty, 5),
        "criteria": criterion_results,
        "metadata": metadata_results,
        "matched_exclusions": exclusion_results,
        "final_score": round(final_score, 5),
    }
    return round(final_score, 5), breakdown


def rank_candidates(
    candidates: list[Candidate],
    terms: ExtractedTerms,
    relevance_profile: dict[str, Any] | None = None,
) -> list[Candidate]:
    if relevance_profile is not None:
        for candidate in candidates:
            candidate.relevance_score, candidate.relevance_score_breakdown = (
                profile_relevance_score(candidate, relevance_profile)
            )
        return sorted(
            candidates,
            key=lambda item: (item.relevance_score, item.year or 0, bool(item.doi)),
            reverse=True,
        )

    vocabulary = set(terms.keywords[:30])
    for phrase in terms.noun_phrases[:20]:
        vocabulary.update(phrase.split())
    if terms.title:
        vocabulary.update(content_tokens(terms.title))
    vocabulary = {token for token in vocabulary if token not in STOPWORDS and len(token) > 2}

    current_year = datetime.now().year
    for candidate in candidates:
        title_tokens = set(content_tokens(candidate.title))
        abstract_tokens = set(content_tokens(candidate.abstract))
        source_tokens = title_tokens | abstract_tokens
        overlap = len(vocabulary & source_tokens)
        title_overlap = len(vocabulary & title_tokens)
        score = 0.0
        if vocabulary:
            score += 0.55 * (title_overlap / max(1, min(len(vocabulary), len(title_tokens))))
            score += 0.35 * (overlap / max(1, min(len(vocabulary), len(source_tokens))))
        if candidate.source_score:
            score += min(math.log1p(max(candidate.source_score, 0.0)) / 10.0, 0.2)
        if candidate.citation_count:
            score += min(math.log10(candidate.citation_count + 1) / 20.0, 0.15)
        if candidate.year:
            age = max(current_year - candidate.year, 0)
            score += max(0.0, 0.08 - age * 0.005)
        if candidate.candidate_pdf_urls or candidate.candidate_landing_page_urls:
            score += 0.1
        if candidate.doi:
            score += 0.03
        candidate.relevance_score = round(score, 5)
        candidate.relevance_score_breakdown = {"mode": "legacy"}
    return sorted(
        candidates,
        key=lambda item: (item.relevance_score, item.year or 0, bool(item.doi)),
        reverse=True,
    )


class PdfResolver:
    def __init__(
        self,
        http: HttpClient,
        unpaywall_email: str | None = None,
        web_searcher: WebPdfSearcher | None = None,
    ) -> None:
        self.http = http
        self.unpaywall_email = unpaywall_email
        self.web_searcher = web_searcher

    def resolve(self, candidate: Candidate, use_web_search: bool = True) -> list[str]:
        urls: list[str] = []
        landing_pages: list[str] = []
        attempts: list[dict[str, Any]] = []
        for url in candidate.candidate_pdf_urls:
            add_url(urls, url)
            if not has_pdf_file_hint(url):
                add_url(landing_pages, url)
            attempts.append({"source": "source_metadata", "status": "candidate", "url": url})

        for url in inferred_pdf_urls(candidate):
            add_url(urls, url)
            attempts.append({"source": "identifier", "status": "candidate", "url": url})

        for url in candidate.candidate_landing_page_urls:
            add_url(landing_pages, url)
        add_url(landing_pages, candidate.landing_page_url)
        add_url(landing_pages, doi_to_url(candidate.doi))

        if candidate.doi and self.unpaywall_email:
            try:
                unpaywall_pdf_urls, unpaywall_landing_urls = self._resolve_unpaywall(candidate.doi)
                for url in unpaywall_pdf_urls:
                    add_url(urls, url)
                for url in unpaywall_landing_urls:
                    add_url(landing_pages, url)
                attempts.append(
                    {
                        "source": "unpaywall",
                        "status": "resolved"
                        if unpaywall_pdf_urls or unpaywall_landing_urls
                        else "no_pdf_url",
                        "url_count": len(unpaywall_pdf_urls),
                        "landing_page_count": len(unpaywall_landing_urls),
                    }
                )
            except Exception as exc:  # pragma: no cover - network defensive path
                attempts.append(
                    {"source": "unpaywall", "status": "error", "reason": str(exc)}
                )
        elif candidate.doi:
            attempts.append(
                {
                    "source": "unpaywall",
                    "status": "skipped",
                    "reason": "No Unpaywall email configured.",
                }
            )

        if self.web_searcher and use_web_search:
            web_pdf_urls, web_landing_pages, web_attempts = self.web_searcher.resolve(candidate)
            for url in web_pdf_urls:
                add_url(urls, url)
            for url in web_landing_pages:
                add_url(landing_pages, url)
            attempts.extend(web_attempts)

        for landing_page_url in landing_pages:
            try:
                landing_urls = self._resolve_landing_page(landing_page_url)
                for url in landing_urls:
                    add_url(urls, url)
                attempts.append(
                    {
                        "source": "landing_page",
                        "status": "resolved" if landing_urls else "no_pdf_url",
                        "url_count": len(landing_urls),
                        "landing_page_url": landing_page_url,
                    }
                )
            except Exception as exc:  # pragma: no cover - network defensive path
                attempts.append(
                    {
                        "source": "landing_page",
                        "status": "error",
                        "reason": str(exc),
                        "landing_page_url": landing_page_url,
                    }
                )

        candidate.pdf_resolution = {
            "resolved_at": utc_now(),
            "urls": urls,
            "landing_page_urls": landing_pages,
            "attempts": attempts,
        }
        return urls

    def _resolve_unpaywall(self, doi: str) -> tuple[list[str], list[str]]:
        encoded_doi = urllib.parse.quote(doi, safe="/")
        data = self.http.get_json(
            f"https://api.unpaywall.org/v2/{encoded_doi}",
            params={"email": self.unpaywall_email},
        )
        urls: list[str] = []
        landing_page_urls: list[str] = []
        if data.get("is_oa") is False:
            return urls, landing_page_urls
        locations = [data.get("best_oa_location")] + ensure_list(data.get("oa_locations"))
        for location in locations:
            if not isinstance(location, dict):
                continue
            add_url(urls, location.get("url_for_pdf"))
            fallback = location.get("url")
            if looks_like_pdf_url(fallback):
                add_url(urls, fallback)
            else:
                add_url(landing_page_urls, fallback)
            add_url(landing_page_urls, location.get("url_for_landing_page"))
        return urls, landing_page_urls

    def _resolve_landing_page(self, landing_page_url: str) -> list[str]:
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }
        body, headers, final_url = self.http.request_bytes(
            landing_page_url,
            headers=headers,
        )
        if not body:
            return []
        content_type = str(headers.get("Content-Type") or headers.get("content-type") or "")
        if is_pdf_response(content_type, body[:8192]):
            return [final_url]
        try:
            html_text = body.decode("utf-8", errors="replace")
        except Exception:
            html_text = body.decode("latin-1", errors="replace")
        return extract_pdf_urls_from_html(html_text, final_url or landing_page_url)


class PdfDownloader:
    def __init__(
        self,
        http: HttpClient,
        output_dir: Path,
        max_pdf_mb: int = 80,
        force_download: bool = False,
        allow_abstract_fallback: bool | None = None,
        abstract_fallback: bool | None = None,
    ) -> None:
        self.http = http
        self.output_dir = output_dir
        self.max_bytes = max_pdf_mb * 1024 * 1024
        self.force_download = force_download
        if allow_abstract_fallback is None:
            allow_abstract_fallback = True if abstract_fallback is None else abstract_fallback
        self.allow_abstract_fallback = allow_abstract_fallback
        self.abstract_fallback = allow_abstract_fallback

    def download(self, candidate: Candidate, urls: list[str]) -> dict[str, Any]:
        filename = build_pdf_filename(candidate)
        destination = self.output_dir / filename
        attempts: list[dict[str, Any]] = []
        retrying_abstract_only = candidate.download.get("status") == "retry_abstract_only"
        if destination.exists() and not self.force_download and not retrying_abstract_only:
            result = {
                "status": "already_exists",
                "path": str(destination),
                "filename": filename,
                "attempts": attempts,
            }
            candidate.download = result
            return result

        for url in urls:
            try:
                result = self.http.download_pdf(url, destination, max_bytes=self.max_bytes)
                result["filename"] = filename
                result["attempts"] = attempts + [{"url": url, "status": "downloaded"}]
                candidate.download = result
                return result
            except Exception as exc:  # pragma: no cover - network defensive path
                attempts.append({"url": url, "status": "failed", "reason": str(exc)})

        if self.allow_abstract_fallback and candidate.abstract:
            try:
                bytes_written = self._write_abstract_pdf(
                    destination,
                    title=candidate.title,
                    authors=candidate.authors,
                    abstract=candidate.abstract,
                )
                result = {
                    "status": "abstract_only",
                    "filename": filename,
                    "path": str(destination),
                    "attempts": attempts
                    + [
                        {
                            "source": "abstract_fallback",
                            "status": "generated",
                            "reason": "Saved abstract as PDF because no PDF was available.",
                        }
                    ],
                    "bytes": bytes_written,
                    "content_type": "application/pdf",
                }
                candidate.download = result
                return result
            except Exception as exc:  # pragma: no cover - PDF generation error
                attempts.append(
                    {
                        "source": "abstract_fallback",
                        "status": "failed",
                        "reason": str(exc),
                    }
                )

        result = {
            "status": "failed",
            "filename": filename,
            "attempts": attempts,
            "reason": "No candidate PDF URL downloaded successfully."
            if urls
            else "No candidate PDF URL available.",
        }
        candidate.download = result
        return result

    def _write_abstract_pdf(
        self,
        destination: Path,
        title: str,
        authors: list[str],
        abstract: str,
    ) -> int:
        lines: list[str] = []
        if title:
            lines.append(title)
            lines.append("")
        if authors:
            lines.append(", ".join(authors))
            lines.append("")
        lines.append("Abstract:")
        lines.append("")
        for paragraph in abstract.strip().splitlines():
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue
            lines.extend(_wrap_text(paragraph, width=80))
        content_line = []
        for line in lines:
            content_line.append(f"({ _pdf_escape(line) }) Tj")
            content_line.append("T*")
        content_stream = "BT\n/F1 12 Tf\n72 760 Td\n" + "\n".join(content_line) + "\nET\n"
        stream_bytes = content_stream.encode("latin-1", errors="replace")

        objs: list[bytes] = []
        objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        objs.append(
            b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        )
        objs.append(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
        )
        objs.append(
            f"5 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        body = b"%PDF-1.4\n" + b"".join(objs)
        offsets = []
        current = len(b"%PDF-1.4\n")
        for obj in objs:
            offsets.append(current)
            current += len(obj)
        xref = [b"xref\n0 6\n0000000000 65535 f \n"]
        for offset in offsets:
            xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
        startxref = len(body)
        trailer = (
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
            + str(startxref).encode("ascii")
            + b"\n%%EOF\n"
        )
        destination.write_bytes(body + b"".join(xref) + trailer)
        return destination.stat().st_size


def should_use_web_search_for_candidate(
    web_searcher: WebPdfSearcher | None,
    max_web_search_candidates: int | None,
    web_search_count: int,
) -> bool:
    if not web_searcher:
        return False
    if max_web_search_candidates == 0:
        return False
    if (
        max_web_search_candidates is not None
        and web_search_count >= max_web_search_candidates
    ):
        return False
    return True


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def download_status_counts(candidates: list[Candidate]) -> Counter[str]:
    return Counter(
        str(candidate.download.get("status") or "pending")
        for candidate in candidates
    )


class ProgressReporter:
    def __init__(
        self,
        total: int,
        *,
        enabled: bool = True,
        label: str = "Candidates",
        min_interval: float = 0.5,
    ) -> None:
        self.total = max(total, 0)
        self.enabled = enabled and self.total > 0 and sys.stderr.isatty()
        self.label = label
        self.min_interval = min_interval
        self.started_at = time.monotonic()
        self.current = 0
        self._last_render_at = 0.0
        self._line_length = 0

    def set_current(
        self,
        current: int,
        candidates: list[Candidate],
        *,
        force: bool = False,
    ) -> None:
        self.current = max(0, min(current, self.total))
        self.render(candidates, force=force)

    def advance(self, candidates: list[Candidate], step: int = 1) -> None:
        self.current = max(0, min(self.current + step, self.total))
        self.render(candidates)

    def render(self, candidates: list[Candidate], *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and self.current < self.total and now - self._last_render_at < self.min_interval:
            return
        self._last_render_at = now
        elapsed = max(now - self.started_at, 0.001)
        rate = self.current / elapsed
        remaining = self.total - self.current
        eta = remaining / rate if rate > 0 else 0
        percent = (self.current / self.total * 100.0) if self.total else 100.0
        counts = download_status_counts(candidates)
        summary_parts = []
        for key in ["downloaded", "already_exists", "abstract_only", "failed", "not_attempted"]:
            count = counts.get(key, 0)
            if count:
                summary_parts.append(f"{key}={count}")
        summary = " ".join(summary_parts) if summary_parts else "pending"
        line = (
            f"{self.label}: {self.current}/{self.total} "
            f"({percent:5.1f}%) | {rate * 60:5.1f}/min | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} | {summary}"
        )
        padding = " " * max(0, self._line_length - len(line))
        sys.stderr.write("\r" + line + padding)
        sys.stderr.flush()
        self._line_length = len(line)

    def clear(self) -> None:
        if not self.enabled or not self._line_length:
            return
        sys.stderr.write("\r" + " " * self._line_length + "\r")
        sys.stderr.flush()
        self._line_length = 0

    def done(self, candidates: list[Candidate]) -> None:
        if not self.enabled:
            return
        self.current = self.total
        self.render(candidates, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()
        self._line_length = 0


class PhaseProgressReporter:
    def __init__(
        self,
        total: int,
        *,
        enabled: bool = True,
        label: str = "Progress",
        min_interval: float = 0.5,
        log_interval: float = 30.0,
    ) -> None:
        self.total = max(total, 0)
        self.enabled = enabled and self.total > 0
        self.tty = sys.stderr.isatty()
        self.label = label
        self.min_interval = min_interval
        self.log_interval = log_interval
        self.started_at = time.monotonic()
        self.current = 0
        self.summary = ""
        self._last_render_at = 0.0
        self._last_log_at = 0.0
        self._line_length = 0

    def set_current(
        self,
        current: int,
        *,
        summary: str | None = None,
        force: bool = False,
    ) -> None:
        self.current = max(0, min(current, self.total))
        if summary is not None:
            self.summary = summary
        self.render(force=force)

    def advance(self, step: int = 1, *, summary: str | None = None) -> None:
        self.current = max(0, min(self.current + step, self.total))
        if summary is not None:
            self.summary = summary
        self.render()

    def progress_line(self) -> str:
        now = time.monotonic()
        elapsed = max(now - self.started_at, 0.001)
        rate = self.current / elapsed
        remaining = self.total - self.current
        eta = remaining / rate if rate > 0 else 0
        percent = (self.current / self.total * 100.0) if self.total else 100.0
        line = (
            f"{self.label}: {self.current}/{self.total} "
            f"({percent:5.1f}%) | {rate * 60:5.1f}/min | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
        )
        if self.summary:
            line += f" | {self.summary}"
        return line

    def render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if self.tty:
            if not force and self.current < self.total and now - self._last_render_at < self.min_interval:
                return
            self._last_render_at = now
            line = self.progress_line()
            padding = " " * max(0, self._line_length - len(line))
            sys.stderr.write("\r" + line + padding)
            sys.stderr.flush()
            self._line_length = len(line)
            return

        if force or self.current >= self.total or now - self._last_log_at >= self.log_interval:
            self._last_log_at = now
            LOGGER.info(self.progress_line())

    def done(self, *, summary: str | None = None) -> None:
        if not self.enabled:
            return
        self.current = self.total
        if summary is not None:
            self.summary = summary
        self.render(force=True)
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._line_length = 0


def resolve_and_download_candidate(
    candidate: Candidate,
    resolver: PdfResolver,
    downloader: PdfDownloader,
    use_web_search: bool,
) -> dict[str, Any]:
    try:
        urls = resolver.resolve(candidate, use_web_search=use_web_search)
        return downloader.download(candidate, urls)
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        result = {
            "status": "failed",
            "filename": build_pdf_filename(candidate),
            "attempts": [],
            "reason": str(exc),
        }
        candidate.download = result
        return result


def mark_downloads_not_attempted(
    candidates: list[Candidate],
    reason: str,
    ledger: DownloadLogLedger | None = None,
    preserve_existing: bool = False,
) -> None:
    records: list[tuple[Candidate, dict[str, Any]]] = []
    for candidate in candidates:
        if not (preserve_existing and candidate.download):
            candidate.download = {
                "status": "not_attempted",
                "reason": reason,
                "filename": build_pdf_filename(candidate),
            }
        records.append((candidate, candidate.download))
    if ledger:
        ledger.record_many(records)


def mark_candidates_not_selected(
    candidates: list[Candidate],
    ledger: DownloadLogLedger | None = None,
) -> None:
    records: list[tuple[Candidate, dict[str, Any]]] = []
    for candidate in candidates:
        candidate.download = {
            "status": "not_selected",
            "reason": "not_selected",
            "filename": build_pdf_filename(candidate),
        }
        records.append((candidate, candidate.download))
    if ledger:
        ledger.record_many(records)


def log_download_result(
    result: dict[str, Any],
    candidate: Candidate,
    progress: ProgressReporter | None = None,
) -> None:
    if progress and progress.enabled:
        LOGGER.debug("Document %s: %s", result.get("status"), candidate.title)
        return
    if is_saved_document_status(result.get("status")):
        LOGGER.info("Document %s: %s", result["status"], candidate.title)
    else:
        LOGGER.info("No document saved for: %s", candidate.title)


def mark_existing_downloads(
    candidates: list[Candidate],
    downloader: PdfDownloader,
    ledger: DownloadLogLedger | None = None,
) -> int:
    if downloader.force_download:
        return 0
    existing_count = 0
    records: list[tuple[Candidate, dict[str, Any]]] = []
    for candidate in candidates:
        if is_terminal_download_status(candidate.download.get("status")):
            continue
        if candidate.download.get("status") == "retry_abstract_only":
            continue
        filename = build_pdf_filename(candidate)
        destination = downloader.output_dir / filename
        if not destination.exists():
            continue
        result = {
            "status": "already_exists",
            "path": str(destination),
            "filename": filename,
            "attempts": [],
        }
        candidate.download = result
        records.append((candidate, result))
        existing_count += 1
    if ledger:
        ledger.record_many(records)
    if existing_count:
        LOGGER.info("Preflight found %d existing documents; skipping resolution.", existing_count)
    return existing_count


def download_log_resume_indices(
    candidates: list[Candidate],
    ledger: DownloadLogLedger | None,
) -> set[int]:
    if not ledger:
        return set()
    indices: set[int] = set()
    for idx, candidate in enumerate(candidates):
        if not is_processed_download_status(candidate.download.get("status")):
            continue
        if ledger.record_for_candidate(candidate):
            indices.add(idx)
    return indices


def download_candidates(
    candidates: list[Candidate],
    *,
    resolver: PdfResolver,
    downloader: PdfDownloader,
    max_downloads: int | None,
    max_web_search_candidates: int | None,
    download_workers: int,
    ledger: DownloadLogLedger | None = None,
    progress_enabled: bool = True,
) -> int:
    resume_indices = download_log_resume_indices(candidates, ledger)
    mark_existing_downloads(candidates, downloader, ledger=ledger)
    downloaded_count = sum(
        1
        for candidate in candidates
        if is_full_document_status(candidate.download.get("status"))
    )
    web_search_count = 0
    worker_count = max(1, download_workers)
    pending_candidates = [
        candidate
        for candidate in candidates
        if not is_terminal_download_status(candidate.download.get("status"))
    ]
    progress_candidates = [
        candidate for idx, candidate in enumerate(candidates) if idx not in resume_indices
    ]
    progress = ProgressReporter(
        len(progress_candidates),
        enabled=progress_enabled,
        label="PDF candidates",
    )
    progress.set_current(
        sum(
            1
            for candidate in progress_candidates
            if is_processed_download_status(candidate.download.get("status"))
        ),
        progress_candidates,
        force=True,
    )

    if worker_count == 1:
        for idx, candidate in enumerate(pending_candidates):
            if max_downloads is not None and downloaded_count >= max_downloads:
                mark_downloads_not_attempted(
                    pending_candidates[idx:],
                    "max_downloads limit reached",
                    ledger=ledger,
                )
                progress.set_current(progress.total, progress_candidates, force=True)
                break
            use_web_search = should_use_web_search_for_candidate(
                resolver.web_searcher,
                max_web_search_candidates,
                web_search_count,
            )
            if use_web_search:
                web_search_count += 1
            result = resolve_and_download_candidate(
                candidate,
                resolver,
                downloader,
                use_web_search,
            )
            if is_full_document_status(result.get("status")):
                downloaded_count += 1
            log_download_result(result, candidate, progress)
            if ledger:
                ledger.record(candidate, result)
            progress.advance(progress_candidates)
        progress.done(progress_candidates)
        return downloaded_count

    LOGGER.info("Downloading documents with %d workers.", worker_count)
    next_index = 0
    pending: dict[Future, Candidate] = {}

    def submit_until_full(executor: ThreadPoolExecutor) -> None:
        nonlocal next_index, web_search_count
        while len(pending) < worker_count and next_index < len(pending_candidates):
            if max_downloads is not None and downloaded_count >= max_downloads:
                return
            candidate = pending_candidates[next_index]
            next_index += 1
            use_web_search = should_use_web_search_for_candidate(
                resolver.web_searcher,
                max_web_search_candidates,
                web_search_count,
            )
            if use_web_search:
                web_search_count += 1
            future = executor.submit(
                resolve_and_download_candidate,
                candidate,
                resolver,
                downloader,
                use_web_search,
            )
            pending[future] = candidate

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        submit_until_full(executor)
        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in done:
                candidate = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - executor defensive path
                    result = {
                        "status": "failed",
                        "filename": build_pdf_filename(candidate),
                        "attempts": [],
                        "reason": str(exc),
                    }
                    candidate.download = result
                if is_full_document_status(result.get("status")):
                    downloaded_count += 1
                log_download_result(result, candidate, progress)
                if ledger:
                    ledger.record(candidate, result)
                progress.advance(progress_candidates)
            submit_until_full(executor)

    if (
        max_downloads is not None
        and downloaded_count >= max_downloads
        and next_index < len(pending_candidates)
    ):
        mark_downloads_not_attempted(
            pending_candidates[next_index:],
            "max_downloads limit reached",
            ledger=ledger,
        )
        progress.set_current(progress.total, progress_candidates, force=True)
    progress.done(progress_candidates)
    return downloaded_count


def is_saved_document_status(status: str | None) -> bool:
    return status in {"downloaded", "already_exists", "abstract_only"}


def is_full_document_status(status: str | None) -> bool:
    return status in {"downloaded", "already_exists"}


def is_terminal_download_status(status: str | None) -> bool:
    return is_saved_document_status(status) or status == "failed"


def is_processed_download_status(status: str | None) -> bool:
    return is_terminal_download_status(status) or status == "not_attempted"


def build_pdf_filename(candidate: Candidate) -> str:
    year = str(candidate.year or "undated")
    first_author = "unknown"
    if candidate.authors:
        first_author = candidate.authors[0].split()[-1]
    title_slug = slugify(candidate.title, max_length=70) or "untitled"
    digest_source = (
        candidate.doi or normalize_title(candidate.title) or candidate.landing_page_url or title_slug
    )
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"{year}_{slugify(first_author, 24)}_{title_slug}_{digest}.pdf"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_text(text: str, width: int = 80) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current:
            candidate = f"{current} {word}"
        else:
            candidate = word
        if len(candidate) > width:
            if current:
                lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def slugify(value: str | None, max_length: int = 80) -> str:
    if not value:
        return ""
    slug = normalize_title(value).replace(" ", "_")
    slug = re.sub(r"_+", "_", slug).strip("_")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")
    return slug


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=str(path.parent),
            encoding="utf-8",
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def candidate_download_identity(candidate: Candidate) -> str:
    if candidate.doi:
        return f"doi:{candidate.doi}"
    title = normalize_title(candidate.title)
    return f"title:{title}|year:{candidate.year or ''}"


def download_log_record(candidate: Candidate, download: dict[str, Any]) -> dict[str, Any]:
    filename = str(download.get("filename") or build_pdf_filename(candidate))
    download_copy = dict(download)
    download_copy["filename"] = filename
    status = str(download_copy.get("status") or "")
    return {
        "key": filename,
        "identity": candidate_download_identity(candidate),
        "filename": filename,
        "title": candidate.title,
        "doi": candidate.doi,
        "year": candidate.year,
        "status": status,
        "path": download_copy.get("path", ""),
        "reason": download_copy.get("reason", ""),
        "updated_at": utc_now(),
        "download": download_copy,
    }


def normalize_download_log_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    download = raw.get("download") if isinstance(raw.get("download"), dict) else dict(raw)
    status = str(raw.get("status") or download.get("status") or "")
    filename = str(raw.get("filename") or download.get("filename") or "")
    title = str(raw.get("title") or "")
    doi = normalize_doi(raw.get("doi") or download.get("doi"))
    identity = str(raw.get("identity") or "")
    if not identity:
        if doi:
            identity = f"doi:{doi}"
        elif title:
            identity = f"title:{normalize_title(title)}|year:{raw.get('year') or ''}"
    if not filename and not identity:
        return None
    download["status"] = status
    if filename:
        download["filename"] = filename
    return {
        "key": str(raw.get("key") or filename or identity),
        "identity": identity,
        "filename": filename,
        "title": title,
        "doi": doi,
        "year": safe_int(raw.get("year")),
        "status": status,
        "path": raw.get("path") or download.get("path", ""),
        "reason": raw.get("reason") or download.get("reason", ""),
        "updated_at": raw.get("updated_at") or raw.get("generated_at") or "",
        "download": download,
    }


def download_log_payload(
    records: list[dict[str, Any]],
    *,
    candidate_cache: dict[str, Any],
    search_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    downloads = [record["download"] for record in records if record.get("download")]
    failed_downloads = [
        {
            "title": record.get("title", ""),
            "doi": record.get("doi"),
            "filename": record.get("filename", ""),
            "reason": record.get("reason") or (record.get("download") or {}).get("reason") or "failed",
            "attempts": (record.get("download") or {}).get("attempts", []),
        }
        for record in records
        if record.get("status") == "failed"
    ]
    return {
        "project": "LitHarvest",
        "generated_at": utc_now(),
        "candidate_cache": candidate_cache,
        "records": records,
        "downloads": downloads,
        "failed_downloads": failed_downloads,
        "search_errors": search_errors,
    }


class DownloadLogLedger:
    def __init__(
        self,
        path: Path,
        *,
        candidate_cache: dict[str, Any],
        search_errors: list[dict[str, Any]],
    ) -> None:
        self.path = path
        self.candidate_cache = candidate_cache
        self.search_errors = search_errors
        self.records_by_key: dict[str, dict[str, Any]] = {}
        self.records_by_filename: dict[str, dict[str, Any]] = {}
        self.records_by_identity: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        if not self.path.exists() or not self.path.is_file():
            return
        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            return
        raw = json.loads(text)
        if not isinstance(raw, dict):
            return
        for item in raw.get("records") or []:
            record = normalize_download_log_record(item)
            if record:
                self._store(record)
        if self.records_by_key:
            return
        for item in raw.get("downloads") or []:
            if isinstance(item, dict):
                record = normalize_download_log_record(item)
                if record:
                    self._store(record)
        for item in raw.get("failed_downloads") or []:
            if isinstance(item, dict):
                failed = dict(item)
                failed["status"] = "failed"
                failed["download"] = {
                    "status": "failed",
                    "filename": failed.get("filename", ""),
                    "attempts": failed.get("attempts", []),
                    "reason": failed.get("reason", "failed"),
                }
                record = normalize_download_log_record(failed)
                if record:
                    self._store(record)

    def _store(self, record: dict[str, Any]) -> None:
        key = str(record.get("key") or record.get("filename") or record.get("identity"))
        if not key:
            return
        record["key"] = key
        self.records_by_key[key] = record
        filename = record.get("filename")
        if filename:
            self.records_by_filename[str(filename)] = record
        identity = record.get("identity")
        if identity:
            self.records_by_identity[str(identity)] = record

    def record_for_candidate(self, candidate: Candidate) -> dict[str, Any] | None:
        filename = build_pdf_filename(candidate)
        return self.records_by_filename.get(filename) or self.records_by_identity.get(
            candidate_download_identity(candidate)
        )

    def record(self, candidate: Candidate, download: dict[str, Any]) -> None:
        self._store(download_log_record(candidate, download))
        self.write()

    def record_many(self, items: list[tuple[Candidate, dict[str, Any]]]) -> None:
        for candidate, download in items:
            self._store(download_log_record(candidate, download))
        if items:
            self.write()

    def write(self) -> None:
        write_json(
            self.path,
            download_log_payload(
                list(self.records_by_key.values()),
                candidate_cache=self.candidate_cache,
                search_errors=self.search_errors,
            ),
        )


def apply_download_log_resume(
    candidates: list[Candidate],
    ledger: DownloadLogLedger,
    downloader: PdfDownloader,
    *,
    retry_failed_downloads: bool = False,
) -> dict[str, int]:
    skipped_saved = 0
    skipped_failed = 0
    retry_abstract_only = 0
    for candidate in candidates:
        record = ledger.record_for_candidate(candidate)
        if not record:
            continue
        status = record.get("status")
        download = dict(record.get("download") or {})
        filename = str(download.get("filename") or record.get("filename") or build_pdf_filename(candidate))
        expected_path = downloader.output_dir / filename
        if is_full_document_status(status):
            raw_path = download.get("path") or record.get("path")
            saved_path = Path(raw_path).expanduser() if raw_path else expected_path
            if not saved_path.exists() and expected_path.exists():
                saved_path = expected_path
            if not saved_path.exists():
                continue
            download["status"] = status
            download["filename"] = filename
            download["path"] = str(saved_path)
            candidate.download = download
            skipped_saved += 1
        elif status == "abstract_only":
            candidate.download = {
                "status": "retry_abstract_only",
                "filename": filename,
                "path": str(expected_path),
                "reason": "Previous run only saved an abstract fallback; retrying full PDF.",
            }
            retry_abstract_only += 1
        elif status == "failed" and not retry_failed_downloads:
            download["status"] = "failed"
            download["filename"] = filename
            download["reason"] = download.get("reason") or record.get("reason") or "Previous download failed."
            download.setdefault("attempts", [])
            candidate.download = download
            skipped_failed += 1
    if skipped_saved or skipped_failed or retry_abstract_only:
        LOGGER.info(
            "Download log resume skipped %d saved and %d failed candidates; retrying %d abstract-only candidates.",
            skipped_saved,
            skipped_failed,
            retry_abstract_only,
        )
    return {
        "saved": skipped_saved,
        "failed": skipped_failed,
        "abstract_only_retry": retry_abstract_only,
    }


def write_csv_report(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "title",
        "authors",
        "year",
        "doi",
        "venue",
        "source_apis",
        "relevance_score",
        "relevance_score_breakdown",
        "landing_page_url",
        "candidate_pdf_urls",
        "candidate_landing_page_urls",
        "resolved_pdf_urls",
        "matched_keywords",
        "download_status",
        "download_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "title": candidate.title,
                    "authors": "; ".join(candidate.authors),
                    "year": candidate.year or "",
                    "doi": candidate.doi or "",
                    "venue": candidate.venue or "",
                    "source_apis": "; ".join(candidate.source_apis),
                    "relevance_score": candidate.relevance_score,
                    "relevance_score_breakdown": json.dumps(
                        candidate.relevance_score_breakdown,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "landing_page_url": candidate.landing_page_url or "",
                    "candidate_pdf_urls": "; ".join(candidate.candidate_pdf_urls),
                    "candidate_landing_page_urls": "; ".join(
                        candidate.candidate_landing_page_urls
                    ),
                    "resolved_pdf_urls": "; ".join(
                        candidate.pdf_resolution.get("urls", [])
                    ),
                    "matched_keywords": "; ".join(candidate.matched_keywords),
                    "download_status": candidate.download.get("status", ""),
                    "download_path": candidate.download.get("path", ""),
                }
            )


def write_bibtex(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for candidate in candidates:
        key = bibtex_key(candidate)
        fields = {
            "title": candidate.title,
            "author": " and ".join(candidate.authors),
            "year": str(candidate.year) if candidate.year else "",
            "journal": candidate.venue or "",
            "doi": candidate.doi or "",
            "url": candidate.landing_page_url or doi_to_url(candidate.doi) or "",
        }
        body = []
        for field_name, value in fields.items():
            if value:
                safe_value = value.replace("{", "").replace("}", "")
                body.append(f"  {field_name} = {{{safe_value}}}")
        entries.append("@article{" + key + ",\n" + ",\n".join(body) + "\n}")
    path.write_text("\n\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def bibtex_key(candidate: Candidate) -> str:
    author = "unknown"
    if candidate.authors:
        author = slugify(candidate.authors[0].split()[-1], 20) or "unknown"
    title_word = "paper"
    title_tokens = content_tokens(candidate.title)
    if title_tokens:
        title_word = title_tokens[0]
    year = candidate.year or "nd"
    digest = hashlib.sha1((candidate.doi or candidate.title).encode("utf-8")).hexdigest()[:6]
    return f"{author}{year}{title_word}{digest}"


def redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in config.items():
        if "key" in key.lower() and value:
            output[key] = "***redacted***"
        else:
            output[key] = value
    return output


def config_fingerprint(path: Path, text: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
        "characters": len(text),
    }


def resolve_candidates_json_path(config: dict[str, Any], logs_dir: Path) -> Path:
    raw_path = config.get("candidates_json") or logs_dir / "candidates.json"
    return Path(raw_path).expanduser()


def search_query_from_dict(raw: dict[str, Any]) -> SearchQuery | None:
    text = normalize_whitespace(str(raw.get("text") or ""))
    if not text:
        return None
    terms = raw.get("terms") or []
    if isinstance(terms, str):
        terms = content_tokens(terms)
    elif isinstance(terms, list):
        terms = [str(term).strip() for term in terms if str(term).strip()]
    else:
        terms = []
    return SearchQuery(
        bucket=str(raw.get("bucket") or "cached").strip() or "cached",
        text=text,
        terms=terms,
    )


def candidate_from_dict(raw: dict[str, Any]) -> Candidate:
    allowed = {field.name for field in dataclasses.fields(Candidate)}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    title = normalize_whitespace(str(kwargs.get("title") or ""))
    if not title:
        raise ValueError("Cached candidate is missing a title.")
    kwargs["title"] = title

    for key in [
        "authors",
        "source_apis",
        "candidate_pdf_urls",
        "candidate_landing_page_urls",
        "matched_keywords",
    ]:
        value = kwargs.get(key)
        if isinstance(value, list):
            kwargs[key] = [str(item) for item in value if str(item).strip()]
        elif value:
            kwargs[key] = [str(value)]
        else:
            kwargs[key] = []

    if not isinstance(kwargs.get("source_ids"), dict):
        kwargs["source_ids"] = {}
    else:
        kwargs["source_ids"] = {
            str(key): str(value)
            for key, value in kwargs["source_ids"].items()
            if value is not None
        }

    for key in ["relevance_score_breakdown", "pdf_resolution", "download"]:
        if not isinstance(kwargs.get(key), dict):
            kwargs[key] = {}

    discovered_via = kwargs.get("discovered_via")
    if isinstance(discovered_via, list):
        kwargs["discovered_via"] = [
            item for item in discovered_via if isinstance(item, dict)
        ]
    else:
        kwargs["discovered_via"] = []

    kwargs["doi"] = normalize_doi(kwargs.get("doi"))
    kwargs["abstract"] = strip_markup(kwargs.get("abstract"))
    kwargs["venue"] = strip_markup(kwargs.get("venue"))
    kwargs["landing_page_url"] = clean_url(kwargs.get("landing_page_url"))
    kwargs["year"] = safe_int(kwargs.get("year"))
    kwargs["relevance_score"] = safe_float(kwargs.get("relevance_score")) or 0.0
    kwargs["source_score"] = safe_float(kwargs.get("source_score"))
    kwargs["citation_count"] = safe_int(kwargs.get("citation_count"))
    return Candidate(**kwargs)


def load_candidate_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    raw = json.loads(text)
    if isinstance(raw, list):
        raw_candidates = raw
        payload: dict[str, Any] = {"candidates": raw_candidates}
    elif isinstance(raw, dict):
        raw_candidates = raw.get("candidates") or []
        payload = raw
    else:
        raise ValueError(f"Candidate cache must contain an object or list: {path}")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return None

    candidates: list[Candidate] = []
    for idx, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate cache item {idx} must be an object.")
        candidates.append(candidate_from_dict(item))
    if not candidates:
        return None

    cached_queries: list[SearchQuery] = []
    for item in payload.get("queries") or []:
        if isinstance(item, dict):
            query = search_query_from_dict(item)
            if query:
                cached_queries.append(query)

    keyword_counts: dict[str, Counter[str]] = {}
    raw_keyword_counts = payload.get("keyword_search_counts") or {}
    if isinstance(raw_keyword_counts, dict):
        for source, counts in raw_keyword_counts.items():
            if isinstance(counts, dict):
                keyword_counts[str(source)] = Counter(
                    {
                        str(keyword): int(count)
                        for keyword, count in counts.items()
                        if safe_int(count) is not None
                    }
                )

    search_errors = payload.get("search_errors") or []
    if not isinstance(search_errors, list):
        search_errors = []
    citation_lookup_errors = payload.get("citation_lookup_errors") or []
    if not isinstance(citation_lookup_errors, list):
        citation_lookup_errors = []
    extraction_errors = payload.get("extraction_errors") or []
    if not isinstance(extraction_errors, list):
        extraction_errors = []
    citation_lookup_progress = payload.get("citation_lookup_progress") or {}
    if not isinstance(citation_lookup_progress, dict):
        citation_lookup_progress = {}

    return {
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "stage": payload.get("stage"),
        "candidates": candidates,
        "queries": cached_queries,
        "search_errors": search_errors,
        "citation_lookup_errors": citation_lookup_errors,
        "extraction_errors": extraction_errors,
        "extraction_count": safe_int(payload.get("extraction_count")),
        "extracted_reference_count": safe_int(payload.get("extracted_reference_count")),
        "citation_lookup_progress": citation_lookup_progress,
        "keyword_search_counts": keyword_counts,
        "candidate_count": len(candidates),
    }


def write_candidate_cache(
    path: Path,
    *,
    draft_path: Path,
    draft_text: str,
    config: dict[str, Any],
    extracted: ExtractedTerms,
    queries: list[SearchQuery],
    search_errors: list[dict[str, Any]],
    keyword_search_counts: dict[str, Counter[str]],
    candidates: list[Candidate],
    relevance_profile: dict[str, Any] | None = None,
) -> None:
    payload = {
        "project": "LitHarvest",
        "version": VERSION,
        "generated_at": utc_now(),
        "stage": "post_search_pre_download",
        "draft": config_fingerprint(draft_path, draft_text),
        "config": redacted_config(config),
        "extracted_terms": dataclasses.asdict(extracted),
        "queries": [dataclasses.asdict(query) for query in queries],
        "search_errors": search_errors,
        "keyword_search_counts": {
            source: dict(counter) for source, counter in keyword_search_counts.items()
        },
        "relevance_scoring": relevance_profile,
        "candidate_count": len(candidates),
        "candidates": [dataclasses.asdict(candidate) for candidate in candidates],
    }
    write_json(path, payload)


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    draft_path = Path(config["draft"]).expanduser()
    output_dir = Path(config["output"]).expanduser()
    logs_dir = Path(config.get("logs") or output_dir / "logs").expanduser()
    candidates_json_path = resolve_candidates_json_path(config, logs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = logs_dir / "manifest.json"
    download_log_path = logs_dir / "download_log.json"

    draft_text = read_text_file(draft_path)
    extracted = extract_keywords(draft_text)
    relevance_profile = None
    if config.get("relevance_scoring"):
        relevance_profile = load_relevance_profile(
            Path(config["relevance_scoring"]).expanduser()
        )
    queries = combine_search_queries(
        configured_search_queries(config.get("extra_queries")),
        build_search_queries(extracted, int(config["max_queries"])),
        int(config["max_queries"]),
    )
    LOGGER.info("Built %d search queries.", len(queries))
    for query in queries:
        LOGGER.info("Query [%s]: %s", query.bucket, query.text)

    search_errors: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    keyword_search_counts: dict[str, Counter[str]] = {}
    keyword_lookup = {normalize_title(keyword): keyword for keyword in extracted.keywords}
    candidate_cache: dict[str, Any] = {
        "path": str(candidates_json_path),
        "status": "not_used",
    }
    if not config.get("dry_run"):
        http = HttpClient(
            timeout=float(config["timeout"]),
            retries=int(config["retries"]),
            backoff=float(config["backoff"]),
            rate_limit_delay=float(config["rate_limit_delay"]),
            email=config.get("email"),
        )

        cached = None
        if not config.get("refresh_candidates"):
            cached = load_candidate_cache(candidates_json_path)
        if cached:
            candidates = cached["candidates"]
            if cached["queries"]:
                queries = cached["queries"]
            search_errors = cached["search_errors"]
            keyword_search_counts = cached["keyword_search_counts"]
            candidate_cache = {
                "path": str(candidates_json_path),
                "status": "loaded",
                "generated_at": cached.get("generated_at"),
                "candidate_count": len(candidates),
            }
            LOGGER.info(
                "Loaded %d candidates from %s; skipping metadata search.",
                len(candidates),
                candidates_json_path,
            )
        else:
            clients = build_clients(config, http)
            for query in queries:
                for client in clients:
                    try:
                        LOGGER.info("Searching %s for %r", client.name, query.text)
                        results = client.search(
                            query,
                            top_k=int(config["top_k_per_query"]),
                            year_from=config.get("year_from"),
                            year_to=config.get("year_to"),
                        )
                        LOGGER.info("%s returned %d candidates.", client.name, len(results))
                        for candidate in results:
                            candidate.matched_keywords = candidate_matches_keywords(candidate, extracted.keywords)
                        query_keywords: set[str] = set()
                        normalized_query_text = normalize_title(query.text)
                        if normalized_query_text in keyword_lookup:
                            query_keywords.add(keyword_lookup[normalized_query_text])
                        for term in query.terms:
                            normalized_term = normalize_title(term)
                            if normalized_term in keyword_lookup:
                                query_keywords.add(keyword_lookup[normalized_term])
                        for keyword in query_keywords:
                            keyword_search_counts.setdefault(client.name, Counter())[keyword] += len(results)
                        candidates.extend(results)
                    except Exception as exc:
                        LOGGER.warning("%s search failed for %r: %s", client.name, query.text, exc)
                        search_errors.append(
                            {
                                "source": client.name,
                                "query": query.text,
                                "bucket": query.bucket,
                                "reason": str(exc),
                            }
                        )

            candidates = merge_candidates(candidates)
            candidates = ensure_candidate_ids(
                rank_candidates(candidates, extracted, relevance_profile)
            )
            write_candidate_cache(
                candidates_json_path,
                draft_path=draft_path,
                draft_text=draft_text,
                config=config,
                extracted=extracted,
                queries=queries,
                search_errors=search_errors,
                keyword_search_counts=keyword_search_counts,
                candidates=candidates,
                relevance_profile=relevance_profile,
            )
            candidate_cache = {
                "path": str(candidates_json_path),
                "status": "saved",
                "candidate_count": len(candidates),
            }
            LOGGER.info(
                "Saved %d post-search candidates to %s before PDF download.",
                len(candidates),
                candidates_json_path,
            )

        if cached and relevance_profile is not None:
            candidates = ensure_candidate_ids(
                rank_candidates(candidates, extracted, relevance_profile)
            )
            write_candidate_cache(
                candidates_json_path,
                draft_path=draft_path,
                draft_text=draft_text,
                config=config,
                extracted=extracted,
                queries=queries,
                search_errors=search_errors,
                keyword_search_counts=keyword_search_counts,
                candidates=candidates,
                relevance_profile=relevance_profile,
            )
            LOGGER.info(
                "Re-ranked %d cached candidates with relevance profile %r.",
                len(candidates),
                relevance_profile["title"],
            )

        ensure_candidate_ids(candidates)
        web_searcher = build_web_pdf_searcher(config, http)
        resolver = PdfResolver(
            http,
            unpaywall_email=config.get("unpaywall_email")
            or os.getenv("UNPAYWALL_EMAIL")
            or config.get("email"),
            web_searcher=web_searcher,
        )
        downloader = PdfDownloader(
            http,
            output_dir=output_dir,
            max_pdf_mb=int(config["max_pdf_mb"]),
            force_download=bool(config.get("force_download")),
            allow_abstract_fallback=bool(config.get("allow_abstract_fallback", True)),
        )
        download_ledger = DownloadLogLedger(
            download_log_path,
            candidate_cache=candidate_cache,
            search_errors=search_errors,
        )
        download_ledger.load()
        if not config.get("force_download"):
            apply_download_log_resume(
                candidates,
                download_ledger,
                downloader,
                retry_failed_downloads=bool(config.get("retry_failed_downloads")),
            )

        selected_candidate_ids = config.get("selected_candidate_ids")
        selected_candidates, unselected_candidates = partition_selected_candidates(
            candidates, selected_candidate_ids
        )
        if selected_candidate_ids is not None:
            mark_candidates_not_selected(unselected_candidates, ledger=download_ledger)

        if config.get("metadata_only"):
            mark_downloads_not_attempted(
                selected_candidates,
                "metadata_only mode",
                ledger=download_ledger,
                preserve_existing=True,
            )
        else:
            download_limit = config.get("max_downloads")
            max_downloads = int(download_limit) if download_limit is not None else None
            web_limit = config.get("web_search_max_candidates")
            max_web_search_candidates = (
                int(web_limit) if web_limit is not None else None
            )
            download_candidates(
                selected_candidates,
                resolver=resolver,
                downloader=downloader,
                max_downloads=max_downloads,
                max_web_search_candidates=max_web_search_candidates,
                download_workers=int(config.get("download_workers") or 1),
                ledger=download_ledger,
                progress_enabled=bool(config.get("progress")),
            )
    else:
        LOGGER.info("Dry run requested; no API searches or downloads were attempted.")
        candidate_cache["status"] = "not_used_dry_run"

    download_records = [
        download_log_record(candidate, candidate.download)
        for candidate in candidates
        if candidate.download
    ]
    downloads = [record["download"] for record in download_records]
    failures = [
        {
            "title": candidate.title,
            "doi": candidate.doi,
            "filename": candidate.download.get("filename", build_pdf_filename(candidate)),
            "reason": candidate.download.get("reason")
            or candidate.download.get("status")
            or "No download attempted.",
            "attempts": candidate.download.get("attempts", []),
        }
        for candidate in candidates
        if candidate.download.get("status") == "failed"
    ]

    manifest = {
        "project": "LitHarvest",
        "version": VERSION,
        "generated_at": utc_now(),
        "draft": config_fingerprint(draft_path, draft_text),
        "config": redacted_config(config),
        "extracted_terms": dataclasses.asdict(extracted),
        "queries": [dataclasses.asdict(query) for query in queries],
        "search_errors": search_errors,
        "candidate_cache": candidate_cache,
        "candidate_count": len(candidates),
        "downloaded_count": sum(
            1
            for candidate in candidates
            if is_saved_document_status(candidate.download.get("status"))
        ),
        "keyword_search_counts": {
            source: dict(counter) for source, counter in keyword_search_counts.items()
        },
        "relevance_scoring": relevance_profile,
        "candidates": [dataclasses.asdict(candidate) for candidate in candidates],
    }
    download_log = download_log_payload(
        download_records,
        candidate_cache=candidate_cache,
        search_errors=search_errors,
    )
    download_log["failed_downloads"] = failures

    write_json(manifest_path, manifest)
    write_json(download_log_path, download_log)
    if config.get("csv_report"):
        write_csv_report(logs_dir / "candidates.csv", candidates)
    if config.get("bibtex"):
        write_bibtex(logs_dir / "candidates.bib", candidates)

    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info("Wrote download log: %s", download_log_path)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "download_log_path": str(download_log_path),
        "candidates_json_path": str(candidates_json_path),
    }


def write_citation_candidate_cache(
    path: Path,
    *,
    config: dict[str, Any],
    core_pdfs: list[Path],
    extractions: list[dict[str, Any]],
    extraction_errors: list[dict[str, Any]],
    lookup_errors: list[dict[str, Any]],
    candidates: list[Candidate],
    stage: str = "citation_lookup_pre_download",
    lookup_progress: dict[str, Any] | None = None,
    extraction_count: int | None = None,
    extracted_reference_count: int | None = None,
) -> None:
    if extraction_count is None:
        extraction_count = len(extractions)
    if extracted_reference_count is None:
        extracted_reference_count = sum(
            len(extraction.get("references") or []) for extraction in extractions
        )
    payload = {
        "project": "LitHarvest",
        "version": VERSION,
        "generated_at": utc_now(),
        "stage": stage,
        "config": redacted_config(config),
        "core_pdfs": [file_fingerprint(path) for path in core_pdfs],
        "extraction_count": extraction_count,
        "extracted_reference_count": extracted_reference_count,
        "extraction_errors": extraction_errors,
        "citation_lookup_errors": lookup_errors,
        "citation_lookup_progress": lookup_progress or {},
        "candidate_count": len(candidates),
        "candidates": [dataclasses.asdict(candidate) for candidate in candidates],
    }
    write_json(path, payload)


def run_citation_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output"]).expanduser()
    logs_dir = Path(config.get("logs") or output_dir / "logs").expanduser()
    candidates_json_path = resolve_candidates_json_path(config, logs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = logs_dir / "manifest.json"
    download_log_path = logs_dir / "download_log.json"

    core_pdfs = collect_core_pdfs(config)
    extractions: list[dict[str, Any]] = []
    extraction_errors: list[dict[str, Any]] = []
    lookup_errors: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    extraction_count = 0
    extracted_reference_count = 0
    candidate_cache: dict[str, Any] = {
        "path": str(candidates_json_path),
        "status": "not_used",
    }

    if config.get("dry_run"):
        LOGGER.info("Dry run requested; no citation extraction, API lookups, or downloads were attempted.")
        candidate_cache["status"] = "not_used_dry_run"
    else:
        http = HttpClient(
            timeout=float(config["timeout"]),
            retries=int(config["retries"]),
            backoff=float(config["backoff"]),
            rate_limit_delay=float(config["rate_limit_delay"]),
            email=config.get("email"),
        )

        cached = None
        if not config.get("refresh_candidates"):
            cached = load_candidate_cache(candidates_json_path)
        lookup_needs_run = False
        if cached:
            candidates = cached["candidates"]
            lookup_errors = cached.get("citation_lookup_errors") or cached.get("search_errors", [])
            extraction_errors = cached.get("extraction_errors", [])
            extraction_count = int(cached.get("extraction_count") or 0)
            extracted_reference_count = int(cached.get("extracted_reference_count") or 0)
            cached_stage = str(cached.get("stage") or "")
            candidate_cache = {
                "path": str(candidates_json_path),
                "status": "loaded",
                "generated_at": cached.get("generated_at"),
                "stage": cached_stage,
                "candidate_count": len(candidates),
            }
            if cached_stage == "citation_lookup_partial":
                lookup_needs_run = True
                progress = cached.get("citation_lookup_progress") or {}
                LOGGER.info(
                    "Loaded partial citation lookup from %s: %s/%s attempts complete.",
                    candidates_json_path,
                    progress.get("completed_lookup_attempts", "?"),
                    progress.get("total_lookup_attempts", "?"),
                )
            else:
                LOGGER.info(
                    "Loaded %d citation candidates from %s; skipping citation extraction and lookup.",
                    len(candidates),
                    candidates_json_path,
                )
        else:
            cached_extraction_count = 0
            fresh_extraction_count = 0
            extraction_progress = PhaseProgressReporter(
                len(core_pdfs),
                enabled=bool(config.get("progress")),
                label="Citation extraction",
            )
            extraction_progress.set_current(
                0,
                summary="cached=0 extracted=0 errors=0",
                force=True,
            )

            for pdf_path in core_pdfs:
                cached_extraction = None
                if not config.get("refresh_citation_extractions"):
                    cached_extraction = load_cached_citation_extraction(pdf_path, logs_dir, config)
                if cached_extraction:
                    extractions.append(cached_extraction)
                    cached_extraction_count += 1
                    extraction_progress.advance(
                        summary=(
                            f"cached={cached_extraction_count} "
                            f"extracted={fresh_extraction_count} errors={len(extraction_errors)}"
                        )
                    )
                    continue

                if not config.get("llm_api_key"):
                    provider = config.get("llm_provider")
                    if provider == "gemini":
                        raise ValueError(
                            "GEMINI_API_KEY or GOOGLE_API_KEY is required for citation extraction. "
                            "Set it in the environment or pass --llm-api-key."
                        )
                    raise ValueError(
                        "OPENAI_API_KEY is required for citation extraction. "
                        "Set it in the environment or pass --llm-api-key."
                    )

                try:
                    if not extraction_progress.enabled:
                        LOGGER.info("Extracting citations from %s", pdf_path)
                    extractions.append(extract_citations_from_core_pdf(pdf_path, logs_dir, config))
                    fresh_extraction_count += 1
                except Exception as exc:
                    LOGGER.warning("Citation extraction failed for %s: %s", pdf_path, exc)
                    extraction_errors.append(
                        {
                            "core_pdf": str(pdf_path),
                            "paper_id": safe_pdf_stem(pdf_path),
                            "reason": str(exc),
                        }
                    )
                extraction_progress.advance(
                    summary=(
                        f"cached={cached_extraction_count} "
                        f"extracted={fresh_extraction_count} errors={len(extraction_errors)}"
                    )
                )
            extraction_progress.done(
                summary=(
                    f"cached={cached_extraction_count} "
                    f"extracted={fresh_extraction_count} errors={len(extraction_errors)}"
                )
            )

            candidates = merge_candidates(candidates_from_citation_extractions(extractions))
            extraction_count = len(extractions)
            extracted_reference_count = sum(
                len(extraction.get("references") or []) for extraction in extractions
            )
            LOGGER.info("Built %d deduplicated citation candidates.", len(candidates))
            lookup_needs_run = True

        if lookup_needs_run:
            clients = build_clients(config, http)

            def write_lookup_checkpoint(
                checkpoint_candidates: list[Candidate],
                checkpoint_errors: list[dict[str, Any]],
                lookup_progress: dict[str, Any],
            ) -> None:
                write_citation_candidate_cache(
                    candidates_json_path,
                    config=config,
                    core_pdfs=core_pdfs,
                    extractions=extractions,
                    extraction_errors=extraction_errors,
                    lookup_errors=checkpoint_errors,
                    candidates=checkpoint_candidates,
                    stage="citation_lookup_partial",
                    lookup_progress=lookup_progress,
                    extraction_count=extraction_count,
                    extracted_reference_count=extracted_reference_count,
                )

            write_lookup_checkpoint(
                candidates,
                lookup_errors,
                citation_lookup_progress_payload(candidates, clients, lookup_errors),
            )
            lookup_config = dict(config)
            lookup_config["_initial_citation_lookup_errors"] = lookup_errors
            candidates, lookup_errors = enrich_citation_candidates(
                candidates,
                clients,
                lookup_config,
                checkpoint=write_lookup_checkpoint,
            )
            write_citation_candidate_cache(
                candidates_json_path,
                config=config,
                core_pdfs=core_pdfs,
                extractions=extractions,
                extraction_errors=extraction_errors,
                lookup_errors=lookup_errors,
                candidates=candidates,
                stage="citation_lookup_pre_download",
                lookup_progress=citation_lookup_progress_payload(candidates, clients, lookup_errors),
                extraction_count=extraction_count,
                extracted_reference_count=extracted_reference_count,
            )
            candidate_cache = {
                "path": str(candidates_json_path),
                "status": "saved",
                "stage": "citation_lookup_pre_download",
                "candidate_count": len(candidates),
            }
            LOGGER.info(
                "Saved %d post-lookup citation candidates to %s before PDF download.",
                len(candidates),
                candidates_json_path,
            )

        ensure_candidate_ids(candidates)
        web_searcher = build_web_pdf_searcher(config, http)
        resolver = PdfResolver(
            http,
            unpaywall_email=config.get("unpaywall_email")
            or os.getenv("UNPAYWALL_EMAIL")
            or config.get("email"),
            web_searcher=web_searcher,
        )
        downloader = PdfDownloader(
            http,
            output_dir=output_dir,
            max_pdf_mb=int(config["max_pdf_mb"]),
            force_download=bool(config.get("force_download")),
            allow_abstract_fallback=bool(config.get("allow_abstract_fallback")),
        )
        download_ledger = DownloadLogLedger(
            download_log_path,
            candidate_cache=candidate_cache,
            search_errors=lookup_errors,
        )
        download_ledger.load()
        if not config.get("force_download"):
            apply_download_log_resume(
                candidates,
                download_ledger,
                downloader,
                retry_failed_downloads=bool(config.get("retry_failed_downloads")),
            )

        selected_candidate_ids = config.get("selected_candidate_ids")
        selected_candidates, unselected_candidates = partition_selected_candidates(
            candidates, selected_candidate_ids
        )
        if selected_candidate_ids is not None:
            mark_candidates_not_selected(unselected_candidates, ledger=download_ledger)

        if config.get("metadata_only"):
            mark_downloads_not_attempted(
                selected_candidates,
                "metadata_only mode",
                ledger=download_ledger,
                preserve_existing=True,
            )
        else:
            download_limit = config.get("max_downloads")
            max_downloads = int(download_limit) if download_limit is not None else None
            web_limit = config.get("web_search_max_candidates")
            max_web_search_candidates = int(web_limit) if web_limit is not None else None
            download_candidates(
                selected_candidates,
                resolver=resolver,
                downloader=downloader,
                max_downloads=max_downloads,
                max_web_search_candidates=max_web_search_candidates,
                download_workers=int(config.get("download_workers") or 1),
                ledger=download_ledger,
                progress_enabled=bool(config.get("progress")),
            )

    download_records = [
        download_log_record(candidate, candidate.download)
        for candidate in candidates
        if candidate.download
    ]
    downloads = [record["download"] for record in download_records]
    failures = [
        {
            "title": candidate.title,
            "doi": candidate.doi,
            "filename": candidate.download.get("filename", build_pdf_filename(candidate)),
            "reason": candidate.download.get("reason")
            or candidate.download.get("status")
            or "No download attempted.",
            "attempts": candidate.download.get("attempts", []),
        }
        for candidate in candidates
        if candidate.download.get("status") == "failed"
    ]

    if extractions:
        extraction_count = len(extractions)
        extracted_reference_count = sum(len(extraction.get("references") or []) for extraction in extractions)
    manifest = {
        "project": "LitHarvest",
        "version": VERSION,
        "generated_at": utc_now(),
        "harvest_mode": "citations",
        "config": redacted_config(config),
        "core_pdfs": [file_fingerprint(path) for path in core_pdfs],
        "citation_extraction": {
            "extractor": config["citation_extractor"],
            "llm_provider": config["llm_provider"],
            "llm_model": config["llm_model"],
            "extraction_count": extraction_count,
            "extracted_reference_count": extracted_reference_count,
            "errors": extraction_errors,
            "artifacts": [extraction.get("metadata", {}) for extraction in extractions],
        },
        "citation_lookup_errors": lookup_errors,
        "candidate_cache": candidate_cache,
        "candidate_count": len(candidates),
        "deduplicated_candidate_count": len(candidates),
        "downloaded_count": sum(
            1
            for candidate in candidates
            if is_saved_document_status(candidate.download.get("status"))
        ),
        "failed_downloads": failures,
        "candidates": [dataclasses.asdict(candidate) for candidate in candidates],
    }
    download_log = download_log_payload(
        download_records,
        candidate_cache=candidate_cache,
        search_errors=lookup_errors,
    )
    download_log["harvest_mode"] = "citations"
    download_log["downloads"] = downloads
    download_log["failed_downloads"] = failures
    download_log["extraction_errors"] = extraction_errors
    download_log["citation_lookup_errors"] = lookup_errors

    write_json(manifest_path, manifest)
    write_json(download_log_path, download_log)
    if config.get("csv_report"):
        write_csv_report(logs_dir / "candidates.csv", candidates)
    if config.get("bibtex"):
        write_bibtex(logs_dir / "candidates.bib", candidates)

    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info("Wrote download log: %s", download_log_path)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "download_log_path": str(download_log_path),
        "candidates_json_path": str(candidates_json_path),
    }


def run_harvest(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("harvest_mode") == "citations":
        return run_citation_pipeline(config)
    return run_pipeline(config)


DEFAULT_CONFIG: dict[str, Any] = {
    "harvest_mode": "draft",
    "core_pdf": [],
    "core_pdf_dir": None,
    "citation_extractor": "direct_pdf",
    "llm_provider": os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
    "llm_model": None,
    "llm_api_key": None,
    "max_output_tokens": 0,
    "max_text_chars": 0,
    "max_queries": 8,
    "top_k_per_query": 10,
    "sources": DEFAULT_SOURCES,
    "year_from": None,
    "year_to": None,
    "email": os.getenv("LITHARVEST_EMAIL"),
    "unpaywall_email": os.getenv("UNPAYWALL_EMAIL"),
    "semantic_scholar_api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
    "web_search_sources": [],
    "web_search_max_candidates": 100,
    "web_search_max_results": 5,
    "web_search_queries_per_candidate": 3,
    "web_search_country": "us",
    "web_search_language": "en",
    "brave_search_api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
    "serpapi_api_key": os.getenv("SERPAPI_API_KEY"),
    "serpapi_engine": "google",
    "timeout": 20.0,
    "retries": 2,
    "backoff": 1.5,
    "rate_limit_delay": 0.1,
    "download_workers": 1,
    "max_pdf_mb": 80,
    "max_downloads": 50,
    "logs": None,
    "candidates_json": None,
    "refresh_candidates": False,
    "refresh_citation_extractions": False,
    "retry_failed_downloads": False,
    "progress": True,
    "metadata_only": False,
    "force_download": False,
    "allow_abstract_fallback": True,
    "no_abstract": False,
    "csv_report": False,
    "bibtex": False,
    "dry_run": False,
    "extra_queries": [],
    "selected_candidate_ids": None,
    "relevance_scoring": None,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect open scholarly PDFs from a draft using API-first metadata sources."
    )
    parser.add_argument(
        "--harvest-mode",
        choices=HARVEST_MODES,
        help="Harvest mode: draft search or citation-seeded core PDF expansion.",
    )
    parser.add_argument("--draft", help="Path to the input draft text or markdown file.")
    parser.add_argument(
        "--relevance-scoring",
        help="Optional generated relevance_scoring.json profile for candidate ranking.",
    )
    parser.add_argument(
        "--core-pdf",
        action="append",
        help="Core PDF to mine for citations. Repeatable. Enables citation mode when no mode is set.",
    )
    parser.add_argument(
        "--core-pdf-dir",
        help="Directory of core PDFs to mine for citations. Enables citation mode when no mode is set.",
    )
    parser.add_argument(
        "--citation-extractor",
        choices=CITATION_EXTRACTORS,
        help="Citation extraction path for core PDFs: direct_pdf or pymupdf_markdown.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=LLM_PROVIDERS,
        help="LLM provider for citation extraction.",
    )
    parser.add_argument("--llm-model", help="LLM model for citation extraction.")
    parser.add_argument(
        "--llm-api-key",
        help="LLM API key for citation extraction. Defaults to provider-specific environment variables.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Optional max output tokens for citation extraction LLM calls.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        help="Optional character cap for PyMuPDF text sent to the model. 0 means no cap.",
    )
    parser.add_argument("--output", help="Folder where downloaded PDFs should be stored.")
    parser.add_argument("--logs", help="Folder for manifest.json and download_log.json.")
    parser.add_argument(
        "--candidates-json",
        help="Post-search candidate checkpoint JSON. Defaults to logs/candidates.json.",
    )
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--max-queries", type=int, help="Maximum search queries to generate.")
    parser.add_argument(
        "--top-k-per-query",
        type=int,
        help="Maximum papers to retrieve from each source for each query.",
    )
    parser.add_argument(
        "--sources",
        help="Comma-separated sources: openalex,semantic_scholar,crossref,europe_pmc.",
    )
    parser.add_argument("--year-from", type=int, help="Earliest publication year to keep.")
    parser.add_argument("--year-to", type=int, help="Latest publication year to keep.")
    parser.add_argument("--email", help="Contact email for polite API usage.")
    parser.add_argument("--unpaywall-email", help="Email for Unpaywall DOI lookups.")
    parser.add_argument("--semantic-scholar-api-key", help="Optional Semantic Scholar API key.")
    parser.add_argument(
        "--web-search-sources",
        help="Comma-separated web search fallbacks for PDF discovery: brave,serpapi.",
    )
    parser.add_argument(
        "--web-search-max-candidates",
        type=int,
        help="Maximum ranked candidates to enrich with web search; 0 disables web search.",
    )
    parser.add_argument(
        "--web-search-max-results",
        type=int,
        help="Maximum web search results to inspect per query.",
    )
    parser.add_argument(
        "--web-search-queries-per-candidate",
        type=int,
        help="Maximum web search queries to run per candidate.",
    )
    parser.add_argument("--web-search-country", help="Country code for web search providers.")
    parser.add_argument("--web-search-language", help="Language code for web search providers.")
    parser.add_argument("--brave-search-api-key", help="Optional Brave Search API key.")
    parser.add_argument("--serpapi-api-key", help="Optional SerpAPI key.")
    parser.add_argument("--serpapi-engine", help="SerpAPI engine, default google.")
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, help="HTTP retry count.")
    parser.add_argument("--backoff", type=float, help="Retry backoff base in seconds.")
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        help="Minimum delay between outbound HTTP requests.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        help="Number of candidates to resolve/download in parallel.",
    )
    parser.add_argument("--max-pdf-mb", type=int, help="Maximum accepted PDF size in MB.")
    parser.add_argument("--max-downloads", type=int, help="Maximum successful PDF downloads.")
    parser.add_argument(
        "--selected-candidate-id",
        dest="selected_candidate_ids",
        action="append",
        help="Stable candidate ID to download. Repeatable; omitted downloads the ranked set.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        default=None,
        help="Search and write metadata without downloading PDFs.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=None,
        help="Overwrite existing PDFs with the same generated filename.",
    )
    abstract_group = parser.add_mutually_exclusive_group()
    abstract_group.add_argument(
        "--allow-abstract-fallback",
        dest="allow_abstract_fallback",
        action="store_true",
        default=None,
        help="Allow saving abstract-only placeholder PDFs when no PDF is downloadable.",
    )
    abstract_group.add_argument(
        "--no-abstract",
        action="store_true",
        default=None,
        help="Disable abstract-only PDF fallback when no full PDF can be downloaded.",
    )
    parser.add_argument(
        "--refresh-candidates",
        action="store_true",
        default=None,
        help="Ignore an existing candidates JSON checkpoint and run metadata searches again.",
    )
    parser.add_argument(
        "--refresh-citation-extractions",
        action="store_true",
        default=None,
        help="Ignore cached citation extraction artifacts and reread core PDFs with the LLM.",
    )
    parser.add_argument(
        "--retry-failed-downloads",
        action="store_true",
        default=None,
        help="Retry candidates marked failed in the existing download log.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=None,
        help="Disable the download progress meter.",
    )
    parser.add_argument(
        "--csv-report",
        action="store_true",
        default=None,
        help="Also write logs/candidates.csv.",
    )
    parser.add_argument(
        "--bibtex",
        action="store_true",
        default=None,
        help="Also write logs/candidates.bib.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Read the draft and build queries without API calls or downloads.",
    )
    parser.add_argument("--quiet", action="store_true", help="Show warnings and errors only.")
    parser.add_argument("--verbose", action="store_true", help="Show debug logging.")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    loaded: dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config).expanduser()
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a JSON object.")
        config.update(loaded)

    explicit_harvest_mode = args.harvest_mode is not None or "harvest_mode" in loaded
    explicit_allow_abstract_fallback = (
        args.allow_abstract_fallback is not None or "allow_abstract_fallback" in loaded
    )
    explicit_no_abstract = args.no_abstract is not None or "no_abstract" in loaded

    for key, value in vars(args).items():
        if key in {"config", "quiet", "verbose"}:
            continue
        if value is None:
            continue
        config[key] = value

    config["harvest_mode"] = str(config.get("harvest_mode") or "draft").strip().lower()
    if not explicit_harvest_mode and (config.get("core_pdf") or config.get("core_pdf_dir")):
        config["harvest_mode"] = "citations"
    if config["harvest_mode"] not in HARVEST_MODES:
        raise ValueError(f"--harvest-mode must be one of: {', '.join(HARVEST_MODES)}.")

    config["citation_extractor"] = str(
        config.get("citation_extractor") or "direct_pdf"
    ).strip().lower()
    if config["citation_extractor"] not in CITATION_EXTRACTORS:
        raise ValueError(
            f"--citation-extractor must be one of: {', '.join(CITATION_EXTRACTORS)}."
        )

    config["llm_provider"] = str(
        config.get("llm_provider") or DEFAULT_LLM_PROVIDER
    ).strip().lower()
    if config["llm_provider"] not in LLM_PROVIDERS:
        raise ValueError(f"--llm-provider must be one of: {', '.join(LLM_PROVIDERS)}.")
    if not config.get("llm_model"):
        config["llm_model"] = llm_provider_default_model(config["llm_provider"])
    if not config.get("llm_api_key"):
        config["llm_api_key"] = llm_provider_env_api_key(config["llm_provider"])

    if isinstance(config.get("core_pdf"), str):
        config["core_pdf"] = [config["core_pdf"]]
    elif config.get("core_pdf") is None:
        config["core_pdf"] = []
    else:
        config["core_pdf"] = [str(path) for path in ensure_list(config.get("core_pdf")) if str(path)]

    allow_abstract_fallback = bool(config.get("allow_abstract_fallback", True))
    if explicit_no_abstract:
        allow_abstract_fallback = not bool(config.get("no_abstract"))
    elif config["harvest_mode"] == "citations" and not explicit_allow_abstract_fallback:
        allow_abstract_fallback = False
    config["allow_abstract_fallback"] = allow_abstract_fallback
    config["no_abstract"] = not allow_abstract_fallback

    if isinstance(config.get("sources"), str):
        config["sources"] = parse_sources(config["sources"])
    else:
        config["sources"] = [
            normalize_source_name(source) for source in config.get("sources", DEFAULT_SOURCES)
        ]

    if isinstance(config.get("web_search_sources"), str):
        config["web_search_sources"] = parse_sources(config["web_search_sources"])
    else:
        config["web_search_sources"] = [
            normalize_source_name(source) for source in config.get("web_search_sources", [])
        ]

    if not config.get("output"):
        raise ValueError("--output is required unless provided in --config.")
    if config["harvest_mode"] == "draft" and not config.get("draft"):
        raise ValueError("--draft is required unless provided in --config.")
    if config["harvest_mode"] == "citations" and not (
        config.get("core_pdf") or config.get("core_pdf_dir")
    ):
        raise ValueError("Citation harvest mode requires --core-pdf or --core-pdf-dir.")
    if config.get("year_from") and config.get("year_to"):
        if int(config["year_from"]) > int(config["year_to"]):
            raise ValueError("--year-from cannot be later than --year-to.")
    if int(config["max_queries"]) < 1:
        raise ValueError("--max-queries must be at least 1.")
    if int(config["top_k_per_query"]) < 1:
        raise ValueError("--top-k-per-query must be at least 1.")
    if int(config["download_workers"]) < 1:
        raise ValueError("--download-workers must be at least 1.")
    if int(config["max_pdf_mb"]) < 1:
        raise ValueError("--max-pdf-mb must be at least 1.")
    if int(config["max_output_tokens"]) < 0:
        raise ValueError("--max-output-tokens must be 0 or greater.")
    if int(config["max_text_chars"]) < 0:
        raise ValueError("--max-text-chars must be 0 or greater.")
    if int(config["web_search_max_candidates"]) < 0:
        raise ValueError("--web-search-max-candidates must be at least 0.")
    if int(config["web_search_max_results"]) < 1:
        raise ValueError("--web-search-max-results must be at least 1.")
    if int(config["web_search_queries_per_candidate"]) < 1:
        raise ValueError("--web-search-queries-per-candidate must be at least 1.")
    return config


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    try:
        config = load_config(args)
        result = run_harvest(config)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    manifest = result["manifest"]
    LOGGER.info(
        "Finished: %d candidates, %d downloaded.",
        manifest["candidate_count"],
        manifest["downloaded_count"],
    )
    print(result["manifest_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
