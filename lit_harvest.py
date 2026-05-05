#!/usr/bin/env python3
"""LitHarvest: draft-to-PDF literature collection pipeline.

This module intentionally avoids Google Scholar scraping and browser
automation. It talks to metadata APIs, resolves open-access PDFs where those
APIs expose them, and records every candidate and download outcome.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import html
import json
import logging
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "0.1.0"
USER_AGENT_NAME = f"LitHarvest/{VERSION}"
DEFAULT_SOURCES = ["openalex", "semantic_scholar", "crossref", "europe_pmc"]
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
class Candidate:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    abstract: str | None = None
    venue: str | None = None
    landing_page_url: str | None = None
    source_apis: list[str] = field(default_factory=list)
    source_ids: dict[str, str] = field(default_factory=dict)
    candidate_pdf_urls: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    source_score: float | None = None
    citation_count: int | None = None
    discovered_via: list[dict[str, Any]] = field(default_factory=list)
    pdf_resolution: dict[str, Any] = field(default_factory=dict)
    download: dict[str, Any] = field(default_factory=dict)
    matched_keywords: list[str] = field(default_factory=list)


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


def add_url(urls: list[str], value: str | None) -> None:
    if not value:
        return
    url = html.unescape(str(value)).strip()
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        return
    if url.lower() not in {existing.lower() for existing in urls}:
        urls.append(url)


def looks_like_pdf_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urllib.parse.urlparse(url)
    return parsed.path.lower().endswith(".pdf") or ".pdf" in parsed.path.lower()


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
        if self.rate_limit_delay <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.rate_limit_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

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
                        raise RuntimeError(
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
                            raise RuntimeError(
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
                                raise RuntimeError(
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


def extract_pdf_urls_from_html(html_text: str, base_url: str) -> list[str]:
    urls: list[str] = []

    # Direct href="...pdf..." links
    for match in re.findall(r"href=[\"']([^\"']*\.pdf[^\"']*)[\"']", html_text, flags=re.IGNORECASE):
        if match:
            absolute = urllib.parse.urljoin(base_url, match)
            urls.append(absolute)

    # Meta tags for PDF URLs
    for match in re.findall(r"<meta[^>]*name=[\"']citation_pdf_url[\"'][^>]*content=[\"']([^\"']*)[\"']", html_text, flags=re.IGNORECASE):
        if match:
            absolute = urllib.parse.urljoin(base_url, match)
            urls.append(absolute)

    # Link rel="alternate" type="application/pdf"
    for match in re.findall(r"<link[^>]*rel=[\"']alternate[\"'][^>]*type=[\"']application/pdf[\"'][^>]*href=[\"']([^\"']*)[\"']", html_text, flags=re.IGNORECASE):
        if match:
            absolute = urllib.parse.urljoin(base_url, match)
            urls.append(absolute)

    # Iframe src attributes that might point to PDFs
    for match in re.findall(r"<iframe[^>]*src=[\"']([^\"']*\.pdf[^\"']*)[\"']", html_text, flags=re.IGNORECASE):
        if match:
            absolute = urllib.parse.urljoin(base_url, match)
            urls.append(absolute)

    # A href links that might be PDFs (even if not ending in .pdf, but containing pdf in path)
    for match in re.findall(r"href=[\"']([^\"']*pdf[^\"']*)[\"']", html_text, flags=re.IGNORECASE):
        if match and not match.endswith('.pdf'):  # Avoid duplicates
            absolute = urllib.parse.urljoin(base_url, match)
            urls.append(absolute)

    # Publisher-specific patterns
    parsed_base = urllib.parse.urlparse(base_url)
    if 'ieeexplore.ieee.org' in parsed_base.netloc:
        # IEEE Xplore: look for specific patterns
        for match in re.findall(r"href=[\"']([^\"']*download[^\"']*)[\"']", html_text, flags=re.IGNORECASE):
            if match and 'pdf' in match.lower():
                absolute = urllib.parse.urljoin(base_url, match)
                urls.append(absolute)
    elif 'academic.oup.com' in parsed_base.netloc:
        # OUP: look for chapter-pdf or similar
        for match in re.findall(r"href=[\"']([^\"']*chapter-pdf[^\"']*)[\"']", html_text, flags=re.IGNORECASE):
            absolute = urllib.parse.urljoin(base_url, match)
            urls.append(absolute)
    elif 'scholarship.law.umn.edu' in parsed_base.netloc:
        # UMN scholarship: look for viewcontent
        for match in re.findall(r"href=[\"']([^\"']*viewcontent[^\"']*)[\"']", html_text, flags=re.IGNORECASE):
            absolute = urllib.parse.urljoin(base_url, match)
            urls.append(absolute)

    return dedupe_preserve(urls)


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
        add_url(pdf_urls, primary_location.get("pdf_url"))
        open_access = item.get("open_access") or {}
        oa_url = open_access.get("oa_url")
        landing_page = primary_location.get("landing_page_url") or doi_to_url(
            normalize_doi(item.get("doi"))
        )
        if looks_like_pdf_url(oa_url):
            add_url(pdf_urls, oa_url)
        elif oa_url and not landing_page:
            landing_page = oa_url
        for location in ensure_list(item.get("locations")):
            if isinstance(location, dict):
                add_url(pdf_urls, location.get("pdf_url"))

        authors = []
        for authorship in ensure_list(item.get("authorships")):
            author = authorship.get("author") if isinstance(authorship, dict) else None
            if isinstance(author, dict) and author.get("display_name"):
                authors.append(author["display_name"])

        candidate = Candidate(
            title=title,
            authors=dedupe_preserve(authors),
            year=safe_int(item.get("publication_year")),
            doi=normalize_doi(item.get("doi")),
            abstract=reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
            venue=venue,
            landing_page_url=landing_page,
            source_apis=[self.name],
            source_ids={self.name: str(item.get("id"))} if item.get("id") else {},
            candidate_pdf_urls=pdf_urls,
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
        open_pdf = item.get("openAccessPdf") or {}
        if isinstance(open_pdf, dict):
            add_url(pdf_urls, open_pdf.get("url"))
        authors = [
            author["name"]
            for author in ensure_list(item.get("authors"))
            if isinstance(author, dict) and author.get("name")
        ]
        paper_id = item.get("paperId")
        return Candidate(
            title=title,
            authors=dedupe_preserve(authors),
            year=safe_int(item.get("year")),
            doi=normalize_doi(external_ids.get("DOI")),
            abstract=strip_markup(item.get("abstract")),
            venue=item.get("venue"),
            landing_page_url=item.get("url"),
            source_apis=[self.name],
            source_ids={self.name: str(paper_id)} if paper_id else {},
            candidate_pdf_urls=pdf_urls,
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
        for link in ensure_list(item.get("link")):
            if not isinstance(link, dict):
                continue
            content_type = str(link.get("content-type") or "").lower()
            url = link.get("URL")
            if "pdf" in content_type or looks_like_pdf_url(url):
                add_url(pdf_urls, url)

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
            candidate_pdf_urls=pdf_urls,
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
        full_text_list = (item.get("fullTextUrlList") or {}).get("fullTextUrl")
        for full_text in ensure_list(full_text_list):
            if not isinstance(full_text, dict):
                continue
            url = full_text.get("url")
            style = str(full_text.get("documentStyle") or "").lower()
            if style == "pdf" or looks_like_pdf_url(url):
                add_url(pdf_urls, url)

        authors = parse_europe_pmc_authors(item.get("authorString"))
        source = item.get("source")
        source_id = item.get("id")
        landing_page = doi_to_url(normalize_doi(item.get("doi")))
        if not landing_page and source and source_id:
            landing_page = f"https://europepmc.org/article/{source}/{source_id}"

        return Candidate(
            title=title,
            authors=authors,
            year=safe_int(item.get("pubYear")),
            doi=normalize_doi(item.get("doi")),
            abstract=strip_markup(item.get("abstractText")),
            venue=item.get("journalTitle"),
            landing_page_url=landing_page,
            source_apis=[self.name],
            source_ids={self.name: f"{source}:{source_id}"}
            if source and source_id
            else {},
            candidate_pdf_urls=pdf_urls,
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


def candidate_keys(candidate: Candidate) -> list[str]:
    keys: list[str] = []
    if candidate.doi:
        keys.append(f"doi:{candidate.doi}")
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


def merge_candidate_into(target: Candidate, incoming: Candidate) -> None:
    target.source_apis = dedupe_preserve(target.source_apis + incoming.source_apis)
    target.source_ids.update({key: value for key, value in incoming.source_ids.items() if value})
    target.candidate_pdf_urls = dedupe_preserve(
        target.candidate_pdf_urls + incoming.candidate_pdf_urls
    )
    target.matched_keywords = dedupe_preserve(
        target.matched_keywords + incoming.matched_keywords
    )
    target.discovered_via.extend(incoming.discovered_via)
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


def rank_candidates(candidates: list[Candidate], terms: ExtractedTerms) -> list[Candidate]:
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
        if candidate.candidate_pdf_urls:
            score += 0.1
        if candidate.doi:
            score += 0.03
        candidate.relevance_score = round(score, 5)
    return sorted(
        candidates,
        key=lambda item: (item.relevance_score, item.year or 0, bool(item.doi)),
        reverse=True,
    )


class PdfResolver:
    def __init__(self, http: HttpClient, unpaywall_email: str | None = None) -> None:
        self.http = http
        self.unpaywall_email = unpaywall_email

    def resolve(self, candidate: Candidate) -> list[str]:
        urls: list[str] = []
        attempts: list[dict[str, Any]] = []
        for url in candidate.candidate_pdf_urls:
            add_url(urls, url)
            attempts.append({"source": "source_metadata", "status": "candidate", "url": url})

        if candidate.doi and self.unpaywall_email:
            try:
                unpaywall_urls = self._resolve_unpaywall(candidate.doi)
                for url in unpaywall_urls:
                    add_url(urls, url)
                attempts.append(
                    {
                        "source": "unpaywall",
                        "status": "resolved" if unpaywall_urls else "no_pdf_url",
                        "url_count": len(unpaywall_urls),
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

        if candidate.landing_page_url:
            try:
                landing_urls = self._resolve_landing_page(candidate.landing_page_url)
                for url in landing_urls:
                    add_url(urls, url)
                attempts.append(
                    {
                        "source": "landing_page",
                        "status": "resolved" if landing_urls else "no_pdf_url",
                        "url_count": len(landing_urls),
                        "landing_page_url": candidate.landing_page_url,
                    }
                )
            except Exception as exc:  # pragma: no cover - network defensive path
                attempts.append(
                    {
                        "source": "landing_page",
                        "status": "error",
                        "reason": str(exc),
                        "landing_page_url": candidate.landing_page_url,
                    }
                )

        candidate.pdf_resolution = {
            "resolved_at": utc_now(),
            "urls": urls,
            "attempts": attempts,
        }
        return urls

    def _resolve_unpaywall(self, doi: str) -> list[str]:
        encoded_doi = urllib.parse.quote(doi, safe="/")
        data = self.http.get_json(
            f"https://api.unpaywall.org/v2/{encoded_doi}",
            params={"email": self.unpaywall_email},
        )
        urls: list[str] = []
        if data.get("is_oa") is False:
            return urls
        locations = [data.get("best_oa_location")] + ensure_list(data.get("oa_locations"))
        for location in locations:
            if not isinstance(location, dict):
                continue
            add_url(urls, location.get("url_for_pdf"))
            fallback = location.get("url")
            if looks_like_pdf_url(fallback):
                add_url(urls, fallback)
        return urls

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
        body, _, _ = self.http.request_bytes(
            landing_page_url,
            headers=headers,
        )
        if not body:
            return []
        try:
            html_text = body.decode("utf-8", errors="replace")
        except Exception:
            html_text = body.decode("latin-1", errors="replace")
        return extract_pdf_urls_from_html(html_text, landing_page_url)


class PdfDownloader:
    def __init__(
        self,
        http: HttpClient,
        output_dir: Path,
        max_pdf_mb: int = 80,
        force_download: bool = False,
    ) -> None:
        self.http = http
        self.output_dir = output_dir
        self.max_bytes = max_pdf_mb * 1024 * 1024
        self.force_download = force_download

    def download(self, candidate: Candidate, urls: list[str]) -> dict[str, Any]:
        filename = build_pdf_filename(candidate)
        destination = self.output_dir / filename
        attempts: list[dict[str, Any]] = []
        if destination.exists() and not self.force_download:
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

        if candidate.abstract:
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


def is_saved_document_status(status: str | None) -> bool:
    return status in {"downloaded", "already_exists", "abstract_only"}


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        "landing_page_url",
        "candidate_pdf_urls",
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
                    "landing_page_url": candidate.landing_page_url or "",
                    "candidate_pdf_urls": "; ".join(candidate.candidate_pdf_urls),
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


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    draft_path = Path(config["draft"]).expanduser()
    output_dir = Path(config["output"]).expanduser()
    logs_dir = Path(config.get("logs") or output_dir / "logs").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    draft_text = read_text_file(draft_path)
    extracted = extract_keywords(draft_text)
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
    if not config.get("dry_run"):
        http = HttpClient(
            timeout=float(config["timeout"]),
            retries=int(config["retries"]),
            backoff=float(config["backoff"]),
            rate_limit_delay=float(config["rate_limit_delay"]),
            email=config.get("email"),
        )
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

        candidates = rank_candidates(merge_candidates(candidates), extracted)
        resolver = PdfResolver(
            http,
            unpaywall_email=config.get("unpaywall_email")
            or os.getenv("UNPAYWALL_EMAIL")
            or config.get("email"),
        )
        downloader = PdfDownloader(
            http,
            output_dir=output_dir,
            max_pdf_mb=int(config["max_pdf_mb"]),
            force_download=bool(config.get("force_download")),
        )

        if config.get("metadata_only"):
            for candidate in candidates:
                candidate.download = {
                    "status": "not_attempted",
                    "reason": "metadata_only mode",
                }
        else:
            download_limit = config.get("max_downloads")
            max_downloads = int(download_limit) if download_limit is not None else None
            downloaded_count = 0
            for candidate in candidates:
                if max_downloads is not None and downloaded_count >= max_downloads:
                    candidate.download = {
                        "status": "not_attempted",
                        "reason": "max_downloads limit reached",
                    }
                    continue
                urls = resolver.resolve(candidate)
                result = downloader.download(candidate, urls)
                if is_saved_document_status(result.get("status")):
                    downloaded_count += 1
                    LOGGER.info("Document %s: %s", result["status"], candidate.title)
                else:
                    LOGGER.info("No document saved for: %s", candidate.title)
    else:
        LOGGER.info("Dry run requested; no API searches or downloads were attempted.")

    downloads = [candidate.download for candidate in candidates if candidate.download]
    failures = [
        {
            "title": candidate.title,
            "doi": candidate.doi,
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
        "candidate_count": len(candidates),
        "downloaded_count": sum(
            1
            for candidate in candidates
            if is_saved_document_status(candidate.download.get("status"))
        ),
        "keyword_search_counts": {
            source: dict(counter) for source, counter in keyword_search_counts.items()
        },
        "candidates": [dataclasses.asdict(candidate) for candidate in candidates],
    }
    download_log = {
        "project": "LitHarvest",
        "generated_at": utc_now(),
        "downloads": downloads,
        "failed_downloads": failures,
        "search_errors": search_errors,
    }

    manifest_path = logs_dir / "manifest.json"
    download_log_path = logs_dir / "download_log.json"
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
    }


DEFAULT_CONFIG: dict[str, Any] = {
    "max_queries": 8,
    "top_k_per_query": 10,
    "sources": DEFAULT_SOURCES,
    "year_from": None,
    "year_to": None,
    "email": os.getenv("LITHARVEST_EMAIL"),
    "unpaywall_email": os.getenv("UNPAYWALL_EMAIL"),
    "semantic_scholar_api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
    "timeout": 20.0,
    "retries": 2,
    "backoff": 1.5,
    "rate_limit_delay": 0.1,
    "max_pdf_mb": 80,
    "max_downloads": 50,
    "logs": None,
    "metadata_only": False,
    "force_download": False,
    "csv_report": False,
    "bibtex": False,
    "dry_run": False,
    "extra_queries": [],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect open scholarly PDFs from a draft using API-first metadata sources."
    )
    parser.add_argument("--draft", help="Path to the input draft text or markdown file.")
    parser.add_argument("--output", help="Folder where downloaded PDFs should be stored.")
    parser.add_argument("--logs", help="Folder for manifest.json and download_log.json.")
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
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, help="HTTP retry count.")
    parser.add_argument("--backoff", type=float, help="Retry backoff base in seconds.")
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        help="Minimum delay between outbound HTTP requests.",
    )
    parser.add_argument("--max-pdf-mb", type=int, help="Maximum accepted PDF size in MB.")
    parser.add_argument("--max-downloads", type=int, help="Maximum successful PDF downloads.")
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
    if args.config:
        config_path = Path(args.config).expanduser()
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a JSON object.")
        config.update(loaded)

    for key, value in vars(args).items():
        if key in {"config", "quiet", "verbose"}:
            continue
        if value is None:
            continue
        config[key] = value

    if isinstance(config.get("sources"), str):
        config["sources"] = parse_sources(config["sources"])
    else:
        config["sources"] = [
            normalize_source_name(source) for source in config.get("sources", DEFAULT_SOURCES)
        ]

    if not config.get("draft"):
        raise ValueError("--draft is required unless provided in --config.")
    if not config.get("output"):
        raise ValueError("--output is required unless provided in --config.")
    if config.get("year_from") and config.get("year_to"):
        if int(config["year_from"]) > int(config["year_to"]):
            raise ValueError("--year-from cannot be later than --year-to.")
    if int(config["max_queries"]) < 1:
        raise ValueError("--max-queries must be at least 1.")
    if int(config["top_k_per_query"]) < 1:
        raise ValueError("--top-k-per-query must be at least 1.")
    if int(config["max_pdf_mb"]) < 1:
        raise ValueError("--max-pdf-mb must be at least 1.")
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
        result = run_pipeline(config)
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
