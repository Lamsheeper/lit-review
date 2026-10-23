"""Collection downloads."""

from __future__ import annotations

from lit_review.core.io import utc_now


import hashlib


import logging

import math


import re


import threading


import urllib.error

import urllib.parse

import urllib.request


from dataclasses import dataclass


from html.parser import HTMLParser

from pathlib import Path

from typing import Any


from lit_review.core.http import HttpClient, is_pdf_response

from lit_review.collection.sources import (
    Candidate,
    add_doi_pdf_urls,
    add_identifier_pdf_urls,
    add_url,
    clean_url,
    dedupe_preserve,
    doi_to_url,
    ensure_list,
    has_pdf_file_hint,
    looks_like_pdf_url,
)
from lit_review.core.text import normalize_whitespace

from lit_review.core.text import content_tokens, normalize_title, slugify


LOGGER = logging.getLogger("litharvest")


@dataclass
class WebSearchResult:
    source: str
    query: str
    rank: int
    url: str
    title: str | None = None
    snippet: str | None = None


def add_absolute_url(urls: list[str], value: str | None, base_url: str) -> None:
    url = clean_url(value)
    if not url:
        return
    absolute = urllib.parse.urljoin(base_url, url)
    add_url(urls, absolute)


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
            name = (attr.get("name") or attr.get("property") or attr.get("itemprop") or "").lower()
            content = attr.get("content")
            if content and (name in PDF_META_NAMES or "pdf" in name or looks_like_pdf_url(content)):
                add_absolute_url(self.urls, content, self.base_url)
            return

        if tag == "link":
            href = attr.get("href")
            rel = attr.get("rel", "").lower()
            content_type = attr.get("type", "").lower()
            title = attr.get("title", "").lower()
            if href and ("pdf" in rel or "pdf" in content_type or "pdf" in title or looks_like_pdf_url(href)):
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


def quote_search_phrase(value: str) -> str:
    escaped = value.replace('"', " ").strip()
    return f'"{normalize_whitespace(escaped)}"'


def build_web_pdf_queries(candidate: Candidate, max_queries: int = 3) -> list[str]:
    queries: list[str] = []
    if candidate.doi:
        queries.append(f"{quote_search_phrase(candidate.doi)} pdf")
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
    result_text = " ".join(part for part in [result.title or "", result.snippet or "", result.url] if part)
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
                attempts.append({"source": "unpaywall", "status": "error", "reason": str(exc)})
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
        if (
            destination.exists()
            and destination.stat().st_size > 0
            and not self.force_download
            and not retrying_abstract_only
        ):
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
            content_line.append(f"({_pdf_escape(line)}) Tj")
            content_line.append("T*")
        content_stream = "BT\n/F1 12 Tf\n72 760 Td\n" + "\n".join(content_line) + "\nET\n"
        stream_bytes = content_stream.encode("latin-1", errors="replace")

        objs: list[bytes] = []
        objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        objs.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
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
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(startxref).encode("ascii") + b"\n%%EOF\n"
        )
        destination.write_bytes(body + b"".join(xref) + trailer)
        return destination.stat().st_size


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
