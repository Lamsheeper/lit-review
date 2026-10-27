#!/usr/bin/env python3
"""LitHarvest: draft-to-PDF literature collection pipeline.

This module intentionally avoids Google Scholar scraping and browser
automation. It talks to metadata APIs, resolves open-access PDFs where those
APIs expose them, and records every candidate and download outcome.
"""

from __future__ import annotations

from lit_review.core.google_auth import use_vertex

import csv
import dataclasses
import hashlib
import json
import math
import os
import re
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


from lit_review.collection.relevance import load_relevance_profile

from lit_review.core.http import HttpClient, VERSION, safe_int

from lit_review.collection.sources import (
    Candidate,
    SearchQuery,
    build_clients,
    clean_url,
    dedupe_preserve,
    doi_to_url,
    normalize_doi,
    normalize_source_name,
    parse_sources,
    safe_float,
)
from lit_review.core.text import normalize_whitespace, strip_markup

from lit_review.collection.downloads import (
    LOGGER,
    PdfDownloader,
    PdfResolver,
    WebPdfSearcher,
    WebSearchProvider,
    WebSearchResult,
    build_pdf_filename,
    slugify,
)
from lit_review.core.text import STOPWORDS, content_tokens, normalize_title
from lit_review.core.io import utc_now

from lit_review.collection.progress import PhaseProgressReporter, ProgressReporter

from lit_review.collection.citations import (
    candidates_from_citation_extractions,
    citation_extraction_paths,
    citation_lookup_progress_payload,
    collect_core_pdfs,
    enrich_citation_candidates,
    extract_citations_from_core_pdf,
    merge_candidates,
    safe_pdf_stem,
    validate_citation_bibliography,
)
from lit_review.core.io import write_json


HARVEST_MODES = ("draft", "citations")
CITATION_EXTRACTORS = ("direct_pdf", "pymupdf_markdown")
LLM_PROVIDERS = ("gemini", "openai")
DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


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


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Draft file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Draft path is not a file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


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

    for keyword in keywords[:max_queries]:
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
        queries.append(SearchQuery(bucket=bucket, text=text, terms=terms or content_tokens(text)))
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
        "open_access": float(bool(candidate.candidate_pdf_urls or candidate.candidate_landing_page_urls)),
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
                term for term in criterion["terms"] if relevance_term_matches(term, field_text[field_name])
            ]
            field_scores[field_name] = float(bool(matches))
            if matches:
                matched_terms[field_name] = matches
        criterion_score = (
            sum(field_scores[field_name] * field_weights[field_name] for field_name in field_weights)
            / field_weight_total
        )
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
            candidate.relevance_score, candidate.relevance_score_breakdown = profile_relevance_score(
                candidate, relevance_profile
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


def should_use_web_search_for_candidate(
    web_searcher: WebPdfSearcher | None,
    max_web_search_candidates: int | None,
    web_search_count: int,
) -> bool:
    if not web_searcher:
        return False
    if max_web_search_candidates == 0:
        return False
    if max_web_search_candidates is not None and web_search_count >= max_web_search_candidates:
        return False
    return True


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
        1 for candidate in candidates if is_full_document_status(candidate.download.get("status"))
    )
    web_search_count = 0
    worker_count = max(1, download_workers)
    pending_candidates = [
        candidate
        for candidate in candidates
        if not is_terminal_download_status(candidate.download.get("status"))
    ]
    progress_candidates = [candidate for idx, candidate in enumerate(candidates) if idx not in resume_indices]
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
                    "candidate_landing_page_urls": "; ".join(candidate.candidate_landing_page_urls),
                    "resolved_pdf_urls": "; ".join(candidate.pdf_resolution.get("urls", [])),
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
            str(key): str(value) for key, value in kwargs["source_ids"].items() if value is not None
        }

    for key in ["relevance_score_breakdown", "pdf_resolution", "download"]:
        if not isinstance(kwargs.get(key), dict):
            kwargs[key] = {}

    discovered_via = kwargs.get("discovered_via")
    if isinstance(discovered_via, list):
        kwargs["discovered_via"] = [item for item in discovered_via if isinstance(item, dict)]
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
        "keyword_search_counts": {source: dict(counter) for source, counter in keyword_search_counts.items()},
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
        relevance_profile = load_relevance_profile(Path(config["relevance_scoring"]).expanduser())
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
                            candidate.matched_keywords = candidate_matches_keywords(
                                candidate, extracted.keywords
                            )
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
            candidates = ensure_candidate_ids(rank_candidates(candidates, extracted, relevance_profile))
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
            candidates = ensure_candidate_ids(rank_candidates(candidates, extracted, relevance_profile))
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
    else:
        LOGGER.info("Dry run requested; no API searches or downloads were attempted.")
        candidate_cache["status"] = "not_used_dry_run"

    download_records = [
        download_log_record(candidate, candidate.download) for candidate in candidates if candidate.download
    ]
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
            1 for candidate in candidates if is_saved_document_status(candidate.download.get("status"))
        ),
        "keyword_search_counts": {source: dict(counter) for source, counter in keyword_search_counts.items()},
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
        extracted_reference_count = sum(len(extraction.get("references") or []) for extraction in extractions)
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

                vertex = config.get("llm_provider") == "gemini" and use_vertex(
                    config.get("llm_api_key"), auth_mode=config.get("auth_mode", "auto"),
                    project=config.get("vertex_project"),
                )
                if not config.get("llm_api_key") and not vertex:
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
        download_log_record(candidate, candidate.download) for candidate in candidates if candidate.download
    ]
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
            1 for candidate in candidates if is_saved_document_status(candidate.download.get("status"))
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
    download_log["downloads"] = [record["download"] for record in download_records]
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
