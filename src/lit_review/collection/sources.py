"""Collection sources."""

from __future__ import annotations


import html


import os

import re


import urllib.error

import urllib.parse

import urllib.request


from dataclasses import dataclass, field


from typing import Any


from lit_review.core.http import HttpClient, safe_int

from lit_review.core.text import normalize_whitespace, strip_markup


DEFAULT_SOURCES = ["openalex", "semantic_scholar", "crossref", "europe_pmc"]


@dataclass
class SearchQuery:
    bucket: str
    text: str
    terms: list[str] = field(default_factory=list)


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
        r"^https?://(?:www\.)?aclanthology\.org/",
        "",
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
        landing_page = primary_location.get("landing_page_url") or doi_to_url(normalize_doi(item.get("doi")))
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

    def _candidate_from_paper(self, item: dict[str, Any], query: SearchQuery) -> Candidate | None:
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

    def _candidate_from_result(self, item: dict[str, Any], query: SearchQuery) -> Candidate | None:
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
            source_ids={self.name: f"{source}:{source_id}"} if source and source_id else {},
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
                    api_key=config.get("semantic_scholar_api_key") or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
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
