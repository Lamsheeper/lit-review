"""Validated settings used by both the command line and guided UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    @model_validator(mode="before")
    @classmethod
    def reject_literal_credentials(cls, value):
        def inspect(item):
            if isinstance(item, dict):
                for key, nested in item.items():
                    if "api_key" in str(key).lower() and not str(key).endswith("_env"):
                        raise ValueError("Use credential environment-variable names, not API keys in configs")
                    inspect(nested)
            elif isinstance(item, list):
                for nested in item:
                    inspect(nested)

        inspect(value)
        return value


class ProviderSettings(Settings):
    provider: Literal["gemini", "openai"] = "gemini"
    model: str = "gemini-2.5-flash"
    base_url: str | None = None
    api_key_env: str | None = None
    auth_mode: Literal["auto", "api_key", "adc"] = "auto"
    vertex_project: str | None = None
    vertex_location: str | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    timeout: float = Field(default=600, gt=0)
    max_tokens: int = Field(default=6000, ge=1)

    @model_validator(mode="after")
    def valid_authentication(self):
        if self.provider == "openai" and (self.auth_mode == "adc" or self.vertex_project):
            raise ValueError("Vertex AI ADC authentication requires the Gemini provider")
        return self

    def transport(self) -> dict:
        key_env = self.api_key_env or ("GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY")
        return {
            "api_key": os.getenv(key_env) or (
                os.getenv("GOOGLE_API_KEY") if self.provider == "gemini" and not self.api_key_env else None
            ),
            "auth_mode": self.auth_mode,
            "vertex_project": self.vertex_project,
            "vertex_location": self.vertex_location,
            "model": self.model,
            "base_url": self.base_url
            or (
                "https://generativelanguage.googleapis.com/v1beta"
                if self.provider == "gemini"
                else "https://api.openai.com/v1"
            ),
            "api_provider": self.provider,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }


class FeatureSettings(ProviderSettings):
    taxonomy: Path
    goal: Path | None = None
    profile: Literal["media_attributes"] | None = None
    batch_features: int = Field(default=25, ge=1)
    input_mode: Literal["pdf", "chunks"] = "pdf"
    max_features_per_pdf: int = Field(default=40, ge=1)
    max_features_per_chunk: int = Field(default=12, ge=1)
    batch_chunks: int = Field(default=4, ge=1)
    max_chunk_chars: int = Field(default=6000, ge=1)
    paper_limit: int | None = Field(default=None, ge=1)
    top_k: int = Field(default=10, ge=1)
    batch_pairs: int = Field(default=25, ge=1)
    candidate_min_similarity: float | None = Field(default=None, ge=-1, le=1)
    category_config: Path | None = None
    embedding_model: str = "gemini-embedding-001"
    embedding_api_key_env: str = "GEMINI_API_KEY"
    embedding_auth_mode: Literal["auto", "api_key", "adc"] | None = None
    embedding_vertex_project: str | None = None
    embedding_vertex_location: str | None = None
    embedding_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    embedding_task_type: str = "CLUSTERING"
    embedding_output_dimensionality: int = Field(default=768, ge=1)
    embedding_batch_size: int = Field(default=50, ge=1)
    cache: Path | None = None
    progress: bool = True


class StageSpec(Settings):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    operation: str
    options: dict = Field(default_factory=dict)


class PipelineConfig(Settings):
    version: Literal[1] = 1
    output_root: Path
    inputs: dict = Field(default_factory=dict)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    profile: Literal["media_attributes"] | None = None
    stages: list[StageSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_stages(self):
        ids = [stage.id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("Stage IDs must be unique")
        return self


class CollectionSettings(Settings):
    draft: Path | None = None
    core_pdf: list[Path] = Field(default_factory=list)
    core_pdf_dir: Path | None = None
    citation_extractor: Literal["direct_pdf", "pymupdf_markdown"] = "direct_pdf"
    llm_provider: Literal["gemini", "openai"] = "gemini"
    llm_model: str | None = None
    api_key_env: str | None = None
    auth_mode: Literal["auto", "api_key", "adc"] = "auto"
    vertex_project: str | None = None
    vertex_location: str | None = None
    max_output_tokens: int = Field(default=0, ge=0)
    max_text_chars: int = Field(default=0, ge=0)
    sources: list[Literal["openalex", "semantic_scholar", "crossref", "europe_pmc"]] = Field(
        default_factory=lambda: ["openalex", "semantic_scholar", "crossref", "europe_pmc"], min_length=1
    )
    max_queries: int = Field(default=8, ge=1)
    top_k_per_query: int = Field(default=10, ge=1)
    year_from: int | None = Field(default=None, ge=0)
    year_to: int | None = Field(default=None, ge=0)
    email: str | None = None
    unpaywall_email: str | None = None
    web_search_sources: list[Literal["brave", "serpapi"]] = Field(default_factory=list)
    web_search_max_candidates: int = Field(default=100, ge=0)
    web_search_max_results: int = Field(default=5, ge=1)
    web_search_queries_per_candidate: int = Field(default=3, ge=1)
    web_search_country: str = "us"
    web_search_language: str = "en"
    serpapi_engine: str = "google"
    timeout: float = Field(default=20, gt=0)
    retries: int = Field(default=2, ge=0)
    backoff: float = Field(default=1.5, ge=0)
    rate_limit_delay: float = Field(default=0.1, ge=0)
    download_workers: int = Field(default=1, ge=1)
    max_pdf_mb: int = Field(default=80, ge=1)
    max_downloads: int = Field(default=50, ge=0)
    candidates_json: Path | None = None
    refresh_candidates: bool = False
    refresh_citation_extractions: bool = False
    retry_failed_downloads: bool = False
    progress: bool = True
    metadata_only: bool = False
    force_download: bool = False
    allow_abstract_fallback: bool = False
    csv_report: bool = False
    bibtex: bool = False
    extra_queries: list[dict] = Field(default_factory=list)
    selected_candidate_ids: list[str] | None = None
    relevance_scoring: Path | None = None

    @model_validator(mode="after")
    def valid_years(self):
        if self.llm_provider == "openai" and (self.auth_mode == "adc" or self.vertex_project):
            raise ValueError("Vertex AI ADC authentication requires the Gemini provider")
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from cannot be later than year_to")
        return self

    @classmethod
    def for_operation(cls, operation: str, options: dict):
        settings = cls.model_validate(options)
        if operation == "collect.search":
            invalid = set(options) & {
                "core_pdf",
                "core_pdf_dir",
                "citation_extractor",
                "llm_provider",
                "llm_model",
                "api_key_env",
                "auth_mode",
                "vertex_project",
                "vertex_location",
                "max_output_tokens",
                "max_text_chars",
                "refresh_citation_extractions",
            }
            if invalid:
                raise ValueError(f"Options do not apply to collection search: {sorted(invalid)}")
            if settings.draft is None:
                raise ValueError("Collection search requires draft")
        elif operation == "collect.citations":
            if settings.draft is not None or settings.extra_queries:
                raise ValueError("Draft/query-generation settings do not apply to citation collection")
            if not (settings.core_pdf or settings.core_pdf_dir):
                raise ValueError("Citation collection requires core_pdf or core_pdf_dir")
        else:
            raise ValueError(f"Unknown collection operation: {operation}")
        return settings


class AuthorSettings(Settings):
    roster: Path
    year_from: int | None = None
    year_to: int | None = None
    max_publications_per_author: int | None = Field(default=None, ge=1)
    include_needs_review: bool = False
    download: bool = False
    listed_pdfs_only: bool = False
    max_downloads_per_author: int = Field(default=20, ge=1)
    max_pdf_mb: int = Field(default=80, ge=1)
    email: str | None = None
    unpaywall_email: str | None = None
    timeout: float = Field(default=30, gt=0)
    retries: int = Field(default=2, ge=0)
    backoff: float = Field(default=1.5, ge=0)
    rate_limit_delay: float = Field(default=0.12, ge=0)

    @model_validator(mode="after")
    def valid_years(self):
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from cannot be later than year_to")
        return self
