"""Shared execution boundary for command-line and UI stages."""

from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path
from typing import Any

from lit_review.core.context import research_context
from lit_review.core.io import read_json, write_json, write_jsonl
from lit_review.core.settings import FeatureSettings, CollectionSettings, AuthorSettings
from lit_review.core.taxonomy import load_taxonomy

FEATURE_OPERATIONS = {
    "features.extract",
    "features.filter",
    "features.annotate",
    "features.merge",
    "features.score",
}
PROVIDER_FIELDS = {
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "auth_mode",
    "vertex_project",
    "vertex_location",
    "reasoning_effort",
    "temperature",
    "timeout",
    "max_tokens",
}
COMMON_FEATURE_FIELDS = PROVIDER_FIELDS | {"taxonomy", "goal", "profile", "progress"}
FEATURE_FIELDS = {
    "features.extract": COMMON_FEATURE_FIELDS
    | {
        "input_mode",
        "paper_limit",
        "batch_chunks",
        "max_chunk_chars",
        "max_features_per_chunk",
        "max_features_per_pdf",
    },
    "features.filter": COMMON_FEATURE_FIELDS | {"batch_features"},
    "features.annotate": COMMON_FEATURE_FIELDS | {"batch_features"},
    "features.score": COMMON_FEATURE_FIELDS | {"batch_features"},
    "features.merge": COMMON_FEATURE_FIELDS
    | {
        "category_config",
        "top_k",
        "candidate_min_similarity",
        "batch_pairs",
        "embedding_model",
        "embedding_api_key_env",
        "embedding_auth_mode",
        "embedding_vertex_project",
        "embedding_vertex_location",
        "embedding_base_url",
        "embedding_task_type",
        "embedding_output_dimensionality",
        "embedding_batch_size",
        "cache",
    },
}
# Primary artifacts are stable contracts, independent of optional figures/reports.
ARTIFACTS = {
    "collect.summary": ("service_summary.json",),
    "collect.search": ("logs/manifest.json",),
    "collect.citations": ("logs/manifest.json",),
    "collect.authors": ("manifest.json", "authors.csv", "publications.csv"),
    "evidence.prepare": ("paper_index.json",),
    "evidence.convert": ("paper_index.json",),
    "evidence.index": ("paper_index.json", "chunks.jsonl", "index.sqlite"),
    "evidence.write": ("report.md",),
    "evidence.search": ("search_results.json",),
    "features.extract": ("features.jsonl", "feature_mentions.jsonl", "paper_features.jsonl"),
    "features.filter": ("filtered_features.jsonl", "rejected_features.jsonl", "curation_decisions.jsonl"),
    "features.annotate": (
        "annotated_features.jsonl",
        "excluded_annotations.jsonl",
        "annotation_decisions.jsonl",
    ),
    "features.merge": (
        "llm_merged_features.jsonl",
        "llm_merge_decisions.jsonl",
        "llm_merge_candidates.jsonl",
    ),
    "features.score": ("scored_features.jsonl", "scoring_decisions.jsonl"),
}


def feature_options(operation: str, options: dict[str, Any]) -> FeatureSettings:
    unknown = set(options) - FEATURE_FIELDS[operation]
    if unknown:
        raise ValueError(f"Options do not apply to {operation}: {', '.join(sorted(unknown))}")
    config = FeatureSettings.model_validate(options)
    if operation == "features.extract":
        invalid = set(options) & (
            {"batch_chunks", "max_chunk_chars", "max_features_per_chunk"}
            if config.input_mode == "pdf"
            else {"max_features_per_pdf"}
        )
        if invalid:
            raise ValueError(f"Options do not apply to {config.input_mode} extraction: {sorted(invalid)}")
    if operation == "features.merge" and config.category_config and config.category_config.exists():
        category = read_json(config.category_config)
        if not isinstance(category, dict) or not category:
            raise ValueError("Category embedding settings must be a nonempty object")
        for family, values in category.items():
            if not isinstance(values, dict) or set(values) - {"text_fields", "text_cleanup"}:
                raise ValueError(
                    f"Merge category {family}: only text_fields/text_cleanup apply; threshold/linkage belong to analysis clustering"
                )
    return config


def _call(function, values: dict):
    accepted = inspect.signature(function).parameters
    return function(**{key: value for key, value in values.items() if key in accepted})


def order_feature_inventory(output_dir: Path, operation: str, taxonomy: dict) -> None:
    import csv
    from lit_review.core.io import read_jsonl, write_csv

    rank = {family["id"]: index for index, family in enumerate(taxonomy["families"])}
    stem = Path(ARTIFACTS[operation][0]).stem

    def ordered(rows):
        return sorted(rows, key=lambda row: rank.get(row.get("category"), len(rank)))

    path = output_dir / (stem + ".jsonl")
    if path.exists():
        write_jsonl(path, ordered(read_jsonl(path)))
    path = output_dir / (stem + ".json")
    if path.exists():
        payload = read_json(path)
        if isinstance(payload, list):
            write_json(path, ordered(payload))
        elif isinstance(payload, dict) and isinstance(payload.get("features"), list):
            payload["features"] = ordered(payload["features"])
            write_json(path, payload)
    path = output_dir / (stem + ".csv")
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        if fields:
            write_csv(path, ordered(rows), fields)


def execute_feature(
    operation: str,
    options: dict,
    *,
    output_dir: Path,
    run_dir: Path | None = None,
    input_file: Path | None = None,
    input_dir: Path | None = None,
    force: bool = False,
    client=None,
    embedding_client=None,
) -> dict:
    from lit_review.features import extraction, filtering, annotation, merging, scoring
    from lit_review.features.records import read_feature_inventory
    from lit_review.embeddings.core import make_embedding_client

    config = feature_options(operation, options)
    taxonomy = load_taxonomy(config.taxonomy)
    families = {family["id"] for family in taxonomy["families"]}
    source = input_file or (input_dir / "features.jsonl" if input_dir else None)
    if source:
        rows = read_feature_inventory(source)
        unknown = {str(row.get("category")) for row in rows} - families
        if unknown:
            raise ValueError(f"Input contains families absent from taxonomy: {', '.join(sorted(unknown))}")
    transport = config.transport()
    kwargs = dict(
        transport,
        output_dir=output_dir,
        run_dir=run_dir,
        input_file=input_file,
        input_dir=input_dir,
        force=force,
        dry_run=False,
        client=client,
        goal_path=config.goal,
        taxonomy_path=config.taxonomy,
        batch_features=config.batch_features,
        max_tokens=config.max_tokens,
        batch_chunks_count=config.batch_chunks,
        paper_limit=config.paper_limit,
        input_mode=config.input_mode,
        max_chunk_chars=config.max_chunk_chars,
        max_features_per_chunk=config.max_features_per_chunk,
        max_features_per_pdf=config.max_features_per_pdf,
        show_progress=config.progress,
    )
    functions = {
        "features.extract": extraction.run_extraction,
        "features.filter": filtering.run_curation,
        "features.annotate": annotation.run_annotation,
        "features.score": scoring.run_scoring,
        "features.merge": merging.run_llm_merge,
    }
    if operation == "features.merge":
        if config.category_config:
            category = read_json(config.category_config)
            if not isinstance(category, dict):
                raise ValueError("Category embedding settings must be an object")
            for family, values in category.items():
                if not isinstance(values, dict) or set(values) - {"text_fields", "text_cleanup"}:
                    raise ValueError(
                        f"Merge category {family}: only text_fields/text_cleanup apply; threshold/linkage belong to analysis clustering"
                    )
        if embedding_client is None and source and rows:
            key = os.getenv(config.embedding_api_key_env)
            if not key and config.embedding_api_key_env == "GEMINI_API_KEY":
                key = os.getenv("GOOGLE_API_KEY")
            embedding_client = make_embedding_client(
                api_key=key,
                auth_mode=config.embedding_auth_mode or config.auth_mode,
                vertex_project=config.embedding_vertex_project or config.vertex_project,
                vertex_location=config.embedding_vertex_location or config.vertex_location,
                model=config.embedding_model,
                base_url=config.embedding_base_url,
                task_type=config.embedding_task_type,
                output_dimensionality=config.embedding_output_dimensionality,
                timeout=config.timeout,
            )
            if embedding_client is None:
                raise ValueError(f"Missing embedding credential: {config.embedding_api_key_env}; or configure Vertex AI ADC")
        kwargs.update(
            category_config=config.category_config,
            top_k=config.top_k,
            candidate_min_similarity=config.candidate_min_similarity,
            batch_pairs=config.batch_pairs,
            cache_path=config.cache,
            embedding_client=embedding_client,
            embedding_model=config.embedding_model,
            embedding_task_type=config.embedding_task_type,
            embedding_output_dimensionality=config.embedding_output_dimensionality,
            embedding_batch_size=config.embedding_batch_size,
        )
    empty = (source is not None and not rows) or (
        operation == "features.extract"
        and run_dir is not None
        and not read_json(run_dir / "paper_index.json", {}).get("papers")
    )
    if empty:
        # An empty, successfully computed inventory is a valid stage result.
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in ARTIFACTS[operation]:
            write_jsonl(output_dir / filename, [])
        stem = {
            "features.extract": "features",
            "features.filter": "filtered_features",
            "features.annotate": "annotated_features",
            "features.merge": "llm_merged_features",
            "features.score": "scored_features",
        }[operation]
        write_json(output_dir / (stem + ".json"), [])
        (output_dir / (stem + ".csv")).write_text("feature_key,category,feature_name\n", encoding="utf-8")
        return {"outputs": {stem: 0}, "failures": []}
    with research_context(taxonomy, config.profile):
        result = _call(functions[operation], kwargs)
    if result.get("failures") or any(
        value for key, value in result.get("counts", {}).items() if key.startswith("failed_")
    ):
        raise RuntimeError(f"{operation} has failed items; rerun to resume completed checkpoints")
    order_feature_inventory(output_dir, operation, taxonomy)
    return result


def execute_operation(
    operation: str, options: dict, *, output_dir: Path, force: bool = False, **runtime
) -> dict:
    if operation in FEATURE_OPERATIONS:
        locations = {
            key: Path(options[key]) for key in ("run_dir", "input_file", "input_dir") if key in options
        }
        return execute_feature(
            operation,
            {k: v for k, v in options.items() if k not in locations},
            output_dir=output_dir,
            force=force,
            **locations,
            **runtime,
        )
    if operation == "collect.summary":
        from lit_review.collection.summary import summarize_logs

        payload = summarize_logs(Path(options["logs_dir"]))
        write_json(output_dir / "service_summary.json", payload)
        return payload
    if operation in {"collect.search", "collect.citations"}:
        from lit_review.collection.harvest import run_harvest, llm_provider_default_model, llm_provider_env_api_key

        config = CollectionSettings.for_operation(operation, options).model_dump(mode="json")
        config.update(
            output=str(output_dir),
            logs=str(output_dir / "logs"),
            dry_run=False,
            harvest_mode="citations" if operation.endswith("citations") else "draft",
        )
        config["llm_model"] = config.get("llm_model") or llm_provider_default_model(config["llm_provider"])
        config["llm_api_key"] = (
            os.getenv(config["api_key_env"]) if config.get("api_key_env")
            else llm_provider_env_api_key(config["llm_provider"])
        )
        config["email"] = config.get("email") or os.getenv("LITHARVEST_EMAIL")
        config["semantic_scholar_api_key"] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        config["unpaywall_email"] = config.get("unpaywall_email") or os.getenv("UNPAYWALL_EMAIL")
        return run_harvest(config)
    if operation == "collect.authors":
        from lit_review.collection.authors import run

        values = AuthorSettings.model_validate(options).model_dump(mode="json")
        values.update(output=str(output_dir))
        values["email"] = values.get("email") or os.getenv("LITHARVEST_EMAIL")
        values["unpaywall_email"] = values.get("unpaywall_email") or os.getenv("UNPAYWALL_EMAIL")
        return run(values)
    from lit_review.evidence import pdf, review

    if operation in {"evidence.prepare", "evidence.convert"}:
        fn = pdf.prepare_direct_pdf_index if operation.endswith("prepare") else pdf.convert_pdfs_to_markdown
        payload = _call(
            fn,
            dict(
                papers_dir=Path(options["papers"]),
                manifest_path=Path(options["manifest"]) if options.get("manifest") else None,
                out_dir=output_dir,
                extractor=options.get("extractor", "auto"),
                force=force,
            ),
        )
        if payload.get("failed_count"):
            raise RuntimeError(f"{operation} failed for {payload['failed_count']} PDFs; rerun to retry")
        if options.get("selection"):
            from lit_review.collection.harvest import candidate_from_dict, stable_candidate_id

            selected = set(read_json(Path(options["selection"]), {}).get("candidate_ids", []))
            candidates = read_json(Path(options["manifest"]), {}).get("candidates", [])
            allowed = {
                str(Path(row["download"]["path"]).resolve())
                for row in candidates
                if stable_candidate_id(candidate_from_dict(row)) in selected
                and row.get("download", {}).get("path")
            }
            payload["papers"] = [
                paper
                for paper in payload["papers"]
                if str(Path(paper.get("pdf_path", "")).resolve()) in allowed
            ]
            payload["paper_count"] = len(payload["papers"])
            write_json(output_dir / "paper_index.json", payload)
        return payload
    if operation == "evidence.index":
        source = Path(options["run_dir"])
        if source.resolve() != output_dir.resolve():
            shutil.copytree(source, output_dir, dirs_exist_ok=True)
        return review.build_index(
            output_dir,
            chunk_words_count=options.get("chunk_words", 700),
            overlap_words=options.get("overlap_words", 80),
            min_chunk_words=options.get("min_chunk_words", 40),
            force=force,
        )
    if operation == "evidence.search":
        results = review.search_index(
            Path(options["run_dir"]) / "index.sqlite", options["query"], limit=int(options.get("limit", 10))
        )
        write_json(output_dir / "search_results.json", results)
        return {"result_count": len(results)}
    if operation == "evidence.write":
        allowed = {
            "run_dir",
            "goal",
            "provider",
            "auth_mode",
            "vertex_project",
            "vertex_location",
            "model",
            "base_url",
            "reasoning_effort",
            "max_questions",
            "per_question",
            "max_evidence",
            "evidence_chars",
            "section_evidence",
            "temperature",
            "api_key_env",
            "profile",
        }
        if set(options) - allowed:
            raise ValueError(f"Unknown report options: {sorted(set(options) - allowed)}")
        values = {k: v for k, v in options.items() if k not in {"goal", "api_key_env", "profile", "provider"}}
        provider = options.get("provider", "openai")
        key_env = options.get("api_key_env") or ("GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY")
        key = os.getenv(key_env)
        if not key and provider == "gemini" and not options.get("api_key_env"):
            key = os.getenv("GOOGLE_API_KEY")
        values.update(
            run_dir=Path(options["run_dir"]),
            goal_path=Path(options["goal"]),
            output_path=output_dir / "report.md",
            api_key=key,
            api_provider=provider,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with research_context({}, options.get("profile")):
            return review.write_report(**values)
    if operation.startswith("analysis."):
        from lit_review.analysis.dispatch import run_analysis

        return run_analysis(operation.split(".", 1)[1], options, output_dir)
    raise ValueError(f"Unknown operation: {operation}")
