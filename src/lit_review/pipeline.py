"""Config-driven, resumable stage execution with content-based provenance."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lit_review import __version__
from lit_review.core.io import read_json, read_jsonl, utc_now, write_json
from lit_review.core.settings import PipelineConfig
from lit_review.core.taxonomy import load_taxonomy
from lit_review.services import ARTIFACTS, FEATURE_OPERATIONS, execute_operation, feature_options

PATH_KEYS = {
    "output_root",
    "papers",
    "manifest",
    "draft",
    "goal",
    "taxonomy",
    "run_dir",
    "input_file",
    "input_dir",
    "roster",
    "category_config",
    "cache",
    "core_pdf",
    "core_pdf_dir",
    "relevance_scoring",
    "known",
    "annotated_features",
    "features",
    "mapped_csv",
    "scored_features",
    "embedding_merge_trace",
    "expected",
    "selection",
    "known_set",
    "scored_jsonl",
    "set_1",
    "set_2",
    "feature_set_1",
    "feature_set_2",
    "logs_dir",
    "candidates_json",
    "overrides",
    "synonym_review",
    "embedding_trace",
}
LOCATION_KEYS = {"run_dir", "input_file", "input_dir"}
ANALYSIS = {
    "analysis.calibrate",
    "analysis.metadata",
    "analysis.cluster",
    "analysis.compare",
    "analysis.tsne",
    "analysis.wikidata",
}
SUPPORTED = set(ARTIFACTS) | ANALYSIS


def _paths(values: Any, base: Path, key: str = "") -> Any:
    if isinstance(values, dict):
        return {name: _paths(value, base, name) for name, value in values.items()}
    if isinstance(values, list):
        return [_paths(value, base, key) for value in values]
    if key in PATH_KEYS and values is not None:
        p = Path(values).expanduser()
        return str((base / p).resolve() if not p.is_absolute() else p.resolve())
    return values


def load_pipeline(path: Path, *, overrides: dict | None = None) -> PipelineConfig:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Pipeline config must be a JSON object")
    payload.update(overrides or {})
    return PipelineConfig.model_validate(_paths(payload, path.resolve().parent))


def _digest_path(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return "missing"
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.rglob("*") if p.is_file() and p.name not in {"manifest.json", ".lock"})
    )
    for file in files:
        digest.update(str(file.relative_to(path) if path.is_dir() else file.name).encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


from functools import lru_cache


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        if "node_modules" not in path.parts:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def producer_state(path: Path) -> dict:
    for folder in (path, *path.parents):
        if folder.parent.name == "stages":
            manifest_path = folder.parent.parent / "manifest.json"
            if manifest_path.exists():
                record = read_json(manifest_path, {}).get("stages", {}).get(folder.name, {})
                return (
                    {"fingerprint": record.get("fingerprint"), "execution_id": record.get("execution_id")}
                    if record
                    else {}
                )
    return {}


def fingerprint(operation: str, options: dict) -> str:
    inputs = {}
    for key, value in options.items():
        if key in PATH_KEYS - {"cache"} and value:
            paths = value if isinstance(value, list) else [value]
            inputs[key] = [
                {"content": _digest_path(Path(path)), "producer": producer_state(Path(path))}
                for path in paths
            ]
    payload = {
        "release": __version__,
        "code": code_fingerprint(),
        "operation": operation,
        "options": options,
        "inputs": inputs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def compile_pipeline(config: PipelineConfig) -> list[dict]:
    allowed_inputs = {
        "run_dir",
        "papers",
        "manifest",
        "input_file",
        "input_dir",
        "taxonomy",
        "goal",
        "draft",
        "core_pdf",
        "core_pdf_dir",
        "roster",
        "relevance_scoring",
        "logs_dir",
    }
    if set(config.inputs) - allowed_inputs:
        raise ValueError(f"Unknown shared inputs: {sorted(set(config.inputs) - allowed_inputs)}")
    plan = []
    evidence = config.inputs.get("run_dir")
    papers = config.inputs.get("papers")
    manifest = config.inputs.get("manifest")
    feature_file = config.inputs.get("input_file")
    feature_dir = config.inputs.get("input_dir")
    for stage in config.stages:
        op = stage.operation
        if op not in SUPPORTED:
            raise ValueError(f"Unknown operation: {op}")
        output = config.output_root / "stages" / stage.id
        options: dict[str, Any] = {}
        if op in FEATURE_OPERATIONS:
            options.update(config.providers.model_dump(exclude_none=True))
            options.update({k: v for k, v in config.inputs.items() if k in {"taxonomy", "goal"}})
            if config.profile:
                options["profile"] = config.profile
            if op == "features.extract":
                options["run_dir"] = evidence
            elif op == "features.filter":
                options["input_dir"] = feature_dir
            else:
                options["input_file"] = feature_file
        elif op in {"evidence.prepare", "evidence.convert"}:
            options.update(papers=papers, manifest=manifest)
        elif op in {"evidence.index", "evidence.write", "evidence.search"}:
            options["run_dir"] = evidence
            if op.endswith("write"):
                provider = config.providers
                options.update(
                    model=provider.model,
                    api_key_env=provider.api_key_env
                    or ("GEMINI_API_KEY" if provider.provider == "gemini" else "OPENAI_API_KEY"),
                    base_url=provider.base_url
                    or (
                        "https://generativelanguage.googleapis.com/v1beta/openai"
                        if provider.provider == "gemini"
                        else "https://api.openai.com/v1"
                    ),
                    reasoning_effort=provider.reasoning_effort,
                    temperature=provider.temperature,
                )
                options["goal"] = config.inputs.get("goal")
        elif op.startswith("collect."):
            options.update(
                {
                    k: v
                    for k, v in config.inputs.items()
                    if k in {"draft", "core_pdf", "core_pdf_dir", "roster", "relevance_scoring"}
                }
            )
        if op == "collect.summary":
            options = (
                {"logs_dir": str(Path(manifest).parent)}
                if manifest
                else {k: v for k, v in config.inputs.items() if k == "logs_dir"}
            )
        if config.profile and op.startswith(("analysis.", "evidence.write")):
            options["profile"] = config.profile
        options.update(stage.options)
        options = {k: v for k, v in options.items() if v is not None}
        if op in FEATURE_OPERATIONS:
            expected_location = (
                {"run_dir"}
                if op == "features.extract"
                else {"input_dir"}
                if op == "features.filter"
                else {"input_file"}
            )
            if set(options) & (LOCATION_KEYS - expected_location):
                raise ValueError(f"Input options do not apply to {op}")
            feature_options(op, {k: v for k, v in options.items() if k not in LOCATION_KEYS})
        required = {
            "collect.summary": {"logs_dir"},
            "evidence.search": {"run_dir", "query"},
            "collect.search": {"draft"},
            "collect.citations": set(),
            "collect.authors": {"roster"},
            "evidence.prepare": {"papers"},
            "evidence.convert": {"papers"},
            "evidence.index": {"run_dir"},
            "evidence.write": {"run_dir", "goal"},
            "features.extract": {"run_dir", "taxonomy"},
            "features.filter": {"input_dir", "taxonomy"},
            "features.annotate": {"input_file", "taxonomy"},
            "features.merge": {"input_file", "taxonomy"},
            "features.score": {"input_file", "taxonomy"},
        }.get(op, set())
        missing = required - set(options)
        if missing:
            raise ValueError(f"Stage {stage.id} requires: {', '.join(sorted(missing))}")
        if op == "collect.citations" and not (options.get("core_pdf") or options.get("core_pdf_dir")):
            raise ValueError("Citation collection requires core_pdf or core_pdf_dir")
        if op in {"evidence.prepare", "evidence.convert", "evidence.index"}:
            evidence = str(output)
        if op in {"collect.search", "collect.citations"}:
            papers = str(output)
            manifest = str(output / "logs/manifest.json")
        if op in FEATURE_OPERATIONS:
            feature_dir = str(output)
            feature_file = str(output / ARTIFACTS[op][0])
        plan.append(
            {
                "id": stage.id,
                "operation": op,
                "options": options,
                "output_dir": str(output),
                "artifacts": [str(output / p) for p in ARTIFACTS.get(op, ())],
            }
        )
    validate_plan(plan)
    return plan


def validate_plan(plan: list[dict]) -> None:
    planned: list[Path] = []
    for stage in plan:
        for key, value in stage["options"].items():
            if key not in PATH_KEYS or not value or key in {"cache"}:
                continue
            paths = value if isinstance(value, list) else [value]
            for path in paths:
                p = Path(path)
                output_dir = Path(stage["output_dir"])
                if p == output_dir or output_dir in p.parents or (p.is_dir() and p in output_dir.parents):
                    raise ValueError(f"Stage {stage['id']}: output must be separate from {key} input")
                future = any(p == root or root in p.parents for root in planned)
                if not p.exists() and not future:
                    raise ValueError(f"Stage {stage['id']}: missing {key}: {p}")
                if p.exists():
                    directory_keys = {"papers", "run_dir", "input_dir", "core_pdf_dir", "logs_dir"}
                    if key in directory_keys and not p.is_dir():
                        raise ValueError(f"Stage {stage['id']}: {key} must be a directory")
                    if key not in directory_keys and not p.is_file():
                        raise ValueError(f"Stage {stage['id']}: {key} must be a file")
                if key == "taxonomy" and p.exists():
                    load_taxonomy(p)
        op = stage["operation"]
        options = stage["options"]
        if op in {"evidence.prepare", "evidence.convert"}:
            unknown = set(options) - {"papers", "manifest", "extractor", "selection"}
            if op.endswith("prepare") and "extractor" in options:
                unknown.add("extractor")
        elif op == "evidence.index":
            unknown = set(options) - {"run_dir", "chunk_words", "overlap_words", "min_chunk_words"}
        elif op == "collect.summary":
            unknown = set(options) - {"logs_dir"}
        elif op == "evidence.search":
            unknown = set(options) - {"run_dir", "query", "limit"}
        elif op in {"collect.search", "collect.citations"}:
            from lit_review.core.settings import CollectionSettings

            CollectionSettings.for_operation(op, options)
            unknown = set()
        elif op == "collect.authors":
            from lit_review.core.settings import AuthorSettings

            AuthorSettings.model_validate(options)
            unknown = set()
        elif op.startswith("analysis."):
            from lit_review.analysis.dispatch import analysis_arguments

            analysis_arguments(op.split(".", 1)[1], options, Path(stage["output_dir"]))
            unknown = set()
        else:
            unknown = set()
        if unknown:
            raise ValueError(f"Stage {stage['id']}: options do not apply: {sorted(unknown)}")
        if any("api_key" in key and not key.endswith("_env") for key in options):
            raise ValueError("Use credential environment-variable names, not API keys in configs")
        planned.append(Path(stage["output_dir"]))


def valid_artifacts(paths: list[str]) -> bool:
    if not paths:
        return False
    try:
        for value in paths:
            p = Path(value)
            if not p.is_file():
                return False
            if p.suffix == ".jsonl":
                read_jsonl(p)
            elif p.suffix == ".json":
                read_json(p)
            elif p.suffix == ".sqlite":
                import sqlite3

                with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as connection:
                    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        return False
        return True
    except (ValueError, OSError):
        return False


def stage_current(stage: dict, record: dict | None) -> bool:
    return bool(
        record
        and record.get("status") in {"completed", "imported"}
        and record.get("fingerprint") == fingerprint(stage["operation"], stage["options"])
        and valid_artifacts(record.get("artifacts", []))
        and record.get("artifact_hashes") == {p: _digest_path(Path(p)) for p in record.get("artifacts", [])}
    )


@contextmanager
def run_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    for attempt in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            try:
                pid = int(lock.read_text())
                os.kill(pid, 0)
            except (ProcessLookupError, ValueError):
                if attempt:
                    raise RuntimeError(f"Run is locked: {root}")
                lock.unlink(missing_ok=True)
                continue
            raise RuntimeError(f"Run already active: {root}")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_pipeline(
    config: PipelineConfig,
    *,
    dry_run: bool = False,
    stages: list[str] | None = None,
    rerun: list[str] | None = None,
    executor=None,
) -> dict:
    plan = compile_pipeline(config)
    ids = {stage["id"] for stage in plan}
    unknown = (set(stages or []) | set(rerun or [])) - ids
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    selected = set(stages or ids)
    if dry_run:
        return {
            "version": 1,
            "dry_run": True,
            "stages": [
                dict(stage, planned_outputs=stage["artifacts"]) for stage in plan if stage["id"] in selected
            ],
        }
    execute = executor or execute_operation
    root = config.output_root
    with run_lock(root):
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path, {"version": 1, "created_at": utc_now(), "stages": {}})
        manifest.update(release=__version__, updated_at=utc_now(), config=config.model_dump(mode="json"))
        records = manifest["stages"]
        invalidated = False
        for stage in plan:
            sid = stage["id"]
            previous = records.get(sid, {})
            changed = not stage_current(stage, previous)
            forced = sid in set(rerun or [])
            if invalidated or changed or forced:
                if previous:
                    previous["status"] = "stale"
                invalidated = True
            if sid not in selected:
                continue
            if not invalidated and not forced:
                continue
            # If an earlier selected prerequisite was omitted, never consume stale data.
            for predecessor in plan[: plan.index(stage)]:
                dep_root = Path(predecessor["output_dir"])
                depends = any(
                    isinstance(v, str) and (Path(v) == dep_root or dep_root in Path(v).parents)
                    for k, v in stage["options"].items()
                    if k in PATH_KEYS
                )
                if depends and not stage_current(predecessor, records.get(predecessor["id"])):
                    write_json(manifest_path, manifest)
                    raise ValueError(
                        f"Stage {sid} needs current prerequisite {predecessor['id']}; include it in --stages"
                    )
            output = Path(stage["output_dir"])
            signature = fingerprint(stage["operation"], stage["options"])
            # Resume failed checkpoints only for exactly the same inputs/settings.
            reset = forced or (previous and previous.get("fingerprint") != signature)
            if reset and output.exists():
                archive = root / "history" / (sid + "-" + utc_now().replace(":", "").replace("+", ""))
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists():
                    archive = archive.with_name(archive.name + "-" + uuid.uuid4().hex[:8])
                shutil.move(str(output), archive)
                if stage["operation"] in {"collect.search", "collect.citations"}:
                    # Downloaded full papers remain reusable when search/selection settings change.
                    output.mkdir(parents=True, exist_ok=True)
                    for paper in archive.glob("*.pdf"):
                        shutil.copy2(paper, output / paper.name)
            record = dict(
                stage,
                status="running",
                started_at=utc_now(),
                fingerprint=signature,
                execution_id=uuid.uuid4().hex,
            )
            records[sid] = record
            write_json(manifest_path, manifest)
            try:
                result = execute(stage["operation"], stage["options"], output_dir=output, force=False)
                artifacts = stage["artifacts"] or [str(p) for p in output.rglob("*") if p.is_file()]
                if not valid_artifacts(artifacts):
                    raise RuntimeError(f"Stage {sid} did not produce valid required artifacts")
                record.update(
                    status="completed",
                    artifacts=artifacts,
                    result=result,
                    artifact_hashes={p: _digest_path(Path(p)) for p in artifacts},
                )
            except BaseException as exc:
                record.update(
                    status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)
                )
                for later in plan[plan.index(stage) + 1 :]:
                    if later["id"] in records:
                        records[later["id"]]["status"] = "stale"
                raise
            finally:
                record["finished_at"] = utc_now()
                write_json(manifest_path, manifest)
        return manifest
