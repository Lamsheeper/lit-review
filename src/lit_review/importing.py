"""Copy validated legacy artifacts into the release layout without altering sources."""

from __future__ import annotations

import shutil
from pathlib import Path

from lit_review.core.io import read_json, read_jsonl, utc_now, write_json, write_jsonl
from lit_review.collection.relevance import validate_relevance_profile
from lit_review.core.taxonomy import validate_taxonomy
from lit_review.features.records import read_feature_inventory
from lit_review.pipeline import _digest_path

INVENTORIES = {
    "features.jsonl": "extract",
    "filtered_features.jsonl": "filter",
    "annotated_features.jsonl": "annotate",
    "llm_merged_features.jsonl": "merge",
    "scored_features.jsonl": "score",
}
INVENTORIES.update(
    {
        Path(name).stem + suffix: stage
        for name, stage in list(INVENTORIES.items())
        for suffix in (".json", ".csv")
    }
)

ARTIFACT_STAGES = {
    "paper_index.json": "prepare",
    "chunks.jsonl": "index",
    "rejected_features.jsonl": "filter",
    "curation_decisions.jsonl": "filter",
    "excluded_annotations.jsonl": "annotate",
    "annotation_decisions.jsonl": "annotate",
    "llm_merge_candidates.jsonl": "merge",
    "llm_merge_decisions.jsonl": "merge",
    "scoring_decisions.jsonl": "score",
    "feature_mentions.jsonl": "extract",
    "paper_features.jsonl": "extract",
}
DEFINITIONS = {
    "draft.md",
    "goal.md",
    "taxonomy.json",
    "relevance_scoring.json",
    "project.json",
    "selection.json",
}


def inspect_import(source: Path, destination: Path) -> dict:
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Import source must be a directory: {source}")
    if destination.exists():
        raise ValueError(f"Import destination already exists: {destination}")
    if source == destination or source in destination.parents:
        raise ValueError("Import destination must be outside the source directory")
    files, unsupported, used = [], [], set()
    inventory_sources = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink() or source not in path.resolve().parents:
            unsupported.append({"path": str(path), "reason": "External or symbolic link"})
            continue
        relative = path.relative_to(source)
        if any(part in {"node_modules", ".git", ".venv", "__pycache__"} for part in relative.parts):
            continue
        name = path.name
        stage = INVENTORIES.get(name) or ARTIFACT_STAGES.get(name)
        if name in DEFINITIONS and path.parent == source:
            target = name
            payload = read_json(path) if path.suffix == ".json" else None
            if name == "taxonomy.json":
                validate_taxonomy(payload)
            elif name == "relevance_scoring.json":
                validate_relevance_profile(payload)
            elif name == "project.json" and (
                not isinstance(payload, dict) or not str(payload.get("name", "")).strip()
            ):
                raise ValueError(f"Invalid project definition: {path}")
            elif name == "selection.json" and (
                not isinstance(payload, dict)
                or not isinstance(payload.get("candidate_ids"), list)
                or any(not isinstance(item, str) for item in payload["candidate_ids"])
            ):
                raise ValueError(f"Invalid candidate selection: {path}")
        elif stage:
            if name in INVENTORIES:
                inventory_id = (stage, path.stem)
                if inventory_id in inventory_sources and inventory_sources[inventory_id] != path.parent:
                    raise ValueError(
                        f"Ambiguous legacy inventory {path.stem}; import a narrower source directory"
                    )
                inventory_sources[inventory_id] = path.parent
                rows = read_feature_inventory(path)
                if any(
                    not str(row.get("category") or "").strip()
                    or not str(row.get("feature_name") or "").strip()
                    for row in rows
                ):
                    raise ValueError(f"Inventory rows require category and feature_name: {path}")
            elif path.suffix == ".jsonl":
                read_jsonl(path)
            else:
                payload = read_json(path)
                if name == "paper_index.json" and (
                    not isinstance(payload, dict) or not isinstance(payload.get("papers"), list)
                ):
                    raise ValueError(f"Invalid paper index: {path}")
            target = f"stages/{stage}/{name}"
        elif name == "manifest.json" and path.parent.name == "logs":
            payload = read_json(path)
            if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
                raise ValueError(f"Invalid collection manifest: {path}")
            target = "stages/collect/logs/manifest.json"
        else:
            if path.suffix in {".json", ".jsonl", ".sqlite"}:
                unsupported.append(
                    {
                        "path": str(relative),
                        "reason": "Legacy configuration, checkpoint, or unsupported artifact; not resumed",
                    }
                )
            continue
        if target in used:
            raise ValueError(f"Ambiguous legacy artifacts for {target}; import a narrower source directory")
        used.add(target)
        files.append({"source": str(path), "destination": target, "sha256": _digest_path(path)})
    canonical = {item["destination"] for item in files}
    for item in sorted(files, key=lambda item: Path(item["source"]).suffix != ".json"):
        path = Path(item["source"])
        if path.name in INVENTORIES and path.suffix != ".jsonl":
            target = str(Path(item["destination"]).with_suffix(".jsonl"))
            if target not in canonical:
                canonical.add(target)
                files.append(dict(item, destination=target, transform="inventory_jsonl"))
    if not files:
        raise ValueError("No supported artifacts found")
    return {
        "version": 1,
        "source": str(source),
        "destination": str(destination),
        "files": files,
        "unsupported": unsupported,
        "checkpoint_policy": "Legacy execution checkpoints are never resumed",
    }


def import_artifacts(source: Path, destination: Path, *, dry_run: bool = False) -> dict:
    report = inspect_import(source, destination)
    if dry_run:
        return dict(report, dry_run=True)
    destination = Path(report["destination"])
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for item in report["files"]:
            src = Path(item["source"])
            if _digest_path(src) != item["sha256"]:
                raise ValueError(f"Source changed during import: {src}")
            target = destination / item["destination"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.get("transform") == "inventory_jsonl":
                write_jsonl(target, read_feature_inventory(src))
            else:
                shutil.copy2(src, target)
        report.update(imported_at=utc_now(), dry_run=False)
        write_json(destination / "import.json", report)
        write_json(
            destination / "manifest.json",
            {
                "version": 1,
                "created_at": utc_now(),
                "provenance": report,
                "stages": {
                    stage: {
                        "status": "imported",
                        "artifacts": [
                            str(destination / item["destination"])
                            for item in report["files"]
                            if item["destination"].startswith(f"stages/{stage}/")
                        ],
                    }
                    for stage in {
                        Path(item["destination"]).parts[1]
                        for item in report["files"]
                        if item["destination"].startswith("stages/")
                    }
                },
            },
        )
    except BaseException:
        shutil.rmtree(destination)
        raise
    return report
