"""Prepare fixed, local copies of inventory exports for the guided UI."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


INVENTORY_FILES = {
    "extract": "features.csv",
    "filter": "filtered_features.csv",
    "annotate": "annotated_features.csv",
    "merge": "llm_merged_features.csv",
    "score": "scored_features.csv",
}


def _contained(project_dir: Path, path: Path) -> Path:
    root = project_dir.resolve(strict=True)
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ValueError("Inventory exports must stay within their project directory")
    return resolved


def prepared_inventory_csv(project_dir: Path, filename: str) -> Path:
    """Find an already prepared inventory; never accept a caller-supplied path."""
    if filename not in INVENTORY_FILES.values():
        raise ValueError("Unknown inventory export")
    path = _contained(project_dir, project_dir / "exports" / filename)
    if not path.is_file():
        raise FileNotFoundError("The prepared CSV is not available")
    return path


def prepare_inventory_csv(project_dir: Path, inventory: str) -> dict[str, str]:
    """Publish a complete CSV copy atomically, then report its actual local path."""
    filename = INVENTORY_FILES.get(inventory)
    if filename is None:
        raise ValueError("Unknown inventory")
    source = _contained(project_dir, project_dir / "run" / "stages" / inventory / filename)
    if not source.is_file():
        raise FileNotFoundError("This inventory has no CSV yet")
    directory = _contained(project_dir, project_dir / "exports")
    directory.mkdir(parents=False, exist_ok=True)
    target = _contained(project_dir, directory / filename)
    temporary = tempfile.NamedTemporaryFile(
        mode="wb", dir=directory, prefix=f".{filename}.", suffix=".tmp", delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary, source.open("rb") as handle:
            shutil.copyfileobj(handle, temporary)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"filename": filename, "path": str(target)}
