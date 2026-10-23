"""Expose research operations through the shared stage runner."""

from __future__ import annotations

import argparse
import importlib
from importlib.util import find_spec
from pathlib import Path

MODULES = {
    "calibrate": "calibration",
    "metadata": "metadata",
    "cluster": "clustering",
    "compare": "comparison",
    "tsne": "visualization",
    "wikidata": "wikidata",
}


def analysis_arguments(name: str, options: dict, output_dir: Path):
    if name not in MODULES:
        raise ValueError(f"Unknown analysis operation: {name}")
    if any(find_spec(dependency) is None for dependency in ("matplotlib", "sklearn")):
        raise ValueError("Analysis requires optional dependencies. Install with: uv sync --extra analysis")
    options = {key: value for key, value in options.items() if key != "profile"}
    module = importlib.import_module("lit_review.analysis." + MODULES[name])
    parser = module.build_parser()
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    unknown = (set(options) - set(actions)) | (set(options) & {"output_dir", "gemini_api_key", "api_key"})
    if unknown:
        raise ValueError(f"Options do not apply to analysis.{name}: {sorted(unknown)}")
    values = dict(options, output_dir=str(output_dir))
    argv = []
    for key in actions:
        value = values.get(key)
        if value is None:
            continue
        action = actions[key]
        if not action.option_strings:
            if isinstance(value, list):
                raise ValueError(f"{key} expects a single input path")
            argv.append(str(value))
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(flag)
        elif isinstance(value, list):
            if isinstance(action, argparse._AppendAction):
                for item in value:
                    argv.extend([flag, str(item)])
            elif action.nargs in {"*", "+"} or isinstance(action.nargs, int):
                argv.extend([flag, *map(str, value)])
            elif key in {"text_fields", "categories", "top_k"}:
                argv.extend([flag, ",".join(map(str, value))])
            else:
                raise ValueError(f"{key} expects a scalar value")
        else:
            argv.extend([flag, str(value)])
    try:
        args = parser.parse_args(argv)
        if hasattr(module, "validate_args"):
            try:
                module.validate_args(args)
            except FileNotFoundError:
                pass  # A preceding stage may produce this input; the runner checks dependencies.
    except SystemExit as exc:
        raise ValueError(f"Invalid analysis.{name} configuration") from exc
    return module, argv


def run_analysis(name: str, options: dict, output_dir: Path) -> dict:
    module, argv = analysis_arguments(name, options, output_dir)
    from lit_review.core.context import research_context

    with research_context({}, options.get("profile")):
        code = module.main(argv)
    if code not in (None, 0):
        raise RuntimeError(f"analysis.{name} exited with {code}")
    return {"operation": name, "output_dir": str(output_dir)}
