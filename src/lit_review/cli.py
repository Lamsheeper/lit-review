"""The supported lit-review command interface."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from lit_review import __version__
from lit_review.core.io import read_json
from lit_review.core.settings import PipelineConfig, StageSpec
from lit_review.pipeline import SUPPORTED, _paths, load_pipeline, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lit-review", description="Evidence-backed literature research pipelines"
    )
    parser.add_argument("--version", action="version", version=__version__)
    groups = parser.add_subparsers(dest="group", required=True)
    for group in ("collect", "evidence", "features", "analysis"):
        sub = groups.add_parser(group)
        commands = sub.add_subparsers(dest="operation", required=True)
        for op in sorted(
            operation.split(".", 1)[1] for operation in SUPPORTED if operation.startswith(group + ".")
        ):
            command = commands.add_parser(op)
            command.add_argument(
                "--config", type=Path, help="JSON stage options (paths relative to this file)"
            )
            command.add_argument(
                "--output-root", type=Path, required=True, help="Run root containing stages and manifest"
            )
            command.add_argument("--dry-run", action="store_true")
            command.add_argument(
                "--force", action="store_true", help="Archive previous output and rerun this stage"
            )
            command.add_argument(
                "--set",
                action="append",
                default=[],
                metavar="KEY=VALUE",
                help="Override an option; JSON values supported",
            )
            if group == "features":
                command.add_argument("--taxonomy", type=Path)
                command.add_argument("--goal", type=Path)
                command.add_argument("--input-file", type=Path)
                command.add_argument("--input-dir", type=Path)
                command.add_argument("--run-dir", type=Path)
    pipeline = groups.add_parser("pipeline").add_subparsers(dest="operation", required=True).add_parser("run")
    pipeline.add_argument("--config", type=Path, required=True)
    pipeline.add_argument("--output-root", type=Path)
    pipeline.add_argument("--dry-run", action="store_true")
    pipeline.add_argument("--stages", help="Comma-separated stage IDs")
    pipeline.add_argument("--rerun", help="Comma-separated stage IDs to rerun with fresh checkpoints")
    importing = groups.add_parser("import", help="Preview or copy supported legacy artifacts")
    importing.add_argument("source", type=Path)
    importing.add_argument("--destination", type=Path, required=True)
    importing.add_argument("--dry-run", action="store_true")
    groups.add_parser("ui", help="Start the local guided application")
    return parser


def main(argv: list[str] | None = None) -> int:
    import signal

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, interrupted)
    args = build_parser().parse_args(argv)
    try:
        if args.group == "ui":
            from lit_review.ui.app import main as serve

            serve()
            return 0
        if args.group == "import":
            from lit_review.importing import import_artifacts

            result = import_artifacts(args.source, args.destination, dry_run=args.dry_run)
        elif args.group == "pipeline":
            overrides = {"output_root": str(args.output_root.resolve())} if args.output_root else None
            config = load_pipeline(args.config, overrides=overrides)
            result = run_pipeline(
                config,
                dry_run=args.dry_run,
                stages=args.stages.split(",") if args.stages else None,
                rerun=args.rerun.split(",") if args.rerun else None,
            )
        else:
            base = args.config.resolve().parent if args.config else Path.cwd()
            options = read_json(args.config, {}) if args.config else {}
            if not isinstance(options, dict):
                raise ValueError("Stage options must be a JSON object")
            options = _paths(options, base)
            for key in ("taxonomy", "goal", "input_file", "input_dir", "run_dir"):
                value = getattr(args, key, None)
                if value is not None:
                    options[key] = str(value.resolve())
            for override in args.set:
                key, separator, value = override.partition("=")
                if not separator:
                    raise ValueError("--set expects KEY=VALUE")
                try:
                    options[key] = json.loads(value)
                except json.JSONDecodeError:
                    options[key] = value
                options.update(_paths({key: options[key]}, Path.cwd()))
            op = args.group + "." + args.operation
            # A single-stage command uses the identical runner and state contract.
            config = PipelineConfig(
                output_root=args.output_root.resolve(),
                stages=[StageSpec(id=args.operation, operation=op, options=options)],
            )
            result = run_pipeline(
                config, dry_run=args.dry_run, rerun=[args.operation] if args.force else None
            )
        print(json.dumps(result, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; matching checkpoints can be resumed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"lit-review: {exc}", file=sys.stderr)
        return 1
