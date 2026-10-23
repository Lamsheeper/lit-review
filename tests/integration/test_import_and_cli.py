import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lit_review.cli import main
from lit_review.core.io import read_json, write_json, write_jsonl
from lit_review.importing import import_artifacts
from lit_review.pipeline import _digest_path, load_pipeline, run_pipeline


class ImportAndCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "legacy"
        self.source.mkdir()
        self.destination = self.root / "imported"

    def test_import_preview_copy_provenance_and_unchanged_originals(self):
        write_jsonl(
            self.source / "filtered_features.jsonl",
            [{"category": "ergonomics", "feature_name": "Neutral posture"}],
        )
        write_jsonl(self.source / "completed_annotation_batches.jsonl", [{"batch_id": "old"}])
        (self.source / "goal.md").write_text("A goal")
        before = _digest_path(self.source)
        preview = import_artifacts(self.source, self.destination, dry_run=True)
        self.assertFalse(self.destination.exists())
        self.assertTrue(preview["unsupported"])
        report = import_artifacts(self.source, self.destination)
        self.assertEqual(_digest_path(self.source), before)
        self.assertEqual(
            (self.destination / "stages/filter/filtered_features.jsonl").read_bytes(),
            (self.source / "filtered_features.jsonl").read_bytes(),
        )
        self.assertEqual(read_json(self.destination / "import.json")["source"], str(self.source))
        self.assertEqual(len(report["files"]), 2)
        self.assertFalse((self.destination / "completed_annotation_batches.jsonl").exists())

    def test_rejects_existing_destination_and_nested_destination(self):
        write_jsonl(self.source / "features.jsonl", [])
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, "exists"):
            import_artifacts(self.source, self.destination)
        with self.assertRaisesRegex(ValueError, "outside"):
            import_artifacts(self.source, self.source / "nested")

    def test_rejects_bad_artifacts_before_creating_destination(self):
        (self.source / "features.jsonl").write_text("broken JSON")
        with self.assertRaises(ValueError):
            import_artifacts(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_rejects_ambiguous_runs(self):
        for folder in ("first", "second"):
            write_jsonl(self.source / folder / "features.jsonl", [])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            import_artifacts(self.source, self.destination)

    def test_external_links_are_reported_without_copying(self):
        write_jsonl(self.source / "features.jsonl", [])
        other = self.root / "outside.json"
        write_json(other, {"value": "private"})
        (self.source / "outside.json").symlink_to(other)
        report = import_artifacts(self.source, self.destination)
        self.assertTrue(any("link" in row["reason"] for row in report["unsupported"]))
        self.assertFalse((self.destination / "outside.json").exists())

    def test_json_and_csv_inventories_create_canonical_jsonl(self):
        write_json(
            self.source / "annotated_features.json",
            {
                "features": [
                    {
                        "category": "ergonomics",
                        "feature_name": "Neutral posture",
                        "annotation": {"formal_definition": "A definition"},
                    }
                ]
            },
        )
        (self.source / "annotated_features.csv").write_text(
            "category,feature_name\nergonomics,Neutral posture\n"
        )
        report = import_artifacts(self.source, self.destination)
        canonical = self.destination / "stages/annotate/annotated_features.jsonl"
        self.assertIn("A definition", canonical.read_text())
        self.assertEqual(sum(item.get("transform") == "inventory_jsonl" for item in report["files"]), 1)

    def test_inventory_requires_identity_fields(self):
        write_jsonl(self.source / "features.jsonl", [{"anything": "invalid"}])
        with self.assertRaisesRegex(ValueError, "category and feature_name"):
            import_artifacts(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_cli_rejects_legacy_options_and_honors_overrides(self):
        taxonomy = self.root / "taxonomy.json"
        write_json(
            taxonomy, {"version": 1, "title": "Test", "families": [{"id": "factors", "label": "Factors"}]}
        )
        source = self.root / "features"
        source.mkdir()
        write_jsonl(source / "features.jsonl", [])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(
                [
                    "features",
                    "filter",
                    "--output-root",
                    str(self.destination),
                    "--taxonomy",
                    str(taxonomy),
                    "--input-dir",
                    str(source),
                    "--set",
                    "model=chosen-model",
                    "--dry-run",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["stages"][0]["options"]["model"], "chosen-model")
        self.assertFalse(self.destination.exists())
        with contextlib.redirect_stderr(io.StringIO()):
            status = main(
                [
                    "features",
                    "filter",
                    "--output-root",
                    str(self.destination),
                    "--taxonomy",
                    str(taxonomy),
                    "--input-dir",
                    str(source),
                    "--set",
                    "merge_style=moderate",
                    "--dry-run",
                ]
            )
        self.assertEqual(status, 1)
        self.assertFalse(self.destination.exists())

    def test_literal_credentials_are_rejected_without_persisting(self):
        config_path = self.root / "config.json"
        draft = self.root / "draft.md"
        draft.write_text("# Example\n\nworkplace ergonomics")
        write_json(
            config_path,
            {
                "version": 1,
                "output_root": "run",
                "stages": [
                    {
                        "id": "collect",
                        "operation": "collect.search",
                        "options": {"draft": "draft.md", "semantic_scholar_api_key": "do-not-record"},
                    }
                ],
            },
        )
        with self.assertRaisesRegex(ValueError, "environment"):
            run_pipeline(load_pipeline(config_path), dry_run=True)
        self.assertFalse((self.root / "run").exists())

    def test_every_shipped_recipe_previews_without_network_or_output_changes(self):
        repo = Path(__file__).resolve().parents[2]
        with patch("urllib.request.urlopen", side_effect=AssertionError("network must not be called")):
            for recipe in sorted((repo / "examples").rglob("pipeline.json")):
                with self.subTest(recipe=recipe):
                    config = load_pipeline(
                        recipe, overrides={"output_root": str(self.root / recipe.parent.name)}
                    )
                    report = run_pipeline(config, dry_run=True)
                    self.assertTrue(report["stages"])
                    self.assertFalse(config.output_root.exists())
