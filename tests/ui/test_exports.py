import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import lit_review.ui.exports as exports


class InventoryExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "project"
        self.project.mkdir()

    def source(self, inventory="merge", content=b"feature_name,paper_count\nFear appeal,3\nLoaded language,1\n"):
        path = self.project / "run/stages" / inventory / exports.INVENTORY_FILES[inventory]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_prepared_copy_reports_its_real_location_and_preserves_inventory_bytes(self):
        for inventory, filename in exports.INVENTORY_FILES.items():
            with self.subTest(inventory=inventory):
                source = self.source(inventory)
                original = source.read_bytes()
                result = exports.prepare_inventory_csv(self.project, inventory)
                prepared = Path(result["path"])
                self.assertTrue(prepared.is_absolute())
                self.assertEqual(prepared, self.project / "exports" / filename)
                self.assertEqual(result["filename"], filename)
                self.assertEqual(prepared.read_bytes(), original)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(exports.prepared_inventory_csv(self.project, filename), prepared)

    def test_empty_inventory_is_copied_without_inventing_records(self):
        source = self.source(content=b"feature_name,paper_count\n")
        prepared = exports.prepare_inventory_csv(self.project, "merge")
        self.assertEqual(Path(prepared["path"]).read_bytes(), source.read_bytes())

    def test_inventory_and_download_names_are_whitelisted(self):
        self.source()
        for inventory in ("../merge", "run/stages/merge", "unknown", "/tmp/secret"):
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                exports.prepare_inventory_csv(self.project, inventory)
        for filename in ("../features.csv", "manifest.json", "exports/features.csv", "secret.csv"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                exports.prepared_inventory_csv(self.project, filename)
        self.assertFalse((self.project / "exports").exists())

    def test_missing_source_does_not_claim_an_export_exists(self):
        with self.assertRaises(FileNotFoundError):
            exports.prepare_inventory_csv(self.project, "merge")
        with self.assertRaises(FileNotFoundError):
            exports.prepared_inventory_csv(self.project, "features.csv")
        self.assertFalse((self.project / "exports").exists())

    def test_source_and_export_symlinks_cannot_escape_the_project(self):
        outside = self.root / "outside"
        outside.mkdir()
        secret = outside / "secret.csv"
        secret.write_bytes(b"outside contents")
        source = self.source()
        source.unlink()
        source.symlink_to(secret)
        with self.assertRaises(ValueError):
            exports.prepare_inventory_csv(self.project, "merge")
        source.unlink()
        self.source()
        (self.project / "exports").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            exports.prepare_inventory_csv(self.project, "merge")
        with self.assertRaises(ValueError):
            exports.prepared_inventory_csv(self.project, "llm_merged_features.csv")
        self.assertEqual(secret.read_bytes(), b"outside contents")
        self.assertEqual(list(outside.iterdir()), [secret])

    def test_failed_publication_preserves_previous_export_and_cleans_up(self):
        source = self.source()
        initial = exports.prepare_inventory_csv(self.project, "merge")
        previous = Path(initial["path"]).read_bytes()
        source.write_bytes(b"feature_name,paper_count\nNew feature,2\n")
        with patch.object(exports.os, "replace", side_effect=OSError("Export publication failed")):
            with self.assertRaisesRegex(OSError, "Export publication failed"):
                exports.prepare_inventory_csv(self.project, "merge")
        self.assertEqual(Path(initial["path"]).read_bytes(), previous)
        self.assertEqual(list((self.project / "exports").glob("*.tmp")), [])

    def test_concurrent_preparations_publish_complete_copies_with_unique_temporary_files(self):
        source = self.source()
        barrier = threading.Barrier(2)
        temporary_paths = []
        replace = exports.os.replace

        def synchronize(first, second):
            temporary_paths.append(Path(first))
            barrier.wait(timeout=5)
            replace(first, second)

        with patch.object(exports.os, "replace", side_effect=synchronize):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [pool.submit(exports.prepare_inventory_csv, self.project, "merge") for _ in range(2)]
                paths = [Path(result.result(timeout=5)["path"]) for result in results]
        self.assertEqual(len(set(temporary_paths)), 2)
        self.assertEqual(paths[0], paths[1])
        self.assertEqual(paths[0].read_bytes(), source.read_bytes())
        self.assertTrue(all(not path.exists() for path in temporary_paths))
