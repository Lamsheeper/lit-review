import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import lit_review.ui.app as application
import lit_review.ui.store as store_module
from lit_review.ui.store import ProjectStore


class StoreConcurrencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = ProjectStore(Path(temporary.name))
        self.root = Path(temporary.name)

    def test_atomic_json_writers_use_distinct_temporary_files(self):
        path = self.root / "selection.json"
        initial = {"candidate_ids": ["initial"]}
        self.store.write_json(path, initial)
        barrier = threading.Barrier(2)
        temporary_paths = []
        replace = store_module.os.replace
        payloads = [{"candidate_ids": [name], "body": name * 4096} for name in ("first", "second")]

        def synchronized_replace(source, target):
            temporary_paths.append(Path(source))
            self.assertEqual(json.loads(path.read_text()), initial)
            barrier.wait(timeout=5)
            replace(source, target)

        with patch.object(store_module.os, "replace", side_effect=synchronized_replace):
            with ThreadPoolExecutor(max_workers=2) as pool:
                writes = [pool.submit(self.store.write_json, path, payload) for payload in payloads]
                for result in writes:
                    result.result(timeout=5)
        self.assertEqual(len(set(temporary_paths)), 2)
        self.assertIn(json.loads(path.read_text()), payloads)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        self.assertTrue(all(not temp.exists() for temp in temporary_paths))

    def test_failed_json_publication_cleans_up_without_replacing_saved_data(self):
        path = self.root / "selection.json"
        self.store.write_json(path, {"candidate_ids": ["saved"]})
        with patch.object(store_module.os, "replace", side_effect=OSError("Publication failed")):
            with self.assertRaisesRegex(OSError, "Publication failed"):
                self.store.write_json(path, {"candidate_ids": ["new"]})
        self.assertEqual(json.loads(path.read_text()), {"candidate_ids": ["saved"]})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def prepare_candidates(self):
        application_store = patch.object(application, "store", self.store, create=True)
        application_store.start()
        self.addCleanup(application_store.stop)
        project = self.store.create_project("Concurrent paper selection")
        project_id = project["id"]
        path = self.store.project_dir(project_id)
        self.store.write_json(
            path / "run/stages/collect/logs/candidates.json",
            {"candidates": [{"title": "First", "doi": "10.1000/first"}, {"title": "Second", "doi": "10.1000/second"}]},
        )
        rows = application.load_candidates(application.candidate_path(path))
        return project_id, path, [row["candidate_id"] for row in rows]

    def test_default_selection_cannot_overwrite_an_explicit_save(self):
        project_id, path, ids = self.prepare_candidates()
        default_writing = threading.Event()
        manual_waiting = threading.Event()
        release_default = threading.Event()
        write_json = self.store.write_json
        transaction = self.store.selection_transaction

        def pause_default(target, payload):
            if target == path / "selection.json" and not manual_waiting.is_set():
                default_writing.set()
                self.assertTrue(release_default.wait(timeout=5))
            write_json(target, payload)

        @contextmanager
        def observe_transaction():
            if default_writing.is_set():
                manual_waiting.set()
            with transaction():
                yield

        with patch.object(self.store, "write_json", side_effect=pause_default), patch.object(
            self.store, "selection_transaction", side_effect=observe_transaction
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                default = pool.submit(application.list_candidates, project_id)
                try:
                    self.assertTrue(default_writing.wait(timeout=5))
                    manual = pool.submit(application.save_selection, project_id, {"candidate_ids": [ids[1]]})
                    self.assertTrue(manual_waiting.wait(timeout=5))
                    self.assertFalse(manual.done())
                finally:
                    release_default.set()
                self.assertEqual(default.result(timeout=5)["total"], 2)
                self.assertEqual(manual.result(timeout=5)["candidate_ids"], [ids[1]])
        self.assertEqual(self.store.read_json(path / "selection.json"), {"candidate_ids": [ids[1]]})
        application.save_selection(project_id, {"candidate_ids": []})
        application.list_candidates(project_id)
        self.assertEqual(self.store.read_json(path / "selection.json"), {"candidate_ids": []})

    def test_concurrent_first_candidate_requests_publish_defaults_once(self):
        project_id, path, ids = self.prepare_candidates()
        start = threading.Barrier(2)
        write_json = self.store.write_json
        defaults = []

        def count_defaults(target, payload):
            if target == path / "selection.json":
                defaults.append(payload)
            write_json(target, payload)

        def request():
            start.wait(timeout=5)
            return application.list_candidates(project_id)

        with patch.object(self.store, "write_json", side_effect=count_defaults):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [pool.submit(request) for _ in range(2)]
                responses = [result.result(timeout=5) for result in results]
        self.assertEqual(len(defaults), 1)
        for response in responses:
            self.assertEqual(response["total"], 2)
            self.assertTrue(all(row["selected"] for row in response["items"]))
        self.assertEqual(set(self.store.read_json(path / "selection.json")["candidate_ids"]), set(ids))
