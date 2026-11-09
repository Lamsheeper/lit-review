import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
import lit_review.ui.app as app_module
from lit_review.core.io import write_json, write_jsonl, utc_now
from lit_review.ui.store import ProjectStore
from lit_review.ui.jobs import JobManager
from lit_review.collection.harvest import DownloadLogLedger
from lit_review.collection.sources import Candidate


class LiveResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        app_module.store = ProjectStore(Path(self.temp.name))
        app_module.jobs = JobManager(app_module.store)
        self.client = TestClient(app_module.app)
        project = self.client.post("/api/projects", json={"name": "Live"}).json()
        self.id = project["id"]
        self.base = "/api/projects/" + self.id
        self.path = app_module.store.project_dir(self.id)

    def job(self, kind):
        return app_module.store.create_job(self.id, kind, self.path / "job.log")

    def test_search_updates_are_read_only_paginated_and_survive_cancellation(self):
        job = self.job("collection_search")
        preview = self.path / "run/stages/search/logs/candidates.live.json"
        write_json(
            preview,
            {
                "complete": False,
                "candidates": [{"title": "First", "year": 2024, "relevance_score": 1}],
                "progress": {"completed": 1, "total": 2},
            },
        )
        response = self.client.get(self.base + "/collection/live?page_size=1").json()
        self.assertEqual(response["total"], 1)
        self.assertTrue(response["partial"])
        self.assertFalse((self.path / "selection.json").exists())
        write_json(
            preview,
            {
                "complete": False,
                "candidates": [
                    {"title": "First", "year": 2024, "relevance_score": 1},
                    {"title": "Second", "year": 2025, "relevance_score": 2},
                ],
                "progress": {"completed": 2, "total": 2},
            },
        )
        response = self.client.get(self.base + "/collection/live?page_size=1&page=2").json()
        self.assertEqual(response["total"], 2)
        self.assertEqual(len(response["items"]), 1)
        app_module.store.update_job(job["id"], status="cancelled")
        self.assertEqual(self.client.get(self.base + "/collection/live").json()["total"], 2)
        self.assertFalse((self.path / "selection.json").exists())

    def test_new_search_does_not_present_old_preview_as_new_results(self):
        preview = self.path / "run/stages/search/logs/candidates.live.json"
        write_json(preview, {"candidates": [{"title": "Old search"}]})
        os.utime(preview, (1, 1))
        self.job("collection_search")
        self.assertEqual(self.client.get(self.base + "/collection/live").json()["total"], 0)

    def test_download_status_appears_before_the_final_manifest(self):
        candidate = Candidate(title="Downloaded paper", doi="10.1000/test")
        write_json(
            self.path / "run/stages/search/logs/candidates.json",
            {"candidates": [{"title": candidate.title, "doi": candidate.doi}]},
        )
        ledger = DownloadLogLedger(
            self.path / "run/stages/collect/logs/download_log.json", candidate_cache={}, search_errors=[]
        )
        self.job("collection_download")
        ledger.record(candidate, {"status": "downloaded", "path": "paper.pdf"})
        preview = self.client.get(self.base + "/collection/live").json()
        self.assertEqual(preview["downloaded"], 1)
        self.assertEqual(preview["items"][0]["download_status"], "downloaded")
        self.assertEqual(
            self.client.get(self.base + "/collection/candidates").json()["items"][0]["download_status"],
            "downloaded",
        )

    def test_feature_batches_arrive_during_extraction_and_old_output_is_hidden(self):
        artifact = self.path / "run/stages/extract/features.jsonl"
        write_jsonl(artifact, [{"feature_name": "Old"}])
        os.utime(artifact, (1, 1))
        job = self.job("direct_pdf_extraction")
        app_module.store.update_job(job["id"], status="running", started_at=utc_now())
        url = self.base + "/stages/extract/results/features.jsonl"
        self.assertEqual(self.client.get(url).json()["total"], 0)
        write_jsonl(artifact, [{"feature_name": "New", "paper_count": 1}])
        self.assertEqual(self.client.get(url).json()["items"][0]["feature_name"], "New")
        write_jsonl(
            artifact,
            [{"feature_name": "New", "paper_count": 2}, {"feature_name": "Another", "paper_count": 1}],
        )
        self.assertEqual(self.client.get(url).json()["total"], 2)
        app_module.store.update_job(job["id"], status="completed")
        self.assertEqual(self.client.get(url).json()["items"][0]["paper_count"], 2)
