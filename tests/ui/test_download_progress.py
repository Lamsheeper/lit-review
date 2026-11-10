import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter

import lit_review.ui.app as app
from lit_review.collection.harvest import DownloadLogLedger, stable_candidate_id
from lit_review.collection.sources import Candidate
from lit_review.core.io import write_json
from lit_review.ui.jobs import JobManager
from lit_review.ui.store import ProjectStore


class DownloadProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        app.store = ProjectStore(Path(self.temporary.name))
        app.jobs = JobManager(app.store)
        self.client = TestClient(app.app)
        self.project = app.store.create_project("PDF progress")
        self.id = self.project["id"]
        self.path = app.store.project_dir(self.id)
        self.base = f"/api/projects/{self.id}"
        self.papers = [Candidate(title=f"Paper {number}", doi=f"10.1000/{number}") for number in range(4)]
        self.ids = [stable_candidate_id(paper) for paper in self.papers]
        self.rows = [{"title": paper.title, "doi": paper.doi} for paper in self.papers]
        write_json(self.path / "run/stages/search/logs/candidates.json", {"candidates": self.rows})
        write_json(self.path / "selection.json", {"candidate_ids": self.ids[:3]})
        self.ledger = DownloadLogLedger(self.path / "run/stages/collect/logs/download_log.json", candidate_cache={}, search_errors=[])

    def queue(self):
        def start(project_id, kind, commands, env):
            return app.store.create_job(project_id, kind, self.path / "jobs/test.log")
        with patch.dict(os.environ, {}, clear=True), patch.object(app.jobs, "start", side_effect=start):
            response = self.client.post(self.base + "/collection/download")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def save_pdf(self, number):
        path = self.path / f"run/stages/collect/paper-{number}.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(str(path))
        self.ledger.record(self.papers[number], {"status": "downloaded", "path": str(path)})
        return path

    def progress(self):
        response = self.client.get(self.base + "/collection/download-progress")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def finalize(self):
        rows = []
        for number, paper in enumerate(self.papers):
            raw = {"title": paper.title, "doi": paper.doi}
            record = self.ledger.record_for_candidate(paper)
            if record:
                raw["download"] = record["download"]
            rows.append(raw)
        write_json(self.path / "run/stages/collect/logs/manifest.json", {"candidates": rows})

    def test_progress_is_selected_only_global_and_read_only(self):
        job = self.queue()
        self.save_pdf(0)
        self.ledger.record(self.papers[1], {"status": "failed", "reason": "No usable PDF"})
        self.ledger.record(self.papers[2], {"status": "not_attempted", "reason": "Limit reached"})
        self.ledger.record(self.papers[3], {"status": "failed", "reason": "Unselected failure"})
        selection = (self.path / "selection.json").read_bytes()
        before = self.progress()
        self.assertEqual((before["selected"], before["ready"], before["failed"], before["unavailable"], before["pending"]), (3, 1, 1, 1, 0))
        self.assertTrue(before["running"])
        self.assertEqual(before["job_id"], job["id"])
        self.assertFalse(before["ready_to_extract"])
        filtered = self.client.get(self.base + "/collection/candidates?page_size=1&query=Paper+0").json()
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(self.progress(), before)
        self.assertEqual((self.path / "selection.json").read_bytes(), selection)

    def test_completed_availability_is_retained_and_requires_recorded_full_pdfs(self):
        job = self.queue()
        for number in range(3):
            self.save_pdf(number)
        running = self.progress()
        self.assertEqual(running["ready"], 3)
        self.assertFalse(running["ready_to_extract"])
        app.store.update_job(job["id"], status="completed")
        self.assertFalse(self.progress()["ready_to_extract"])
        self.finalize()
        completed = self.progress()
        self.assertTrue(completed["complete"])
        self.assertTrue(completed["ready_to_extract"])
        self.assertEqual((completed["ready"], completed["pending"]), (3, 0))
        self.assertEqual(self.progress(), completed)

    def test_cancel_retry_does_not_count_old_failures_as_new_attempts(self):
        first = self.queue()
        self.save_pdf(0)
        self.ledger.record(self.papers[1], {"status": "failed", "reason": "First attempt"})
        app.store.update_job(first["id"], status="cancelled")
        paused = self.progress()
        self.assertEqual((paused["ready"], paused["failed"], paused["pending"]), (1, 1, 1))
        second = self.queue()
        retrying = self.progress()
        self.assertEqual((retrying["ready"], retrying["failed"], retrying["processed"], retrying["pending"]), (1, 0, 1, 2))
        self.assertEqual(retrying["job_id"], second["id"])
        self.ledger.record(self.papers[1], {"status": "failed", "reason": "A new failed lookup"})
        self.assertEqual(self.progress()["failed"], 1)
        self.assertEqual(app.store.read_json(self.path / "selection.json")["candidate_ids"], self.ids[:3])

    def test_cached_old_failure_is_unavailable_after_completed_resume(self):
        self.save_pdf(0)
        self.ledger.record(self.papers[1], {"status": "failed", "reason": "Earlier failure"})
        self.ledger.record(self.papers[2], {"status": "not_attempted", "reason": "Earlier limit"})
        job = self.queue()
        self.assertEqual((self.progress()["failed"], self.progress()["pending"]), (0, 2))
        app.store.update_job(job["id"], status="completed")
        completed = self.progress()
        self.assertEqual((completed["ready"], completed["failed"], completed["unavailable"]), (1, 0, 2))
        self.assertTrue(completed["complete"])
        self.assertFalse(completed["ready_to_extract"])

    def test_zero_selection_never_claims_completion(self):
        write_json(self.path / "selection.json", {"candidate_ids": []})
        progress = self.progress()
        self.assertEqual((progress["selected"], progress["ready"], progress["pending"]), (0, 0, 0))
        self.assertFalse(progress["complete"])
        self.assertFalse(progress["ready_to_extract"])
        self.assertFalse(progress["running"])

    def test_missing_or_invalid_file_never_counts_as_ready(self):
        self.ledger.record(self.papers[0], {"status": "downloaded", "path": str(self.path / "missing.pdf")})
        invalid = self.path / "invalid.pdf"
        invalid.write_text("Not a PDF")
        self.ledger.record(self.papers[1], {"status": "already_exists", "path": str(invalid)})
        self.save_pdf(2)
        progress = self.progress()
        self.assertEqual((progress["ready"], progress["unavailable"], progress["pending"]), (1, 2, 0))
        self.assertFalse(progress["ready_to_extract"])

    def test_csv_preparation_returns_actual_local_copy_and_download(self):
        source = self.path / "run/stages/merge/llm_merged_features.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("feature_name,paper_count\nTechnique,3\n")
        response = self.client.post(self.base + "/exports/csv", json={"inventory": "merge"})
        self.assertEqual(response.status_code, 200, response.text)
        prepared = response.json()
        self.assertEqual(Path(prepared["path"]), self.path / "exports/llm_merged_features.csv")
        self.assertEqual(Path(prepared["path"]).read_bytes(), source.read_bytes())
        downloaded = self.client.get(prepared["download_url"])
        self.assertEqual(downloaded.content, source.read_bytes())
        self.assertEqual(self.client.post(self.base + "/exports/csv", json={"inventory": "../../private"}).status_code, 422)
        self.assertEqual(self.client.get(self.base + "/exports/prepared/secret.csv").status_code, 422)
