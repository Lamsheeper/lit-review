import os
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter

import lit_review.ui.app as application
from lit_review.collection.harvest import candidate_from_dict, stable_candidate_id
from lit_review.core.io import write_json
from lit_review.pipeline import run_pipeline
from lit_review.services import execute_operation
from lit_review.ui.jobs import JobManager
from lit_review.ui.store import ProjectStore
from tests.integration.test_pipeline import ResearchClient, Embeddings


class ProcessingUiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        application.SESSION_CREDENTIALS.clear()
        application.store = ProjectStore(Path(self.temp.name))
        application.jobs = JobManager(application.store)
        self.client = TestClient(application.app)
        self.project = self.client.post("/api/projects", json={"name": "Ergonomics"}).json()
        self.id = self.project["id"]
        self.path = application.store.project_dir(self.id)
        self.pdf = self.path / "run/stages/collect/workplace.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(str(self.pdf))
        self.candidate = {
            "title": "Workplace",
            "year": 2026,
            "doi": "10.1000/workplace",
            "download": {"status": "downloaded", "path": str(self.pdf)},
        }
        self.candidate_id = stable_candidate_id(candidate_from_dict(self.candidate))
        write_json(self.path / "run/stages/collect/logs/manifest.json", {"candidates": [self.candidate]})
        write_json(self.path / "run/stages/collect/logs/candidates.json", {"candidates": [self.candidate]})
        write_json(self.path / "selection.json", {"candidate_ids": [self.candidate_id]})
        (self.path / "draft.md").write_text("# Ergonomics")
        (self.path / "goal.md").write_text("Extract ergonomic factors")
        write_json(
            self.path / "taxonomy.json",
            {"version": 1, "title": "Ergonomics", "families": [{"id": "ergonomics", "label": "Ergonomics"}]},
        )
        self.model = ResearchClient()

    def executor(self, operation, options, **kwargs):
        runtime = (
            {"client": self.model, "embedding_client": Embeddings()}
            if operation.startswith("features.")
            else {}
        )
        return execute_operation(operation, options, **kwargs, **runtime)

    def complete(self):
        return run_pipeline(
            application.feature_pipeline_config(self.id),
            stages=["prepare", "extract", "filter", "annotate", "merge"],
            executor=self.executor,
        )

    def test_stage_readiness_export_optional_score_and_settings_invalidation(self):
        self.complete()
        status = self.client.get(f"/api/projects/{self.id}/workflow").json()["steps"]
        self.assertTrue(status["merge"]["complete"])
        self.assertTrue(status["results"]["complete"])
        self.assertFalse(status["score"]["complete"])
        response = self.client.get(f"/api/projects/{self.id}/stages/merge/results/llm_merged_features.jsonl")
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(
            self.client.get(
                f"/api/projects/{self.id}/stages/merge/exports/llm_merged_features.csv"
            ).status_code,
            200,
        )
        response = self.client.put(
            f"/api/projects/{self.id}/stages/filter/config", json={"batch_features": 5}
        )
        self.assertEqual(response.status_code, 200)
        status = self.client.get(f"/api/projects/{self.id}/workflow").json()["steps"]
        self.assertTrue(status["extract"]["complete"])
        self.assertFalse(status["filter"]["complete"])
        self.assertTrue(status["merge"]["stale"])
        self.assertEqual(self.client.post(f"/api/projects/{self.id}/stages/merge/run").status_code, 409)

    def test_preview_and_package_job_commands_keep_credentials_out_of_files(self):
        self.complete()
        before = sorted(self.path.rglob("*"))
        response = self.client.post(f"/api/projects/{self.id}/stages/merge/preview")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(before, sorted(self.path.rglob("*")))
        application.SESSION_CREDENTIALS["OPENAI_API_KEY"] = "session-secret"
        self.client.put(
            f"/api/projects/{self.id}/stages/score/config", json={"provider": "openai", "model": "test"}
        )
        with patch.object(application.jobs, "start", return_value={"id": "fake"}) as start:
            self.assertEqual(self.client.post(f"/api/projects/{self.id}/stages/score/run").status_code, 200)
        commands = start.call_args.args[2]
        self.assertEqual(commands[0][:3], [sys.executable, "-m", "lit_review"])
        self.assertNotIn("session-secret", str(commands))
        self.assertEqual(start.call_args.args[3]["OPENAI_API_KEY"], "session-secret")
        config_path = Path(commands[0][commands[0].index("--config") + 1])
        self.assertNotIn("session-secret", config_path.read_text())

    def test_extraction_can_launch_with_adc_and_no_api_key(self):
        response = self.client.put(
            f"/api/projects/{self.id}/extraction/config",
            json={"auth_mode": "adc", "vertex_project": "test-project", "vertex_location": "us-central1"},
        )
        self.assertEqual(response.status_code, 200)
        with patch.dict(os.environ, {}, clear=True), patch.object(
            application.jobs, "start", return_value={"id": "fake"}
        ) as start:
            response = self.client.post(f"/api/projects/{self.id}/extraction/run")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(start.call_args.args[3], {})
        commands = start.call_args.args[2]
        path = Path(commands[0][commands[0].index("--config") + 1])
        from lit_review.pipeline import compile_pipeline, load_pipeline

        plan = compile_pipeline(load_pipeline(path))
        extraction = next(stage for stage in plan if stage["id"] == "extract")
        self.assertEqual(extraction["options"]["auth_mode"], "adc")
        self.assertEqual(extraction["options"]["vertex_project"], "test-project")

    def test_processing_can_launch_with_adc_and_no_api_key(self):
        self.complete()
        response = self.client.put(
            f"/api/projects/{self.id}/stages/filter/config",
            json={"auth_mode": "adc", "vertex_project": "test-project"},
        )
        self.assertEqual(response.status_code, 200)
        with patch.dict(os.environ, {}, clear=True), patch.object(
            application.jobs, "start", return_value={"id": "fake"}
        ) as start:
            response = self.client.post(f"/api/projects/{self.id}/stages/filter/run")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(start.call_args.args[3], {})

    def test_selection_changes_invalidate_extraction(self):
        self.complete()
        write_json(self.path / "selection.json", {"candidate_ids": []})
        status = self.client.get(f"/api/projects/{self.id}/workflow").json()["steps"]
        self.assertFalse(status["extract"]["complete"])
        self.assertFalse(status["merge"]["complete"])

    def test_new_search_results_replace_older_collection_rankings(self):
        import os

        searched = self.path / "run/stages/search/logs/candidates.json"
        write_json(searched, {"candidates": []})
        downloaded = self.path / "run/stages/collect/logs/candidates.json"
        os.utime(downloaded, ns=(1, 1))
        self.assertEqual(application.candidate_path(self.path), searched)

    def test_concurrent_job_creation_has_one_winner(self):
        def create(_):
            try:
                application.store.create_job(self.id, "filter", self.path / "jobs/job.log")
                return True
            except ValueError:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(create, range(2))), 1)

    def test_cancel_stops_subsequent_commands(self):
        marker = self.path / "should-not-exist"
        job = application.jobs.start(
            self.id,
            "test",
            [
                [sys.executable, "-c", "import time; time.sleep(10)"],
                [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
            ],
        )
        deadline = time.monotonic() + 5
        while application.store.get_job(job["id"])["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.02)
        application.jobs.cancel(job["id"])
        while (
            application.store.get_job(job["id"])["status"] in {"queued", "running", "cancelling"}
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        self.assertEqual(application.store.get_job(job["id"])["status"], "cancelled")
        self.assertFalse(marker.exists())
