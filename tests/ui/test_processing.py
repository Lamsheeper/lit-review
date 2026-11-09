import json
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
from lit_review.pipeline import load_pipeline, run_pipeline
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
        self.assertTrue(status["merge"]["materials"]["has_saved_inventory"])
        self.assertTrue(status["results"]["materials"]["has_saved_inventory"])
        # Saved results remain inspectable; new work still requires current predecessors.
        self.assertEqual(
            self.client.get(
                f"/api/projects/{self.id}/stages/merge/results/llm_merged_features.jsonl"
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(
                f"/api/projects/{self.id}/stages/merge/exports/llm_merged_features.csv"
            ).status_code,
            200,
        )
        self.assertEqual(self.client.post(f"/api/projects/{self.id}/stages/merge/run").status_code, 409)

    def test_preview_and_package_job_commands_keep_credentials_out_of_files(self):
        self.complete()
        before = sorted(self.path.rglob("*"))
        response = self.client.post(f"/api/projects/{self.id}/stages/merge/preview")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(before, sorted(self.path.rglob("*")))
        application.SESSION_CREDENTIALS["OPENAI_API_KEY"] = "session-secret"
        application.SESSION_CREDENTIALS["GEMINI_API_KEY"] = "embedding-test-key"
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
        with (
            patch("google.auth.default", return_value=(object(), "test-project")),
            patch.dict(os.environ, {}, clear=True),
            patch.object(application.jobs, "start", return_value={"id": "fake"}) as start,
        ):
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
        with (
            patch("google.auth.default", return_value=(object(), "test-project")),
            patch.dict(os.environ, {}, clear=True),
            patch.object(application.jobs, "start", return_value={"id": "fake"}) as start,
        ):
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

    def run_job_inline(self, project_id, kind, commands, env_overrides):
        """Exercise real checkpointing with local provider fixtures, without subprocess auth."""
        job = application.store.create_job(project_id, kind, self.path / "jobs/refine-test.log")
        application.store.update_job(job["id"], status="running")
        command = commands[0]
        config = load_pipeline(Path(command[command.index("--config") + 1]))
        stages = command[command.index("--stages") + 1].split(",")
        try:
            run_pipeline(config, stages=stages, executor=self.executor)
            application.store.update_job(job["id"], status="completed")
        except Exception as exc:
            application.store.update_job(job["id"], status="failed", error=str(exc))
        return application.store.get_job(job["id"])

    def complete_extraction(self):
        run_pipeline(
            application.feature_pipeline_config(self.id),
            stages=["prepare", "extract"],
            executor=self.executor,
        )

    def test_guided_refinement_runs_filter_then_annotation_in_one_job(self):
        self.complete_extraction()
        application.SESSION_CREDENTIALS["GEMINI_API_KEY"] = "local-fixture-key"
        with patch.dict(os.environ, {}, clear=True), patch.object(
            application.jobs, "start", side_effect=self.run_job_inline
        ) as start:
            response = self.client.post(f"/api/projects/{self.id}/refine/run")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["kind"], "refine")
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(start.call_args.args[2][0][-1], "filter,annotate")
        steps = self.client.get(f"/api/projects/{self.id}/workflow").json()["steps"]
        self.assertTrue(steps["filter"]["complete"])
        self.assertTrue(steps["annotate"]["complete"])
        self.assertEqual(steps["filter"]["materials"]["refinement_phase"], "complete")
        self.assertFalse(steps["merge"]["complete"])

    def test_guided_refinement_preflights_annotation_credentials_before_filter(self):
        self.complete_extraction()
        application.SESSION_CREDENTIALS["GEMINI_API_KEY"] = "local-fixture-key"
        self.client.put(
            f"/api/projects/{self.id}/stages/annotate/config",
            json={"provider": "openai", "model": "test", "auth_mode": "api_key"},
        )
        with patch.dict(os.environ, {}, clear=True), patch.object(application.jobs, "start") as start:
            readiness = self.client.get(f"/api/projects/{self.id}/readiness?action=refine")
            response = self.client.post(f"/api/projects/{self.id}/refine/run")
        self.assertFalse(readiness.json()["ready"])
        self.assertEqual(response.status_code, 409)
        self.assertIn("OpenAI", response.text)
        start.assert_not_called()
        self.assertFalse((self.path / "run/stages/filter/filtered_features.jsonl").exists())
        self.assertEqual(list((self.path / "configs").glob("refine_*.json")), [])

    def test_guided_refinement_resumes_annotation_without_rerunning_valid_filter(self):
        run_pipeline(
            application.feature_pipeline_config(self.id),
            stages=["prepare", "extract", "filter"],
            executor=self.executor,
        )
        before = json.loads((self.path / "run/manifest.json").read_text())["stages"]["filter"]
        self.model.prompts.clear()
        application.SESSION_CREDENTIALS["GEMINI_API_KEY"] = "local-fixture-key"
        with patch.dict(os.environ, {}, clear=True), patch.object(
            application.jobs, "start", side_effect=self.run_job_inline
        ):
            response = self.client.post(f"/api/projects/{self.id}/refine/run")
        self.assertEqual(response.json()["status"], "completed")
        after = json.loads((self.path / "run/manifest.json").read_text())["stages"]["filter"]
        self.assertEqual(before["execution_id"], after["execution_id"])
        self.assertTrue(any(prompt.startswith("Write polished") for prompt in self.model.prompts))
        self.assertFalse(any(prompt.startswith("Filter extracted") for prompt in self.model.prompts))

    def test_guided_refinement_annotation_failure_keeps_filter_for_retry(self):
        self.complete_extraction()
        application.SESSION_CREDENTIALS["GEMINI_API_KEY"] = "local-fixture-key"
        actual_executor = self.executor

        def fail_annotation(operation, options, **kwargs):
            if operation == "features.annotate":
                raise RuntimeError("Provider unavailable during automatic annotation")
            return actual_executor(operation, options, **kwargs)

        with patch.dict(os.environ, {}, clear=True), patch.object(
            application.jobs, "start", side_effect=self.run_job_inline
        ), patch.object(self, "executor", side_effect=fail_annotation):
            response = self.client.post(f"/api/projects/{self.id}/refine/run")
        self.assertEqual(response.json()["status"], "failed")
        steps = self.client.get(f"/api/projects/{self.id}/workflow").json()["steps"]
        self.assertTrue(steps["filter"]["complete"])
        self.assertFalse(steps["filter"]["stale"])
        self.assertFalse(steps["annotate"]["complete"])
        self.assertEqual(steps["filter"]["materials"]["refinement_phase"], "annotate")
        self.assertIn("Provider unavailable", steps["filter"]["materials"]["refinement_error"])
        self.assertEqual(steps["filter"]["action_label"], "Retry refinement")
        before = json.loads((self.path / "run/manifest.json").read_text())["stages"]["filter"]
        self.model.prompts.clear()
        with patch.dict(os.environ, {}, clear=True), patch.object(
            application.jobs, "start", side_effect=self.run_job_inline
        ):
            response = self.client.post(f"/api/projects/{self.id}/refine/run")
        self.assertEqual(response.json()["status"], "completed")
        after = json.loads((self.path / "run/manifest.json").read_text())["stages"]["filter"]
        self.assertEqual(before["execution_id"], after["execution_id"])
        self.assertFalse(any(prompt.startswith("Filter extracted") for prompt in self.model.prompts))

    def test_guided_refinement_progress_and_overlap_are_one_action(self):
        self.complete_extraction()
        job = application.store.create_job(self.id, "refine", self.path / "jobs/progress.log")
        for phase in ("filter", "annotate"):
            statuses = {"filter": {"complete": phase == "annotate"}, "annotate": {"complete": False}}
            with patch.object(application, "processing_status", return_value=statuses):
                steps = self.client.get(f"/api/projects/{self.id}/workflow").json()["steps"]
            self.assertTrue(steps["filter"]["running"])
            self.assertTrue(steps["annotate"]["running"])
            self.assertFalse(steps["filter"]["complete"])
            self.assertEqual(steps["filter"]["materials"]["refinement_phase"], phase)
        response = self.client.post(f"/api/projects/{self.id}/refine/run")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(application.store.list_jobs(self.id)), 1)
        self.assertEqual(application.jobs.cancel(job["id"])["status"], "cancelling")
        application.store.update_job(job["id"], status="cancelled")
        steps = self.client.get(f"/api/projects/{self.id}/workflow").json()["steps"]
        self.assertFalse(steps["filter"]["running"])
        self.assertEqual(steps["filter"]["materials"]["refinement_status"], "cancelled")

    def test_guided_refinement_rejects_stale_upstream_extraction(self):
        with patch.object(application.jobs, "start") as start:
            response = self.client.post(f"/api/projects/{self.id}/refine/run")
        self.assertEqual(response.status_code, 409)
        self.assertIn("Complete extract", response.text)
        start.assert_not_called()

    def test_guided_refinement_restart_reuses_filter_after_interrupted_annotation(self):
        self.complete_extraction()
        actual_executor = self.executor

        def interrupted_annotation(operation, options, **kwargs):
            if operation == "features.annotate":
                raise KeyboardInterrupt()
            return actual_executor(operation, options, **kwargs)

        job = application.store.create_job(self.id, "refine", self.path / "jobs/interrupted.log")
        application.store.update_job(job["id"], status="running")
        with self.assertRaises(KeyboardInterrupt):
            run_pipeline(
                application.feature_pipeline_config(self.id),
                stages=["filter", "annotate"],
                executor=interrupted_annotation,
            )
        before = json.loads((self.path / "run/manifest.json").read_text())["stages"]["filter"]
        application.store = ProjectStore(Path(self.temp.name))
        application.jobs = JobManager(application.store)
        self.assertEqual(application.store.get_job(job["id"])["status"], "interrupted")
        steps = self.client.get(f"/api/projects/{self.id}/workflow").json()["steps"]
        self.assertTrue(steps["filter"]["complete"])
        self.assertFalse(steps["annotate"]["complete"])
        self.assertEqual(steps["filter"]["materials"]["refinement_status"], "interrupted")
        application.SESSION_CREDENTIALS["GEMINI_API_KEY"] = "local-fixture-key"
        self.model.prompts.clear()
        with patch.dict(os.environ, {}, clear=True), patch.object(
            application.jobs, "start", side_effect=self.run_job_inline
        ):
            response = self.client.post(f"/api/projects/{self.id}/refine/run")
        self.assertEqual(response.json()["status"], "completed")
        after = json.loads((self.path / "run/manifest.json").read_text())["stages"]["filter"]
        self.assertEqual(before["execution_id"], after["execution_id"])
        self.assertFalse(any(prompt.startswith("Filter extracted") for prompt in self.model.prompts))
