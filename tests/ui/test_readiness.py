import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from pypdf import PdfWriter
import lit_review.ui.app as app
from lit_review.ui.store import ProjectStore
from lit_review.ui.jobs import JobManager
from lit_review.ui.readiness import provider_requirements


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        app.SESSION_CREDENTIALS.clear()
        self.addCleanup(app.SESSION_CREDENTIALS.clear)
        app.store = ProjectStore(Path(self.temp.name))
        app.jobs = JobManager(app.store)
        self.client = TestClient(app.app)

    def test_missing_credentials_block_start_before_creating_workspace(self):
        with patch.object(app, "generate_documents") as generate, patch.object(app.jobs, "start") as start:
            result = self.client.post("/api/projects/start", json={"request": "Emotion concepts"})
        self.assertEqual(result.status_code, 409, result.text)
        self.assertFalse(result.json()["detail"]["readiness"]["ready"])
        self.assertEqual(self.client.get("/api/projects").json(), [])
        generate.assert_not_called()
        start.assert_not_called()

    def test_openai_alone_reports_the_future_embedding_credential(self):
        app.SESSION_CREDENTIALS["OPENAI_API_KEY"] = "secret-openai"
        result = self.client.get("/api/readiness")
        self.assertFalse(result.json()["ready"])
        self.assertEqual([r["id"] for r in result.json()["requirements"]], ["GEMINI_API_KEY"])
        self.assertIn("merging", result.text)
        self.assertNotIn("secret-openai", result.text)

    def test_sufficient_credentials_do_not_require_pdfs_before_search(self):
        app.SESSION_CREDENTIALS["GEMINI_API_KEY"] = "secret-gemini"
        self.assertTrue(self.client.get("/api/readiness").json()["ready"])
        result = self.client.post("/api/projects/start", json={"request": "Emotion concepts"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertNotIn("secret-gemini", result.text)

    def test_bad_adc_configuration_is_caught_before_provider_calls(self):
        with (
            patch.dict(
                os.environ,
                {
                    "GOOGLE_GENAI_USE_VERTEXAI": "true",
                    "GOOGLE_CLOUD_PROJECT": "example-project",
                    "GOOGLE_APPLICATION_CREDENTIALS": "/missing/credentials.json",
                },
            ),
            patch("google.auth.default") as auth,
        ):
            result = self.client.get("/api/readiness")
        self.assertFalse(result.json()["ready"])
        self.assertIn("credentials file", result.text)
        auth.assert_not_called()

    def test_existing_adc_login_can_satisfy_both_model_requirements(self):
        with (
            patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "example-project"}),
            patch("google.auth.default", return_value=(object(), "example-project")),
        ):
            self.assertTrue(self.client.get("/api/readiness").json()["ready"])

    def test_custom_embedding_key_and_explicit_key_mode_are_respected(self):
        missing = provider_requirements(
            {"embedding_api_key_env": "MY_EMBEDDING_KEY"}, lambda name: None, embedding=True
        )
        self.assertEqual(missing[0]["id"], "MY_EMBEDDING_KEY")
        with patch.dict(
            os.environ,
            {"GOOGLE_GENAI_USE_VERTEXAI": "true", "GOOGLE_APPLICATION_CREDENTIALS": "/missing.json"},
        ):
            self.assertEqual(
                provider_requirements({"provider": "gemini", "auth_mode": "api_key"}, lambda name: "a-key"),
                [],
            )

    def make_pdf_project(self):
        app.SESSION_CREDENTIALS["GEMINI_API_KEY"] = "test-key"
        project = self.client.post("/api/projects", json={"name": "PDF inputs"}).json()
        path = app.store.project_dir(project["id"])
        (path / "goal.md").write_text("Extract concepts")
        app.store.write_json(
            path / "taxonomy.json",
            {"version": 1, "title": "Concepts", "families": [{"id": "concepts", "label": "Concepts"}]},
        )
        pdf = path / "run/stages/collect/example.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(str(pdf))
        candidate = {
            "title": "Example",
            "doi": "10.1000/example",
            "download": {"status": "downloaded", "path": str(pdf)},
        }
        qid = app.stable_candidate_id(app.candidate_from_dict(candidate))
        app.store.write_json(path / "selection.json", {"candidate_ids": [qid]})
        app.store.write_json(path / "run/stages/collect/logs/manifest.json", {"candidates": [candidate]})
        return project, path, pdf

    def test_deleted_pdf_blocks_before_queuing_even_if_manifest_says_downloaded(self):
        project, path, pdf = self.make_pdf_project()
        pdf.unlink()
        with patch.object(app.jobs, "start") as start:
            result = self.client.post("/api/projects/" + project["id"] + "/extraction/run")
        self.assertEqual(result.status_code, 409, result.text)
        self.assertIn("PDF", result.text)
        start.assert_not_called()
        self.assertFalse(
            self.client.get("/api/projects/" + project["id"] + "/workflow").json()["steps"]["candidates"][
                "complete"
            ]
        )

    def test_corrupt_pdf_is_detected_before_the_extraction_job(self):
        project, path, pdf = self.make_pdf_project()
        pdf.write_bytes(b"%PDF-1.4\ninvalid contents")
        with patch.object(app.jobs, "start") as start:
            result = self.client.post("/api/projects/" + project["id"] + "/extraction/run")
        self.assertEqual(result.status_code, 409, result.text)
        start.assert_not_called()

    def test_readable_selected_pdf_is_ready_without_uploading_extra_files(self):
        project, path, pdf = self.make_pdf_project()
        result = self.client.get("/api/projects/" + project["id"] + "/readiness?action=extract")
        self.assertTrue(result.json()["ready"], result.text)
