import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import lit_review_ui.app as app_module
from lit_review_ui.jobs import JobManager
from lit_review_ui.store import ProjectStore


class UiApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        app_module.SESSION_CREDENTIALS.clear()
        app_module.store = ProjectStore(Path(self.temp.name))
        app_module.jobs = JobManager(app_module.store)
        self.client = TestClient(app_module.app)

    def tearDown(self):
        self.temp.cleanup()

    def test_project_documents_taxonomy_and_candidate_preselection(self):
        response = self.client.post("/api/projects", json={"name": "API Review"})
        self.assertEqual(response.status_code, 200)
        project = response.json()
        project_id = project["id"]

        response = self.client.put(
            f"/api/projects/{project_id}/documents",
            json={"draft_markdown": "# Draft", "goal_markdown": "# Goal"},
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.put(
            f"/api/projects/{project_id}/taxonomy",
            json={
                "version": 1,
                "title": "Test",
                "families": [
                    {"id": "test_family", "label": "Test family", "description": "", "aliases": []}
                ],
            },
        )
        self.assertEqual(response.status_code, 200)

        path = app_module.store.project_dir(project_id)
        candidates = {
            "candidates": [
                {"title": "First", "year": 2024, "relevance_score": 0.9},
                {"title": "Second", "year": 2023, "relevance_score": 0.8},
            ]
        }
        (path / "papers/logs/candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
        app_module.store.write_json(path / "configs/collection.json", {"max_downloads": 1})

        response = self.client.get(f"/api/projects/{project_id}/collection/candidates?page_size=10")
        self.assertEqual(response.status_code, 200)
        rows = response.json()["items"]
        self.assertEqual(sum(bool(row["selected"]) for row in rows), 1)
        self.assertTrue(rows[0]["candidate_id"].startswith("paper_"))

    def test_project_delete_removes_project_and_refuses_active_jobs(self):
        project = self.client.post("/api/projects", json={"name": "Delete API"}).json()
        path = app_module.store.project_dir(project["id"])
        job = app_module.store.create_job(project["id"], "collection_search", path / "jobs" / "job.log")

        response = self.client.delete(f"/api/projects/{project['id']}")
        self.assertEqual(response.status_code, 409)
        self.assertTrue(path.exists())

        app_module.store.update_job(job["id"], status="completed")
        response = self.client.delete(f"/api/projects/{project['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["deleted"])
        self.assertFalse(path.exists())
        self.assertEqual(self.client.get(f"/api/projects/{project['id']}").status_code, 404)

    def test_document_generation_creates_relevance_scoring_profile(self):
        project = self.client.post("/api/projects", json={"name": "Generated"}).json()
        generated = {
            "draft_markdown": "# Draft",
            "goal_markdown": "# Goal",
            "taxonomy": {
                "version": 1,
                "title": "Generated taxonomy",
                "families": [
                    {
                        "id": "generated_family",
                        "label": "Generated family",
                        "description": "Generated.",
                        "aliases": [],
                    }
                ],
            },
            "relevance_scoring": {
                "version": 1,
                "title": "Generated relevance",
                "description": "Generated for this project.",
                "criteria": [
                    {
                        "id": "topic_match",
                        "label": "Topic match",
                        "description": "Matches the requested topic.",
                        "weight": 5,
                        "terms": ["requested topic"],
                    }
                ],
                "field_weights": {"title": 3, "abstract": 2, "matched_keywords": 1},
                "metadata_weights": {
                    "source_relevance": 0,
                    "citation_impact": 0,
                    "recency": 0,
                    "open_access": 0,
                    "doi": 0,
                },
                "exclusions": [],
            },
        }

        class FakeClient:
            def chat(self, *args, **kwargs):
                return json.dumps(generated)

        app_module.SESSION_CREDENTIALS["GEMINI_API_KEY"] = "test-key"
        with mock.patch.object(app_module, "make_llm_client", return_value=FakeClient()):
            response = self.client.post(
                f"/api/projects/{project['id']}/documents/generate",
                json={"request": "Find requested topic features"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["relevance_scoring"]["title"], "Generated relevance")
        path = app_module.store.project_dir(project["id"]) / "relevance_scoring.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["criteria"][0]["id"], "topic_match")

    def test_built_frontend_is_served(self):
        if not app_module.FRONTEND_DIST.exists():
            self.skipTest("Frontend bundle has not been built.")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Lit Review Studio", response.text)

    def test_session_credential_status_never_returns_secret(self):
        response = self.client.put(
            "/api/settings/session-credentials",
            json={"credentials": {"GEMINI_API_KEY": "super-secret"}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("super-secret", response.text)
        self.assertEqual(response.json()["GEMINI_API_KEY"]["source"], "session")
        self.assertEqual(
            set(response.json()),
            {"GEMINI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"},
        )

    def test_session_credentials_reject_unused_provider_keys(self):
        response = self.client.put(
            "/api/settings/session-credentials",
            json={"credentials": {"OPENAI_API_KEY": "unused-secret"}},
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("unused-secret", response.text)

    def test_raw_result_pagination_and_export(self):
        project = self.client.post("/api/projects", json={"name": "Results"}).json()
        path = app_module.store.project_dir(project["id"])
        feature_path = path / "feature_extract/features.jsonl"
        feature_path.write_text(
            "\n".join(
                json.dumps({"category": "alpha", "feature_name": name, "paper_count": count})
                for name, count in [("One", 1), ("Two", 2)]
            )
            + "\n",
            encoding="utf-8",
        )

        response = self.client.get(
            f"/api/projects/{project['id']}/results/features?page_size=1&sort=paper_count&order=desc"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 2)
        self.assertEqual(response.json()["items"][0]["feature_name"], "Two")
        export = self.client.get(f"/api/projects/{project['id']}/exports/features.jsonl")
        self.assertEqual(export.status_code, 200)
        self.assertIn('"feature_name": "One"', export.text)

    def test_config_redaction_preserves_api_key_env_not_secret(self):
        path = Path(self.temp.name) / "config.json"
        app_module.write_config(
            path,
            {"api_key": "secret", "api_key_env": "GEMINI_API_KEY", "model": "test"},
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("api_key", payload)
        self.assertEqual(payload["api_key_env"], "GEMINI_API_KEY")

    def test_extraction_prompt_preview_uses_current_direct_pdf_prompt(self):
        project = self.client.post("/api/projects", json={"name": "Prompt Preview"}).json()
        project_id = project["id"]
        path = app_module.store.project_dir(project_id)
        (path / "goal.md").write_text("# Goal\nExtract the exact target features.\n", encoding="utf-8")
        taxonomy = {
            "version": 1,
            "title": "Prompt taxonomy",
            "families": [
                {
                    "id": "prompt_family",
                    "label": "Prompt family",
                    "description": "Features relevant to prompt preview.",
                    "aliases": [],
                }
            ],
        }
        app_module.store.write_json(path / "taxonomy.json", taxonomy)
        pdf_path = path / "papers" / "preview.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        app_module.store.write_json(
            path / "papers/logs/manifest.json",
            {
                "candidates": [
                    {
                        "title": "Preview Paper",
                        "year": 2026,
                        "doi": "10.1000/preview",
                        "download": {"status": "downloaded", "path": str(pdf_path)},
                    }
                ]
            },
        )

        response = self.client.post(
            f"/api/projects/{project_id}/extraction/prompt-preview",
            json={"max_features_per_pdf": 7},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["attachment_filename"], "preview.pdf")
        self.assertIn("taxonomy extraction agent", payload["system_prompt"])
        self.assertIn("Preview Paper", payload["user_prompt"])
        self.assertIn("prompt_family", payload["user_prompt"])
        self.assertIn("Return at most 7 features from this PDF", payload["user_prompt"])

    def test_relevance_scoring_profile_lifecycle_and_harvest_config(self):
        project = self.client.post("/api/projects", json={"name": "Relevance"}).json()
        profile = {
            "version": 1,
            "title": "Topic relevance",
            "description": "Rank papers about the requested topic.",
            "criteria": [
                {
                    "id": "topic_match",
                    "label": "Topic match",
                    "description": "Matches the requested topic.",
                    "weight": 5,
                    "terms": ["requested topic"],
                }
            ],
            "field_weights": {"title": 3, "abstract": 2, "matched_keywords": 1},
            "metadata_weights": {
                "source_relevance": 0,
                "citation_impact": 0,
                "recency": 0,
                "open_access": 0,
                "doi": 0,
            },
            "exclusions": [],
        }
        response = self.client.put(
            f"/api/projects/{project['id']}/relevance-scoring",
            json=profile,
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f"/api/projects/{project['id']}/relevance-scoring")
        self.assertEqual(response.json()["criteria"][0]["id"], "topic_match")
        config = app_module.harvest_config(project["id"], metadata_only=True)
        self.assertTrue(config["relevance_scoring"].endswith("relevance_scoring.json"))

    def test_relevance_scoring_rejects_unknown_signal(self):
        project = self.client.post("/api/projects", json={"name": "Invalid relevance"}).json()
        response = self.client.put(
            f"/api/projects/{project['id']}/relevance-scoring",
            json={
                "version": 1,
                "title": "Invalid",
                "description": "Invalid signal.",
                "criteria": [
                    {
                        "id": "topic",
                        "label": "Topic",
                        "description": "Topic.",
                        "weight": 1,
                        "terms": ["topic"],
                    }
                ],
                "field_weights": {"title": 1},
                "metadata_weights": {"prestige": 1},
                "exclusions": [],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_workflow_status_guides_cached_rescoring_and_next_steps(self):
        project = self.client.post("/api/projects", json={"name": "Guided workflow"}).json()
        project_id = project["id"]
        path = app_module.store.project_dir(project_id)

        workflow = self.client.get(f"/api/projects/{project_id}/workflow").json()["steps"]
        self.assertFalse(workflow["define"]["complete"])
        self.assertEqual(workflow["collect"]["action_label"], "Find and rank papers")

        (path / "draft.md").write_text("# Draft\n", encoding="utf-8")
        (path / "goal.md").write_text("# Goal\n", encoding="utf-8")
        app_module.store.write_json(
            path / "taxonomy.json",
            {
                "version": 1,
                "title": "Workflow taxonomy",
                "families": [
                    {
                        "id": "workflow_family",
                        "label": "Workflow family",
                        "description": "Workflow.",
                        "aliases": [],
                    }
                ],
            },
        )
        app_module.store.write_json(
            path / "relevance_scoring.json",
            {
                "version": 1,
                "title": "Workflow relevance",
                "description": "Workflow relevance.",
                "criteria": [
                    {
                        "id": "workflow_match",
                        "label": "Workflow match",
                        "description": "Matches workflow.",
                        "weight": 1,
                        "terms": ["workflow"],
                    }
                ],
                "field_weights": {"title": 1, "abstract": 1, "matched_keywords": 1},
                "metadata_weights": {},
                "exclusions": [],
            },
        )
        app_module.store.write_json(
            path / "papers/logs/candidates.json",
            {"candidates": [{"title": "Cached paper", "doi": "10.1000/cached"}]},
        )

        workflow = self.client.get(f"/api/projects/{project_id}/workflow").json()["steps"]
        self.assertTrue(workflow["define"]["complete"])
        self.assertTrue(workflow["collect"]["complete"])
        self.assertTrue(workflow["collect"]["materials"]["uses_cached_candidates"])
        self.assertEqual(workflow["collect"]["action_label"], "Re-score cached candidates")
        self.assertFalse(workflow["candidates"]["complete"])

        job = app_module.store.create_job(
            project_id,
            "collection_search",
            path / "jobs" / "workflow-test.log",
        )
        workflow = self.client.get(f"/api/projects/{project_id}/workflow").json()["steps"]
        self.assertTrue(workflow["collect"]["running"])
        app_module.store.update_job(job["id"], status="completed")

        candidate_id = app_module.stable_candidate_id(
            app_module.candidate_from_dict({"title": "Cached paper", "doi": "10.1000/cached"})
        )
        app_module.store.write_json(path / "selection.json", {"candidate_ids": [candidate_id]})
        app_module.store.write_json(
            path / "papers/logs/manifest.json",
            {
                "candidates": [
                    {
                        "title": "Cached paper",
                        "doi": "10.1000/cached",
                        "download": {"status": "downloaded"},
                    }
                ]
            },
        )
        (path / "feature_extract/features.jsonl").write_text(
            json.dumps({"feature_name": "Feature"}) + "\n",
            encoding="utf-8",
        )

        workflow = self.client.get(f"/api/projects/{project_id}/workflow").json()["steps"]
        self.assertTrue(workflow["candidates"]["complete"])
        self.assertTrue(workflow["extract"]["complete"])
        self.assertTrue(workflow["results"]["complete"])

    def test_workflow_status_marks_refresh_as_new_metadata_search(self):
        project = self.client.post("/api/projects", json={"name": "Refresh workflow"}).json()
        path = app_module.store.project_dir(project["id"])
        app_module.store.write_json(
            path / "papers/logs/candidates.json",
            {"candidates": [{"title": "Cached paper"}]},
        )
        response = self.client.put(
            f"/api/projects/{project['id']}/collection/config",
            json={"refresh_candidates": True},
        )
        self.assertEqual(response.status_code, 200)

        collect = self.client.get(
            f"/api/projects/{project['id']}/workflow"
        ).json()["steps"]["collect"]
        self.assertEqual(collect["materials"]["run_mode"], "search_metadata")
        self.assertFalse(collect["materials"]["uses_cached_candidates"])
        self.assertEqual(collect["action_label"], "Find and rank papers")

    def test_collection_config_validates_year_range(self):
        project = self.client.post("/api/projects", json={"name": "Validation"}).json()
        response = self.client.put(
            f"/api/projects/{project['id']}/collection/config",
            json={"year_from": 2025, "year_to": 2020},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
