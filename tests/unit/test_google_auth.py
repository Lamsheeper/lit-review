import datetime
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from google.auth.exceptions import DefaultCredentialsError
from google.oauth2.credentials import Credentials

from lit_review.core.google_auth import ADCAuth, use_vertex, vertex_model_url
from lit_review.core.llm import make_llm_client
from lit_review.core.settings import ProviderSettings
from lit_review.embeddings.core import make_embedding_client, VertexEmbeddingClient, GeminiEmbeddingClient


class GoogleAuthTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_explicit_modes_and_project_take_precedence_over_environment(self):
        self.assertFalse(use_vertex("key"))
        self.assertFalse(use_vertex())
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "env-project"}):
            self.assertFalse(use_vertex("key"))
            self.assertTrue(use_vertex())
            self.assertTrue(use_vertex("key", project="explicit-project"))
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "true"}):
            self.assertTrue(use_vertex("key"))
            self.assertFalse(use_vertex("key", auth_mode="api_key", project="project"))
        self.assertTrue(use_vertex("key", auth_mode="adc"))
        with self.assertRaises(ValueError):
            use_vertex(auth_mode="unknown")

    def test_vertex_endpoints_validate_project_and_support_global(self):
        with patch.dict(
            os.environ, {"GOOGLE_CLOUD_PROJECT": "env-project", "GOOGLE_CLOUD_LOCATION": "global"}
        ):
            url = vertex_model_url(None, None, "models/gemini-2.5-flash")
            self.assertEqual(
                url,
                "https://aiplatform.googleapis.com/v1/projects/env-project/locations/global/publishers/google/models/gemini-2.5-flash",
            )
            self.assertIn(
                "https://us-central1-aiplatform.googleapis.com/v1/projects/explicit/",
                vertex_model_url("explicit", "us-central1", "gemini-2.5-flash"),
            )
        with self.assertRaisesRegex(ValueError, "requires vertex_project"):
            vertex_model_url(None, None, "gemini-2.5-flash")
        with self.assertRaisesRegex(ValueError, "Invalid Vertex AI project"):
            vertex_model_url("project/../../other", "global", "gemini-2.5-flash")

    def test_adc_discovers_once_and_refreshes_expired_tokens(self):
        credentials = Credentials(token="expired", quota_project_id="billing-project")
        credentials.expiry = datetime.datetime.now(datetime.timezone.utc).replace(
            tzinfo=None
        ) - datetime.timedelta(hours=1)

        def refresh(request):
            credentials.token = "fresh-token"
            credentials.expiry = datetime.datetime.now(datetime.timezone.utc).replace(
                tzinfo=None
            ) + datetime.timedelta(hours=1)

        auth = ADCAuth()
        with (
            patch("google.auth.default", return_value=(credentials, "project")) as discover,
            patch.object(credentials, "refresh", side_effect=refresh) as renew,
        ):
            first = auth.headers("https://aiplatform.googleapis.com")
            self.assertEqual(first["authorization"], "Bearer fresh-token")
            self.assertEqual(first["x-goog-user-project"], "billing-project")
            auth.headers("https://aiplatform.googleapis.com")
            self.assertEqual(renew.call_count, 1)
            credentials.expiry -= datetime.timedelta(hours=2)
            auth.headers("https://aiplatform.googleapis.com")
            self.assertEqual(renew.call_count, 2)
            discover.assert_called_once_with(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    def test_missing_adc_has_actionable_error_without_credential_details(self):
        with patch("google.auth.default", side_effect=DefaultCredentialsError("private credential detail")):
            with self.assertRaisesRegex(RuntimeError, "gcloud auth application-default login") as error:
                ADCAuth().headers("https://aiplatform.googleapis.com")
            self.assertNotIn("private credential detail", str(error.exception))

    def test_vertex_generation_preserves_pdf_json_and_continuation_support(self):
        client = make_llm_client(
            api_key="ignored-key",
            model="models/gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_provider="gemini",
            reasoning_effort="low",
            auth_mode="adc",
            vertex_project="test-project",
            vertex_location="global",
        )
        response = {
            "candidates": [{"content": {"parts": [{"text": "partial"}]}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 10},
        }
        with (
            patch.object(ADCAuth, "headers", return_value={"authorization": "Bearer adc-token"}),
            patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send,
        ):
            result = client.chat_result(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Read this"},
                            {"type": "file", "file": {"file_data": "data:application/pdf;base64,cGRm"}},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
        request = send.call_args.args[0]
        self.assertIn(
            "aiplatform.googleapis.com/v1/projects/test-project/locations/global/", request.full_url
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer adc-token")
        self.assertNotIn("ignored-key", str(request.headers))
        payload = json.loads(request.data)
        self.assertEqual(payload["contents"][0]["parts"][1]["inline_data"]["data"], "cGRm")
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.usage["promptTokenCount"], 10)

    def test_api_key_mode_keeps_developer_api_even_with_cloud_environment(self):
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "true", "GOOGLE_CLOUD_PROJECT": "project"}):
            client = make_llm_client(
                api_key="key",
                model="gemini-2.5-flash",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                reasoning_effort=None,
                api_provider="gemini",
                auth_mode="api_key",
            )
        response = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        with (
            patch("google.auth.default") as discover,
            patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send,
        ):
            self.assertEqual(client.chat([{"role": "user", "content": "hello"}]), "ok")
        discover.assert_not_called()
        self.assertEqual(send.call_args.args[0].get_header("X-goog-api-key"), "key")
        self.assertTrue(
            send.call_args.args[0].full_url.startswith("https://generativelanguage.googleapis.com/")
        )

    def test_embedding_routing_cache_only_mode_and_vertex_prediction_contract(self):
        options = dict(
            api_key=None,
            model="gemini-embedding-001",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            task_type="CLUSTERING",
            output_dimensionality=768,
            timeout=30,
        )
        self.assertIsNone(make_embedding_client(**options))
        self.assertIsInstance(make_embedding_client(**dict(options, api_key="key")), GeminiEmbeddingClient)
        client = make_embedding_client(
            **options, auth_mode="adc", vertex_project="project", vertex_location="us-central1"
        )
        self.assertIsInstance(client, VertexEmbeddingClient)
        response = {"predictions": [{"embeddings": {"values": [0.1, 0.2]}}]}
        with (
            patch.object(ADCAuth, "headers", return_value={"authorization": "Bearer token"}),
            patch(
                "urllib.request.urlopen",
                side_effect=lambda *a, **kw: io.BytesIO(json.dumps(response).encode()),
            ) as send,
        ):
            self.assertEqual(client.embed_texts(["first", "second"]), [[0.1, 0.2], [0.1, 0.2]])
        self.assertEqual(send.call_count, 2)
        request = send.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/models/gemini-embedding-001:predict"))
        self.assertEqual(
            json.loads(request.data)["instances"], [{"content": "second", "task_type": "CLUSTERING"}]
        )
        self.assertEqual(json.loads(request.data)["parameters"]["outputDimensionality"], 768)

    def test_provider_settings_reject_openai_adc_and_support_google_api_key_alias(self):
        with self.assertRaisesRegex(ValueError, "requires the Gemini provider"):
            ProviderSettings(provider="openai", auth_mode="adc")
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "alias-key"}):
            self.assertEqual(ProviderSettings().transport()["api_key"], "alias-key")
            self.assertIsNone(ProviderSettings(api_key_env="CUSTOM_KEY").transport()["api_key"])

    def test_citation_extraction_uses_vertex_and_preserves_schema(self):
        from lit_review.collection.citations import call_gemini_citation_extraction

        config = dict(
            llm_model="gemini-2.5-flash",
            auth_mode="adc",
            vertex_project="project",
            vertex_location="us-central1",
            timeout=30,
            retries=0,
            backoff=0,
        )
        response = {"candidates": [{"content": {"parts": [{"text": '{"paper_id":"seed","references":[]}'}]}}]}
        with (
            patch.object(ADCAuth, "headers", return_value={"authorization": "Bearer adc"}),
            patch("lit_review.collection.citations.post_json", return_value=response) as send,
        ):
            result = call_gemini_citation_extraction([], config, "seed")
        self.assertEqual(result.parsed["references"], [])
        self.assertTrue(send.call_args.args[0].endswith("/models/gemini-2.5-flash:generateContent"))
        self.assertEqual(send.call_args.kwargs["headers"]["authorization"], "Bearer adc")
        self.assertIn("responseJsonSchema", send.call_args.args[1]["generationConfig"])

    def test_report_service_uses_adc_instead_of_offline_mode_without_a_key(self):
        from lit_review.services import execute_operation

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "index.sqlite").touch()
            goal = root / "goal.md"
            goal.write_text("Review workplace posture factors.")
            response = {
                "candidates": [{"content": {"parts": [{"text": "# Report"}]}, "finishReason": "STOP"}]
            }

            def write_report(**kwargs):
                # Exercise the structured-result method used by the report writer.
                result = kwargs["client"].chat_result([{"role": "user", "content": "Write report"}])
                kwargs["output_path"].write_text(result.content)
                return result.content

            with (
                patch.object(ADCAuth, "headers", return_value={"authorization": "Bearer adc"}),
                patch(
                    "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())
                ) as send,
                patch("lit_review.evidence.review.llm_plan_goal", return_value={"questions": ["posture"]}),
                patch("lit_review.evidence.review.select_evidence", return_value=[]),
                patch("lit_review.evidence.review.write_llm_report", side_effect=write_report),
            ):
                trace = execute_operation(
                    "evidence.write",
                    {
                        "run_dir": str(root),
                        "goal": str(goal),
                        "provider": "gemini",
                        "auth_mode": "adc",
                        "vertex_project": "project",
                    },
                    output_dir=root / "output",
                )
            self.assertTrue(trace["llm_used"])
            self.assertIn("/projects/project/", send.call_args.args[0].full_url)
            self.assertEqual((root / "output/report.md").read_text(), "# Report")

    def test_all_model_analysis_commands_accept_cloud_options(self):
        from importlib import import_module

        for name in ("calibration", "clustering", "metadata", "comparison", "visualization"):
            with self.subTest(name=name):
                parser = import_module("lit_review.analysis." + name).build_parser()
                args, _ = parser.parse_known_args(
                    [
                        "a",
                        "b",
                        "--auth-mode",
                        "adc",
                        "--vertex-project",
                        "project",
                        "--vertex-location",
                        "global",
                    ]
                )
                self.assertEqual(args.auth_mode, "adc")
                self.assertEqual(args.vertex_project, "project")
                self.assertEqual(args.vertex_location, "global")
