import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from google.auth.exceptions import RefreshError

import lit_review.ui.app as app
from lit_review.ui.google_cloud import GoogleCloudSetup
from lit_review.ui.jobs import JobManager
from lit_review.ui.store import ProjectStore


URL = "https://accounts.google.com/o/oauth2/auth?client_id=example&state=random"
SETTINGS = {"enabled": True, "project": "example-project", "location": "us-central1"}
BASE = "/api/settings/google-cloud"


class CloudSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.root = Path(self.temp.name)
        app.store = ProjectStore(self.root)
        app.jobs = JobManager(app.store)
        app.SESSION_CREDENTIALS.clear()
        self.addCleanup(app.SESSION_CREDENTIALS.clear)
        self.client = TestClient(app.app)
        self.addCleanup(lambda: app.cloud_setup().close())

    def save(self, **overrides):
        result = self.client.put(BASE, json={**SETTINGS, **overrides})
        self.assertEqual(result.status_code, 200, result.text)
        return result.json()

    def fake_gcloud(self, exit_code=0, prefix="", timeout=600):
        original_popen = subprocess.Popen
        app.cloud_setup().login_timeout = timeout
        script = (
            "import sys\n"
            f"print({prefix!r}, flush=True)\n"
            f"print({URL!r}, flush=True)\n"
            "print('Enter authorization code: ', end='', flush=True)\n"
            "code = sys.stdin.readline().strip()\n"
            f"sys.exit({exit_code} if code == 'private-auth-code' else 1)\n"
        )
        self.commands, self.processes = [], []

        def start(command, **kwargs):
            self.commands.append(command)
            process = original_popen([sys.executable, "-u", "-c", script], **kwargs)
            self.processes.append(process)
            self.addCleanup(lambda: process.wait(timeout=3))
            self.addCleanup(lambda: process.terminate() if process.poll() is None else None)
            return process

        finder = patch("lit_review.ui.google_cloud.shutil.which", return_value="/fake/gcloud")
        launcher = patch("lit_review.ui.google_cloud.subprocess.Popen", side_effect=start)
        finder.start()
        launcher.start()
        self.addCleanup(finder.stop)
        self.addCleanup(launcher.stop)

    def wait_status(self, status):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            result = self.client.get(BASE).json()["login"]
            if result and result["status"] == status:
                return result
            threading.Event().wait(0.01)
        self.fail(f"Expected {status}, got {result}")

    def test_saved_defaults_apply_to_readiness_jobs_and_restart_without_api_keys(self):
        self.save()
        self.assertEqual(
            app.credential_env(),
            {
                "GOOGLE_CLOUD_PROJECT": "example-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
            },
        )
        with patch("google.auth.default", return_value=(object(), "example-project")):
            self.assertTrue(self.client.get("/api/readiness").json()["ready"])
        self.assertEqual(json.loads((self.root / "google_cloud.json").read_text()), SETTINGS)
        # Simulate a fresh process with no inherited Cloud variables.
        for key in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_GENAI_USE_VERTEXAI"):
            os.environ.pop(key, None)
        app.store = ProjectStore(self.root)
        loaded = self.client.get(BASE).json()
        self.assertTrue(loaded["enabled"])
        self.assertEqual(loaded["project"], "example-project")
        self.assertEqual(os.environ["GOOGLE_GENAI_USE_VERTEXAI"], "true")
        self.save(enabled=False)
        self.assertNotIn("GOOGLE_CLOUD_PROJECT", os.environ)
        self.assertNotIn("GOOGLE_GENAI_USE_VERTEXAI", os.environ)

    def test_invalid_settings_do_not_change_environment_or_write_files(self):
        for settings in (
            {"project": "../bad"},
            {"location": "https://evil.example"},
            {"project": ""},
            {"location": ""},
            {"access_token": "never-store-this"},
        ):
            with self.subTest(settings=settings):
                response = self.client.put(BASE, json={**SETTINGS, **settings})
                self.assertEqual(response.status_code, 422)
                self.assertFalse((self.root / "google_cloud.json").exists())
                self.assertNotIn("GOOGLE_CLOUD_PROJECT", os.environ)

    def test_browser_login_streams_url_and_accepts_code_without_exposing_secrets(self):
        self.save()
        self.fake_gcloud(prefix="private-refresh-token\nhttps://evil.example/authorize")
        response = self.client.post(BASE + "/login")
        self.assertEqual(response.status_code, 200)
        login = self.wait_status("awaiting_code")
        self.assertEqual(login["authorization_url"], URL)
        self.assertEqual(self.client.post(BASE + "/login").status_code, 409)
        self.assertEqual(self.client.put(BASE, json=SETTINGS).status_code, 422)
        response = self.client.post(BASE + f"/login/{login['id']}/code", json={"code": "private-auth-code"})
        self.assertEqual(response.status_code, 200, response.text)
        self.wait_status("succeeded")
        status = self.client.get(BASE)
        self.assertNotIn("private-auth-code", status.text)
        self.assertNotIn("private-refresh-token", status.text)
        self.assertIsNone(status.json()["login"]["authorization_url"])
        self.assertIn("--no-launch-browser", self.commands[0])
        self.assertIn("--project=example-project", self.commands[0])
        self.assertNotIn("private-auth-code", str(self.commands))
        self.assertEqual(json.loads((self.root / "google_cloud.json").read_text()), SETTINGS)

    def test_failed_login_clears_url_and_can_restart(self):
        self.save()
        self.fake_gcloud(exit_code=1, prefix="private-error")
        self.client.post(BASE + "/login")
        login = self.wait_status("awaiting_code")
        self.client.post(BASE + f"/login/{login['id']}/code", json={"code": "private-auth-code"})
        result = self.wait_status("failed")
        self.assertNotIn("private-error", str(result))
        self.assertIsNone(result["authorization_url"])
        response = self.client.post(BASE + "/login")
        self.assertEqual(response.status_code, 200)
        current = self.wait_status("awaiting_code")
        self.assertNotEqual(current["id"], login["id"])
        self.assertEqual(
            self.client.post(BASE + f"/login/{login['id']}/code", json={"code": "stale"}).status_code, 409
        )

    def test_cancel_and_timeout_stop_the_login_process(self):
        self.save()
        self.fake_gcloud(timeout=0.5)
        self.client.post(BASE + "/login")
        login = self.wait_status("awaiting_code")
        response = self.client.delete(BASE + f"/login/{login['id']}")
        self.assertEqual(response.json()["status"], "cancelled")
        self.assertIsNone(response.json()["authorization_url"])
        self.processes[0].wait(timeout=3)
        self.client.post(BASE + "/login")
        self.wait_status("expired")
        self.processes[1].wait(timeout=3)

    def test_missing_gcloud_and_external_credentials_have_actionable_errors(self):
        self.save()
        with patch("lit_review.ui.google_cloud.shutil.which", return_value=None):
            response = self.client.post(BASE + "/login")
        self.assertEqual(response.status_code, 409)
        self.assertIn("Install the Google Cloud CLI", response.text)
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/external.json"}):
            response = self.client.post(BASE + "/login")
        self.assertEqual(response.status_code, 409)
        self.assertIn("GOOGLE_APPLICATION_CREDENTIALS", response.text)

    def test_code_validation_and_origin_checks_do_not_launch_or_write(self):
        for method, path, data in (
            ("put", BASE, SETTINGS),
            ("post", BASE + "/login", None),
            ("post", BASE + "/check", None),
            ("delete", BASE + "/login/fake", None),
        ):
            with self.subTest(path=path), patch("lit_review.ui.google_cloud.subprocess.Popen") as launch:
                response = self.client.request(
                    method, path, json=data, headers={"Origin": "https://evil.example"}
                )
                self.assertEqual(response.status_code, 403)
                launch.assert_not_called()
        self.save()
        self.fake_gcloud()
        self.client.post(BASE + "/login")
        login = self.wait_status("awaiting_code")
        for code in ("", "first\nsecond", "first second", "x" * 5000):
            response = self.client.post(BASE + f"/login/{login['id']}/code", json={"code": code})
            self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.get(BASE).json()["login"]["status"], "awaiting_code")

    def test_credential_check_refreshes_adc_without_returning_tokens(self):
        self.save()
        credentials = Mock(token="private-access-token")
        with patch("google.auth.default", return_value=(credentials, "example-project")):
            response = self.client.post(BASE + "/check")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["verified"])
        credentials.refresh.assert_called_once()
        self.assertNotIn("private-access-token", response.text)
        with patch("google.auth.default", side_effect=RefreshError("private-details")):
            response = self.client.post(BASE + "/check")
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("private-details", response.text)

    def test_corrupt_saved_settings_can_be_repaired_from_the_app(self):
        (self.root / "google_cloud.json").write_text("broken json")
        setup = GoogleCloudSetup(self.root)
        self.assertTrue(setup.status()["configuration_error"])
        app.store.google_cloud = setup
        self.assertEqual(self.save()["configuration_error"], "")
