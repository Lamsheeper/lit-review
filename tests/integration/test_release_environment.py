"""The release verifier must work with CI's uv-managed interpreters."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "verify_release", Path(__file__).resolve().parents[2] / "scripts" / "verify_release.py"
)
verify_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_release)


class ReleaseEnvironmentTests(unittest.TestCase):
    def test_isolated_interpreter_finds_standard_library_outside_checkout(self):
        with tempfile.TemporaryDirectory(prefix="lit-review-environment-test-") as temporary:
            root = Path(temporary).resolve()
            environment = root / "environment"
            python = verify_release.create_environment(environment)
            # uv probes interpreters in isolated mode before installing a wheel.
            result = subprocess.run(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import encodings, json, sys; print(json.dumps({'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'version': list(sys.version_info[:2])}))",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            runtime = json.loads(result.stdout)
            self.assertEqual(Path(runtime["prefix"]).resolve(), environment)
            self.assertNotEqual(Path(runtime["base_prefix"]).resolve(), environment)
            self.assertEqual(runtime["version"], list(sys.version_info[:2]))
