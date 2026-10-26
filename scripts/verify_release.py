"""Verify built archives and an installed wheel from outside the checkout."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path


def create_environment(path: Path) -> Path:
    # A copied uv-managed Python 3.10 binary can lose its stdlib path on Linux.
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(path)
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert "lit_review/cli.py" in names, "Missing package code"
        assert "lit_review/ui/frontend/dist/index.html" in names, (
            "Build the frontend before building distributions"
        )
        assert any(name.startswith("lit_review/ui/frontend/dist/assets/") for name in names), (
            "Missing frontend assets"
        )
        assert not any(
            "node_modules/" in name
            or name.startswith(("review_runs/", "faculty_publications/", "collected_"))
            for name in names
        ), "Local data leaked into wheel"
        entries = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        content = archive.read(entries).decode()
        assert "lit-review = lit_review.cli:main" in content
        assert "lit-extract" not in content, "Legacy entry points remain"


def main() -> None:
    distribution = Path(sys.argv[1] if len(sys.argv) > 1 else "dist").resolve()
    wheels = sorted(distribution.glob("lit_review-0.2.0-*.whl"))
    sources = sorted(distribution.glob("lit_review-0.2.0.tar.gz"))
    assert wheels and sources, "Build both wheel and source distribution first"
    wheel = wheels[-1]
    verify_wheel(wheel)
    dependencies = sysconfig.get_paths()["purelib"]
    with tempfile.TemporaryDirectory(prefix="lit-review-release-") as temporary:
        root = Path(temporary).resolve()
        environment = root / "environment"
        python = create_environment(environment)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)], check=True, cwd=root
        )
        env = dict(
            os.environ,
            PYTHONPATH=dependencies,
            MPLCONFIGDIR=str(root / "matplotlib"),
            LIT_REVIEW_UI_PROJECTS_DIR=str(root / "projects"),
        )
        subprocess.run(
            [
                str(python),
                "-c",
                "import lit_review; from pathlib import Path; assert Path(lit_review.__file__).is_relative_to(Path('environment').resolve()); print(lit_review.__version__)",
            ],
            check=True,
            cwd=root,
            env=env,
        )
        executable = environment / ("Scripts/lit-review.exe" if os.name == "nt" else "bin/lit-review")
        subprocess.run([str(executable), "--version"], check=True, cwd=root, env=env)
        subprocess.run(
            [str(executable), "features", "merge", "--help"],
            check=True,
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                "from fastapi.testclient import TestClient; from lit_review.ui.app import app;\nwith TestClient(app) as client:\n assert client.get('/').status_code == 200\n assert client.get('/api/projects').json() == []\n",
            ],
            check=True,
            cwd=root,
            env=env,
        )
        extraction = root / "source"
        extraction.mkdir()
        with tarfile.open(sources[-1]) as archive:
            for member in archive.getmembers():
                destination = (extraction / member.name).resolve()
                assert destination.is_relative_to(extraction), "Unsafe source archive path"
                assert not member.issym() and not member.islnk(), "Unexpected archive link"
            archive.extractall(extraction)
        source_root = next(extraction.iterdir())
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(root / "rebuilt"),
                str(source_root),
            ],
            check=True,
            cwd=root,
            stdout=subprocess.DEVNULL,
        )
        verify_wheel(next((root / "rebuilt").glob("*.whl")))
    print("Release verified: wheel install, CLI, bundled UI, and source rebuild.")


if __name__ == "__main__":
    main()
