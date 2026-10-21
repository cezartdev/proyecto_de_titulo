"""Comprueba entradas públicas desde una copia sin datos ni artefactos privados."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def downloaded_repo(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    for directory in ("src", "config"):
        shutil.copytree(
            root / directory,
            tmp_path / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for filename in ("pyproject.toml", "uv.lock", "README.md", ".python-version"):
        shutil.copy2(root / filename, tmp_path / filename)
    return tmp_path


def test_supervisor_plan_needs_only_standard_library(downloaded_repo: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "src/truss_research/evaluacion/recorrido.py",
            "--plan",
            "--interpolation",
        ],
        cwd=downloaded_repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["stages"][-1] == "interpolation"
    assert not (downloaded_repo / "reports").exists()
    assert not (downloaded_repo / "artifacts").exists()


def test_installed_cli_plans_without_excel_or_weights(downloaded_repo: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "truss_research",
            "ejecutar-estudio",
            "--destino",
            "artifacts/download_check",
            "--plan",
        ],
        cwd=downloaded_repo,
        env={**os.environ, "PYTHONPATH": str(downloaded_repo / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    assert not (downloaded_repo / "artifacts").exists()
    assert not (downloaded_repo / "reports").exists()


def test_full_campaign_rejects_download_without_authorized_excel(
    downloaded_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from truss_research.evaluacion.campana import Campaign

    monkeypatch.chdir(downloaded_repo)
    with pytest.raises(FileNotFoundError, match="Falta el Excel autorizado"):
        Campaign(
            argparse.Namespace(
                mode="full",
                estudio=Path("artifacts/download_check"),
                config=Path("config/study.json"),
                cpu_workers=1,
                accelerator="cpu",
            )
        )
    assert not (downloaded_repo / "artifacts").exists()
    assert not (downloaded_repo / "reports").exists()
