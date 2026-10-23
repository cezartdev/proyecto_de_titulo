"""Presentación sobre artefactos mínimos sintéticos, sin preparar ni entrenar."""

import ast
import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procedencia import confirmations_digest
from truss_research.evaluacion.informe_busqueda import write_export_provenance
from truss_research.evaluacion.presentacion import (
    NOTEBOOK_DIRECTORY,
    NOTEBOOK_NAMES,
    export_notebooks,
    notebook_context,
    render_notebooks,
    verify_presentation_report,
)
from truss_research.optimizacion.estudio_estado import (
    STUDY_SCHEMA_VERSION,
    artifact_signatures,
)
from truss_research.optimizacion.protocolo import load_protocol
from truss_research.surrogates.experimentos import write_json


@pytest.fixture
def presentation(tmp_path: Path) -> tuple[Path, Path]:
    study, report, data = (tmp_path / name for name in ("study", "report", "data"))
    for directory in (study, report, data, study / "evolution"):
        directory.mkdir(exist_ok=True)
    config = load_protocol(Path("config/smoke.json"), (study, data))
    write_json(
        study / "study.json",
        {"schema_version": STUDY_SCHEMA_VERSION, "config": asdict(config)},
    )
    frozen = {
        "signature": "synthetic",
        "selected": ["dummy", "dummy", "ridge"],
        "test_labels_used_for_training": False,
    }
    write_json(study / "freeze.json", frozen)
    write_json(
        study / "evolution/closed.json", {"phase": "complete", "signature": "synthetic"}
    )
    full = pd.DataFrame(
        [
            [0, 3, 1, 1, 2, 1, 1, 1000, 0.5, 0.01],
            [1, 4, 1, 1, 2, 1, 1, 1200, 0.6, 0.04],
        ],
        columns=[*INPUT_COLUMNS, *OUTPUT_COLUMNS],
    ).assign(iter_id=["{0;0}", "{1;0}"])
    for name in ("train", "full_clean"):
        full.to_csv(data / f"{name}.csv", index=False)
    pd.DataFrame(
        {
            "iter_id": full.iter_id,
            "representative_iter_id": full.iter_id,
            "split": "train",
        }
    ).to_csv(data / "design_membership.csv", index=False)
    write_json(
        data / "manifest.json",
        {
            "design_audit": {"rows": 3, "unique_designs": 2, "repeated_rows": 1},
            "design_overlap": {"train_to_test_shared_designs": 0},
            "limits": {"displacement_limit_m": 20 / 700, "utilization_limit": 1.0},
            "diagnostics": {
                name: {
                    "rows": 2,
                    "generations": [index],
                    "web_config_counts": {"0": 1, "1": 1},
                }
                for index, name in enumerate(
                    ("full", "train", "calibration", "validation", "test")
                )
            },
        },
    )
    pd.DataFrame(
        [
            {
                "split": split,
                "model": model,
                "role": "regressor",
                "budget": budget,
                "mean_auc": 1.0,
                "mean_normalized_auc": 1.0,
            }
            for split in ("validation", "test")
            for model in ("dummy", "ridge")
            for budget in config.budgets
        ]
    ).to_csv(report / "predictor_utility_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": split,
                "experiment": experiment,
                "strategy": strategy,
                "budget": budget,
                "mean_auc": 1.0,
            }
            for split in ("validation", "test")
            for experiment in ("A", "B")
            for strategy in ("random", "greedy")
            for budget in config.budgets
        ]
    ).to_csv(report / "comparison.csv", index=False)
    for name in (
        "category_coverage",
        "physical_provenance",
        "predictor_utility_paired_intervals",
        "prediction_utility",
        "predictive_model_summary",
        "predictive_seed_metrics",
        "predictive_category_metrics",
        "predictive_paired_errors",
        "predictor_costs",
        "predictive_inference_costs",
        "termination_counts",
        "target_attainment",
        "policy_costs",
    ):
        pd.DataFrame({"evidence": ["synthetic"]}).to_csv(
            report / f"{name}.csv", index=False
        )
    pd.DataFrame(columns=["policy", "searches_to_amortize"]).to_csv(
        report / "amortization.csv", index=False
    )
    for name in ("RESULTS", "EVIDENCE"):
        (report / f"{name}.md").write_text("# Evidencia sintética", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "model": "dummy",
                "seed": 42,
                "split": split,
                "target": target,
                "observed": value,
                "predicted": value + 0.01,
                "error": 0.01,
            }
            for split in ("validation", "test")
            for target in OUTPUT_COLUMNS
            for value in (1.0, 2.0)
        ]
    ).to_csv(report / "prediction_errors.csv", index=False)
    pd.DataFrame(
        {
            "model": ["dummy"],
            "seed": [42],
            "device": ["cpu"],
            "tuning_seconds": [0.1],
            "crossfit_seconds": [0.2],
            "fit_seconds": [0.1],
        }
    ).to_csv(report / "predictor_costs.csv", index=False)
    pd.DataFrame(
        [
            {"model": model, "split": split, "mean": error}
            for model, error in (("dummy", 0.1), ("ridge", 0.2))
            for split in ("validation", "test")
        ]
    ).to_csv(report / "predictive_model_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "dummy",
                "seed": 42,
                "split": split,
                "individual_median_seconds": 0.01,
                "batch_median_seconds": 0.02,
                "batch_rows": 2,
            }
            for split in ("validation", "test")
        ]
    ).to_csv(report / "predictive_inference_costs.csv", index=False)
    (study / "evolution/test").mkdir(parents=True)
    pd.DataFrame(columns=["budget", "strategy", "baseline", "experiment"]).to_csv(
        study / "evolution/test/paired_intervals.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "split": split,
                "experiment": mode,
                "strategy": strategy,
                "budget": max(config.budgets),
                "scenario": 0,
                "policy_seed": 42,
                "query": query,
                "hypervolume": query * 0.1,
            }
            for split in ("validation", "test")
            for mode in ("A", "B")
            for strategy in ("random", "greedy")
            for query in (config.evolution.population_size, max(config.budgets))
        ]
    ).to_csv(report / "search_curves.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": split,
                "experiment": mode,
                "strategy": "random",
                "mass_kg": 1000,
                "util_ratio": 0.5,
                "z_disp_m": 0.01,
                "is_pareto": True,
            }
            for split in ("validation", "test")
            for mode in ("A", "B")
        ]
    ).to_csv(report / "search_fronts.csv", index=False)
    for name in ("resource_tasks", "research_cost"):
        pd.DataFrame({"scope": ["synthetic"]}).to_csv(
            report / f"{name}.csv", index=False
        )
    write_export_provenance(study, report, "notebooks")
    return study, report


def test_canonical_notebook_sources_preserve_spanish_characters() -> None:
    import re

    for name in NOTEBOOK_NAMES:
        notebook = nbformat.read(NOTEBOOK_DIRECTORY / f"{name}.ipynb", as_version=4)
        text = "\n".join(cell.source for cell in notebook.cells)
        assert "validación" in text
        assert "\ufffd" not in text
        assert re.search(r"\w\?\w", text) is None


def test_canonical_notebooks_execute_only_presentation_cells(
    presentation: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ejecuta las cuatro fuentes con tablas pequeñas y conserva los artefactos."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython import display

    study, report = presentation
    monkeypatch.setenv("TRUSS_STUDY", str(study))
    monkeypatch.setenv("TRUSS_REPORT", str(report))
    monkeypatch.setattr(display, "display", lambda *values: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    before = artifact_signatures(study)
    tables_before = artifact_signatures(report)
    for name in NOTEBOOK_NAMES:
        notebook = nbformat.read(NOTEBOOK_DIRECTORY / f"{name}.ipynb", as_version=4)
        nbformat.validate(notebook)
        namespace = {}
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            assert cell.execution_count is None and not cell.outputs
            source = "\n".join(
                line for line in cell.source.splitlines() if not line.startswith("%")
            )
            ast.parse(source)
            exec(compile(source, name, "exec"), namespace)
        plt.close("all")
    assert artifact_signatures(study) == before
    assert artifact_signatures(report) == tables_before


def test_notebook_discovers_unique_closed_study_without_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion import presentacion

    monkeypatch.delenv("TRUSS_STUDY", raising=False)
    monkeypatch.setattr(presentacion, "NOTEBOOK_DIRECTORY", tmp_path / "notebooks")
    study = tmp_path / "artifacts/campana_002"
    (study / "evolution").mkdir(parents=True)
    write_json(study / "study.json", {})
    write_json(study / "evolution/closed.json", {"phase": "complete"})
    incomplete = tmp_path / "artifacts/incompleto"
    incomplete.mkdir()
    write_json(incomplete / "study.json", {})
    assert presentacion.resolve_notebook_study() == study.resolve()
    monkeypatch.setenv("TRUSS_STUDY", str(tmp_path / "inexistente"))
    with pytest.raises(ValueError, match="No hay estudio"):
        presentacion.resolve_notebook_study()
    monkeypatch.setenv("TRUSS_STUDY", str(study))
    assert presentacion.resolve_notebook_study() == study.resolve()


@pytest.mark.parametrize("count", [0, 2])
def test_notebook_discovery_requires_unambiguous_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    from truss_research.evaluacion import presentacion

    monkeypatch.delenv("TRUSS_STUDY", raising=False)
    monkeypatch.setattr(presentacion, "NOTEBOOK_DIRECTORY", tmp_path / "notebooks")
    for index in range(count):
        study = tmp_path / "artifacts" / f"campana_{index}"
        (study / "evolution").mkdir(parents=True)
        write_json(study / "study.json", {})
        write_json(study / "evolution/closed.json", {"phase": "complete"})
    with pytest.raises(ValueError, match="TRUSS_STUDY"):
        presentacion.resolve_notebook_study()


def test_notebook_preserves_existing_default_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion import presentacion

    monkeypatch.delenv("TRUSS_STUDY", raising=False)
    monkeypatch.setattr(presentacion, "NOTEBOOK_DIRECTORY", tmp_path / "notebooks")
    default = tmp_path / presentacion.DEFAULT_STUDY
    default.mkdir(parents=True)
    write_json(default / "study.json", {})
    assert presentacion.resolve_notebook_study() == default.resolve()


def test_report_integrity_and_explicit_context(
    presentation: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    study, report = presentation
    monkeypatch.setenv("TRUSS_STUDY", str(study))
    monkeypatch.setenv("TRUSS_REPORT", str(report))
    context = notebook_context()
    assert context.report_dir == report
    assert context.frozen["selected"][0] == "dummy"
    (report / "comparison.csv").write_text("alterado", encoding="utf-8")
    with pytest.raises(ValueError, match="Cambió una salida"):
        notebook_context()


def test_wrong_freeze_or_provenance_is_rejected(
    presentation: tuple[Path, Path],
) -> None:
    study, report = presentation
    path = report / "provenance.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    assert original["physical_provenance_signature"] == confirmations_digest()
    for changed in ({"frozen": {}}, {"physical_provenance_signature": "old"}):
        write_json(path, original | changed)
        with pytest.raises(ValueError, match="cierre o procedencia"):
            verify_presentation_report(study, report)


def test_legacy_presentation_is_rejected(presentation: tuple[Path, Path]) -> None:
    study, report = presentation
    path = study / "study.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["config"]["experiment"] = "catalog"
    write_json(path, state)
    with pytest.raises(ValueError, match="evolución"):
        export_notebooks(study, report / "legacy")


def test_renderer_uses_canonical_sources_and_preserves_study(
    presentation: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Comprueba el cableado de exportación sin iniciar kernels ni una campaña."""
    from nbclient import NotebookClient

    study, report = presentation
    before = artifact_signatures(study)
    visited: list[str] = []

    async def execute(client: NotebookClient, **kwargs: dict[str, str]) -> None:
        environment = kwargs["env"]
        assert environment["TRUSS_REPORT"] == str(report.resolve())
        visited.append(client.nb.cells[0].source)

    def execute_sync(client: NotebookClient, **kwargs: dict[str, str]) -> None:
        import asyncio

        asyncio.run(execute(client, **kwargs))

    monkeypatch.setattr(NotebookClient, "async_execute", execute)
    monkeypatch.setattr(NotebookClient, "execute", execute_sync)
    render_notebooks(study, report)
    write_export_provenance(study, report, "notebooks")
    assert len(visited) == len(NOTEBOOK_NAMES)
    assert len(list(report.glob("*.html"))) == len(NOTEBOOK_NAMES)
    provenance = json.loads((report / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["presentation_complete"] is True
    assert "provenance.json" not in provenance["outputs"]
    for name in NOTEBOOK_NAMES:
        notebook = nbformat.read(report / f"{name}.ipynb", as_version=4)
        assert (
            notebook.metadata["source_sha256"]
            == provenance["notebook_sources"][f"{name}.ipynb"]
        )
    verify_presentation_report(study, report)
    assert artifact_signatures(study) == before


def test_renderer_executes_a_minimal_kernel(
    presentation: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verifica ejecución y HTML con un notebook de dos celdas sin datos reales."""
    import truss_research.evaluacion.presentacion as module

    study, report = presentation
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Ejecución sintética"),
            nbformat.v4.new_code_cell(
                '%matplotlib inline\n%config InlineBackend.figure_formats = {"png", "svg"}\n'
                'import os\nimport matplotlib.pyplot as plt\nprint(os.environ["TRUSS_REPORT"])\n'
                "plt.plot([0, 1], [0, 1])\nplt.show()",
                metadata={"figure_name": "prueba_sintetica"},
            ),
        ]
    )
    nbformat.write(source, source_dir / "minimal.ipynb")
    monkeypatch.setattr(module, "NOTEBOOK_DIRECTORY", source_dir)
    monkeypatch.setattr(module, "NOTEBOOK_NAMES", ("minimal",))
    before = artifact_signatures(study)
    render_notebooks(study, report)
    exported = nbformat.read(report / "minimal.ipynb", as_version=4)
    assert exported.cells[1].execution_count == 1
    assert str(report.resolve()) in exported.cells[1].outputs[0].text
    assert (report / "minimal.html").is_file()
    assert len(list(report.glob("minimal_prueba_sintetica_*.png"))) == 1
    assert len(list(report.glob("minimal_prueba_sintetica_*.svg"))) == 1
    assert set(pd.read_csv(report / "figures.csv")["format"]) == {"png", "svg"}
    assert artifact_signatures(study) == before


def test_dev_opens_complete_presentation_and_sets_report(
    presentation: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ignora una exportación parcial más reciente y abre el informe correcto."""
    from subprocess import CompletedProcess

    import truss_research.__main__ as cli

    study, report = presentation
    completed = report.with_name("presentacion_001")
    report.rename(completed)
    for name in NOTEBOOK_NAMES:
        for extension in ("ipynb", "html"):
            (completed / f"{name}.{extension}").write_text(
                "synthetic", encoding="utf-8"
            )
    write_export_provenance(study, completed, "notebooks")
    partial = completed.with_name("presentacion_002")
    partial.mkdir()
    write_json(partial / "provenance.json", {"presentation_complete": False})
    monkeypatch.setattr(cli, "results_root", lambda directory: completed.parent)

    def launch(
        command: list[str], env: dict[str, str], check: bool
    ) -> CompletedProcess[str]:
        assert command[-1] == str(completed)
        assert env["TRUSS_REPORT"] == str(completed.resolve())
        assert env["TRUSS_STUDY"] == str(study.resolve())
        assert check is False
        return CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", launch)
    cli.run_jupyter_command(Namespace(estudio=study))
