"""Exportación de notebooks ejecutados sin alterar fuentes ni estudios firmados."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from truss_research.datos.procesados import JsonValue

if TYPE_CHECKING:
    from truss_research.optimizacion.estudio_estado import StudyConfig

from truss_research.evaluacion.informe_estudio import (
    latest_study_report,
    signed_records,
)

DEFAULT_STUDY = Path("artifacts/estudio")

CELL_TIMEOUT_SECONDS = 180

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotebookContext:
    study_dir: Path
    report_dir: Path
    config: StudyConfig
    state: dict[str, JsonValue]
    records: dict[str, Path]
    sensitivity_dir: Path | None = None
    benchmark_dir: Path | None = None
    learning_curves_dir: Path | None = None


def notebook_context() -> NotebookContext:
    """Comparte la lectura del estudio entre los cuatro notebooks sin entrenar."""
    from truss_research.evaluacion.benchmark import latest_benchmark
    from truss_research.evaluacion.curvas_aprendizaje import latest_learning_curves
    from truss_research.evaluacion.sensibilidad import latest_sensitivity
    from truss_research.optimizacion.estudio_estado import load_study_config, read_study

    root = Path(__file__).resolve().parents[3]
    selected = (
        Path(os.environ["TRUSS_STUDY"])
        if "TRUSS_STUDY" in os.environ
        else root / DEFAULT_STUDY
    )
    study = resolve_study(selected)
    report = latest_study_report(study)
    if report is None:
        raise ValueError(
            "Falta un informe vigente; completar make study o ejecutar make report"
        )
    state = read_study(study)
    records = {
        record["key"]: Path(record["directory"])
        for record in state["tasks"]
        if record["status"] == "complete"
    }
    return NotebookContext(
        study,
        report,
        load_study_config(study),
        state,
        records,
        latest_sensitivity(study),
        latest_benchmark(study) if (study / "freeze.json").is_file() else None,
        latest_learning_curves(study) if (study / "freeze.json").is_file() else None,
    )


def resolve_study(directory: Path | None = None) -> Path:
    selected = directory if directory is not None else DEFAULT_STUDY
    if not (selected / "study.json").is_file():
        raise ValueError(
            "No hay un estudio disponible; ejecutar make prepare y make study. Para presentar una comprobación, usar ESTUDIO con el directorio que informa make smoke."
        )
    logger.info("Estudio presentado: %s", selected.resolve())
    return selected.resolve()


def export_notebooks(study: Path, destination: Path) -> Path:
    """Ejecuta en memoria y exporta HTML, notebooks con salidas y figuras PNG."""
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    if destination.exists():
        raise ValueError("La presentación ya existe; elegir otro destino")
    if latest_study_report(study) is None:
        raise ValueError(
            "Falta un informe vigente; ejecutar make report y revisar el estudio"
        )
    records = signed_records(study)
    if not any(record["key"] == "decision/validation" for record in records):
        raise ValueError("Completar make study antes de exportar los cuatro notebooks")
    root = Path(__file__).resolve().parents[3]
    environment = {
        **os.environ,
        "TRUSS_STUDY": str(study.resolve()),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    destination.mkdir(parents=True, exist_ok=False)
    for source in sorted((root / "notebooks").glob("*.ipynb")):
        logger.info("Presentando %s", source.name)
        notebook = nbformat.read(source, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=CELL_TIMEOUT_SECONDS,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        try:
            if os.name == "nt":
                with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
                    runner.run(client.async_execute(env=environment))
            else:
                client.execute(env=environment)
        except Exception as error:
            raise RuntimeError(
                f"Falló {source.name}; presentación incompleta en {destination}: {error}"
            ) from error
        notebook.metadata["study_source"] = str(study.resolve())
        nbformat.write(notebook, destination / source.name)
        html, _ = HTMLExporter().from_notebook_node(notebook)
        (destination / f"{source.stem}.html").write_text(html, encoding="utf-8")
        for number, cell in enumerate(notebook.cells):
            for output_number, output in enumerate(cell.get("outputs", [])):
                png = output.get("data", {}).get("image/png")
                if png:
                    (
                        destination / f"{source.stem}_{number}_{output_number}.png"
                    ).write_bytes(base64.b64decode(png))
    return destination
