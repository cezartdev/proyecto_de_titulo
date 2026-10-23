"""Presentación canónica de artefactos cerrados, sin entrenamiento ni selección."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procedencia import confirmations_digest
from truss_research.datos.procesados import JsonValue

if TYPE_CHECKING:
    from truss_research.optimizacion.estudio_estado import StudyConfig

from truss_research import DEFAULT_STUDY

NOTEBOOK_DIRECTORY = Path(__file__).resolve().parents[3] / "notebooks"
NOTEBOOK_NAMES = (
    "01_datos",
    "02_prediccion",
    "03_seleccion",
    "04_conclusiones",
)
CELL_TIMEOUT_SECONDS = 180
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotebookContext:
    study_dir: Path
    report_dir: Path
    config: StudyConfig
    frozen: dict[str, JsonValue]


def resolve_study(directory: Path | None = None) -> Path:
    selected = directory if directory is not None else DEFAULT_STUDY
    if not (selected / "study.json").is_file():
        raise ValueError(
            "No hay estudio disponible; seleccionar ESTUDIO y completar make study"
        )
    return selected.resolve()


def resolve_notebook_study() -> Path:
    """Respeta la selección explícita y reconoce un único cierre disponible."""
    if "TRUSS_STUDY" in os.environ:
        return resolve_study(Path(os.environ["TRUSS_STUDY"]))
    default = NOTEBOOK_DIRECTORY.parent / DEFAULT_STUDY
    if (default / "study.json").is_file():
        return resolve_study(default)
    candidates = sorted(
        path.parent
        for path in default.parent.glob("*/study.json")
        if (path.parent / "evolution/closed.json").is_file()
        and read_artifact(path.parent / "evolution/closed.json").get("phase")
        == "complete"
    )
    if len(candidates) != 1:
        raise ValueError(
            "No hay un único estudio cerrado disponible; definir TRUSS_STUDY "
            f"con el estudio que se desea presentar. Candidatos: {candidates}"
        )
    return resolve_study(candidates[0])


def verify_presentation_report(study: Path, report: Path) -> None:
    """Comprueba procedencia y salidas declaradas antes de presentarlas."""
    provenance = read_artifact(report / "provenance.json")
    closed = read_artifact(study / "evolution" / "closed.json")
    if (
        Path(str(provenance.get("study", ""))).resolve() != study.resolve()
        or provenance.get("frozen") != read_artifact(study / "freeze.json")
        or provenance.get("physical_provenance_signature") != confirmations_digest()
        or closed.get("phase") != "complete"
        or closed.get("signature")
        != read_artifact(study / "freeze.json").get("signature")
    ):
        raise ValueError(
            "El informe no corresponde al cierre o procedencia vigente; ejecutar make report"
        )
    outputs = provenance.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("El informe no declara salidas firmadas")
    if any(
        not (report / name).is_file() or source_digest(report / name) != digest
        for name, digest in outputs.items()
    ):
        raise ValueError(
            "Cambió una salida del informe; crear otra exportación con make report"
        )


def notebook_context() -> NotebookContext:
    """Lee un informe vigente sin recalcular resultados."""
    from truss_research.evaluacion.registro import results_root
    from truss_research.optimizacion.estudio_estado import load_study_config

    study = resolve_notebook_study()
    config = load_study_config(study)
    if config.experiment != "evolution":
        raise ValueError(
            "La presentación canónica requiere el protocolo evolution vigente"
        )
    candidates = (
        [Path(os.environ["TRUSS_REPORT"])]
        if "TRUSS_REPORT" in os.environ
        else sorted(results_root(study).glob("informe_[0-9][0-9][0-9]"), reverse=True)
    )
    if not candidates:
        raise ValueError("Falta un informe cerrado; completar make final y make report")
    report = candidates[0].resolve()
    verify_presentation_report(study, report)
    return NotebookContext(study, report, config, read_artifact(study / "freeze.json"))


def export_notebooks(study: Path, destination: Path) -> Path:
    """Exporta el cierre vigente mediante las fuentes que se abren en Jupyter."""
    from truss_research.evaluacion.informe_busqueda import export_search
    from truss_research.optimizacion.estudio_estado import load_study_config

    if load_study_config(study).experiment != "evolution":
        raise ValueError(
            "La presentación canónica requiere el protocolo evolution vigente"
        )
    return export_search(study, destination, "notebooks")


def render_notebooks(study: Path, destination: Path) -> None:
    """Ejecuta solo lectura de tablas y gráficos en las cuatro fuentes canónicas."""
    import nbformat
    import pandas as pd
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    environment = {
        **os.environ,
        "TRUSS_STUDY": str(study.resolve()),
        "TRUSS_REPORT": str(destination.resolve()),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    figures: list[dict[str, str | int]] = []
    for name in NOTEBOOK_NAMES:
        source = NOTEBOOK_DIRECTORY / f"{name}.ipynb"
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(
            notebook,
            timeout=CELL_TIMEOUT_SECONDS,
            kernel_name="python3",
            resources={"metadata": {"path": str(NOTEBOOK_DIRECTORY.parent)}},
        )
        logger.info("Presentando %s", source.name)
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
        notebook.metadata["source_sha256"] = source_digest(source)
        nbformat.write(notebook, destination / source.name)
        html, _ = HTMLExporter().from_notebook_node(notebook)
        (destination / f"{name}.html").write_text(html, encoding="utf-8")
        for number, cell in enumerate(notebook.cells):
            for output_number, output in enumerate(cell.get("outputs", [])):
                label = cell.get("metadata", {}).get("figure_name", str(number))
                for mime, extension in (("image/png", "png"), ("image/svg+xml", "svg")):
                    value = output.get("data", {}).get(mime)
                    if not value:
                        continue
                    payload = "".join(value) if isinstance(value, list) else value
                    filename = f"{name}_{label}_{output_number}.{extension}"
                    content = (
                        base64.b64decode(payload)
                        if extension == "png"
                        else payload.encode("utf-8")
                    )
                    (destination / filename).write_bytes(content)
                    figures.append(
                        {
                            "notebook": name,
                            "cell": number,
                            "figure": label,
                            "format": extension,
                            "path": filename,
                        }
                    )
    pd.DataFrame(
        figures, columns=["notebook", "cell", "figure", "format", "path"]
    ).to_csv(destination / "figures.csv", index=False)
