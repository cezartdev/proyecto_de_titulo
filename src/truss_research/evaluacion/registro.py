"""Logs y duración de invocaciones e intentos, sin configuración al importar."""

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import gmtime, perf_counter

from truss_research.datos.artefactos import replace_artifact
from truss_research.datos.procesados import JsonValue


def results_root(study: Path) -> Path:
    """Resultados junto al estudio; los estudios temporales permanecen aislados."""
    parent = study.resolve().parent
    root = parent.parent if parent.name == "artifacts" else parent
    return root / "reports" / study.name


@contextmanager
def execution_log(
    path: Path, label: str, details: dict[str, JsonValue] | None = None
) -> Iterator[dict[str, JsonValue]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"%(asctime)sZ %(levelname)s %(process)d [{label}] %(name)s: %(message)s"
    )
    formatter.converter = gmtime
    handler.setFormatter(formatter)
    root = logging.getLogger()
    previous = root.level
    root.setLevel(min(previous, logging.INFO))
    root.addHandler(handler)
    record: dict[str, JsonValue] = {
        "label": label,
        "pid": os.getpid(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "details": details or {},
    }
    metadata = path.with_suffix(".execution.json")
    started = perf_counter()
    try:
        save_execution(metadata, record)
        root.info("Inicio de %s", label)
        yield record
        record["status"] = "complete"
    except BaseException as error:
        from truss_research.surrogates.convergencia import NumericalFitError

        if isinstance(error, NumericalFitError):
            record["numerical_diagnostics"] = error.attempts
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        handler.handle(
            logging.LogRecord(
                __name__,
                logging.ERROR,
                __file__,
                0,
                f"Falló {label}",
                (),
                sys.exc_info(),
            )
        )
        root.error("Falló %s: %s", label, error)
        raise
    finally:
        record.update(
            finished_at_utc=datetime.now(UTC).isoformat(),
            wall_seconds=perf_counter() - started,
        )
        root.info(
            "Fin de %s: %s, %.3f s", label, record["status"], record["wall_seconds"]
        )
        root.removeHandler(handler)
        handler.close()
        root.setLevel(previous)
        save_execution(metadata, record)


def save_execution(path: Path, record: dict[str, JsonValue]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    replace_artifact(temporary, path)


def next_log(root: Path, command: str) -> Path:
    numbers = [
        int(path.stem.removeprefix(command + "_"))
        for path in root.glob(command + "_*.log")
        if path.stem.removeprefix(command + "_").isdigit()
    ]
    return root / f"{command}_{max(numbers, default=0) + 1:03d}.log"


def attempt_logs(destination: Path) -> tuple[Path, Path]:
    return destination.with_suffix(".log"), destination.with_suffix(".execution.json")
