"""Rutas relocatables y lectura del formato vigente de manifiestos firmados."""

import hashlib
import json
import logging
import os
from pathlib import Path
from time import sleep
from typing import cast

from truss_research.datos.procesados import JsonValue, read_manifest
from truss_research.datos.schemas import (
    CALIBRATION_FILENAME,
    DESIGN_MEMBERSHIP_FILENAME,
    FULL_FILENAME,
    SPLIT_FILENAMES,
)

PATH_FORMAT = "relative-v1"
REPLACE_ATTEMPTS = 6
REPLACE_RETRY_SECONDS = 0.02
logger = logging.getLogger(__name__)


def replace_artifact(temporary: Path, destination: Path) -> None:
    """Conserva escritura atómica ante bloqueos transitorios del sistema de archivos."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            logger.warning(
                "Archivo temporalmente bloqueado: %s; reintento %s",
                destination,
                attempt + 1,
            )
            sleep(REPLACE_RETRY_SECONDS * 2**attempt)


PATH_KEYS = frozenset(
    (
        "data_dir",
        "output_dir",
        "predictor_dir",
        "resume_dir",
        "study_path",
        "reuse_dir",
        "preparation_dir",
        "source_predictor",
        "directory",
        "attempts",
        "measurement_files",
        "log_files",
        "benchmark_dir",
        "telemetry_file",
        "winner",
        "study",
    )
)


def encode_paths(value: JsonValue, context: tuple[Path, str]) -> JsonValue:
    root, key = context
    if key in ("measurement_signatures", "log_signatures") and isinstance(value, dict):
        return {
            Path(os.path.relpath(Path(name).resolve(), root)).as_posix(): item
            for name, item in value.items()
        }
    if isinstance(value, dict):
        return {name: encode_paths(item, (root, name)) for name, item in value.items()}
    if isinstance(value, list):
        return [encode_paths(item, context) for item in value]
    if key in PATH_KEYS and isinstance(value, str):
        return Path(os.path.relpath(Path(value).resolve(), root)).as_posix()
    return value


def decode_paths(value: JsonValue, context: tuple[Path, str]) -> JsonValue:
    root, key = context
    if key in ("measurement_signatures", "log_signatures") and isinstance(value, dict):
        return {str((root / name).resolve()): item for name, item in value.items()}
    if isinstance(value, dict):
        return {name: decode_paths(item, (root, name)) for name, item in value.items()}
    if isinstance(value, list):
        return [decode_paths(item, context) for item in value]
    if key in PATH_KEYS and isinstance(value, str):
        return str((root / value).resolve())
    return value


def read_artifact(path: Path) -> dict[str, JsonValue]:
    value: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path}")
    format_name = value.pop("_path_format", None)
    if format_name not in (None, PATH_FORMAT):
        raise ValueError("Formato de rutas de artefactos desconocido")
    if format_name == PATH_FORMAT:
        return cast(
            dict[str, JsonValue], decode_paths(value, (path.parent.resolve(), ""))
        )
    return value


def portable_payload(value: JsonValue, destination: Path) -> JsonValue:
    if not isinstance(value, dict):
        return value
    encoded = cast(
        dict[str, JsonValue], encode_paths(value, (destination.parent.resolve(), ""))
    )
    return {**encoded, "_path_format": PATH_FORMAT}


def dataset_identity(directory: Path) -> str:
    manifest = dict(read_manifest(directory))
    manifest.pop("source", None)
    files = dict(SPLIT_FILENAMES)
    files["full"] = FULL_FILENAME
    if (
        manifest.get("partition_policy")
        == "unique_design_first_appearance_chronological"
    ):
        files["events"] = "population_events.csv"
    if float(manifest.get("calibration_fraction", 0)) > 0:
        files["calibration"] = CALIBRATION_FILENAME
    membership = directory / DESIGN_MEMBERSHIP_FILENAME
    if membership.is_file():
        files["membership"] = membership.name
    hashes = {
        key: hashlib.sha256(
            (directory / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for key, name in files.items()
    }
    payload = {"identity_version": 2, "manifest": manifest, "tables": hashes}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
