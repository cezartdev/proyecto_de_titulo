"""Importación explícita de predictores completos conservando su procedencia."""

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from shutil import copy2, copytree

from truss_research.datos.artefactos import PATH_KEYS, portable_payload, read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    StudySession,
    TaskRecord,
    artifact_signatures,
    environment_versions,
    load_study_config,
    read_study,
    source_signature,
    study_lock,
    verify_task,
)
from truss_research.surrogates.experimentos import data_signature, write_json

IMPORT_FILE = "predictor_import.json"
ORIGIN_DIRECTORY = "predictor_origin"
COPY_DIRECTORIES = ("tasks", "preparation", "studies", "execution")
SOURCE_ROOT = Path(__file__).resolve().parents[3]
ORCHESTRATION_CHANGES = {
    "src/truss_research/__main__.py": frozenset(("main",)),
    "src/truss_research/optimizacion/estudio.py": frozenset(("run_study",)),
    "src/truss_research/optimizacion/estudio_estado.py": frozenset(
        ("StudySession", "open_session")
    ),
    "src/truss_research/evaluacion/busqueda_evolutiva.py": frozenset(
        ("run_evolution_study",)
    ),
    "src/truss_research/evaluacion/informe_busqueda.py": frozenset(
        ("export_search", "verify_closed_search")
    ),
}
NEW_ORCHESTRATION = frozenset(
    (
        "src/truss_research/evaluacion/reutilizacion_predictores.py",
        "src/truss_research/optimizacion/paralelismo_politicas.py",
    )
)


@dataclass(frozen=True)
class PredictorImport:
    source: Path
    destination: Path
    source_manifest: Path


def current_sources(root: Path) -> dict[str, str]:
    paths = [
        *sorted((root / "src").rglob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in paths
    }


def definitions(source: str) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def verify_source_manifest(manifest: dict[str, JsonValue], expected: str) -> None:
    files = manifest["source_files"]
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if identity != expected or manifest["source_signature"] != expected:
        raise ValueError("El inventario de código no corresponde al estudio original")
    current = current_sources(SOURCE_ROOT)
    changed = {name for name in files if files[name] != current.get(name)}
    if (
        changed - ORCHESTRATION_CHANGES.keys()
        or set(current) - set(files) - NEW_ORCHESTRATION
    ):
        raise ValueError(
            "Cambió código científico o dependencias; no reutilizar predictores"
        )
    for name in changed:
        previous = manifest["orchestration_sources"][name]
        if (
            hashlib.sha256(previous.replace("\r\n", "\n").encode()).hexdigest()
            != files[name]
        ):
            raise ValueError("La fuente original no coincide con su firma")
        old = definitions(previous)
        new = definitions((SOURCE_ROOT / name).read_text(encoding="utf-8"))
        if any(
            new.get(key) != value
            for key, value in old.items()
            if key not in ORCHESTRATION_CHANGES[name]
        ):
            raise ValueError(f"Cambió una definición ajena a la orquestación: {name}")


def verified_records(state: dict[str, JsonValue]) -> dict[str, dict[str, TaskRecord]]:
    records = {}
    for section in ("tasks", "preparation_tasks"):
        records[section] = {}
        for raw in state[section]:
            record = TaskRecord(**raw)
            verify_task(record)
            records[section][record.key] = record
    for run in state["execution_runs"]:
        if run.get("telemetry_file") and source_digest(
            Path(run["telemetry_file"])
        ) != run.get("telemetry_signature"):
            raise ValueError("Cambió la telemetría predictiva original")
    return records


def inspect_import(request: PredictorImport) -> dict[str, JsonValue]:
    source, destination = request.source.resolve(), request.destination.resolve()
    if (
        destination.exists()
        or source == destination
        or destination.is_relative_to(source)
    ):
        raise ValueError("La importación requiere un destino nuevo fuera del origen")
    state = read_study(source)
    manifest = json.loads(request.source_manifest.read_text(encoding="utf-8"))
    if source_digest(source / "study.json") != manifest["study_signature"]:
        raise ValueError("El estudio original cambió desde el inventario previo")
    verify_source_manifest(manifest, state["source_signature"])
    config = load_study_config(source)
    if (
        state.get("test_started")
        or (source / "freeze.json").exists()
        or (source / IMPORT_FILE).exists()
    ):
        raise ValueError(
            "Importar únicamente predictores originales anteriores a prueba"
        )
    if state["versions"] != environment_versions() or state[
        "dataset_signature"
    ] != data_signature(config.data_dir):
        raise ValueError("Cambió el entorno o dataset del estudio original")
    records = verified_records(state)
    expected = {
        f"predictor/{name}/{seed}" for name in config.models for seed in config.seeds
    }
    if set(records["tasks"]) != expected:
        raise ValueError("Faltan familias o semillas predictivas completas")
    from truss_research.evaluacion.benchmark_multiobjetivo import verify_benchmark

    utility = source / "evolution/predictor_utility/validation"
    verify_benchmark(utility)
    return {
        "source": str(source),
        "study_signature": source_digest(source / "study.json"),
        "source_signature": state["source_signature"],
        "source_manifest": manifest,
        "validation_files": artifact_signatures(utility),
        "discarded_policy_files": artifact_signatures(source / "evolution/policies"),
        "selected": read_artifact(source / "evolution/search.json")["selected"],
        "predictors": len(records["tasks"]),
        "preparations": len(records["preparation_tasks"]),
        "policy_restart": "all_from_zero",
        "storage": "independent_copy",
        "copy_bytes": sum(
            path.stat().st_size
            for name in (*COPY_DIRECTORIES, "evolution/predictor_utility/validation")
            for path in (source / name).rglob("*")
            if path.is_file()
        )
        + (source / "study.json").stat().st_size,
        "predictive_cost_scope": "reused_original_training_not_new_execution",
    }


def import_predictors(request: PredictorImport, dry_run: bool = False) -> Path:
    if not (request.source / "study.json").is_file():
        raise ValueError("No existe un estudio de origen para importar")
    with study_lock(request.source):
        receipt = inspect_import(request)
        if dry_run:
            print(
                json.dumps(
                    {
                        key: receipt[key]
                        for key in (
                            "source",
                            "predictors",
                            "preparations",
                            "policy_restart",
                            "storage",
                            "copy_bytes",
                        )
                    },
                    indent=2,
                )
            )
            return request.destination
        state = read_study(request.source)
        config = replace(
            load_study_config(request.source), output_dir=request.destination.resolve()
        )
        with study_lock(config.output_dir):
            relocations = copy_predictive_files(request)
            copied = relocate_paths(
                state, (request.source.resolve(), config.output_dir)
            )
            records = copied_records(copied)
            archive = config.output_dir / ORIGIN_DIRECTORY
            archive.mkdir()
            copy2(request.source / "study.json", archive / "study_original.json")
            copytree(
                request.source / "evolution/predictor_utility/validation",
                archive / "validation",
            )
            receipt["discarded_policies"] = discarded_policy_costs(request.source)
            receipt["metadata_relocations"] = relocations
            receipt["storage"] = "independent_copy"
            receipt["local_files"] = {
                name: artifact_signatures(config.output_dir / name)
                for name in (*COPY_DIRECTORIES, ORIGIN_DIRECTORY)
            }
            session = StudySession(
                config,
                state["dataset_signature"],
                source_signature(),
                state["versions"],
                tasks=records["tasks"],
                preparation_tasks=records["preparation_tasks"],
                execution_runs=copied["execution_runs"],
            )
            session.save()
            receipt["predictive_state_signature"] = predictive_state_signature(
                read_study(config.output_dir), config.output_dir
            )
            write_json(config.output_dir / IMPORT_FILE, receipt)
            session.predictor_import_signature = source_digest(
                config.output_dir / IMPORT_FILE
            )
            session.save()
    return config.output_dir


def verify_imported_predictors(config: StudyConfig) -> None:
    state = read_study(config.output_dir)
    marker = config.output_dir / IMPORT_FILE
    if source_digest(marker) != state.get("predictor_import_signature"):
        raise ValueError("Cambió la procedencia de los predictores importados")
    receipt = read_artifact(marker)
    if asdict(
        replace(
            config,
            output_dir=config.output_dir.resolve(),
            data_dir=config.data_dir.resolve(),
        )
    ) != asdict(load_study_config(config.output_dir)):
        raise ValueError("La configuración difiere de la copia predictiva")
    if receipt.get("storage") != "independent_copy":
        raise ValueError("La importación requiere copias independientes")
    if (
        source_digest(config.output_dir / ORIGIN_DIRECTORY / "study_original.json")
        != receipt["study_signature"]
    ):
        raise ValueError("Cambió la evidencia local del estudio original")
    if (
        predictive_state_signature(state, config.output_dir)
        != receipt["predictive_state_signature"]
    ):
        raise ValueError("Los registros predictivos difieren de su procedencia")
    verified_records(state)
    for name, expected in receipt["local_files"].items():
        if artifact_signatures(config.output_dir / name) != expected:
            raise ValueError(f"Cambió la copia local de predictores: {name}")


def verify_reused_validation(config: StudyConfig, selected: tuple[str, ...]) -> None:
    marker = config.output_dir / IMPORT_FILE
    if not marker.exists():
        return
    import pandas as pd

    receipt = read_artifact(marker)
    original = config.output_dir / ORIGIN_DIRECTORY / "validation"
    current = config.output_dir / "evolution/predictor_utility/validation"
    for name in (
        "summary.csv",
        "family_summary.csv",
        "paired_intervals.csv",
        "decisions.csv",
    ):
        pd.testing.assert_frame_equal(
            pd.read_csv(original / name), pd.read_csv(current / name), check_exact=True
        )
    if list(selected) != receipt["selected"]:
        raise ValueError("La selección predictiva cambió al reutilizar resultados")


def relocate_path(value: str, roots: tuple[Path, Path]) -> str:
    source, destination = roots
    path = Path(value)
    if path.is_absolute() and path.resolve().is_relative_to(source):
        return str(destination / path.resolve().relative_to(source))
    return value


def relocate_paths(
    value: JsonValue, roots: tuple[Path, Path], key: str = ""
) -> JsonValue:
    if isinstance(value, dict):
        if key in ("measurement_signatures", "log_signatures"):
            return {relocate_path(name, roots): item for name, item in value.items()}
        return {name: relocate_paths(item, roots, name) for name, item in value.items()}
    if isinstance(value, list):
        return [relocate_paths(item, roots, key) for item in value]
    if key in PATH_KEYS and isinstance(value, str):
        return relocate_path(value, roots)
    return value


def copy_predictive_files(request: PredictorImport) -> dict[str, JsonValue]:
    source, destination = request.source.resolve(), request.destination.resolve()
    relocations = {}
    for name in COPY_DIRECTORIES:
        original = source / name
        if not original.exists():
            continue
        for path in [original, *original.rglob("*")]:
            if (
                path.is_symlink()
                or path.is_junction()
                or not path.resolve().is_relative_to(source)
            ):
                raise ValueError("No copiar enlaces ni archivos externos al estudio")
        signatures = artifact_signatures(original)
        target = destination / name
        copytree(original, target)
        if artifact_signatures(target) != signatures:
            raise ValueError("La copia de predictores no coincide con el origen")
        for relative, signature in signatures.items():
            path = target / relative
            if path.suffix != ".json":
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("_path_format") != "relative-v1":
                continue
            payload = relocate_paths(
                read_artifact(original / relative), (source, destination)
            )
            if portable_payload(payload, path) != raw:
                write_json(path, payload)
                relocations[path.relative_to(destination).as_posix()] = {
                    "original": signature,
                    "copy": source_digest(path),
                }
    return relocations


def copied_records(state: dict[str, JsonValue]) -> dict[str, dict[str, TaskRecord]]:
    for section in ("tasks", "preparation_tasks"):
        for record in state[section]:
            record["signatures"] = artifact_signatures(Path(record["directory"]))
            for name in ("measurement_signatures", "log_signatures"):
                record[name] = {
                    path: source_digest(Path(path)) for path in record[name]
                }
    return verified_records(state)


def predictive_state_signature(state: dict[str, JsonValue], root: Path) -> str:
    payload = {
        name: state[name]
        for name in ("config", "tasks", "preparation_tasks", "execution_runs")
    }
    return hashlib.sha256(
        json.dumps(
            portable_payload(payload, root / "study.json"), sort_keys=True
        ).encode()
    ).hexdigest()


def discarded_policy_costs(source: Path) -> list[dict[str, JsonValue]]:
    rows = []
    for path in (source / "evolution/policies").glob("*/*/run.json"):
        row = read_artifact(path)
        rows.append(
            {
                "policy": str(path.parent.relative_to(source / "evolution/policies")),
                "scope": "discarded_sequential_attempt_checkpoint_lower_bound",
                "completed_steps": row["completed_steps"],
                "wall_seconds": row["wall_seconds"],
                "cpu_seconds": row["cpu_seconds"],
            }
        )
    return rows
