"""Carga de particiones existentes sin volver a preparar ni dividir los datos."""

import json
from pathlib import Path

import pandas as pd

from truss_research.cerchas.schemas import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    StructuralLimits,
)
from truss_research.datos.auditoria import design_keys
from truss_research.datos.particion import verify_partition
from truss_research.datos.schemas import (
    CALIBRATION_FILENAME,
    FULL_FILENAME,
    IDENTIFIER_COLUMNS,
    MANIFEST_FILENAME,
    PARTITION_POLICIES,
    SPLIT_FILENAMES,
    DatasetError,
    SplitProtocol,
    TableSplit,
)
from truss_research.datos.validacion import (
    add_identifiers,
    validate_values,
)

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


def read_manifest(directory: Path) -> dict[str, JsonValue]:
    payload: JsonValue = json.loads(
        (directory / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise DatasetError("El manifiesto de preparación debe ser un objeto JSON")
    return payload


def load_structural_limits(directory: Path) -> StructuralLimits:
    """Recupera el criterio físico firmado en la preparación del dataset."""
    raw = read_manifest(directory).get("limits")
    if not isinstance(raw, dict):
        raise DatasetError("Faltan los límites físicos del dataset")
    required = ("span_m", "serviceability_divisor", "utilization_limit")
    if not set(required) <= raw.keys():
        raise DatasetError("Los límites físicos están incompletos")
    return StructuralLimits(**{name: raw[name] for name in required})


def manifest_protocol(payload: dict[str, JsonValue]) -> SplitProtocol:
    policy = payload.get("partition_policy")
    if policy not in PARTITION_POLICIES.values():
        raise DatasetError("Protocolo de partición no reconocido en el manifiesto")
    return next(
        protocol for protocol, value in PARTITION_POLICIES.items() if value == policy
    )


def verify_manifest_groups(
    tables: dict[str, pd.DataFrame], payload: dict[str, JsonValue]
) -> None:
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise DatasetError("El manifiesto no contiene las asignaciones de generaciones")
    for name, table in tables.items():
        diagnostic = diagnostics.get(name)
        if not isinstance(diagnostic, dict):
            raise DatasetError(f"Falta el diagnóstico de {name} en el manifiesto")
        if sorted(table["generation"].unique().tolist()) != diagnostic.get(
            "generations"
        ):
            raise DatasetError(
                f"Los generaciones de {name} no coinciden con el manifiesto"
            )


def load_processed_table(path: Path) -> pd.DataFrame:
    """Valida valores e identificadores almacenados antes de usar una tabla."""
    table = pd.read_csv(path, float_precision="round_trip")
    required = {*INPUT_COLUMNS, *OUTPUT_COLUMNS, *IDENTIFIER_COLUMNS}
    missing = required - set(table.columns)
    if missing or table.empty:
        raise DatasetError(
            f"Tabla procesada vacía o incompleta en {path}: {sorted(missing)}"
        )
    normalized = add_identifiers(table)
    if (
        not table[["generation", "solution"]]
        .eq(normalized[["generation", "solution"]])
        .all()
        .all()
    ):
        raise DatasetError(f"Identificadores derivados inconsistentes en {path}")
    return validate_values(normalized)


def load_prepared_split(directory: Path) -> TableSplit:
    """Lee los tres conjuntos para verificar integridad, sin ajustar modelos."""
    tables = {
        name: load_processed_table(directory / filename)
        for name, filename in SPLIT_FILENAMES.items()
    }
    manifest = read_manifest(directory)
    if float(manifest.get("calibration_fraction", 0)) > 0:
        tables["calibration"] = load_processed_table(directory / CALIBRATION_FILENAME)
    split = TableSplit(
        tables["train"], tables["validation"], tables["test"], tables.get("calibration")
    )
    combined = pd.concat(tables.values(), ignore_index=True)
    full = load_processed_table(directory / FULL_FILENAME)
    columns = [*INPUT_COLUMNS, *OUTPUT_COLUMNS]
    if (
        not full.iter_id.is_unique
        or not combined.iter_id.is_unique
        or set(full.iter_id) != set(combined.iter_id)
        or not full.set_index("iter_id")[columns]
        .sort_index()
        .eq(combined.set_index("iter_id")[columns].sort_index())
        .all()
        .all()
    ):
        raise DatasetError("La tabla completa difiere de sus particiones")
    protocol = manifest_protocol(manifest)
    verify_partition(combined, split, protocol)
    validate_prepared_structure(combined, manifest)
    verify_manifest_groups(tables, manifest)
    if protocol == "first_appearance":
        from truss_research.datos.auditoria import audit_designs
        from truss_research.datos.schemas import DatasetSpec
        from truss_research.datos.temporal import EVENTS_FILE, design_id
        from truss_research.datos.validacion import validate_structure

        events = load_processed_table(directory / EVENTS_FILE)
        validate_structure(events, DatasetSpec(**manifest["dataset"]))
        if audit_designs(events).conflicting_designs:
            raise DatasetError("Los eventos contienen respuestas contradictorias")
        computed = full.apply(design_id, axis=1)
        if "design_id" not in full or not computed.eq(full.design_id).all():
            raise DatasetError("Identidades temporales inconsistentes")
        if not full.first_generation.eq(full.generation).all():
            raise DatasetError("La primera aparición difiere del representante")
        if not events.apply(design_id, axis=1).eq(events.design_id).all():
            raise DatasetError("Un evento tiene identidad de diseño incorrecta")
        if (
            len(events)
            != manifest["dataset"]["expected_generations"]
            * manifest["dataset"]["solutions_per_generation"]
        ):
            raise DatasetError("El registro de apariciones está incompleto")
        first = events.groupby("design_id").generation.min()
        if not full.design_id.map(first).eq(full.generation).all() or set(
            first.index
        ) != set(full.design_id):
            raise DatasetError("Las primeras apariciones no coinciden con los eventos")
    return split


def validate_prepared_structure(
    table: pd.DataFrame, manifest: dict[str, JsonValue]
) -> None:
    audit = manifest.get("design_audit")
    if not isinstance(audit, dict) or audit.get("unique_designs") != len(table):
        raise DatasetError("El número de diseños únicos no coincide con el manifiesto")
    if not design_keys(table).is_unique:
        raise DatasetError("La tabla contiene diseños repetidos")


def load_design_split(directory: Path) -> TableSplit:
    return load_prepared_split(directory)
