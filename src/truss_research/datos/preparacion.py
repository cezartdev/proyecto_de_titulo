"""Orquestación y exportación de la preparación con manifiesto reproducible."""

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd

from truss_research.cerchas.schemas import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    WEB_CONFIG_NAMES,
)
from truss_research.datos.auditoria import (
    audit_designs,
    audit_split_overlap,
    design_membership,
    select_unique_designs,
)
from truss_research.datos.particion import (
    describe_table,
    reserve_calibration,
    split_dataset,
)
from truss_research.datos.schemas import (
    CALIBRATION_FILENAME,
    COLUMN_MAPPING,
    DESIGN_MEMBERSHIP_FILENAME,
    FULL_FILENAME,
    ID_COLUMN,
    MANIFEST_FILENAME,
    PARTITION_POLICIES,
    RAW_ID_COLUMN,
    SHEET_NAME,
    SPLIT_FILENAMES,
    DatasetError,
    PreparationConfig,
    PreparationResult,
    TableSplit,
)
from truss_research.datos.temporal import (
    EVENTS_FILE,
    TEMPORAL_SCHEMA,
    design_id,
    temporal_designs,
    temporal_split,
)
from truss_research.datos.validacion import (
    add_identifiers,
    add_threshold_indicators,
    validate_structure,
    validate_values,
)

logger = logging.getLogger(__name__)


def source_digest(source: Path) -> str:
    with source.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_dataset(config: PreparationConfig) -> PreparationResult:
    if config.destination.exists():
        raise ValueError(
            "El destino de datos ya existe; no se reemplazan preparaciones"
        )
    digest = source_digest(config.source)
    table = add_identifiers(
        normalize_columns(load_raw_table(config.source, config.sheet_name))
    )
    validate_structure(table, config.dataset)
    table = add_threshold_indicators(validate_values(table), config.limits)
    source_table = table
    design_audit = audit_designs(table)
    table = (
        temporal_designs(table)
        if config.split.protocol == "first_appearance"
        else select_unique_designs(table)
    )
    split = prepared_partition(table, config)
    overlaps = audit_split_overlap(split)
    if any(overlaps.values()):
        logger.warning("Diseños compartidos entre conjuntos: %s", overlaps)
    tables = {"full": table, **split.as_mapping()}
    diagnostics = {
        name: describe_table(subset, name) for name, subset in tables.items()
    }
    config.destination.mkdir(parents=True, exist_ok=False)
    filenames = {"full": FULL_FILENAME, **SPLIT_FILENAMES}
    if split.calibration is not None:
        filenames["calibration"] = CALIBRATION_FILENAME
    paths = {
        name: config.destination / filename for name, filename in filenames.items()
    }
    for name, subset in tables.items():
        subset.to_csv(paths[name], index=False, lineterminator="\n")
    paths["design_membership"] = config.destination / DESIGN_MEMBERSHIP_FILENAME
    design_membership(source_table, split).to_csv(
        paths["design_membership"], index=False, lineterminator="\n"
    )
    if config.split.protocol == "first_appearance":
        events = source_table.sort_values(["generation", "solution"]).copy()
        events["design_id"] = events.apply(design_id, axis=1)
        paths["population_events"] = config.destination / EVENTS_FILE
        events.to_csv(paths["population_events"], index=False, lineterminator="\n")
    paths["manifest"] = config.destination / MANIFEST_FILENAME
    manifest = {
        "schema_version": TEMPORAL_SCHEMA,
        "source": config.source.name,
        "source_sha256": digest,
        "sheet_name": config.sheet_name,
        "dataset": asdict(config.dataset),
        "split": asdict(config.split),
        "calibration_fraction": config.calibration_fraction,
        "limits": {
            **asdict(config.limits),
            "displacement_limit_m": config.limits.displacement_limit_m,
        },
        "input_columns": INPUT_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "web_config_names": WEB_CONFIG_NAMES,
        "identifier_semantics": "successive_generation_and_solution",
        "geometry": "symmetric_half_truss_subdivisions; fixed_span_variable_heights_and_sections",
        "partition_policy": PARTITION_POLICIES[config.split.protocol],
        "design_audit": asdict(design_audit),
        "design_overlap": overlaps,
        "diagnostics": {name: asdict(value) for name, value in diagnostics.items()},
        "outputs": {name: path.name for name, path in paths.items()},
        "temporal_semantics": {
            "design_availability": "first_generation",
            "solution_order": "population_index_not_evaluation_order",
            "population_rows": len(source_table),
            "boundaries": {
                name: {
                    "min_generation": int(subset.generation.min()),
                    "max_generation": int(subset.generation.max()),
                }
                for name, subset in split.as_mapping().items()
            },
        }
        if config.split.protocol == "first_appearance"
        else None,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Preparación completa: %s filas en %s", len(table), config.destination)
    return PreparationResult(table, split, diagnostics, paths)


def prepared_partition(table: pd.DataFrame, config: PreparationConfig) -> TableSplit:
    """Separa generaciones completas; calibra aleatoriamente solo en el control por diseño."""
    if config.split.protocol == "first_appearance":
        return temporal_split(table, config)
    return reserve_calibration(
        split_dataset(table, config.split),
        config.calibration_fraction,
        config.split.seed,
    )


def load_raw_table(source: Path, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    if not source.is_file():
        raise DatasetError(f"No se encontró el Excel de origen: {source}")
    try:
        table = pd.read_excel(source, sheet_name=sheet_name, engine="openpyxl")
    except (OSError, ValueError, BadZipFile) as error:
        raise DatasetError(
            f"No se pudo leer {source}, hoja {sheet_name}: {error}"
        ) from error
    if table.empty:
        raise DatasetError("La hoja de origen está vacía")
    return table


def normalize_columns(table: pd.DataFrame) -> pd.DataFrame:
    required = {RAW_ID_COLUMN, *COLUMN_MAPPING}
    missing = required - set(table.columns)
    if missing:
        raise DatasetError(f"Faltan columnas obligatorias: {sorted(missing)}")
    if not table.columns.is_unique:
        raise DatasetError("Los nombres de columnas deben ser únicos")
    extras = [column for column in table.columns if column not in required]
    unexpected = [
        column
        for column in extras
        if not str(column).startswith("Unnamed:") or not table[column].isna().all()
    ]
    if unexpected:
        raise DatasetError(
            f"Columnas inesperadas o separadores con datos: {unexpected}"
        )
    return (
        table.drop(columns=extras)
        .rename(columns={RAW_ID_COLUMN: ID_COLUMN, **COLUMN_MAPPING})
        .copy()
    )


def ensure_dataset(config: PreparationConfig) -> Path:
    """Reutiliza únicamente una preparación íntegra del mismo origen y protocolo."""
    from truss_research.datos.procesados import load_prepared_split, read_manifest

    if not config.destination.exists():
        prepare_dataset(config)
        return config.destination
    manifest = read_manifest(config.destination)
    expected = {
        "source_sha256": source_digest(config.source),
        "sheet_name": config.sheet_name,
        "dataset": asdict(config.dataset),
        "split": asdict(config.split),
        "calibration_fraction": config.calibration_fraction,
        "limits": {
            **asdict(config.limits),
            "displacement_limit_m": config.limits.displacement_limit_m,
        },
        "input_columns": list(INPUT_COLUMNS),
        "output_columns": list(OUTPUT_COLUMNS),
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "La preparación existente pertenece a otro origen o protocolo; ejecutar make reset-study para reiniciar los derivados experimentales"
        )
    split = load_prepared_split(config.destination)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("El manifiesto no contiene los archivos de salida")
    for filename in outputs.values():
        if not (config.destination / str(filename)).is_file():
            raise ValueError("La preparación existente está incompleta")
    logger.info(
        "Se reutilizan los datos existentes: %s (%s diseños)",
        config.destination,
        sum(len(table) for table in split.as_mapping().values()),
    )
    return config.destination
