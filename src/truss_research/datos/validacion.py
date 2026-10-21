"""Validación de integridad independiente de los umbrales de cumplimiento."""

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import (
    DESIGN_BOUNDS,
    INPUT_COLUMNS,
    INTEGER_COLUMNS,
    OUTPUT_COLUMNS,
    StructuralLimits,
)
from truss_research.datos.schemas import (
    COMPLIANCE_COLUMNS,
    GENERATION_COLUMN,
    ID_COLUMN,
    ID_PATTERN,
    SOLUTION_COLUMN,
    DatasetError,
    DatasetSpec,
)


def add_identifiers(table: pd.DataFrame) -> pd.DataFrame:
    identifiers = table[ID_COLUMN].astype("string").str.strip()
    if not identifiers.str.fullmatch(ID_PATTERN).fillna(False).all():
        raise DatasetError(
            "Identificadores inválidos: se exige el formato completo {generación;solución}"
        )
    extracted = identifiers.str.extract(ID_PATTERN)
    try:
        extracted = extracted.astype("int64")
    except (ValueError, OverflowError) as error:
        raise DatasetError(
            "Los identificadores exceden el rango entero admitido"
        ) from error
    result = table.copy()
    result[ID_COLUMN] = identifiers
    result[[GENERATION_COLUMN, SOLUTION_COLUMN]] = extracted
    return result


def validate_structure(table: pd.DataFrame, spec: DatasetSpec) -> None:
    if sorted(table[GENERATION_COLUMN].unique()) != list(
        range(spec.expected_generations)
    ):
        raise DatasetError(f"Se esperaban {spec.expected_generations} generaciones")
    if table.duplicated([GENERATION_COLUMN, SOLUTION_COLUMN]).any():
        raise DatasetError("Hay identificadores de generación/solución duplicados")
    counts = table.groupby(GENERATION_COLUMN).size()
    if not counts.eq(spec.solutions_per_generation).all():
        raise DatasetError(
            f"Cada generación debe tener {spec.solutions_per_generation} soluciones"
        )
    if not table[SOLUTION_COLUMN].between(0, spec.solutions_per_generation - 1).all():
        raise DatasetError(
            "Las soluciones deben cubrir el intervalo 0..N-1 de cada generación"
        )


def validate_values(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    columns = (*INPUT_COLUMNS, *OUTPUT_COLUMNS)
    try:
        result[list(columns)] = result[list(columns)].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise DatasetError(
            f"Las entradas y respuestas deben ser numéricas: {error}"
        ) from error
    if not np.isfinite(result[list(columns)].to_numpy(dtype=float)).all():
        raise DatasetError(
            "Las entradas y respuestas no pueden contener nulos ni infinitos"
        )
    for column in INTEGER_COLUMNS:
        if not result[column].mod(1).eq(0).all():
            raise DatasetError(f"La columna {column} requiere valores enteros")
    for column, (lower, upper) in DESIGN_BOUNDS.items():
        if not result[column].between(lower, upper).all():
            raise DatasetError(
                f"La columna {column} debe estar entre {lower} y {upper}"
            )
    if not result["mass_kg"].gt(0).all():
        raise DatasetError("La masa debe ser positiva")
    if result[["util_ratio", "z_disp_m"]].lt(0).any().any():
        raise DatasetError("La utilización y el desplazamiento deben ser no negativos")
    result[list(INTEGER_COLUMNS)] = result[list(INTEGER_COLUMNS)].astype("int64")
    return result


def add_threshold_indicators(
    table: pd.DataFrame, limits: StructuralLimits
) -> pd.DataFrame:
    result = table.copy()
    util_column, z_column, combined_column = COMPLIANCE_COLUMNS
    result[util_column] = result["util_ratio"].le(limits.utilization_limit)
    result[z_column] = (
        True
        if limits.displacement_limit_m is None
        else result["z_disp_m"].le(limits.displacement_limit_m)
    )
    result[combined_column] = result[util_column] & result[z_column]
    return result
