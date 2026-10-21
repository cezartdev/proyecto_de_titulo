"""Primera aparición, disponibilidad causal y pliegues progresivos por generación."""

import hashlib
import json

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.schemas import DatasetError, PreparationConfig, TableSplit

TEMPORAL_PROTOCOL = "first_appearance"
EVENTS_FILE = "population_events.csv"
TEMPORAL_SCHEMA = 3


def design_id(row: pd.Series) -> str:
    values = [float(row[name]).hex() for name in INPUT_COLUMNS]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def temporal_designs(table: pd.DataFrame) -> pd.DataFrame:
    from truss_research.datos.auditoria import select_unique_designs

    ordered = table.sort_values(["generation", "solution"], kind="stable")
    unique = select_unique_designs(ordered).sort_values(["generation", "solution"])
    unique = unique.reset_index(drop=True)
    unique["design_id"] = unique.apply(design_id, axis=1)
    unique["first_generation"] = unique.generation
    return unique


def temporal_split(table: pd.DataFrame, config: PreparationConfig) -> TableSplit:
    fractions = np.array(
        [
            config.split.train_fraction * (1 - config.calibration_fraction),
            config.split.train_fraction * config.calibration_fraction,
            config.split.validation_fraction,
            config.split.test_fraction,
        ]
    )
    if np.any(fractions <= 0):
        raise DatasetError("El protocolo temporal requiere cuatro bloques positivos")
    cuts = np.floor(
        np.cumsum(fractions)[:-1] * config.dataset.expected_generations
    ).astype(int)
    labels = np.searchsorted(cuts, table.first_generation.to_numpy(), side="right")
    parts = [table.loc[labels == index].copy() for index in range(4)]
    if any(part.empty for part in parts):
        raise DatasetError("Primera aparición dejó un bloque vacío; revisar cobertura")
    return TableSplit(parts[0], parts[2], parts[3], parts[1])


def first_appearance_folds(
    table: pd.DataFrame, folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    generations = np.array(sorted(table.generation.unique()))
    if folds < 2 or len(generations) < folds + 1:
        raise DatasetError("Generaciones insuficientes para pliegues progresivos")
    blocks = np.array_split(generations, folds + 1)
    result = [
        (
            np.flatnonzero(table.generation.lt(block[0])),
            np.flatnonzero(table.generation.isin(block)),
        )
        for block in blocks[1:]
    ]
    if any(not len(fitting) or not len(held) for fitting, held in result):
        raise DatasetError("Un pliegue progresivo quedó vacío")
    return result


def verify_temporal_split(split: TableSplit) -> None:
    parts = (split.training, split.calibration, split.validation, split.testing)
    if any(part is None or part.empty for part in parts):
        raise DatasetError("Falta un bloque temporal")
    if any(
        left.generation.max() >= right.generation.min()
        for left, right in zip(parts[:-1], parts[1:], strict=True)
    ):
        raise DatasetError("Las primeras apariciones cruzan los cortes temporales")
