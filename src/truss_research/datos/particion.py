"""Partición estratificada por diseños únicos y diagnóstico de categorías."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from truss_research.cerchas.schemas import WEB_CONFIG_VALUES
from truss_research.datos.auditoria import audit_split_overlap, design_keys
from truss_research.datos.schemas import (
    GENERATION_COLUMN,
    DatasetError,
    SplitConfig,
    SplitProtocol,
    TableDiagnostics,
    TableSplit,
)

logger = logging.getLogger(__name__)


def split_by_design(table: pd.DataFrame, config: SplitConfig) -> TableSplit:
    """Divide diseños únicos y estratifica por categoría, sin agrupar generaciones."""
    if not design_keys(table).is_unique:
        raise DatasetError(
            "La partición por diseño requiere una tabla sin diseños repetidos"
        )
    remainder = config.validation_fraction + config.test_fraction
    try:
        training, temporary = train_test_split(
            table,
            test_size=remainder,
            random_state=config.seed,
            stratify=table["web_config"],
        )
        validation, testing = train_test_split(
            temporary,
            test_size=config.test_fraction / remainder,
            random_state=config.seed,
            stratify=temporary["web_config"],
        )
    except ValueError as error:
        raise DatasetError(
            f"No hay suficientes diseños por categoría para estratificar: {error}"
        ) from error
    split = TableSplit(training.copy(), validation.copy(), testing.copy())
    verify_partition(table, split, "design")
    return split


def split_dataset(table: pd.DataFrame, config: SplitConfig) -> TableSplit:
    return split_by_design(table, config)


def reserve_calibration(split: TableSplit, fraction: float, seed: int) -> TableSplit:
    """Separa calibración antes de ajustar transformadores o seleccionar modelos."""
    if fraction == 0:
        return split
    training, calibration = train_test_split(
        split.training,
        test_size=fraction,
        random_state=seed,
        stratify=split.training["web_config"],
    )
    return TableSplit(
        training.copy(), split.validation, split.testing, calibration.copy()
    )


def verify_partition(
    table: pd.DataFrame, split: TableSplit, protocol: SplitProtocol = "design"
) -> None:
    if protocol == "first_appearance":
        from truss_research.datos.temporal import verify_temporal_split

        verify_temporal_split(split)
    subsets = list(split.as_mapping().values())
    group_sets = [set(subset[GENERATION_COLUMN]) for subset in subsets]
    if sum(len(subset) for subset in subsets) != len(table):
        raise DatasetError("La partición no conserva todas las filas")
    if set.union(*group_sets) != set(table[GENERATION_COLUMN]):
        raise DatasetError("La partición no conserva todos los generaciones")
    combined = pd.concat(subsets, ignore_index=True)
    if combined["iter_id"].duplicated().any() or set(combined["iter_id"]) != set(
        table["iter_id"]
    ):
        raise DatasetError(
            "La partición debe conservar exactamente los identificadores originales"
        )
    if not design_keys(combined).is_unique or any(audit_split_overlap(split).values()):
        raise DatasetError(
            "La partición por diseño contiene diseños repetidos o compartidos"
        )


def describe_table(table: pd.DataFrame, name: str) -> TableDiagnostics:
    counts = table["web_config"].value_counts()
    distribution = {value: int(counts.get(value, 0)) for value in WEB_CONFIG_VALUES}
    absent = [value for value, count in distribution.items() if count == 0]
    if absent:
        logger.warning("Categorías Web Config ausentes en %s: %s", name, absent)
    return TableDiagnostics(
        rows=len(table),
        generations=tuple(
            sorted(int(value) for value in table[GENERATION_COLUMN].unique())
        ),
        web_config_counts=distribution,
    )


type IndexArray = npt.NDArray[np.int64]


def training_folds(
    table: pd.DataFrame, folds: int, seed: int
) -> list[tuple[IndexArray, IndexArray]]:
    if folds < 2 or table["web_config"].value_counts().min() < folds:
        raise DatasetError("No hay suficientes diseños por categoría para los pliegues")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(table, table["web_config"]))
