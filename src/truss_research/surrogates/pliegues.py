"""Pliegues internos por diseño o por generaciones con purga explícita."""

import pandas as pd

from truss_research.datos.particion import IndexArray, training_folds
from truss_research.datos.particion_agrupada import (
    forward_generation_folds,
    purged_group_folds,
)
from truss_research.surrogates.configuracion import PredictorConfig


def predictor_folds(
    config: PredictorConfig, table: pd.DataFrame
) -> list[tuple[IndexArray, IndexArray]]:
    from truss_research.datos.temporal import first_appearance_folds
    from truss_research.surrogates.mlp_temporal import temporal_predictor

    if temporal_predictor(config):
        return first_appearance_folds(table, config.folds)
    seed = config.cv_seed if config.cv_seed is not None else config.seed
    if config.cv_protocol == "chronological":
        return forward_generation_folds(
            table,
            pd.read_csv(config.group_membership, float_precision="round_trip"),
            config.folds,
        )
    if config.group_membership is not None:
        return purged_group_folds(
            table,
            pd.read_csv(config.group_membership, float_precision="round_trip"),
            (config.folds, seed),
        )
    return training_folds(table, config.folds, seed)


def eligible_indices(config: PredictorConfig, table: pd.DataFrame) -> IndexArray:
    """Cobertura explícita: excluye calentamiento y filas purgadas."""
    import numpy as np

    return np.unique(
        np.concatenate([held for _, held in predictor_folds(config, table)])
    )
