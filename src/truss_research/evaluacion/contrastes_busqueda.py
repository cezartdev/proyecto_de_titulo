"""Contrastes predefinidos con remuestreo cruzado de semillas y escenarios."""

import numpy as np
import pandas as pd

from truss_research.datos.procesados import JsonValue
from truss_research.optimizacion.estrategias_busqueda import STRATEGY_DEFINITIONS
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.surrogates.configuracion import FloatArray

PRIMARY_CONTRASTS = (
    ("rl", "contextual_bandit"),
    ("rl_assisted", "contextual_bandit_assisted"),
    ("contextual_bandit", "random"),
    ("rl", "random"),
    ("contextual_bandit_assisted", "greedy"),
    ("rl_assisted", "greedy"),
    ("de", "evolution"),
    ("de_assisted", "greedy"),
)


def difference_matrix(group: pd.DataFrame, pair: tuple[str, str]) -> FloatArray:
    left, right = (
        group.loc[group.strategy.eq(name)].pivot(
            index="policy_seed", columns="scenario", values="auc"
        )
        for name in pair
    )
    if left.empty or right.empty or not left.columns.equals(right.columns):
        raise ValueError("Faltan escenarios emparejados para el contraste")
    if STRATEGY_DEFINITIONS[pair[1]].learned:
        if not left.index.equals(right.index):
            raise ValueError("Faltan semillas de aprendizaje emparejadas")
        matrix = (left - right).to_numpy()
    else:
        if len(right) != 1:
            raise ValueError("El control debe tener una ejecución por escenario")
        matrix = left.to_numpy() - right.to_numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("Contraste incompleto o no finito")
    return matrix


def paired_search_intervals(table: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.analysis.bootstrap_seed)
    output: list[dict[str, JsonValue]] = []
    for (experiment, budget), group in table.groupby(["experiment", "budget"]):
        for strategy, baseline in PRIMARY_CONTRASTS:
            if not {strategy, baseline} <= set(config.search.strategies):
                continue
            matrix = difference_matrix(group, (strategy, baseline))
            samples = [
                matrix[rng.integers(len(matrix), size=len(matrix))][
                    :, rng.integers(matrix.shape[1], size=matrix.shape[1])
                ].mean()
                for _ in range(config.analysis.bootstrap_repeats)
            ]
            output.append(
                {
                    "experiment": experiment,
                    "budget": budget,
                    "strategy": strategy,
                    "baseline": baseline,
                    "mean_auc_difference": float(matrix.mean()),
                    "lower": float(np.quantile(samples, 0.025)),
                    "upper": float(np.quantile(samples, 0.975)),
                    "training_seed_pairs": len(matrix),
                    "scenario_pairs": matrix.shape[1],
                    "scope": "exploratory_crossed_seeds_scenarios_single_historical_trajectory",
                }
            )
    return pd.DataFrame(
        output,
        columns=(
            "experiment",
            "budget",
            "strategy",
            "baseline",
            "mean_auc_difference",
            "lower",
            "upper",
            "training_seed_pairs",
            "scenario_pairs",
            "scope",
        ),
    )
