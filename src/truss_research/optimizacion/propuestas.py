"""Presupuestos y soporte de propuestas previos a cualquier evaluación."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from truss_research.cerchas.schemas import (
    DESIGN_BOUNDS,
    INPUT_COLUMNS,
    INTEGER_COLUMNS,
    WEB_CONFIG_VALUES,
)
from truss_research.datos.distancias import design_distances
from truss_research.optimizacion.contratos import BoolArray
from truss_research.surrogates.configuracion import FloatArray


@dataclass(frozen=True)
class ProposalParameters:
    pool_size: int = 32
    max_regeneration: int = 10
    support_quantile: float = 0.95
    uncertainty_multiplier: float = 1.0
    evaluation_episodes: int = 10
    max_evaluations: int = 256
    max_generated: int = 4096

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.pool_size,
                self.max_regeneration,
                self.evaluation_episodes,
                self.max_evaluations,
                self.max_generated,
            )
        ):
            raise ValueError("Los tamaños del piloto deben ser enteros positivos")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in (self.support_quantile, self.uncertainty_multiplier)
        ):
            raise ValueError("Los parámetros predictivos del piloto deben ser finitos")
        if not 0 < self.support_quantile <= 1 or self.uncertainty_multiplier < 0:
            raise ValueError(
                "Cuantil de cobertura o multiplicador de incertidumbre inválidos"
            )


def design_identity(row: FloatArray) -> tuple[float, ...]:
    """Clave exacta en memoria, sin discretizar alturas ni incorporar respuestas."""
    return tuple(float(value) for value in row)


@dataclass(frozen=True)
class TrainingSupport:
    features: FloatArray
    scales: tuple[float, ...]
    thresholds: dict[int, float]

    @classmethod
    def fit(cls, features: FloatArray, quantile: float) -> "TrainingSupport":
        values = np.asarray(features, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != len(INPUT_COLUMNS)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "La cobertura requiere siete entradas finitas de entrenamiento"
            )
        if not 0 < quantile <= 1 or len(values) < 2:
            raise ValueError("Cobertura insuficiente o cuantil inválido")
        scales = tuple(np.ptp(values, axis=0))
        thresholds: dict[int, float] = {}
        for category in np.unique(values[:, 0]):
            group = values[values[:, 0] == category]
            if len(group) < 2:
                continue
            distances = design_distances(group, group, scales)
            np.fill_diagonal(distances, np.inf)
            thresholds[int(category)] = float(
                np.quantile(distances.min(axis=1), quantile)
            )
        return cls(values.copy(), scales, thresholds)

    def distances(self, features: FloatArray) -> FloatArray:
        distances = design_distances(features, self.features, self.scales)
        distances[features[:, None, 0] != self.features[None, :, 0]] = np.inf
        return distances.min(axis=1)

    def accepts(self, features: FloatArray) -> BoolArray:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(INPUT_COLUMNS):
            raise ValueError("Las propuestas requieren siete entradas")
        lower, upper = np.array([DESIGN_BOUNDS[name] for name in INPUT_COLUMNS]).T
        integer_indices = [INPUT_COLUMNS.index(name) for name in INTEGER_COLUMNS]
        valid = (
            np.isfinite(values).all(axis=1)
            & (values >= lower).all(axis=1)
            & (values <= upper).all(axis=1)
        )
        valid &= np.isin(values[:, 0], WEB_CONFIG_VALUES)
        valid &= (
            values[:, integer_indices] == np.floor(values[:, integer_indices])
        ).all(axis=1)
        thresholds = np.array(
            [
                self.thresholds.get(int(value), -1.0) if np.isfinite(value) else -1.0
                for value in values[:, 0]
            ]
        )
        return valid & (self.distances(values) <= thresholds)
