"""Evaluadores privados: ninguna respuesta se obtiene al generar propuestas."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.distancias import design_distances
from truss_research.surrogates.configuracion import FloatArray
from truss_research.surrogates.experimentos import predict

CONSERVATIVE_SIGNS = np.array([1.0, -1.0, 1.0])


@dataclass(frozen=True)
class Evaluation:
    features: FloatArray
    responses: FloatArray
    distances: FloatArray
    evidence: str
    member_predictions: int = 0


class UnsupportedProposal(ValueError):
    """La evaluación histórica no puede asociar esta topología a una respuesta."""


class SearchEvaluator(Protocol):
    def evaluate(self, features: FloatArray) -> Evaluation: ...


@dataclass(frozen=True)
class SurrogateEvaluator:
    models: tuple[Pipeline, ...]
    risk: float = 1.0

    def __post_init__(self) -> None:
        if not self.models or not np.isfinite(self.risk) or self.risk < 0:
            raise ValueError("Se requieren modelos y un riesgo finito no negativo")

    @property
    def uncertainty_scope(self) -> str:
        return (
            "ensemble_disagreement_not_physical_guarantee"
            if len(self.models) > 1
            else "unavailable_single_model_point_prediction"
        )

    def evaluate(self, features: FloatArray) -> Evaluation:
        table = pd.DataFrame(features, columns=INPUT_COLUMNS)
        estimates = np.stack(
            [predict(model, table).to_numpy() for model in self.models]
        )
        deviation = estimates.std(axis=0, ddof=0)
        values = estimates.mean(axis=0) + self.risk * deviation * CONSERVATIVE_SIGNS
        return Evaluation(
            features.copy(),
            values,
            np.zeros(len(features)),
            "surrogate_estimated",
            len(features) * len(self.models),
        )


@dataclass(frozen=True)
class HistoricalEvaluator:
    features: FloatArray
    responses: FloatArray
    scales: tuple[float, ...]

    def evaluate(self, features: FloatArray) -> Evaluation:
        distances = design_distances(features, self.features, self.scales)
        distances[features[:, None, 0] != self.features[None, :, 0]] = np.inf
        indices = distances.argmin(axis=1)
        selected = distances[np.arange(len(features)), indices]
        if not np.isfinite(selected).all():
            raise UnsupportedProposal(
                "El archivo histórico carece de la topología propuesta"
            )
        return Evaluation(
            self.features[indices].copy(),
            self.responses[indices].copy(),
            selected,
            "historical_observed",
        )


@dataclass(frozen=True)
class SyntheticEvaluator:
    """Control matemático con objetivos en conflicto; no representa un solver de cerchas."""

    def evaluate(self, features: FloatArray) -> Evaluation:
        height = features[:, 2]
        values = np.column_stack(
            (1000 + 500 * height, 0.2 + 0.2 * height, 0.025 / (1 + height))
        )
        return Evaluation(
            features.copy(), values, np.zeros(len(features)), "synthetic_exact"
        )
