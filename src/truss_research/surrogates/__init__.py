"""Entrenamiento reproducible de modelos sustitutos tabulares."""

from truss_research.surrogates.configuracion import (
    ModelParameters,
    PredictorConfig,
    PredictorResult,
)
from truss_research.surrogates.experimentos import load_predictor, train_predictor

__all__ = [
    "ModelParameters",
    "PredictorConfig",
    "PredictorResult",
    "train_predictor",
    "load_predictor",
]
