"""Adquisición secuencial por ganancia de hipervolumen y desacuerdo del ensemble."""

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.distancias import design_distances
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.contratos import EpisodeObservation, StrategyName
from truss_research.optimizacion.estrategias import RankingStrategy, available_actions
from truss_research.surrogates.configuracion import FloatArray

MIN_ENSEMBLE_MEMBERS = 2
SAMPLE_DDOF = 1
DEFAULT_EXPLORATION_BETA = 1.0
ENSEMBLE_STRATEGIES: tuple[StrategyName, ...] = ("ensemble_mean", "ensemble_ucb")
ONLINE_STRATEGIES: tuple[StrategyName, ...] = (
    "online_ensemble_mean",
    "online_ensemble_ucb",
)
RESIDUAL_NEIGHBORS = 5
RESIDUAL_DISTANCE_POWER = 2
RESIDUAL_LENGTH_SCALE = 0.25


@dataclass(frozen=True)
class AdaptiveSettings:
    beta: float = DEFAULT_EXPLORATION_BETA
    online_residual: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.online_residual) is not bool
            or isinstance(self.beta, bool)
            or not isinstance(self.beta, (int, float))
            or not isfinite(self.beta)
            or self.beta < 0
        ):
            raise ValueError("Parámetros adaptativos inválidos")


def ensemble_disagreement(values: FloatArray) -> FloatArray:
    """Evita dispersión artificial por redondeo cuando los miembros son idénticos."""
    if (
        values.ndim < 2
        or len(values) < MIN_ENSEMBLE_MEMBERS
        or not np.isfinite(values).all()
    ):
        raise ValueError("El desacuerdo requiere al menos dos miembros finitos")
    spread = values.std(axis=0, ddof=SAMPLE_DDOF)
    return np.where(np.ptp(values, axis=0) == 0, 0.0, spread)


@dataclass(frozen=True)
class AcquisitionContext:
    space: ObjectiveSpace
    member_predictions: FloatArray
    beta: float = DEFAULT_EXPLORATION_BETA
    feature_scales: tuple[float, ...] = ()
    online_residual: bool = False

    def __post_init__(self) -> None:
        values = np.array(self.member_predictions, dtype=float, copy=True)
        if type(self.online_residual) is not bool:
            raise ValueError("La corrección residual debe ser booleana")
        if self.online_residual and (
            len(self.feature_scales) != len(INPUT_COLUMNS)
            or not np.isfinite(self.feature_scales).all()
            or min(self.feature_scales) < 0
        ):
            raise ValueError("La corrección residual requiere escalas de entrenamiento")
        if (
            values.ndim != 3
            or values.shape[0] < MIN_ENSEMBLE_MEMBERS
            or values.shape[1] < 1
            or values.shape[2] != len(OUTPUT_COLUMNS)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Se requieren predicciones finitas: miembros × candidatos × respuestas"
            )
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, (int, float))
            or not isfinite(self.beta)
            or self.beta < 0
        ):
            raise ValueError("Beta debe ser finito y no negativo")
        values.setflags(write=False)
        object.__setattr__(self, "member_predictions", values)


class EnsembleStrategy:
    def __init__(self, context: AcquisitionContext) -> None:
        self._context = context
        self.diagnostics: dict[str, JsonValue] = {}

    def predictions_for(self, observation: EpisodeObservation) -> FloatArray:
        original = self._context.member_predictions
        if not self._context.online_residual:
            return original
        known = np.flatnonzero(~observation.available)
        if not len(known):
            return original
        distances = design_distances(
            observation.features,
            observation.features[known],
            self._context.feature_scales,
        )
        residuals = observation.observed[known][None, :, :] - original[:, known, :]
        corrections = np.empty_like(original)
        for index, row in enumerate(distances):
            neighbors = np.argsort(row, kind="stable")[:RESIDUAL_NEIGHBORS]
            exact = neighbors[row[neighbors] == 0]
            if len(exact):
                corrections[:, index, :] = residuals[:, exact, :].mean(axis=1)
                continue
            min_distance = float(row[neighbors].min())
            weights = (min_distance / row[neighbors]) ** RESIDUAL_DISTANCE_POWER
            decay = 1.0 / (
                1.0 + (min_distance / RESIDUAL_LENGTH_SCALE) ** RESIDUAL_DISTANCE_POWER
            )
            corrections[:, index, :] = decay * np.average(
                residuals[:, neighbors, :], weights=weights, axis=1
            )
        return original + corrections

    def _gains_from(
        self, observation: EpisodeObservation, predictions: FloatArray
    ) -> FloatArray:
        actions = available_actions(observation)
        if predictions.shape[1:] != observation.predictions.shape:
            raise ValueError("El ensemble y la observación no están alineados")
        observed = observation.observed[~observation.available]
        current = self._context.space.hypervolume(observed)
        return np.array(
            [
                [
                    max(
                        0.0,
                        self._context.space.hypervolume(
                            np.vstack((observed, member[index]))
                        )
                        - current,
                    )
                    for index in actions
                ]
                for member in predictions
            ],
            dtype=float,
        )

    def gains(self, observation: EpisodeObservation) -> FloatArray:
        return self._gains_from(observation, self.predictions_for(observation))

    def choose(self, observation: EpisodeObservation) -> int:
        actions = available_actions(observation)
        predictions = self.predictions_for(observation)
        gains = self._gains_from(observation, predictions)
        mean = gains.mean(axis=0)
        disagreement = ensemble_disagreement(gains)
        scores = mean + self._context.beta * disagreement
        if float(scores.max()) <= 0.0:
            mean_observation = replace(
                observation, predictions=predictions.mean(axis=0)
            )
            action = RankingStrategy(mean_observation, self._context.space).best_among(
                actions
            )
            position = int(np.flatnonzero(actions == action)[0])
        else:
            position = int(np.argmax(scores))
            action = int(actions[position])
        response_std = ensemble_disagreement(
            self._context.member_predictions[:, action]
        )
        self.diagnostics = {
            "acquisition_mean_gain": float(mean[position]),
            "acquisition_gain_std": float(disagreement[position]),
            "acquisition_beta": self._context.beta,
            "acquisition_score": float(scores[position]),
            "ensemble_members": len(gains),
            **{
                f"prediction_std_{name}": float(value)
                for name, value in zip(OUTPUT_COLUMNS, response_std, strict=True)
            },
        }
        original = self._context.member_predictions[:, action].mean(axis=0)
        corrected = predictions[:, action].mean(axis=0)
        self.diagnostics.update(
            {
                f"prediction_before_{name}": float(original[i])
                for i, name in enumerate(OUTPUT_COLUMNS)
            }
        )
        self.diagnostics.update(
            {
                f"prediction_after_{name}": float(corrected[i])
                for i, name in enumerate(OUTPUT_COLUMNS)
            }
        )
        self.diagnostics["online_residual"] = self._context.online_residual
        return action
