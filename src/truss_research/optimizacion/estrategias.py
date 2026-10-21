"""Referencias que operan solo sobre la observación pública del episodio."""

import numpy as np
import numpy.typing as npt
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.pareto import ObjectiveSpace, meets_thresholds
from truss_research.optimizacion.contratos import EpisodeObservation


def available_actions(observation: EpisodeObservation) -> npt.NDArray[np.int64]:
    if observation.terminal:
        raise ValueError("No hay acciones válidas en un episodio terminado")
    return np.flatnonzero(observation.selectable)


class RandomStrategy:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def choose(self, observation: EpisodeObservation) -> int:
        return int(self._rng.choice(available_actions(observation)))


def constraint_violation(
    predictions: npt.NDArray[np.float64], space: ObjectiveSpace
) -> npt.NDArray[np.float64]:
    utilization = np.maximum(0.0, predictions[:, 1] - space.limits.utilization_limit)
    if space.limits.displacement_limit_m is None:
        return utilization
    displacement = (
        np.maximum(0.0, predictions[:, 2] - space.limits.displacement_limit_m)
        / space.limits.displacement_limit_m
    )
    return utilization + displacement


class RankingStrategy:
    def __init__(self, observation: EpisodeObservation, space: ObjectiveSpace) -> None:
        self._order = pareto_ranking(observation.predictions, space)

    def best_among(self, candidates: npt.NDArray[np.int64]) -> int:
        allowed = {int(index) for index in candidates}
        return int(next(index for index in self._order if int(index) in allowed))

    def choose(self, observation: EpisodeObservation) -> int:
        return self.best_among(available_actions(observation))


def pareto_ranking(
    predictions: npt.NDArray[np.float64], space: ObjectiveSpace
) -> npt.NDArray[np.int64]:
    """Orden fijo por cumplimiento, violación, dominancia y diversidad."""
    objectives = space.transform(predictions)[:, space.active_dimensions]
    feasible = meets_thresholds(predictions, space.limits)
    ranks = np.zeros(len(objectives), dtype=int)
    crowding = np.zeros(len(objectives), dtype=float)
    for mask in (feasible, ~feasible):
        subset = np.flatnonzero(mask)
        if not len(subset):
            continue
        fronts = NonDominatedSorting().do(objectives[subset])
        for rank, local_indices in enumerate(fronts):
            indices = subset[local_indices]
            ranks[indices] = rank
            crowding[indices] = get_crowding_function("cd").do(objectives[indices])
    violation = constraint_violation(predictions, space)
    return np.lexsort(
        (np.arange(len(objectives)), -crowding, ranks, violation, ~feasible)
    )


class GreedyStrategy:
    def __init__(self, space: ObjectiveSpace) -> None:
        self._space = space

    def choose(self, observation: EpisodeObservation) -> int:
        actions = available_actions(observation)
        observed = observation.observed[~observation.available]
        current = self._space.hypervolume(observed)
        scores = np.array(
            [
                self._space.hypervolume(
                    np.vstack([observed, observation.predictions[index]])
                )
                - current
                for index in actions
            ],
            dtype=float,
        )
        if float(scores.max()) <= 0.0:
            return RankingStrategy(observation, self._space).best_among(actions)
        return int(actions[int(np.argmax(scores))])


class SafeGreedyStrategy(GreedyStrategy):
    def __init__(self, space: ObjectiveSpace, bounds: tuple[float, ...]) -> None:
        super().__init__(space)
        if len(bounds) != 3 or not np.isfinite(bounds).all() or min(bounds) < 0:
            raise ValueError("Cotas calibradas inválidas")
        self.bounds = np.asarray(bounds)
        self.diagnostics: dict[str, JsonValue] = {}

    def choose(self, observation: EpisodeObservation) -> int:
        actions = available_actions(observation)
        upper = observation.predictions[actions] + self.bounds
        conservative = (upper[:, 1] <= self._space.limits.utilization_limit) & (
            True
            if self._space.limits.displacement_limit_m is None
            else upper[:, 2] <= self._space.limits.displacement_limit_m
        )
        conservative &= meets_thresholds(
            observation.predictions[actions], self._space.limits
        )
        fallback = not conservative.any()
        eligible = actions if fallback else actions[conservative]
        observed = observation.observed[~observation.available]
        current = self._space.hypervolume(observed)
        scores = np.array(
            [
                self._space.hypervolume(
                    np.vstack((observed, observation.predictions[index]))
                )
                - current
                for index in eligible
            ],
            dtype=float,
        )
        self.diagnostics = {
            "safety_fallback": fallback,
            "conservative_candidates": int(conservative.sum()),
            "nominal_marginal_coverage": 0.90,
        }
        if float(scores.max()) <= 0.0:
            return RankingStrategy(observation, self._space).best_among(eligible)
        return int(eligible[int(np.argmax(scores))])
