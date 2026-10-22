"""Referencias que operan solo sobre la observación pública del episodio."""

import numpy as np
import numpy.typing as npt
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from truss_research.evaluacion.pareto import ObjectiveSpace, meets_thresholds


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
