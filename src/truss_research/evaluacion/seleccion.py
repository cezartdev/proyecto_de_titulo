"""Reportes retrospectivos; sus referencias completas no llegan a las políticas."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from pymoo.indicators.igd_plus import IGDPlus
from scipy.integrate import trapezoid

from truss_research.evaluacion.pareto import ObjectiveSpace, meets_thresholds

type FloatArray = npt.NDArray[np.float64]

QUALITY_TARGET = 0.9


@dataclass(frozen=True)
class EpisodeReport:
    responses: FloatArray
    selected_indices: tuple[int, ...]
    volumes: tuple[float, ...]
    space: ObjectiveSpace


def selection_metrics(report: EpisodeReport) -> dict[str, float | int | str | None]:
    reference = report.space.front(report.responses)
    observed = report.responses[list(report.selected_indices)]
    front = report.space.front(observed)
    reference_volume = report.space.hypervolume(reference)
    useful = reference_volume > 0
    fractions = (
        np.asarray(report.volumes) / reference_volume
        if useful
        else np.zeros(len(report.volumes))
    )
    hits = (
        np.flatnonzero(fractions >= QUALITY_TARGET)
        if useful
        else np.array([], dtype=int)
    )
    observed_points, reference_points = (
        set(map(tuple, front)),
        set(map(tuple, reference)),
    )
    relative_reason = (
        None if useful else "empty_reference_front_or_zero_reference_hypervolume"
    )
    return {
        "reference_front_size": len(reference),
        "observed_front_size": len(front),
        "queries": len(observed),
        "hypervolume": report.volumes[-1],
        "hypervolume_gap": max(0.0, reference_volume - report.volumes[-1]),
        "raw_auc": float(trapezoid(np.r_[0.0, report.volumes]) / len(report.volumes)),
        "reference_hypervolume": reference_volume,
        "relative_hypervolume": float(fractions[-1]) if useful else None,
        "normalized_auc": float(trapezoid(np.r_[0.0, fractions]) / len(fractions))
        if useful
        else None,
        "queries_to_target": len(set(report.selected_indices[: int(hits[0] + 1)]))
        if len(hits)
        else None,
        "attempts_to_target": int(hits[0] + 1) if len(hits) else None,
        "quality_target": QUALITY_TARGET,
        "relative_metric_reason": relative_reason,
        "pareto_recall": len(observed_points & reference_points) / len(reference_points)
        if reference_points
        else None,
        "igd_plus": float(
            IGDPlus(
                report.space.transform(reference)[:, report.space.active_dimensions]
            ).do(report.space.transform(front)[:, report.space.active_dimensions])
        )
        if len(reference) and len(front)
        else None,
        "observed_failures": int(
            np.sum(
                ~meets_thresholds(
                    report.responses[np.unique(report.selected_indices)],
                    report.space.limits,
                )
            )
        ),
        "attempted_failures": int(
            np.sum(~meets_thresholds(observed, report.space.limits))
        ),
        "outside_reference_count": int(
            np.sum(
                meets_thresholds(report.responses, report.space.limits)
                & np.any(
                    report.space.transform(report.responses)[
                        :, report.space.active_dimensions
                    ]
                    > report.space.reference[list(report.space.active_dimensions)],
                    axis=1,
                )
            )
        ),
    }


def paired_differences(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(
        index="episode_id", columns="strategy", values="normalized_auc"
    )
    if "random" not in pivot:
        raise ValueError("Falta la referencia aleatoria")
    return pivot.subtract(pivot["random"], axis=0).drop(columns="random").reset_index()
