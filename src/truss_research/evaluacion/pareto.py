"""Objetivos y métricas multiobjetivo con escalas fijadas en entrenamiento."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from truss_research.cerchas.schemas import StructuralLimits
from truss_research.datos.procesados import JsonValue

type FloatArray = npt.NDArray[np.float64]

OBJECTIVE_SIGNS = np.array([1.0, -1.0, 1.0])

OBJECTIVE_COUNT = len(OBJECTIVE_SIGNS)

REFERENCE_MARGIN = 0.1

RESPONSE_COUNT = 3


def validate_responses(values: FloatArray) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[1] != RESPONSE_COUNT
        or not np.isfinite(result).all()
    ):
        raise ValueError("Se requieren tres respuestas finitas por diseño")
    return result


def meets_thresholds(
    values: FloatArray, limits: StructuralLimits = StructuralLimits()
) -> npt.NDArray[np.bool_]:
    values = validate_responses(values)
    return (
        (values[:, 0] > 0)
        & (values[:, 1] >= 0)
        & (values[:, 1] <= limits.utilization_limit)
        & (values[:, 2] >= 0)
        & (
            True
            if limits.displacement_limit_m is None
            else values[:, 2] <= limits.displacement_limit_m
        )
    )


@dataclass(frozen=True)
class ObjectiveSpace:
    minimum: tuple[float, ...]
    ranges: tuple[float, ...]
    reference_physical: tuple[float, ...]
    limits: StructuralLimits = StructuralLimits()
    active_dimensions: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if self.active_dimensions not in ((0, 1, 2), (0, 2)):
            raise ValueError("Formulación multiobjetivo desconocida")
        values = (*self.minimum, *self.ranges, *self.reference_physical)
        if (
            len(self.minimum) != OBJECTIVE_COUNT
            or len(self.ranges) != OBJECTIVE_COUNT
            or len(self.reference_physical) != OBJECTIVE_COUNT
        ):
            raise ValueError(
                "El espacio debe tener tres objetivos: masa, utilización y desplazamiento"
            )
        if not np.isfinite(values).all() or any(value <= 0 for value in self.ranges):
            raise ValueError(
                "Escalas y referencia deben ser finitas y rangos positivos"
            )

    @classmethod
    def fit(
        cls, responses: FloatArray, limits: StructuralLimits = StructuralLimits()
    ) -> Self:
        values = validate_responses(responses)
        if len(values) == 0:
            raise ValueError("La normalización requiere entrenamiento no vacío")
        objectives = values * OBJECTIVE_SIGNS
        minimum, spans = objectives.min(axis=0), np.ptp(objectives, axis=0)
        mass_margin = spans[0] if spans[0] > 0 else values[:, 0].max()
        reference = (
            float(values[:, 0].max() + REFERENCE_MARGIN * mass_margin),
            0.0,
            limits.displacement_limit_m
            if limits.displacement_limit_m is not None
            else float(
                values[:, 2].max()
                + REFERENCE_MARGIN * (spans[2] or values[:, 2].max() or 1.0)
            ),
        )
        return cls(
            tuple(minimum), tuple(np.where(spans > 0, spans, 1.0)), reference, limits
        )

    def transform(self, responses: FloatArray) -> FloatArray:
        return (
            validate_responses(responses) * OBJECTIVE_SIGNS - self.minimum
        ) / self.ranges

    @property
    def reference(self) -> FloatArray:
        return (
            np.asarray(self.reference_physical) * OBJECTIVE_SIGNS - self.minimum
        ) / self.ranges

    def front(self, responses: FloatArray) -> FloatArray:
        values = validate_responses(responses)
        feasible = values[meets_thresholds(values, self.limits)]
        if not len(feasible):
            return np.empty((0, RESPONSE_COUNT))
        indices = NonDominatedSorting().do(
            self.transform(feasible)[:, self.active_dimensions],
            only_non_dominated_front=True,
        )
        return np.unique(feasible[indices], axis=0)

    def hypervolume(self, responses: FloatArray) -> float:
        front = self.front(responses)
        if not len(front):
            return 0.0
        objectives = self.transform(front)[:, self.active_dimensions]
        reference = self.reference[list(self.active_dimensions)]
        eligible = objectives[np.all(objectives <= reference, axis=1)]
        return float(HV(ref_point=reference).do(eligible)) if len(eligible) else 0.0


def binary_compliance(
    actual: npt.NDArray[np.bool_], estimated: npt.NDArray[np.bool_]
) -> dict[str, JsonValue]:
    if actual.shape != estimated.shape:
        raise ValueError("Las respuestas de cumplimiento deben estar alineadas")
    false_pass = int(np.sum(estimated & ~actual))
    predicted_pass = int(np.sum(estimated))
    return {
        "count": len(actual),
        "true_pass": int(np.sum(estimated & actual)),
        "false_pass": false_pass,
        "false_fail": int(np.sum(~estimated & actual)),
        "true_fail": int(np.sum(~estimated & ~actual)),
        "actual_fail_count": int(np.sum(~actual)),
        "fail_class_absent": bool(actual.all()),
        "false_pass_fraction": false_pass / predicted_pass if predicted_pass else None,
    }


def compliance_report(
    observed: FloatArray,
    predicted: FloatArray,
    limits: StructuralLimits = StructuralLimits(),
) -> dict[str, JsonValue]:
    observed, predicted = validate_responses(observed), validate_responses(predicted)
    if observed.shape != predicted.shape:
        raise ValueError("Las respuestas de cumplimiento deben estar alineadas")
    report = binary_compliance(
        meets_thresholds(observed, limits), meets_thresholds(predicted, limits)
    )
    invalid = (predicted[:, 0] <= 0) | (predicted[:, 1:] < 0).any(axis=1)
    report["physically_invalid_prediction_count"] = int(invalid.sum())
    report["utilization"] = binary_compliance(
        (observed[:, 1] >= 0) & (observed[:, 1] <= limits.utilization_limit),
        (predicted[:, 1] >= 0) & (predicted[:, 1] <= limits.utilization_limit),
    )
    report["displacement"] = binary_compliance(
        (observed[:, 2] >= 0)
        & (
            True
            if limits.displacement_limit_m is None
            else observed[:, 2] <= limits.displacement_limit_m
        ),
        (predicted[:, 2] >= 0)
        & (
            True
            if limits.displacement_limit_m is None
            else predicted[:, 2] <= limits.displacement_limit_m
        ),
    )
    return report
