"""Optimización GPR verificable, con una recuperación determinista y acotada."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline

from truss_research.datos.procesados import JsonValue
from truss_research.surrogates.configuracion import FloatArray, ModelParameters

logger = logging.getLogger(__name__)
BOUND_TOLERANCE = 1e-8


class GprObjective(Protocol):
    def __call__(
        self, theta: FloatArray, eval_gradient: bool = True
    ) -> tuple[float, FloatArray]: ...


class NumericalFitError(ValueError):
    def __init__(self, attempts: list[dict[str, JsonValue]]) -> None:
        self.attempts = attempts
        super().__init__(
            "GPR no convergió tras una recuperación limitada; ajuste rechazado"
        )


@dataclass
class GprOptimizer:
    parameters: ModelParameters
    attempts: list[dict[str, JsonValue]] = field(default_factory=list, init=False)

    def __call__(
        self, objective: GprObjective, initial_theta: FloatArray, bounds: FloatArray
    ) -> tuple[FloatArray, float]:
        p = self.parameters
        theta = np.asarray(initial_theta, dtype=float).copy()
        for number, lines in enumerate(
            (p.gpr_max_line_search, p.gpr_recovery_line_search), start=1
        ):
            result = minimize(
                lambda point: objective(point, eval_gradient=True),
                theta,
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={
                    "maxiter": p.gpr_max_iter,
                    "maxfun": p.gpr_max_fun,
                    "maxls": lines,
                    "ftol": p.gpr_ftol,
                    "gtol": p.gpr_gtol,
                },
            )
            finite = bool(
                np.isfinite(result.fun)
                and np.isfinite(result.x).all()
                and np.isfinite(result.jac).all()
            )
            inside = bool(
                np.all(result.x >= bounds[:, 0] - BOUND_TOLERANCE)
                and np.all(result.x <= bounds[:, 1] + BOUND_TOLERANCE)
            )
            successful = bool(result.success and finite and inside)
            record: dict[str, JsonValue] = {
                "attempt": number,
                "success": successful,
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "objective": float(result.fun) if finite else None,
                "gradient_max_abs": float(np.abs(result.jac).max()) if finite else None,
                "theta": result.x.tolist() if finite else None,
                "near_bound": bool(
                    np.isclose(result.x, bounds[:, 0], atol=BOUND_TOLERANCE).any()
                    or np.isclose(result.x, bounds[:, 1], atol=BOUND_TOLERANCE).any()
                ),
                "max_line_search": lines,
            }
            self.attempts.append(record)
            logger.info("Optimización GPR: %s", record)
            if successful:
                return np.asarray(result.x), float(result.fun)
            if not finite or not inside:
                break
            theta = np.asarray(result.x).copy()
        raise NumericalFitError(self.attempts)


def convergence_records(model: Pipeline) -> list[dict[str, JsonValue]]:
    regressor = model.named_steps["regressor"]
    if hasattr(regressor, "components_"):
        return [
            {**record, "target_index": target}
            for target, (_, component) in enumerate(regressor.components_)
            for record in convergence_records(component)
        ]
    estimators = getattr(regressor, "estimators_", [regressor])
    return [
        {
            "target_index": index,
            "optimizer": "lbfgsb_bounded_recovery_v1",
            "parameters": {
                key: value
                for key, value in asdict(estimator.optimizer.parameters).items()
                if key.startswith("gpr_")
            },
            "attempts": estimator.optimizer.attempts,
        }
        for index, estimator in enumerate(estimators)
        if isinstance(getattr(estimator, "optimizer", None), GprOptimizer)
    ]
