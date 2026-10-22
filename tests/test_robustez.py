"""Recuperación numérica, precisión de CPU y evidencia descriptiva verificables."""

import gc
import json
import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.procedencia import PhysicalConfirmation, load_confirmations
from truss_research.datos.procesados import load_design_split
from truss_research.evaluacion.registro import execution_log
from truss_research.evaluacion.robustez import coverage_tables, export_error_diagnostics
from truss_research.surrogates import convergencia, explicabilidad
from truss_research.surrogates.configuracion import (
    FloatArray,
    ModelParameters,
    PredictorConfig,
)
from truss_research.surrogates.convergencia import GprOptimizer, NumericalFitError
from truss_research.surrogates.experimentos import (
    managed_study,
    train_predictor,
    tune_predictor,
)


@pytest.mark.parametrize("mode", ["recover", "fail", "nonfinite"])
def test_gpr_recovery_is_bounded_and_rejects_invalid_result(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    calls: list[int] = []

    def result(
        fun: Callable[[FloatArray], tuple[float, FloatArray]],
        initial: FloatArray,
        **kwargs: bool | str | FloatArray | dict[str, float | int],
    ) -> OptimizeResult:
        calls.append(kwargs["options"]["maxls"])
        return OptimizeResult(
            x=np.array([0.5]),
            fun=float("nan") if mode == "nonfinite" else 1.0,
            jac=np.array([0.0]),
            nit=2,
            nfev=3,
            status=0 if len(calls) == 2 and mode == "recover" else 2,
            success=len(calls) == 2 and mode == "recover",
            message="terminación simulada",
        )

    monkeypatch.setattr(convergencia, "minimize", result)
    optimizer = GprOptimizer(ModelParameters())
    if mode == "recover":
        theta, score = optimizer(
            lambda theta, eval_gradient=True: (1.0, theta),
            np.array([0.0]),
            np.array([[-1.0, 1.0]]),
        )
        assert score == 1 and theta[0] == 0.5
        assert optimizer.attempts[-1]["success"] is True
    else:
        with pytest.raises(NumericalFitError) as error:
            optimizer(
                lambda theta, eval_gradient=True: (1.0, theta),
                np.array([0.0]),
                np.array([[-1.0, 1.0]]),
            )
        assert error.value.attempts == optimizer.attempts
    assert calls == ([50] if mode == "nonfinite" else [50, 100])


def test_gpr_optimizer_converges_on_known_bounded_objective() -> None:
    optimizer = GprOptimizer(ModelParameters())
    theta, score = optimizer(
        lambda point, eval_gradient=True: (float((point**2).sum()), 2 * point),
        np.array([0.7]),
        np.array([[-1.0, 1.0]]),
    )
    assert score < 1e-12 and abs(theta[0]) < 1e-6
    assert len(optimizer.attempts) == 1


def test_sqlite_owner_closes_after_exception_and_can_resume(
    design_data: Path, tmp_path: Path
) -> None:
    table = load_design_split(design_data).training
    config = PredictorConfig(
        design_data,
        tmp_path / "run",
        model="ridge",
        trials=2,
        study_path=tmp_path / "study.sqlite",
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ResourceWarning)
        with pytest.raises(RuntimeError, match="interrupción"):
            with managed_study(config, table) as study:
                study.optimize(lambda trial: float(trial.number), n_trials=1)
                raise RuntimeError("interrupción")
        with managed_study(config, table) as study:
            assert len(study.trials) == 1
            study.optimize(lambda trial: float(trial.number), n_trials=1)
        del study
        gc.collect()
    assert not [
        warning for warning in captured if issubclass(warning.category, ResourceWarning)
    ]
    config.study_path.rename(tmp_path / "closed.sqlite")


def test_rejected_gpr_trials_consume_budget(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.surrogates import experimentos

    config = PredictorConfig(
        design_data,
        tmp_path / "run",
        model="gpr",
        trials=2,
        study_path=tmp_path / "study.sqlite",
    )
    table = load_design_split(design_data).training

    def reject(
        config: PredictorConfig, table: pd.DataFrame
    ) -> tuple[pd.DataFrame, float]:
        raise NumericalFitError([{"attempt": 1, "success": False}])

    monkeypatch.setattr(experimentos, "cross_fitted_predictions", reject)
    with pytest.raises(ValueError, match="Todos los ensayos"):
        tune_predictor(config, table)
    with managed_study(config, table) as study:
        assert len(study.trials) == 2
        assert all(trial.state.name == "FAIL" for trial in study.trials)
        assert all(trial.user_attrs["convergence_failure"] for trial in study.trials)


def test_execution_log_preserves_failure_and_restores_handlers(tmp_path: Path) -> None:
    path = tmp_path / "attempt.log"
    previous = list(logging.getLogger().handlers)
    with pytest.raises(NumericalFitError):
        with execution_log(path, "predictor/gpr/42"):
            raise NumericalFitError([{"attempt": 1, "success": False}])
    state = json.loads(path.with_suffix(".execution.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed" and state["wall_seconds"] >= 0
    assert state["numerical_diagnostics"][0]["success"] is False
    assert "Traceback" in path.read_text(encoding="utf-8")
    assert logging.getLogger().handlers == previous


def test_tail_diagnostics_keep_designs_and_separate_seeds(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "model": ["ridge"] * 4,
            "seed": [42, 42, 43, 43],
            "iter_id": ["a", "b", "a", "b"],
            "web_config": [0, 3, 0, 3],
        }
    )
    for target in OUTPUT_COLUMNS:
        table[target] = 1.0
        table[f"pred_{target}"] = [1.0, 11.0, 1.0, 11.0]
    export_error_diagnostics(table, tmp_path)
    errors = pd.read_csv(tmp_path / "prediction_errors.csv")
    summary = pd.read_csv(tmp_path / "error_tail_summary.csv")
    assert len(errors) == 12 and len(summary) == 6
    assert summary.unique_designs.eq(2).all()
    assert summary.worst_squared_error_share.eq(1).all()
    assert summary.mae_without_worst_diagnostic.eq(0).all()
    assert summary.mae.eq(5).all()


def test_coverage_reports_absent_and_sparse_categories(design_data: Path) -> None:
    coverage, generations = coverage_tables(load_design_split(design_data))
    assert len(coverage) == 15 and coverage.unique_designs.ge(0).all()
    assert coverage.scarce_evidence.eq(coverage.unique_designs.lt(10)).all()
    assert "shared_across_partitions" in generations


@pytest.mark.parametrize(
    "values",
    [
        {"status": "confirmed"},
        {"status": "confirmed", "evidence_source": "Documento primario"},
        {
            "status": "confirmed",
            "evidence_source": "Documento primario",
            "confirmed_at": "20261017",
        },
    ],
)
def test_physical_confirmation_requires_source_and_canonical_date(
    values: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        PhysicalConfirmation(element="Cargas y apoyos", **values)


def test_physical_confirmation_accepts_evidence_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    confirmed = PhysicalConfirmation(
        "Cargas y apoyos", "confirmed", "Documento primario, sección 2", "2026-10-17"
    )
    assert confirmed.status == "confirmed"
    path = tmp_path / "physical.json"
    path.write_text(
        json.dumps(
            {"confirmations": [{"element": "material"}, {"element": "material"}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicadas"):
        load_confirmations(path)


@pytest.mark.parametrize("failure", ["exception", "nonfinite", "shape", "additivity"])
def test_shap_tree_failure_uses_valid_permutation(
    design_data: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    tmp_path = design_data.parent
    config = PredictorConfig(
        design_data,
        tmp_path / "predictor",
        model="random_forest",
        folds=2,
        parameters=ModelParameters(trees=2),
    )
    train_predictor(config)

    def broken_tree(
        model: "Pipeline", data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> tuple[FloatArray, FloatArray]:
        if failure == "exception":
            raise ValueError("fallo de árbol controlado")
        n = len(data[1])
        values = np.zeros((n, len(explicabilidad.INPUT_COLUMNS), len(OUTPUT_COLUMNS)))
        bases = np.zeros((n, len(OUTPUT_COLUMNS)))
        if failure == "nonfinite":
            values[:] = np.nan
        if failure == "shape":
            values = values[:, :1]
        return values, bases

    monkeypatch.setattr(explicabilidad, "tree_values", broken_tree)
    destination = tmp_path / "explanation"
    explicabilidad.explain_model(
        explicabilidad.ExplanationConfig(
            design_data,
            config.output_dir,
            destination,
        )
    )
    metadata = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert metadata["method"] == (
        "permutation_after_tree_additivity_failure"
        if failure == "additivity"
        else "permutation_after_tree_error"
    )
    assert metadata["fallback_reason"]
    assert len(metadata["final_additivity_error_by_target"]) == 3
    assert metadata["warnings"] == []


def test_shap_invalid_permutation_is_rejected(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PredictorConfig(design_data, tmp_path / "predictor", model="svr")
    train_predictor(config)

    def invalid_values(
        model: "Pipeline",
        data: tuple[pd.DataFrame, pd.DataFrame],
        config: explicabilidad.ExplanationConfig,
    ) -> tuple[FloatArray, FloatArray]:
        n = len(data[1])
        return (
            np.zeros((n, len(explicabilidad.INPUT_COLUMNS), len(OUTPUT_COLUMNS))),
            np.zeros((n, len(OUTPUT_COLUMNS))),
        )

    monkeypatch.setattr(explicabilidad, "permutation_values", invalid_values)
    destination = tmp_path / "explanation"
    with pytest.raises(ValueError, match="no reconstruyen"):
        explicabilidad.explain_model(
            explicabilidad.ExplanationConfig(
                design_data, config.output_dir, destination
            )
        )
    assert not destination.exists()
