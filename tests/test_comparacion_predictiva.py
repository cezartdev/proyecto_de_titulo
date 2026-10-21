"""Controles sintéticos del estimador pareado y la dependencia por generaciones."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.evaluacion.comparacion_predictiva import (
    bootstrap_means,
    export_prediction_tables,
    paired_losses,
    predictive_contrasts,
)
from truss_research.optimizacion.contratos import AnalysisParameters
from truss_research.optimizacion.evaluadores import SurrogateEvaluator


@pytest.fixture
def predictions() -> pd.DataFrame:
    frames = []
    for model, offset in (("dummy", 2.0), ("ridge", 1.0), ("svr", 0.5)):
        for seed in (42, 43):
            table = pd.DataFrame(
                {
                    "iter_id": [f"d{i}" for i in range(8)],
                    "generation": np.arange(8) // 2,
                    "web_config": 1,
                    "model": model,
                    "seed": seed,
                    "split": "test",
                }
            )
            for name in OUTPUT_COLUMNS:
                table[name] = np.arange(8, dtype=float)
                table[f"pred_{name}"] = table[name] + offset
            frames.append(table)
    return pd.concat(frames, ignore_index=True)


def test_losses_average_seeds_without_multiplying_sample(
    predictions: pd.DataFrame,
) -> None:
    losses = paired_losses(predictions, np.ones(3))
    assert len(losses) == 24
    table = predictive_contrasts(losses, AnalysisParameters(bootstrap_repeats=40))
    result = table.loc[table.model.eq("svr") & table.reference_model.eq("dummy")]
    assert result["count"].eq(8).all()
    np.testing.assert_allclose(result.mean_difference, -1.5)
    np.testing.assert_allclose(result.ci_lower, -1.5)
    assert set(table.contrast) == {"primary_control"}


def test_blocks_preserve_correlated_errors() -> None:
    settings = AnalysisParameters(bootstrap_repeats=500)
    values = np.array([[0.0], [0.0], [10.0], [10.0]])
    blocked = bootstrap_means(values, np.array([0, 0, 1, 1]), settings)
    individual = bootstrap_means(values, np.arange(4), settings)
    assert set(blocked.ravel()) <= {0.0, 5.0, 10.0}
    assert blocked.std() > individual.std()
    np.testing.assert_array_equal(
        blocked, bootstrap_means(values, np.array([0, 0, 1, 1]), settings)
    )


def test_unequal_blocks_weight_designs() -> None:
    samples = bootstrap_means(
        np.array([[0.0], [0.0], [0.0], [12.0]]),
        np.array([0, 0, 0, 1]),
        AnalysisParameters(bootstrap_repeats=100),
    )
    assert set(samples.ravel()) == {0.0, 3.0, 12.0}


def test_incomplete_or_conflicting_predictions_fail(predictions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="mismos diseños"):
        paired_losses(predictions.iloc[1:], np.ones(3))
    conflicting = predictions.copy()
    conflicting.loc[0, "mass_kg"] += 1
    with pytest.raises(ValueError, match="observaciones"):
        paired_losses(conflicting, np.ones(3))
    with pytest.raises(ValueError, match="duplicadas"):
        paired_losses(pd.concat([predictions, predictions.iloc[[0]]]), np.ones(3))


def test_seed_dispersion_does_not_cancel_errors(predictions: pd.DataFrame) -> None:
    table = predictions.copy()
    for name in OUTPUT_COLUMNS:
        table.loc[table.seed.eq(43), f"pred_{name}"] = (
            2 * table[name] - table[f"pred_{name}"]
        )
    losses = paired_losses(table, np.ones(3))
    assert losses.loc[losses.model.eq("dummy"), "mass_kg"].eq(2).all()


def test_single_block_is_not_a_confidence_interval(predictions: pd.DataFrame) -> None:
    losses = paired_losses(predictions.assign(generation=0), np.ones(3))
    settings = AnalysisParameters(bootstrap_repeats=20)
    result = predictive_contrasts(losses, settings)
    blocked = result.loc[result.resampling.eq("generation_block")]
    assert blocked.ci_lower.isna().all()
    assert blocked.interval_status.eq("insufficient_blocks").all()
    with pytest.raises(ValueError, match="generaciones"):
        replace(settings, predictive_block_generations=0)


def test_export_contains_physical_metrics_and_conditional_protocol(
    predictions: pd.DataFrame, tmp_path: Path
) -> None:
    export_prediction_tables(
        predictions, (np.ones(3), AnalysisParameters(bootstrap_repeats=20)), tmp_path
    )
    metrics = pd.read_csv(tmp_path / "predictive_seed_metrics.csv")
    assert {"bias", "p95_absolute_error", "normalized_mae", "count"} <= set(metrics)
    assert metrics["count"].eq(8).all()
    summary = pd.read_csv(tmp_path / "predictive_model_summary.csv")
    assert summary.adjustment_seeds.eq(2).all()
    assert summary.loc[summary.model.eq("dummy"), "mean"].iloc[0] == 2.0
    assert (tmp_path / "predictive_paired_errors.csv").is_file()
    assert (tmp_path / "predictive_protocol.json").is_file()


def test_single_model_reports_absent_disagreement() -> None:
    assert (
        SurrogateEvaluator((Pipeline([]),)).uncertainty_scope
        == "unavailable_single_model_point_prediction"
    )
