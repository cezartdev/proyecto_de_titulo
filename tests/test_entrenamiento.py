"""Pruebas de separación de conjuntos, artefactos y comparación reproducible."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import load_prepared_split
from truss_research.datos.schemas import DatasetError
from truss_research.evaluacion import load_comparison, regression_metrics
from truss_research.surrogates import ModelParameters, PredictorConfig, train_predictor


def test_dummy_uses_only_training_and_persists_pipeline(
    design_data: Path, tmp_path: Path
) -> None:
    split = load_prepared_split(design_data)
    result = train_predictor(
        PredictorConfig(design_data, tmp_path / "dummy", model="dummy", folds=2)
    )
    expected = split.training.loc[:, list(OUTPUT_COLUMNS)].mean().to_numpy()
    predictions = result.predictions.loc[:, [f"pred_{name}" for name in OUTPUT_COLUMNS]]
    np.testing.assert_allclose(
        predictions.to_numpy(), np.tile(expected, (len(predictions), 1))
    )
    saved = joblib.load(result.output_dir / "model.joblib")
    encoder = saved.named_steps["preprocessing"].named_transformers_["web_config"]
    assert set(encoder.categories_[0]) == set(split.training["web_config"])
    np.testing.assert_allclose(
        saved.predict(split.validation.loc[:, list(INPUT_COLUMNS)]),
        predictions.to_numpy(),
    )
    metadata = json.loads((result.output_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["evaluation_split"] == "validation" and metadata["fit_rows"] == len(
        split.training
    )
    assert set(result.predictions["iter_id"]) == set(split.validation["iter_id"])
    assert set(result.predictions["iter_id"]).isdisjoint(split.testing["iter_id"])
    with pytest.raises(ValueError, match="ya existe"):
        train_predictor(
            PredictorConfig(design_data, result.output_dir, model="dummy", folds=2)
        )


def test_forest_is_reproducible_and_comparison_does_not_train(
    design_data: Path, tmp_path: Path
) -> None:
    config = dict(data_dir=design_data, folds=2, parameters=ModelParameters(trees=3))
    first = train_predictor(PredictorConfig(output_dir=tmp_path / "first", **config))
    second = train_predictor(PredictorConfig(output_dir=tmp_path / "second", **config))
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    comparison = load_comparison([first.output_dir, second.output_dir])
    assert len(comparison) == 6 and set(comparison["run"]) == {"first", "second"}


def test_overlapping_designs_and_corrupt_ids_fail(design_data: Path) -> None:
    train = pd.read_csv(design_data / "train.csv")
    train.iloc[:3].to_csv(design_data / "val.csv", index=False)
    with pytest.raises(DatasetError, match="tabla completa"):
        load_prepared_split(design_data)
    train.loc[0, "generation"] = 999
    train.to_csv(design_data / "train.csv", index=False)
    with pytest.raises(DatasetError, match="inconsistentes"):
        load_prepared_split(design_data)


def test_metrics_report_undefined_r2_for_constant_target() -> None:
    observed = pd.DataFrame([[1.0, 0.5, 0.1], [1.0, 0.5, 0.1]], columns=OUTPUT_COLUMNS)
    metrics = regression_metrics(observed, observed.copy())
    assert metrics["r2"].isna().all()
    assert metrics[["mae", "rmse"]].eq(0).all().all()
    with pytest.raises(ValueError, match="índices alineados"):
        regression_metrics(observed, observed.set_axis([2, 3]))


def test_loader_rejects_removed_design(design_data: Path) -> None:
    table = pd.read_csv(design_data / "train.csv")
    table = table.iloc[1:]
    table.to_csv(design_data / "train.csv", index=False)
    with pytest.raises(DatasetError, match="tabla completa"):
        load_prepared_split(design_data)


def test_equivalent_numeric_csv_types_preserve_values(design_data: Path) -> None:
    original = load_prepared_split(design_data).training
    table = pd.read_csv(design_data / "train.csv", float_precision="round_trip")
    table["mass_kg"] = table.mass_kg.astype(float)
    table.to_csv(design_data / "train.csv", index=False)
    loaded = load_prepared_split(design_data).training
    np.testing.assert_array_equal(
        loaded.mass_kg.to_numpy(), original.mass_kg.to_numpy()
    )


def test_comparison_rejects_different_data(design_data: Path, tmp_path: Path) -> None:
    train_predictor(
        PredictorConfig(design_data, tmp_path / "original", model="dummy", folds=2)
    )
    table = pd.read_csv(design_data / "train.csv", float_precision="round_trip")
    table["mass_kg"] += 1.0
    table.to_csv(design_data / "train.csv", index=False)
    full = pd.read_csv(design_data / "full_clean.csv", float_precision="round_trip")
    full.loc[full.iter_id.isin(table.iter_id), "mass_kg"] += 1.0
    full.to_csv(design_data / "full_clean.csv", index=False)
    train_predictor(
        PredictorConfig(design_data, tmp_path / "modified", model="dummy", folds=2)
    )
    with pytest.raises(ValueError, match="particiones diferentes"):
        load_comparison([tmp_path / "original", tmp_path / "modified"])


def test_invalid_configuration_and_missing_data_leave_no_outputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="enteros deben ser positivos"):
        ModelParameters(trees=0)
    destination = tmp_path / "run"
    with pytest.raises(OSError):
        train_predictor(PredictorConfig(tmp_path / "missing", destination))
    assert not destination.exists()
