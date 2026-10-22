"""Invariantes científicas del protocolo con calibración y adaptación por episodio."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.artefactos import dataset_identity
from truss_research.datos.preparacion import prepare_dataset
from truss_research.datos.procesados import load_design_split
from truss_research.datos.schemas import RAW_ID_COLUMN, DatasetSpec, PreparationConfig
from truss_research.optimizacion.entorno import area_reward, marginal_reward
from truss_research.surrogates.calibracion import SplitConformal
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import (
    load_predictor,
    read_run,
    train_predictor,
)


@pytest.fixture
def calibrated_data(raw_table: pd.DataFrame, tmp_path: Path) -> Path:
    table = pd.concat([raw_table.iloc[[0]]] * 500, ignore_index=True)
    table[RAW_ID_COLUMN] = [f"{{{i // 5};{i % 5}}}" for i in table.index]
    table["Web Config"] = (table.index // 5) % 5
    table["L-R Height"] = 0.2 + table.index * 0.002
    table["Mass"] = 1000.0 + table.index * 10.0
    table["Average Util"] = 0.2 + table.index * 0.001
    table["Z Disp"] = 0.01 + table.index * 0.0001
    source = tmp_path / "source.xlsx"
    table.to_excel(source, sheet_name="TOTAL DATA", index=False)
    destination = tmp_path / "calibrated"
    prepare_dataset(
        PreparationConfig(
            source, destination, dataset=DatasetSpec(100, 5), calibration_fraction=0.2
        )
    )
    return destination


def test_calibration_is_disjoint_and_signed(calibrated_data: Path) -> None:
    split = load_design_split(calibrated_data)
    assert len(split.training) == 320
    assert len(split.calibration) == 80
    sets = [set(table.iter_id) for table in split.as_mapping().values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 500
    before = dataset_identity(calibrated_data)
    path = calibrated_data / "calibration.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert dataset_identity(calibrated_data) != before


def test_conformal_quantile_and_insufficient_calibration() -> None:
    residuals = np.column_stack(
        (np.arange(20), np.arange(20) * 0.1, np.arange(20) * 0.001)
    )
    conformal = SplitConformal(residuals)
    np.testing.assert_allclose(conformal.widths(0.90), residuals[18])
    np.testing.assert_allclose(conformal.widths(0.95), residuals[19])
    with pytest.raises(ValueError, match="insuficiente"):
        SplitConformal(residuals[:10]).widths(0.95)


def test_plain_oof_uses_outer_folds_instead_of_final_search_folds(
    calibrated_data: Path, tmp_path: Path
) -> None:
    result = train_predictor(
        PredictorConfig(
            calibrated_data,
            tmp_path / "plain",
            model="dummy",
            folds=2,
            final_tuning_folds=3,
        )
    )
    assert len(read_run(result.output_dir)["fold_model_signatures"]) == 2
    assert pd.read_csv(result.output_dir / "training_oof.csv")["fold"].nunique() == 2


@pytest.mark.parametrize("budget", [1, 2, 8, 24])
def test_marginal_reward_equals_auc_including_initial_query(budget: int) -> None:
    volumes = np.r_[
        0.0, np.cumsum(np.random.default_rng(42).choice([0.0, 0.1, 0.2], budget))
    ]
    trapezoidal = sum(
        area_reward(volumes[t - 1], volumes[t], (budget, 2.0))
        for t in range(1, budget + 1)
    )
    marginal = sum(
        marginal_reward(volumes[t - 1], volumes[t], (t, budget, 2.0))
        for t in range(1, budget + 1)
    )
    assert marginal == pytest.approx(trapezoidal)
    assert marginal_reward(0.5, 0.5, (1, budget, 2.0)) == 0
    assert marginal_reward(0.0, 0.5, (1, budget, 0.0)) == 0


def test_targetwise_nested_excludes_calibration_and_outer_fold(
    calibrated_data: Path, tmp_path: Path
) -> None:
    result = train_predictor(
        PredictorConfig(
            calibrated_data,
            tmp_path / "targetwise",
            model="targetwise",
            folds=2,
            nested_folds=2,
            final_tuning_folds=2,
            trials=1,
            candidate_models=("dummy", "ridge"),
        )
    )
    metadata = read_run(result.output_dir)
    model = load_predictor(result.output_dir)
    split = load_design_split(calibrated_data)
    assert set(metadata["fit_ids"]) == set(split.training.iter_id)
    assert set(metadata["fit_ids"]).isdisjoint(metadata["calibration_ids"])
    assert len(metadata["targetwise_selection"]) == 3
    assert metadata["trial_count"] >= 1
    for _, component in model.named_steps["regressor"].components_:
        assert (
            np.asarray(
                component.predict(split.validation.loc[:, list(INPUT_COLUMNS)])
            ).ndim
            == 1
        )
    from truss_research.datos.artefactos import read_artifact

    for fold in read_artifact(result.output_dir / "nested_folds.json")["folds"]:
        assert set(fold["fit_ids"]).isdisjoint(fold["held_ids"])
        assert set(fold["fit_ids"]).isdisjoint(metadata["calibration_ids"])
    assert (result.output_dir / "prediction_intervals.csv").is_file()
