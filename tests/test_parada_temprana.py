"""Parada real de CatBoost, reajuste y aislamiento de los pliegues externos."""

from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import load_design_split
from truss_research.optimizacion.protocolo import load_protocol
from truss_research.surrogates.configuracion import (
    ModelName,
    ModelParameters,
    PredictorConfig,
)
from truss_research.surrogates.especializacion import scalar_predictor, scalar_search
from truss_research.surrogates.experimentos import managed_study, tune_predictor
from truss_research.surrogates.modelos import build_predictor
from truss_research.surrogates.parada_temprana import BoostFit, stopping_records
from truss_research.surrogates.pliegues import predictor_folds
from truss_research.surrogates.poda import PruningParameters
from truss_research.surrogates.validacion import fit_outer_fold


@pytest.fixture(params=["catboost", "xgboost", "lightgbm"])
def stopping_config(
    design_data: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> PredictorConfig:
    return PredictorConfig(
        data_dir=design_data,
        output_dir=tmp_path / "model",
        model=cast(ModelName, request.param),
        folds=2,
        trials=1,
        study_path=tmp_path / "search.sqlite",
        parameters=ModelParameters(trees=30, depth=2, early_stopping_rounds=3),
        pruning=PruningParameters(enabled=True),
    )


@pytest.fixture
def fixed_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "truss_research.surrogates.experimentos.suggested_parameters",
        lambda trial, config: config.parameters,
    )


def test_real_stopping_and_full_data_refit(stopping_config: PredictorConfig) -> None:
    table = load_design_split(stopping_config.data_dir).training
    x = table.loc[:, list(INPUT_COLUMNS)]
    y = table.loc[:, list(OUTPUT_COLUMNS)].to_numpy()
    model = build_predictor(stopping_config)
    model.named_steps["regressor"].fit_evaluated(
        BoostFit(x, y, (x, 2 * y.mean(axis=0) - y))
    )
    records = stopping_records(model)
    assert len(records) == 3
    assert all(record["executed_trees"] <= 30 for record in records)
    assert any(record["early_stopped"] for record in records)
    assert all(
        record["retained_trees"] <= record["executed_trees"] for record in records
    )
    assert any(record["stop_reason"] == "patience" for record in records)
    counts = tuple(record["retained_trees"] for record in records)
    final = build_predictor(replace(stopping_config, refit_trees=counts))
    final.fit(x, y)
    assert [record["executed_trees"] for record in stopping_records(final)] == list(
        counts
    )
    assert all(not record["early_stopped"] for record in stopping_records(final))
    assert np.isfinite(final.predict(x)).all()
    fixed = build_predictor(stopping_config)
    fixed.fit(x, y)
    assert all(record["maximum_trees"] == 30 for record in stopping_records(fixed))
    assert all(not record["early_stopped"] for record in stopping_records(fixed))


def test_winning_trial_medians_and_resume(
    stopping_config: PredictorConfig, fixed_parameters: None
) -> None:
    table = load_design_split(stopping_config.data_dir).training
    selected, trials = tune_predictor(stopping_config, table)
    assert trials.state.tolist() == ["COMPLETE"]
    with managed_study(stopping_config, table) as study:
        records = study.best_trial.user_attrs["early_stopping"]
    expected = tuple(
        int(median(fold[i]["retained_trees"] for fold in records)) for i in range(3)
    )
    assert selected.refit_trees == expected
    resumed, trials = tune_predictor(stopping_config, table)
    assert resumed.refit_trees == expected
    assert len(trials) == 1
    with pytest.raises(ValueError, match="otra configuración"):
        tune_predictor(
            replace(
                stopping_config,
                parameters=replace(stopping_config.parameters, early_stopping_rounds=0),
            ),
            table,
        )


def test_scalar_search_and_refit(
    stopping_config: PredictorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "truss_research.surrogates.experimentos.suggested_parameters",
        lambda trial, config: config.parameters,
    )
    table = load_design_split(stopping_config.data_dir).training
    selected, trials = scalar_search(stopping_config, table, 1)
    assert trials.state.tolist() == ["COMPLETE"]
    assert len(selected.refit_trees) == 1
    model = scalar_predictor(selected)
    model.fit(table.loc[:, list(INPUT_COLUMNS)], table[OUTPUT_COLUMNS[1]].to_numpy())
    assert stopping_records(model)[0]["executed_trees"] == selected.refit_trees[0]


def test_outer_evaluation_never_controls_stopping(
    stopping_config: PredictorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.surrogates import experimentos

    table = load_design_split(stopping_config.data_dir).training
    config = replace(stopping_config, nested_folds=2)
    fitting, held = predictor_folds(config, table)[0]
    allowed = set(table.iloc[fitting].iter_id)
    forbidden = set(table.iloc[held].iter_id)
    observed = []
    original = experimentos.fit_search_fold

    def inspect(
        model: Pipeline,
        candidate: PredictorConfig,
        tables: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        train, evaluation = tables
        assert set(train.iter_id) <= allowed
        assert set(evaluation.iter_id) <= allowed
        assert not set(evaluation.iter_id) & forbidden
        assert not set(train.iter_id) & set(evaluation.iter_id)
        observed.append(len(evaluation))
        original(model, candidate, tables)

    monkeypatch.setattr(experimentos, "fit_search_fold", inspect)
    monkeypatch.setattr(
        experimentos, "suggested_parameters", lambda trial, config: config.parameters
    )
    result = fit_outer_fold(config, table, 0)
    assert len(observed) == 2
    assert result.provenance["refit_trees"]
    assert all(
        not record["early_stopped"] for record in result.provenance["early_stopping"]
    )


def test_protocol_and_reference(tmp_path: Path) -> None:
    locations = (tmp_path / "study", tmp_path / "data")
    active = load_protocol(Path("config/study.json"), locations)
    from truss_research.evaluacion.entrenamiento_individual import (
        TrainingRequest,
        training_config,
    )

    reference = training_config(
        TrainingRequest("catboost", destination=tmp_path, early_stopping_rounds=0)
    )
    assert active.parameters.early_stopping_rounds == 30
    assert reference.parameters.early_stopping_rounds == 0
    assert reference.pruning == active.pruning


def test_parallel_preparation_preserves_selected_trees(
    stopping_config: PredictorConfig, fixed_parameters: None
) -> None:
    from truss_research.surrogates.preparacion import (
        assemble_preparation,
        load_prepared_predictor,
        prepare_outer_fold,
        prepare_search,
    )

    config = replace(stopping_config, nested_folds=2)
    root = config.output_dir.parent
    search = root / "prepared_search"
    prepare_search(config, search)
    folds = (root / "fold_0", root / "fold_1")
    for number, destination in enumerate(folds):
        prepare_outer_fold(config, destination, number)
    destination = root / "assembled"
    assemble_preparation(config, (search, folds), destination)
    prepared = load_prepared_predictor(replace(config, preparation_dir=destination))
    selected, _ = tune_predictor(
        replace(config, nested_folds=None), load_design_split(config.data_dir).training
    )
    assert prepared.refit_trees == selected.refit_trees
    assert all(fold["refit_trees"] for fold in prepared.provenance)


def test_model_export_preserves_refit_and_predictions(
    stopping_config: PredictorConfig, fixed_parameters: None
) -> None:
    from truss_research.surrogates.experimentos import (
        config_from_run,
        fit_predictor,
        load_predictor,
        read_run,
    )

    config = replace(stopping_config, nested_folds=2)
    result = fit_predictor(config)
    stored = config_from_run(config.output_dir)
    records = read_run(config.output_dir)["early_stopping"]
    assert tuple(record["executed_trees"] for record in records) == stored.refit_trees
    assert all(not record["early_stopped"] for record in records)
    restored = load_predictor(config.output_dir)
    features = load_design_split(config.data_dir).validation.loc[:, list(INPUT_COLUMNS)]
    np.testing.assert_allclose(
        restored.predict(features), result.model.predict(features)
    )


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_patience(value: int) -> None:
    with pytest.raises(ValueError, match="paciencia"):
        ModelParameters(early_stopping_rounds=value)


def test_mlp_stops_and_preserves_best_epoch() -> None:
    from truss_research.surrogates.mlp import TorchRegressor

    generator = np.random.default_rng(42)
    features = generator.normal(size=(20, 7))
    responses = generator.normal(size=(20, 3))
    parameters = ModelParameters(epochs=20, patience=2, learning_rate=1e-30)
    model = TorchRegressor(parameters)
    model.fit(features, responses)
    record = model.early_stopping_[0]
    assert record["executed_epochs"] == 3
    assert record["retained_epoch"] == 1
    assert record["early_stopped"]
    assert record["stop_reason"] == "patience"
    assert record["evaluation"] == "internal_training_reserve"
    reference = TorchRegressor(replace(parameters, epochs=1))
    reference.fit(features, responses)
    np.testing.assert_allclose(model.predict(features), reference.predict(features))
