"""Poda real sobre ajustes mínimos y conservación de ensayos e identidad."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procesados import load_design_split
from truss_research.optimizacion.protocolo import load_protocol
from truss_research.surrogates.configuracion import (
    BASE_MODEL_NAMES,
    ModelName,
    ModelParameters,
    PredictorConfig,
)
from truss_research.surrogates.experimentos import managed_study, tune_predictor
from truss_research.surrogates.poda import (
    FoldProgress,
    PruningParameters,
    create_pruner,
    report_fold,
)


@pytest.fixture
def pruning_config(design_data: Path, tmp_path: Path) -> PredictorConfig:
    return PredictorConfig(
        design_data,
        tmp_path / "run",
        model="ridge",
        folds=3,
        trials=7,
        study_path=tmp_path / "resume.sqlite",
        pruning=PruningParameters(enabled=True),
    )


@pytest.fixture
def predictable_trials(monkeypatch: pytest.MonkeyPatch) -> list[PredictorConfig]:
    fitted: list[PredictorConfig] = []

    def parameters(trial: optuna.Trial, config: PredictorConfig) -> ModelParameters:
        return replace(config.parameters, regularization=1 if trial.number < 5 else 1e9)

    def build(config: PredictorConfig) -> Pipeline:
        fitted.append(config)
        constant = np.full(3, config.parameters.regularization)
        return Pipeline(
            [("regressor", DummyRegressor(strategy="constant", constant=constant))]
        )

    monkeypatch.setattr(
        "truss_research.surrogates.experimentos.suggested_parameters", parameters
    )
    monkeypatch.setattr("truss_research.surrogates.experimentos.build_predictor", build)
    return fitted


@pytest.fixture(
    params=[name for name in BASE_MODEL_NAMES if name not in ("dummy", "tabpfn")]
)
def pruning_family(request: pytest.FixtureRequest) -> ModelName:
    return cast(ModelName, request.param)


def test_pruning_avoids_last_fold_and_selects_only_complete(
    design_data: Path, tmp_path: Path, predictable_trials: list[PredictorConfig]
) -> None:
    config = PredictorConfig(
        design_data,
        tmp_path / "run",
        model="ridge",
        folds=3,
        trials=6,
        study_path=tmp_path / "search.sqlite",
        pruning=PruningParameters(enabled=True),
    )
    table = load_design_split(design_data).training
    selected, trials = tune_predictor(config, table)
    assert trials.state.tolist() == ["COMPLETE"] * 5 + ["PRUNED"]
    assert len(predictable_trials) == 17
    assert selected.parameters.regularization == 1
    summary = read_artifact(tmp_path / "search.pruning.json")
    assert summary["states"]["PRUNED"] == 1
    assert summary["reported_completed_folds"] == 17
    assert summary["skipped_folds"] == 1
    with managed_study(config, table) as study:
        assert study.trials[-1].user_attrs["completed_folds"] == 2
        assert 1 not in study.trials[-1].intermediate_values
        assert study.best_trial.state == optuna.trial.TrialState.COMPLETE
    baseline = replace(
        config, study_path=tmp_path / "baseline.sqlite", pruning=PruningParameters()
    )
    _, unpruned = tune_predictor(baseline, table)
    assert unpruned.state.eq("COMPLETE").all()
    with pytest.raises(ValueError, match="otra configuración"):
        with managed_study(replace(config, pruning=PruningParameters()), table):
            pass


def test_every_search_family_prunes(
    pruning_config: PredictorConfig,
    predictable_trials: list[PredictorConfig],
    pruning_family: ModelName,
) -> None:
    config = replace(pruning_config, model=pruning_family, trials=6)
    _, trials = tune_predictor(config, load_design_split(config.data_dir).training)
    assert trials.state.tolist() == ["COMPLETE"] * 5 + ["PRUNED"]
    assert len(predictable_trials) == 17


def test_scalar_search_uses_same_pruning(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.surrogates.especializacion import scalar_search

    fitted: list[PredictorConfig] = []

    def parameters(trial: optuna.Trial, config: PredictorConfig) -> ModelParameters:
        return replace(config.parameters, regularization=1 if trial.number < 5 else 1e9)

    def build(config: PredictorConfig) -> Pipeline:
        fitted.append(config)
        return Pipeline(
            [
                (
                    "regressor",
                    DummyRegressor(
                        strategy="constant", constant=config.parameters.regularization
                    ),
                )
            ]
        )

    monkeypatch.setattr(
        "truss_research.surrogates.experimentos.suggested_parameters", parameters
    )
    monkeypatch.setattr(
        "truss_research.surrogates.especializacion.scalar_predictor", build
    )
    config = PredictorConfig(
        design_data,
        tmp_path / "run",
        model="ridge",
        folds=3,
        trials=6,
        pruning=PruningParameters(enabled=True),
    )
    _, trials = scalar_search(config, load_design_split(design_data).training, 0)
    assert trials.state.tolist() == ["COMPLETE"] * 5 + ["PRUNED"]
    assert len(fitted) == 17


def test_no_pruning_before_two_folds_or_after_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = PruningParameters(enabled=True)
    study = optuna.create_study(pruner=create_pruner(parameters))
    trial = study.ask()
    monkeypatch.setattr(trial, "should_prune", lambda: True)
    report_fold(trial, FoldProgress((100.0,), 3), parameters)
    assert not study.trials[0].intermediate_values
    with pytest.raises(optuna.TrialPruned):
        report_fold(trial, FoldProgress((100.0, 100.0), 3), parameters)
    completed = study.ask()
    monkeypatch.setattr(completed, "should_prune", lambda: True)
    report_fold(completed, FoldProgress((100.0, 100.0), 2), parameters)
    assert completed.user_attrs["skipped_folds"] == 0


def test_resume_retains_pruned_trials_and_budget(
    pruning_config: PredictorConfig,
    predictable_trials: list[PredictorConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import truss_research.surrogates.experimentos as experiments

    config = pruning_config
    tmp_path = config.output_dir.parent
    original = experiments.save_sampler

    def stop(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        original(study, trial)
        if trial.state == optuna.trial.TrialState.PRUNED:
            raise KeyboardInterrupt

    monkeypatch.setattr(experiments, "save_sampler", stop)
    table = load_design_split(config.data_dir).training
    with pytest.raises(KeyboardInterrupt):
        tune_predictor(config, table)
    assert read_artifact(tmp_path / "resume.pruning.json")["trials"] == 6
    monkeypatch.setattr(experiments, "save_sampler", original)
    _, resumed = tune_predictor(config, table)
    _, continuous = tune_predictor(
        replace(config, study_path=tmp_path / "continuous.sqlite"), table
    )
    assert resumed.state.tolist() == ["COMPLETE"] * 5 + ["PRUNED"] * 2
    pd.testing.assert_frame_equal(
        resumed[["state", "value"]], continuous[["state", "value"]]
    )
    assert len(predictable_trials) == 38


@pytest.mark.parametrize(
    "parameters",
    [
        {"enabled": "true"},
        {"startup_trials": 0},
        {"startup_trials": True},
        {"minimum_folds": 1},
    ],
)
def test_invalid_pruning_parameters(parameters: dict[str, str | int | bool]) -> None:
    with pytest.raises(ValueError):
        PruningParameters(**parameters)


def test_protocol_and_diagnostic_propagate_pruning(tmp_path: Path) -> None:
    from truss_research.evaluacion.entrenamiento_individual import (
        TrainingRequest,
        training_config,
    )

    config = load_protocol(Path("config/study.json"), (tmp_path, tmp_path))
    baseline = training_config(
        TrainingRequest("catboost", destination=tmp_path, pruning=False)
    )
    assert config.pruning.enabled and not baseline.pruning.enabled
    assert config.pruning.startup_trials == 5 and config.pruning.minimum_folds == 2
    request = TrainingRequest("catboost", destination=tmp_path)
    assert training_config(request).pruning == config.pruning


def test_pruning_survives_study_loading(tmp_path: Path) -> None:
    from truss_research.optimizacion.estudio_estado import (
        StudySession,
        load_study_config,
    )

    config = load_protocol(Path("config/study.json"), (tmp_path, tmp_path))
    StudySession(config, "dataset", "source", {}).save()
    assert load_study_config(tmp_path).pruning == config.pruning


def test_fit_diagnostic_disables_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion.entrenamiento_individual import (
        TrainingRequest,
        execute_training,
    )

    captured: list[PredictorConfig] = []
    monkeypatch.setattr(
        "truss_research.evaluacion.entrenamiento_individual.train_predictor",
        captured.append,
    )
    monkeypatch.setattr(
        "truss_research.surrogates.ajuste_directo.fit_diagnostic", captured.append
    )
    config = load_protocol(Path("config/study.json"), (tmp_path, tmp_path))
    execute_training(TrainingRequest("catboost"), config)
    execute_training(TrainingRequest("catboost", mode="fit"), config)
    assert captured[0].pruning.enabled
    assert not captured[1].pruning.enabled and captured[1].trials == 0
