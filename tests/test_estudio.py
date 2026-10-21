"""Estudio pequeño de extremo a extremo; nunca usa datos experimentales reales."""

from dataclasses import replace
from pathlib import Path

import optuna
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.datos.procesados import load_design_split
from truss_research.evaluacion.informe_estudio import read_object, summarize_study
from truss_research.optimizacion.estudio import prediction_tasks, run_study
from truss_research.optimizacion.estudio_cierre import family_selection
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    load_study_config,
    open_session,
    study_lock,
)
from truss_research.surrogates.configuracion import ModelParameters, PredictorConfig
from truss_research.surrogates.experimentos import (
    cross_fitted_predictions,
    train_predictor,
    write_json,
)


@pytest.fixture
def small_study(design_data: Path, tmp_path: Path) -> StudyConfig:
    return StudyConfig(
        tmp_path / "study",
        data_dir=design_data,
        models=("dummy", "ridge", "random_forest"),
        seeds=(42, 43),
        outer_folds=2,
        trials=0,
        catalog_size=4,
        budgets=(2, 3),
        episodes_per_seed=1,
        rl_steps=6,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=3,
        background_size=4,
        permutation_rounds=1,
        parameters=ModelParameters(trees=3, depth=2),
    )


def test_common_folds_ignore_training_seed(design_data: Path, tmp_path: Path) -> None:
    table = load_design_split(design_data).training
    config = PredictorConfig(
        design_data, tmp_path / "model", model="dummy", folds=2, cv_seed=42
    )
    first = cross_fitted_predictions(config, table)[0]
    second = cross_fitted_predictions(replace(config, seed=43), table)[0]
    pd.testing.assert_series_equal(first["fold"], second["fold"])
    assert len(prediction_tasks(StudyConfig(tmp_path / "full"))) == 55


def test_optuna_reuses_budget_and_rejects_protocol(
    design_data: Path, tmp_path: Path
) -> None:
    config = PredictorConfig(
        design_data,
        tmp_path / "first",
        model="random_forest",
        folds=2,
        trials=2,
        cv_seed=42,
        study_path=tmp_path / "search.sqlite",
    )
    train_predictor(config)
    repeated = replace(config, output_dir=tmp_path / "second")
    train_predictor(repeated)
    from truss_research.surrogates.experimentos import managed_study

    with managed_study(config, load_design_split(design_data).training) as study:
        assert len(study.trials) == 2
    with pytest.raises(ValueError, match="otra configuración"):
        train_predictor(replace(config, output_dir=tmp_path / "wrong", cv_seed=43))
    with managed_study(config, load_design_split(design_data).training) as study:
        assert len(study.trials) == 2


def test_interrupted_trial_consumes_attempt(design_data: Path, tmp_path: Path) -> None:
    from truss_research.surrogates.experimentos import managed_study, tune_predictor

    config = PredictorConfig(
        design_data,
        tmp_path / "model",
        model="random_forest",
        folds=2,
        trials=1,
        study_path=tmp_path / "search.sqlite",
    )
    table = load_design_split(design_data).training
    with managed_study(config, table) as study:
        study.ask()
    with pytest.raises(ValueError, match="Todos los ensayos fallaron"):
        tune_predictor(config, table)
    with managed_study(config, table) as study:
        assert len(study.trials) == 1
        assert study.trials[0].state == optuna.trial.TrialState.FAIL


def test_study_lock_and_configuration(small_study: StudyConfig) -> None:
    with study_lock(small_study.output_dir):
        with pytest.raises(ValueError, match="Otro coordinador"):
            with study_lock(small_study.output_dir):
                pass
    run_study(small_study, "predictores")
    loaded = load_study_config(small_study.output_dir)
    assert loaded.parameters == small_study.parameters
    with pytest.raises(ValueError, match="ya existe"):
        run_study(small_study, "predictores")
    with pytest.raises(ValueError, match="Cambió"):
        run_study(replace(small_study, trials=1), "predictores", True)
    before = read_object(small_study.output_dir / "study.json")
    run_study(small_study, "predictores", True)
    assert (
        before["tasks"] == read_object(small_study.output_dir / "study.json")["tasks"]
    )


def test_family_selection_averages_seeds(small_study: StudyConfig) -> None:
    run_study(small_study, "predictores")
    config = replace(
        small_study,
        output_dir=small_study.output_dir.resolve(),
        data_dir=small_study.data_dir.resolve(),
    )
    session = open_session(config, True)
    scores = {"dummy": (0.01, 0.99), "ridge": (0.2, 0.2), "random_forest": (0.3, 0.3)}
    for key, task in session.tasks.items():
        _, name, seed = key.split("/")
        directory = Path(task.directory)
        metadata = read_object(directory / "run.json")
        metadata["normalized_mae"] = scores[name][small_study.seeds.index(int(seed))]
        write_json(directory / "run.json", metadata)
        task.signatures = artifact_signatures(directory)
    winner, table = family_selection(session)
    assert winner == Path(session.tasks["predictor/ridge/42"].directory)
    assert table.iloc[0]["model"] == "ridge"


def test_final_requires_all_validation_tasks(small_study: StudyConfig) -> None:
    run_study(small_study, "predictores")
    with pytest.raises(ValueError, match="Falta completar"):
        run_study(small_study, "final", True)
    assert not (small_study.output_dir / "freeze.json").exists()


def test_full_study_freezes_and_final_never_fits(
    small_study: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_study(small_study, "validacion")
    root = small_study.output_dir
    assert not (root / "freeze.json").exists()
    initial = read_object(root / "study.json")
    completed = {record["key"]: record for record in initial["tasks"]}
    selection_path = Path(completed["decision/validation"]["directory"])
    table = pd.read_csv(selection_path / "summary.csv")
    assert len(table) == 60
    assert set(small_study.rl.variants) <= set(table.strategy)
    assert len(table[table["strategy"] == "random"]) == 4
    assert table["queries"].eq(table["budget"]).all()
    assert set(
        table.loc[table["strategy"] == "rl_ablation", "policy_training_seed"]
    ) == {42, 43}

    def forbidden_fit(
        self: Pipeline, features: pd.DataFrame, responses: pd.DataFrame
    ) -> None:
        raise AssertionError("El cierre no puede ajustar modelos")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    run_study(small_study, "final", True)
    frozen = (root / "freeze.json").read_bytes()
    run_study(small_study, "final", True)
    assert (root / "freeze.json").read_bytes() == frozen
    report = summarize_study(root, root.parent / "rebuilt")
    assert (report / "REPORT.md").is_file()
    assert set(pd.read_csv(report / "family_comparison.csv")["split"]) == {
        "validation",
        "test",
    }
    assert (report / "selection_traces.csv").is_file()
    assert (report / "shap_stability.csv").is_file()
    assert read_object(root / "study.json")["test_started"] is True
    tampered = root / "freeze.json"
    value = read_object(tampered)
    value["winner"] = "otro predictor"
    write_json(tampered, value)
    with pytest.raises(ValueError, match="cierre congelado"):
        run_study(small_study, "final", True)
    with pytest.raises(ValueError, match="predictor congelado"):
        summarize_study(root, root.parent / "bad_freeze")


def test_failed_task_retries_in_new_directory(
    small_study: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.optimizacion.estudio as study_module

    original = study_module.dispatch_task
    attempts = 0

    def fail_once(
        session: study_module.StudySession,
        task: study_module.StudyTask,
        destination: Path,
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Interrupción de infraestructura")
        original(session, task, destination)

    monkeypatch.setattr(study_module, "dispatch_task", fail_once)
    with pytest.raises(RuntimeError, match="tareas fallidas"):
        run_study(small_study, "predictores")
    run_study(small_study, "predictores", True)
    task = read_object(small_study.output_dir / "study.json")["tasks"][0]
    assert task["status"] == "complete"
    assert len(task["attempts"]) == 2


def test_summary_rejects_changed_artifacts(small_study: StudyConfig) -> None:
    run_study(small_study, "predictores")
    task = read_object(small_study.output_dir / "study.json")["tasks"][0]
    path = Path(task["directory"]) / "metrics.csv"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cambió un artefacto"):
        summarize_study(
            small_study.output_dir, small_study.output_dir.parent / "tampered"
        )


def test_optuna_sampler_continues_after_interruption(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.surrogates.experimentos as experiments

    config = PredictorConfig(
        design_data,
        tmp_path / "model",
        model="random_forest",
        folds=2,
        trials=2,
        study_path=tmp_path / "interrupted.sqlite",
    )
    table = load_design_split(design_data).training
    original = experiments.save_sampler

    def interrupt(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        original(study, trial)
        raise KeyboardInterrupt("Interrupción después de persistir un ensayo")

    monkeypatch.setattr(experiments, "save_sampler", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiments.tune_predictor(config, table)
    monkeypatch.setattr(experiments, "save_sampler", original)
    resumed = experiments.tune_predictor(config, table)[1]
    continuous = experiments.tune_predictor(
        replace(config, study_path=tmp_path / "continuous.sqlite"), table
    )[1]
    columns = [
        name for name in resumed if name.startswith("params_") or name == "value"
    ]
    pd.testing.assert_frame_equal(resumed[columns], continuous[columns])


def test_latest_report_rejects_changed_table(small_study: StudyConfig) -> None:
    from truss_research.evaluacion.informe_estudio import latest_study_report

    run_study(small_study, "predictores")
    report = latest_study_report(small_study.output_dir)
    assert report is not None
    (report / "family_comparison.csv").write_bytes(b"alterado")
    with pytest.raises(ValueError, match="tabla o figura"):
        latest_study_report(small_study.output_dir)
