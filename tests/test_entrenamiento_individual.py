"""Contrato del entrenamiento manual aislado, sin ejecutar cargas extensas."""

import json
from pathlib import Path

import pytest

from truss_research.datos.artefactos import read_artifact
from truss_research.evaluacion.entrenamiento_individual import (
    TrainingRequest,
    run_individual_training,
    training_config,
    training_plan,
)
from truss_research.optimizacion.variantes_rl import PolicyVariant
from truss_research.surrogates.configuracion import ModelName, PredictorConfig
from truss_research.surrogates.experimentos import data_signature, write_json


def test_plan_preserves_budget_without_creating_files(tmp_path: Path) -> None:
    request = TrainingRequest("catboost", destination=tmp_path / "unused", seed=99)
    config = training_config(request)
    plan = training_plan(request, config)
    assert plan["trials"] == 30
    assert plan["outer_folds"] == 5 and plan["inner_folds"] == 3
    assert plan["seed"] == 99
    assert config.seeds == (42, 43, 44, 45, 46)
    assert not plan["shap"] and not plan["test_used"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "model",
    [
        "rl",
        "rl_dueling",
        "rl_ppo",
        "evolution_contextual_bandit",
        "evolution_contextual_bandit_assisted",
    ],
)
def test_policy_requires_explicit_predictor(model: str) -> None:
    with pytest.raises(ValueError, match="predictor ya entrenado"):
        TrainingRequest(model)


def test_assisted_bandit_requires_distinct_auxiliary_argument(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cheap-predictor"):
        TrainingRequest("evolution_contextual_bandit_assisted", predictor_dir=tmp_path)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("catboost", "cpu"),
        ("mlp", "cpu"),
        ("tabpfn", "cuda"),
        ("targetwise", "cpu+cuda"),
    ],
)
def test_devices_follow_family(model: ModelName, expected: str) -> None:
    request = TrainingRequest(model, accelerator="cuda")
    assert (
        training_plan(request, training_config(request))["training_device"] == expected
    )


def test_dummy_produces_resources_and_refuses_overwrite(
    design_data: Path, tmp_path: Path
) -> None:
    request = TrainingRequest(
        "dummy",
        protocol=Path("config/smoke.json"),
        data_dir=design_data,
        destination=tmp_path / "single",
    )
    destination = run_individual_training(request)
    summary = read_artifact(destination / "summary.json")
    assert summary["status"] == "complete"
    assert (destination / "run/model.joblib").is_file()
    assert (destination / "run.resources.json").is_file()
    assert (destination / "phase_times.csv").is_file()
    assert not (destination / "study.json").exists()
    assert not (destination / "run/test_predictions.csv").exists()
    original = (destination / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="destino ya existe"):
        run_individual_training(request)
    assert (destination / "summary.json").read_bytes() == original


def test_failed_training_is_recorded(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(config: PredictorConfig) -> None:
        raise RuntimeError("fallo controlado")

    monkeypatch.setattr(
        "truss_research.evaluacion.entrenamiento_individual.train_predictor", fail
    )
    request = TrainingRequest(
        "dummy", data_dir=design_data, destination=tmp_path / "failed"
    )
    with pytest.raises(RuntimeError, match="fallo controlado"):
        run_individual_training(request)
    assert read_artifact(tmp_path / "failed/summary.json")["status"] == "failed"


def test_rl_dispatch_preserves_cpu_and_protocol(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.optimizacion.politica import PolicyConfig

    protocol = tmp_path / "catalog.json"
    protocol.write_text(
        json.dumps(
            {
                "extends": str(Path("config/study.json").resolve()),
                "study": {
                    "experiment": "catalog",
                    "split_protocol": "design",
                    "budgets": [12, 24],
                },
            }
        ),
        encoding="utf-8",
    )

    predictor = tmp_path / "predictor"
    predictor.mkdir()
    (predictor / "model.joblib").write_bytes(b"unused_test_descriptor")
    write_json(
        predictor / "run.json", {"dataset_signature": data_signature(design_data)}
    )
    captured: list[PolicyConfig] = []
    monkeypatch.setattr(
        "truss_research.evaluacion.entrenamiento_individual.train_policy",
        captured.append,
    )
    request = TrainingRequest(
        "rl_ppo",
        protocol=protocol,
        data_dir=design_data,
        predictor_dir=predictor,
        destination=tmp_path / "ppo",
        seed=99,
    )
    run_individual_training(request)
    assert len(captured) == 1
    assert captured[0].device == "cpu" and captured[0].algorithm == "ppo"
    assert captured[0].seed == 99 and captured[0].parameters.transitions == 50_000
    assert captured[0].validation_seeds == (42, 43, 44, 45, 46)
    assert captured[0].predictor_dir == predictor.resolve()


def test_load_test_protocol_matches_manual_budget() -> None:
    from truss_research.evaluacion.carga_cpu import load_cpu_protocol

    protocol = load_cpu_protocol()
    assert protocol.workers == (4, 6, 8, 12)
    assert protocol.budget_seconds == 300


def test_training_overrides_preserve_canonical_protocol(tmp_path: Path) -> None:
    request = TrainingRequest(
        "xgboost", destination=tmp_path, pruning=False, early_stopping_rounds=0
    )
    configured = training_config(request)
    original = training_config(TrainingRequest("xgboost", destination=tmp_path))
    assert not configured.pruning.enabled
    assert configured.parameters.early_stopping_rounds == 0
    assert original.pruning.enabled
    assert original.parameters.early_stopping_rounds == 30
    assert configured.trials == original.trials
    assert configured.seeds == original.seeds


@pytest.mark.parametrize("model", ["targetwise", "rl", "rl_ppo"])
def test_fit_mode_rejects_models_with_internal_selection(
    model: ModelName | PolicyVariant,
) -> None:
    with pytest.raises(ValueError, match="requieren protocol"):
        TrainingRequest(model, mode="fit")


def test_fit_mode_has_no_search_or_oof(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_protocol(config: PredictorConfig) -> None:
        raise AssertionError("No debe ejecutarse entrenamiento con búsqueda u OOF")

    monkeypatch.setattr(
        "truss_research.evaluacion.entrenamiento_individual.train_predictor",
        reject_protocol,
    )
    request = TrainingRequest(
        "dummy", data_dir=design_data, destination=tmp_path / "fit", mode="fit"
    )
    plan = training_plan(request, training_config(request))
    assert plan["trials"] == 0 and plan["outer_folds"] is None
    assert plan["fit_calls"] == 1
    directory = run_individual_training(request)
    diagnostic = read_artifact(directory / "run/diagnostic.json")
    assert diagnostic["fit_calls"] == 1 and diagnostic["trials"] == 0
    assert not diagnostic["eligible_for_campaign"] and not diagnostic["oof"]
    assert not diagnostic["test_used"]
    assert (directory / "run/model.joblib").is_file()
    assert (directory / "run/metrics.csv").is_file()
    assert (directory / "phase_times.csv").is_file()
    assert not (directory / "run/run.json").exists()
    assert not list(directory.glob("*.sqlite*"))
    assert not (directory / "run/training_oof.csv").exists()
