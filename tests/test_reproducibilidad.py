"""Portabilidad, planificación sin efectos y exclusión estricta de pliegues."""

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.__main__ import main
from truss_research.datos.artefactos import dataset_identity, read_artifact
from truss_research.datos.procesados import load_design_split
from truss_research.evaluacion.auditoria_estudio import (
    AuditConfig,
    audit_study,
    nearest_designs,
)
from truss_research.evaluacion.informe_estudio import (
    latest_study_report,
    read_object,
)
from truss_research.optimizacion.estudio import run_study
from truss_research.optimizacion.estudio_estado import StudyConfig, load_study_config
from truss_research.optimizacion.protocolo import load_protocol, study_plan
from truss_research.surrogates.configuracion import ModelParameters, PredictorConfig
from truss_research.surrogates.experimentos import write_json
from truss_research.surrogates.validacion import nested_cross_fitted


def test_atomic_artifact_retries_transient_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.datos import artefactos

    original = Path.replace
    calls: list[Path] = []

    def blocked_once(path: Path, target: Path) -> Path:
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("bloqueo transitorio simulado")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", blocked_once)
    monkeypatch.setattr(artefactos, "sleep", lambda seconds: None)
    target = tmp_path / "run.json"
    write_json(target, {"value": 42})
    assert read_artifact(target)["value"] == 42 and len(calls) == 2


def test_atomic_artifact_preserves_previous_file_on_permanent_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.datos import artefactos

    target = tmp_path / "run.json"
    write_json(target, {"value": 1})
    calls: list[Path] = []

    def blocked(path: Path, target: Path) -> Path:
        calls.append(path)
        raise PermissionError("bloqueo persistente simulado")

    monkeypatch.setattr(Path, "replace", blocked)
    monkeypatch.setattr(artefactos, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError, match="persistente"):
        write_json(target, {"value": 2})
    assert len(calls) == artefactos.REPLACE_ATTEMPTS
    assert read_artifact(target)["value"] == 1


def test_plan_without_data_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "absent"

    def forbidden(config: StudyConfig, phase: str, resume: bool) -> None:
        raise AssertionError("No debe iniciar el coordinador")

    monkeypatch.setattr("truss_research.optimizacion.estudio.run_study", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "truss_research",
            "ejecutar-estudio",
            "--destino",
            str(target),
            "--fase",
            "todo",
            "--plan",
        ],
    )
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total_tasks"] == 297
    assert result["tuning_trials"] == 240
    assert result["nested_additional_trials"] == 1200
    assert result["execution"]["numeric_threads_per_worker"] == 1
    assert result["execution"]["nested_pools"] is False
    assert not target.exists()


def test_profiles_and_unknown_settings(tmp_path: Path) -> None:
    locations = (tmp_path / "study", tmp_path / "data")
    full = load_protocol(Path("config/study.json"), locations)
    assert full.label == "posterior" and full.budgets == (32, 64, 128)
    assert full.experiment == "evolution" and full.split_protocol == "first_appearance"
    smoke = load_protocol(Path("config/smoke.json"), locations)
    assert smoke.label == "comprobacion" and smoke.rl_steps == 100
    assert (
        full.execution.cpu_workers in (4, 6, 8, 12) and smoke.execution.cpu_workers == 2
    )
    nested = load_protocol(Path("config/study.json"), locations)
    assert nested.nested_folds == 3 and nested.catalog_size == 32
    assert nested.label == "posterior"
    final_plan = json.loads(study_plan(nested, "final"))
    assert final_plan["policy_transitions"] == 0
    assert json.loads(study_plan(nested, "todo"))["policy_transitions"] == 1_000_000
    assert smoke.evolution.population_size == 4 and smoke.budgets == (7, 11)
    assert len(full.search.strategies) == 9 and len(full.search.policies) == 4
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"study": {"unknown": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="desconocidos"):
        load_protocol(bad, locations)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[]", "objeto"),
        ("null", "objeto"),
        ('{"unknown": {}}', "Secciones JSON"),
        ('{"study": []}', "Campos desconocidos"),
        ('{"parameters": []}', "Parámetros predictivos"),
        ('{"evolution": []}', "Parámetros evolutivos"),
        ('{"execution": []}', "Parámetros de ejecución"),
        ('{"execution": {"unknown": 2}}', "Parámetros de ejecución"),
        ('{"execution": {"cpu_workers": 0}}', "entero positivo"),
        ('{"study": {"seeds": "42"}}', "lista JSON"),
        ('{"study":', "Expecting value"),
    ],
)
def test_invalid_json_protocols(tmp_path: Path, content: str, message: str) -> None:
    protocol = tmp_path / "invalid.json"
    protocol.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_protocol(protocol, (tmp_path / "study", tmp_path / "data"))


@pytest.mark.parametrize("version", [3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_retired_study_formats_are_rejected(tmp_path: Path, version: int) -> None:
    from truss_research.optimizacion.estudio_estado import read_study

    (tmp_path / "study.json").write_text(
        json.dumps({"schema_version": version}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="compatible"):
        read_study(tmp_path)


def test_identity_ignores_location_but_detects_changes(
    design_data: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "relocated_data"
    shutil.copytree(design_data, copy)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["source"] = "otro/directorio/fixture.xlsx"
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert dataset_identity(copy) == dataset_identity(design_data)
    with (copy / "train.csv").open("a", encoding="utf-8") as stream:
        stream.write("alterado\n")
    assert dataset_identity(copy) != dataset_identity(design_data)


def test_nested_selection_excludes_outer_rows(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.surrogates.validacion as nested_module

    table = load_design_split(design_data).training
    tuning_ids: list[set[str]] = []
    fitting_ids: list[set[str]] = []
    tune = nested_module.tune_predictor
    fit = Pipeline.fit

    def track_tuning(
        config: PredictorConfig, rows: pd.DataFrame
    ) -> tuple[PredictorConfig, pd.DataFrame]:
        tuning_ids.append(set(rows["iter_id"]))
        assert config.nested_folds is None and config.folds == 2
        return tune(config, rows)

    def track_fit(
        self: Pipeline, features: pd.DataFrame, responses: pd.DataFrame
    ) -> Pipeline:
        fitted = set(table.loc[features.index, "iter_id"])
        assert fitted <= tuning_ids[-1]
        fitting_ids.append(fitted)
        return fit(self, features, responses)

    monkeypatch.setattr(nested_module, "tune_predictor", track_tuning)
    monkeypatch.setattr(Pipeline, "fit", track_fit)
    config = PredictorConfig(
        design_data,
        tmp_path / "unused",
        model="random_forest",
        trials=1,
        folds=2,
        nested_folds=2,
    )
    result = nested_cross_fitted(config, table)
    assert set(result.predictions["iter_id"]) == set(table["iter_id"])
    for number, fold in enumerate(result.provenance):
        held = set(fold["held_ids"])
        assert held.isdisjoint(tuning_ids[number])
        assert all(
            held.isdisjoint(fitted)
            for fitted in fitting_ids[number * 3 : (number + 1) * 3]
        )
        assert set(fold["fit_ids"]) == tuning_ids[number] == fitting_ids[number * 3 + 2]
        assert held | tuning_ids[number] == set(table["iter_id"])
    assert not config.output_dir.exists()
    assert len(fitting_ids) == 6


def test_relocation_resume_freeze_audit_without_fit(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion import benchmark
    from truss_research.evaluacion.recursos import MeasurementParameters
    from truss_research.optimizacion.contratos import AnalysisParameters

    export_root = tmp_path / "benchmarks"
    monkeypatch.setattr(benchmark, "benchmark_export_root", lambda study: export_root)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copytree(design_data, bundle / "data")
    config = StudyConfig(
        bundle / "study",
        data_dir=bundle / "data",
        models=("dummy",),
        seeds=(42,),
        outer_folds=2,
        inner_folds=2,
        trials=0,
        catalog_size=4,
        budgets=(2,),
        episodes_per_seed=1,
        rl_steps=4,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=2,
        background_size=2,
        permutation_rounds=1,
        parameters=ModelParameters(trees=2),
        analysis=AnalysisParameters(catalog_sizes=(4, 8), bootstrap_repeats=10),
        measurement=MeasurementParameters(
            warmup=0, inference_repeats=1, selection_repeats=1, batch_sizes=(1, 4, 8)
        ),
    )
    run_study(config)
    frozen = (config.output_dir / "freeze.json").read_bytes()
    relocated = tmp_path / "new_location"
    shutil.move(str(bundle), relocated)
    root = relocated / "study"

    def forbidden_fit(
        self: Pipeline, features: pd.DataFrame, responses: pd.DataFrame
    ) -> None:
        raise AssertionError(
            "La lectura, reanudación completa y auditoría no ajustan predictores"
        )

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    loaded = load_study_config(root)
    assert loaded.data_dir == relocated / "data"
    assert latest_study_report(root) is not None
    run_study(loaded, "todo", True)
    assert (root / "freeze.json").read_bytes() == frozen
    benchmark.run_benchmark(root, export_root / "benchmark_001")
    audit_study(AuditConfig(root, loaded.data_dir, tmp_path / "audit"))
    assert (tmp_path / "audit/AUDIT.md").is_file()
    assert (root / "freeze.json").read_bytes() == frozen


def test_proximity_transformer_uses_training_only(design_data: Path) -> None:
    split = load_design_split(design_data)
    first, thresholds = nearest_designs(split)
    changed = replace(split, validation=split.validation.copy())
    changed.validation.loc[:, "lr_height_m"] += 100
    second, other_thresholds = nearest_designs(changed)
    assert thresholds == other_thresholds
    pd.testing.assert_frame_equal(
        first.loc[first["split"].eq("test")], second.loc[second["split"].eq("test")]
    )
    assert (
        second.loc[second["split"].eq("validation"), "distance"].min()
        > first["distance"].max()
    )


def test_relative_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    payload = {
        "config": {"data_dir": str(tmp_path.parent / "data")},
        "directory": str(tmp_path / "task"),
    }
    write_json(path, payload)
    assert str(tmp_path) not in path.read_text(encoding="utf-8")
    assert read_artifact(path) == payload


def test_interrupted_policy_resumes_after_relocation(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.optimizacion.entrenamiento as training

    bundle = tmp_path / "interrupted_bundle"
    bundle.mkdir()
    shutil.copytree(design_data, bundle / "data")
    config = StudyConfig(
        bundle / "study",
        data_dir=bundle / "data",
        models=("dummy",),
        seeds=(42,),
        outer_folds=2,
        trials=0,
        catalog_size=4,
        budgets=(2,),
        episodes_per_seed=1,
        rl_steps=4,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=2,
        background_size=2,
        permutation_rounds=1,
    )
    save = training.save_policy_run

    def interrupt(
        run: training.PolicyRun, checkpoint: training.TorchRLCheckpoint
    ) -> None:
        save(run, checkpoint)
        if run.config.use_predictions and checkpoint.step == 2:
            raise RuntimeError("Interrupción de prueba tras checkpoint firmado")

    monkeypatch.setattr(training, "save_policy_run", interrupt)
    with pytest.raises(RuntimeError, match="tareas fallidas"):
        run_study(config, "validacion")
    monkeypatch.setattr(training, "save_policy_run", save)
    relocated = tmp_path / "resumed_bundle"
    shutil.move(str(bundle), relocated)
    root = relocated / "study"
    run_study(load_study_config(root), "validacion", True)
    state = read_object(root / "study.json")
    record = next(task for task in state["tasks"] if task["key"] == "policy/rl/42")
    assert record["status"] == "complete" and len(record["attempts"]) == 2
    run = read_object(Path(record["directory"]) / "run.json")
    assert run["completed_steps"] == 4
    assert run["config"]["resume_dir"] == record["attempts"][0]
    assert not state["test_started"]
