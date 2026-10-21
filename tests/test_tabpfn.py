"""Contrato del comparador fijo y flujo sintético sin descargas ni pesos reales."""

import json
import multiprocessing as mp
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter, sleep
from types import ModuleType
from typing import Self

import joblib
import numpy as np
import pandas as pd
import pytest
from test_estandarizacion import analysis_data as analysis_data
from test_estandarizacion import small_config

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.evaluacion.recursos import MeasurementParameters
from truss_research.evaluacion.trabajos_analisis import execute_analysis_jobs
from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    load_study_config,
)
from truss_research.optimizacion.protocolo import load_protocol
from truss_research.optimizacion.trabajos_predictivos import (
    JobOutcome,
    JobResources,
    run_measured_job,
)
from truss_research.surrogates.configuracion import MODEL_NAMES, PredictorConfig
from truss_research.surrogates.modelos import build_predictor
from truss_research.surrogates.persistencia import load_model, save_model
from truss_research.surrogates.tabpfn import (
    TabPFNParameters,
    verify_tabpfn,
)


@pytest.mark.parametrize("available", [False, True])
def test_pilot_command_detects_device_without_training(
    available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from truss_research.__main__ import DEFAULT_PROTOCOL, run_tabpfn_command
    from truss_research.evaluacion import piloto_tabpfn as pilot

    selected: list[str] = []

    def capture(config: StudyConfig, destination: Path) -> Path:
        selected.append(config.tabpfn.device)
        return destination

    monkeypatch.setattr("torch.cuda.is_available", lambda: available)
    monkeypatch.setattr(pilot, "run_pilot", capture)
    run_tabpfn_command(
        argparse.Namespace(
            command="probar-tabpfn",
            config=DEFAULT_PROTOCOL,
            datos=None,
            accelerator="auto",
        )
    )
    assert selected == ["cuda" if available else "cpu"]


@pytest.mark.parametrize("requested", ["cpu", "cuda"])
def test_explicit_pilot_device_skips_automatic_detection(
    requested: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion.piloto_tabpfn import pilot_accelerator

    def unexpected_probe() -> bool:
        pytest.fail("Una elección explícita no debe ejecutar autodetección")

    monkeypatch.setattr("torch.cuda.is_available", unexpected_probe)
    assert pilot_accelerator(requested) == requested


class FakeScalarRegressor:
    fit_calls = 0

    def __init__(self, **parameters: str | int | bool | list[int] | None) -> None:
        self.parameters = parameters

    def fit(self, features: pd.DataFrame, responses: np.ndarray) -> Self:
        import torch

        torch.rand(1)
        type(self).fit_calls += 1
        assert list(features.columns) == list(INPUT_COLUMNS)
        self.mean_ = float(np.mean(responses))
        self.fit_rows_ = len(features)
        x = features["lr_height_m"].to_numpy()
        self.coefficients_ = (
            np.polyfit(x, responses, 1)
            if np.ptp(x) > 0
            else np.array([0.0, self.mean_])
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.polyval(self.coefficients_, features["lr_height_m"])


@pytest.fixture
def fake_tabpfn(monkeypatch: pytest.MonkeyPatch) -> None:
    tabpfn, loading = ModuleType("tabpfn"), ModuleType("tabpfn.model_loading")
    tabpfn.TabPFNRegressor = FakeScalarRegressor
    loading.save_fitted_tabpfn_model = joblib.dump
    loading.load_fitted_tabpfn_model = lambda path, device: joblib.load(path)
    monkeypatch.setitem(sys.modules, "tabpfn", tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.model_loading", loading)

    import truss_research.evaluacion.piloto_tabpfn as pilot
    import truss_research.surrogates.tabpfn as adapter

    monkeypatch.setattr(adapter, "verify_tabpfn", lambda settings: None)
    monkeypatch.setattr(adapter, "tabpfn_slot", lambda settings: nullcontext())
    monkeypatch.setattr(
        pilot, "require_pilot", lambda config: {"status": "synthetic_test_double"}
    )


def test_fixed_profile_and_no_hpo(tmp_path: Path) -> None:
    config = load_protocol(Path("config/study.json"), (tmp_path, tmp_path))
    smoke = load_protocol(Path("config/smoke.json"), (tmp_path, tmp_path))
    assert len(config.models) == 11 and config.models == MODEL_NAMES
    assert config.tabpfn_enabled and smoke.tabpfn_enabled
    with pytest.raises(ValueError, match="Optuna"):
        PredictorConfig(model="tabpfn", trials=1)
    assert TabPFNParameters(device="cuda").device == "cuda"
    with pytest.raises(ValueError, match="cpu/cuda"):
        TabPFNParameters(device="invalid")
    invalid = tmp_path / "bad.json"
    invalid.write_text('{"tabpfn":{"unknown":true}}')
    with pytest.raises(ValueError, match="desconocidos"):
        load_protocol(invalid, (tmp_path, tmp_path))


def test_adapter_and_signed_states(
    fake_tabpfn: None, raw_table: pd.DataFrame, tmp_path: Path
) -> None:
    from truss_research.datos.schemas import COLUMN_MAPPING

    features = raw_table.rename(columns=COLUMN_MAPPING).loc[:, list(INPUT_COLUMNS)]
    responses = np.tile(np.array([1200.0, 0.45, 0.03]), (len(features), 1))
    model = build_predictor(PredictorConfig(model="tabpfn"))
    import torch

    state = torch.random.get_rng_state().clone()
    model.fit(features, responses)
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(model.named_steps["regressor"].estimators_) == 3
    scalar = model.named_steps["regressor"].estimators_[0]
    assert scalar.parameters["n_estimators"] == 8
    assert scalar.parameters["categorical_features_indices"] == [0]
    assert scalar.parameters["n_preprocessing_jobs"] == 1
    expected = model.predict(features)
    save_model(model, tmp_path)
    assert len(list(tmp_path.glob("*.tabpfn_fit"))) == 3
    descriptor = joblib.load(tmp_path / "model.joblib")
    assert not hasattr(descriptor.named_steps["regressor"], "estimators_")
    np.testing.assert_allclose(load_model(tmp_path).predict(features), expected)
    assert torch.equal(state, torch.random.get_rng_state())
    state = tmp_path / "mass_kg.tabpfn_fit"
    state.write_bytes(state.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="estado ajustado"):
        load_model(tmp_path)


def test_checkpoint_and_dependency_fail_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.surrogates.tabpfn as adapter

    settings = TabPFNParameters(checkpoint_dir=str(tmp_path))
    monkeypatch.setattr(adapter, "version", lambda name: "wrong")
    with pytest.raises(ValueError, match="instalado"):
        verify_tabpfn(settings)
    monkeypatch.setattr(adapter, "version", lambda name: adapter.PACKAGE_VERSION)
    with pytest.raises(ValueError, match="checkpoint local"):
        verify_tabpfn(settings)
    settings.checkpoint.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="hash"):
        verify_tabpfn(settings)


def test_targetwise_selects_and_persists_tabpfn(
    fake_tabpfn: None, raw_table: pd.DataFrame, tmp_path: Path
) -> None:
    from truss_research.datos.schemas import COLUMN_MAPPING

    features = raw_table.rename(columns=COLUMN_MAPPING).loc[:, list(INPUT_COLUMNS)]
    features["lr_height_m"] = np.linspace(0.5, 1.5, len(features))
    x = features["lr_height_m"].to_numpy()
    responses = np.column_stack((1000 + x * 100, 0.3 + x * 0.1, 0.01 + x * 0.001))
    config = PredictorConfig(
        model="targetwise",
        data_dir=tmp_path / "synthetic_data",
        folds=2,
        trials=0,
        candidate_models=("dummy", "tabpfn"),
    )
    model = build_predictor(config)
    model.fit(features, responses)
    assert [row["family"] for row in model.named_steps["regressor"].provenance_] == [
        "tabpfn"
    ] * 3
    save_model(model, tmp_path)
    assert len(list(tmp_path.rglob("*.tabpfn_fit"))) == 3
    np.testing.assert_allclose(load_model(tmp_path).predict(features), responses)
    path = next(tmp_path.rglob("*.tabpfn_fit"))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="componentes"):
        load_model(tmp_path)


@dataclass(frozen=True)
class TimedJob:
    config: StudyConfig
    destination: Path
    serial: bool = False


def run_timed_job(job: TimedJob) -> JobOutcome:
    def work() -> None:
        start = perf_counter()
        sleep(0.05)
        job.destination.mkdir()
        (job.destination / "times.json").write_text(
            json.dumps({"start": start, "finish": perf_counter()})
        )

    return run_measured_job(
        JobResources(
            job.destination.name,
            job.destination,
            job.config.data_dir,
            job.config.measurement,
        ),
        work,
    )


def test_serial_batch_after_parallel_jobs(analysis_data: Path, tmp_path: Path) -> None:
    config = replace(
        small_config(analysis_data, tmp_path / "study"),
        execution=ExecutionParameters(cpu_workers=2),
    )
    root = tmp_path / "jobs"
    jobs = [
        TimedJob(config, root / name, serial)
        for name, serial in (("a", False), ("b", False), ("c", True), ("d", True))
    ]
    execute_analysis_jobs(jobs, run_timed_job, root)
    times = [json.loads((job.destination / "times.json").read_text()) for job in jobs]
    assert times[2]["start"] >= max(row["finish"] for row in times[:2])
    assert times[3]["start"] >= times[2]["finish"]
    before = artifact_signatures(root / "c")
    execute_analysis_jobs(jobs, run_timed_job, root)
    assert artifact_signatures(root / "c") == before


def test_eligible_tabpfn_complete_synthetic_flow(
    fake_tabpfn: None,
    analysis_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import truss_research.evaluacion.trabajos_analisis as jobs
    from truss_research.evaluacion.auditoria_estudio import AuditConfig, audit_study
    from truss_research.evaluacion.benchmark import run_benchmark
    from truss_research.evaluacion.curvas_aprendizaje import run_learning_curves
    from truss_research.evaluacion.informe_estudio import summarize_study
    from truss_research.evaluacion.registro import results_root
    from truss_research.evaluacion.seleccion_adaptativa import run_adaptive_selection
    from truss_research.evaluacion.sensibilidad import run_sensitivity
    from truss_research.optimizacion.estudio import run_study

    monkeypatch.setattr(jobs, "cpu_pool", lambda execution: nullcontext(None))
    checkpoint = tmp_path / "weights"
    checkpoint.mkdir()
    settings = TabPFNParameters(checkpoint_dir=str(checkpoint))
    settings.checkpoint.write_bytes(b"synthetic_test_weights")
    config = replace(
        small_config(analysis_data, tmp_path / "study"),
        tabpfn=settings,
        models=("dummy", "tabpfn"),
        calibration_required=True,
        evolution=replace(
            small_config(analysis_data, tmp_path / "study").evolution,
            initialization_ablation=True,
        ),
        measurement=MeasurementParameters(
            warmup=0,
            inference_repeats=1,
            selection_repeats=1,
            batch_sizes=(1, 8),
            cpu_repeats=1,
            cpu_min_seconds=0.01,
            cpu_min_ticks=1,
            cpu_max_wall_seconds=0.5,
            cpu_probe_seconds=0.05,
        ),
    )
    config = replace(
        config,
        analysis=replace(
            config.analysis,
            generalization=replace(
                config.analysis.generalization, control_models=("dummy", "tabpfn")
            ),
            learning_curves=replace(
                config.analysis.learning_curves, models=("dummy", "tabpfn")
            ),
        ),
    )
    FakeScalarRegressor.fit_calls = 0
    run_study(config, "validacion", False)
    assert FakeScalarRegressor.fit_calls == 9
    run_study(config, "validacion", True)
    assert FakeScalarRegressor.fit_calls == 9
    assert load_study_config(config.output_dir) == replace(
        config,
        output_dir=config.output_dir.resolve(),
        data_dir=config.data_dir.resolve(),
    )
    from truss_research.evaluacion.informe_estudio import signed_records

    tab_root = next(
        Path(r["directory"])
        for r in signed_records(config.output_dir)
        if r["key"] == "predictor/tabpfn/42"
    )
    before = artifact_signatures(tab_root)
    run_study(config, "final", True)
    frozen = read_artifact(config.output_dir / "freeze.json")
    assert "/predictor/tabpfn/" in frozen["winner"].replace("\\", "/")
    assert artifact_signatures(tab_root) == before
    root = results_root(config.output_dir)
    run_adaptive_selection(config.output_dir, root / "adaptativa_001")
    assert FakeScalarRegressor.fit_calls == 9
    sensitivity = run_sensitivity(config.output_dir, root / "sensibilidad_001")
    grouped = pd.read_csv(sensitivity / "generalization/selection.csv")
    assert "tabpfn" in set(grouped.model)
    assert grouped.loc[grouped.model.eq("tabpfn"), "selected_on_validation"].all()
    run_learning_curves(config.output_dir, root / "curvas_001")
    full = read_artifact(next((root / "curvas_001").glob("tabpfn_*_full/run.json")))
    assert full["main_fit_reused"] is True
    assert len(list((root / "curvas_001").glob("tabpfn_*/run.json"))) == 5
    assert FakeScalarRegressor.fit_calls == 27
    run_benchmark(config.output_dir, root / "benchmark_001")
    report = summarize_study(config.output_dir, root / "informe_001")
    predictive = pd.read_csv(report / "predictive_runs.csv")
    assert "tabpfn" in set(predictive.model)
    assert "tabpfn" in set(pd.read_csv(report / "ranking_runs.csv").model)
    audit_study(AuditConfig(config.output_dir, analysis_data, root / "auditoria_001"))
    for name in ("errors_by_distance", "prediction_utility"):
        table = pd.read_csv(root / "auditoria_001" / f"{name}.csv")
        assert "tabpfn" in set(table.model)
    from truss_research.evaluacion.presentacion import export_notebooks

    export_notebooks(config.output_dir, root / "presentacion_001")
    assert len(list((root / "presentacion_001").glob("*.html"))) == 4


def test_fixed_configuration_round_trip(tmp_path: Path) -> None:
    from truss_research.surrogates.experimentos import config_from_run, write_json

    settings = TabPFNParameters(
        n_estimators=2, checkpoint_dir=str(tmp_path / "weights")
    )
    config = PredictorConfig(model="tabpfn", seed=7, tabpfn=settings)
    write_json(tmp_path / "run.json", {"config": asdict(config)})
    restored = config_from_run(tmp_path)
    assert restored.seed == 7 and restored.tabpfn == settings


def test_plan_before_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.evaluacion.piloto_tabpfn as pilot
    from truss_research.evaluacion.plan_analisis import analysis_plan

    def missing_data(config: StudyConfig) -> dict:
        raise FileNotFoundError("Datos pendientes de preparar")

    monkeypatch.setattr(pilot, "require_pilot", missing_data)
    config = load_protocol(
        Path("config/study.json"), (tmp_path / "data", tmp_path / "study")
    )
    result = analysis_plan(config)
    assert result["preflight"] == "requires_make_prepare"
    assert result["tabpfn"]["pilot"]["status"] == "required"
    assert not config.output_dir.exists() and not config.data_dir.exists()


def test_pilot_required_and_tamper_detection(
    analysis_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.evaluacion.piloto_tabpfn as pilot
    from truss_research.surrogates.experimentos import write_json

    monkeypatch.setattr(pilot, "verify_tabpfn", lambda settings: None)
    settings = TabPFNParameters(checkpoint_dir=str(tmp_path / "weights"))
    config = replace(
        small_config(analysis_data, tmp_path / "study"),
        tabpfn=settings,
        models=("dummy", "tabpfn"),
    )
    with pytest.raises(ValueError, match="piloto"):
        pilot.require_pilot(config)
    root = settings.checkpoint.parent / "pilots/pilot_001"
    root.mkdir(parents=True)
    artifact = root / "proof.json"
    artifact.write_text('{"synthetic":true}')
    write_json(
        root / "pilot.json",
        {
            **pilot.pilot_identity(config),
            "status": "complete",
            "rows": [{"fit_seconds": 0.1}],
            "pilot_artifact_signatures": artifact_signatures(root),
        },
    )
    assert pilot.require_pilot(config)["status"] == "complete"
    changed = replace(config, tabpfn=replace(settings, n_estimators=1))
    with pytest.raises(ValueError, match="piloto"):
        pilot.require_pilot(changed)
    artifact.write_text('{"synthetic":false}')
    with pytest.raises(ValueError, match="cambiaron"):
        pilot.require_pilot(config)


def test_pilot_budget_stops_worker_and_preserves_failure(
    analysis_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.evaluacion.piloto_tabpfn as pilot

    class Worker:
        exitcode = None
        stopped = False

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return not self.stopped

        def join(self, seconds: float) -> None:
            pass

    class Context:
        def Process(self, **arguments: tuple) -> Worker:
            return worker

    worker = Worker()
    monkeypatch.setattr(pilot, "verify_tabpfn", lambda settings: None)
    monkeypatch.setattr(pilot.mp, "get_context", lambda kind: Context())
    monkeypatch.setattr(
        pilot, "stop_pilot_process", lambda process: setattr(process, "stopped", True)
    )
    ticks = iter((0.0, 601.0, 602.0))
    monkeypatch.setattr(pilot, "perf_counter", lambda: next(ticks))
    config = small_config(analysis_data, tmp_path / "study")
    root = tmp_path / "pilot"
    with pytest.raises(RuntimeError, match="diez minutos"):
        pilot.run_pilot(config, root)
    assert worker.stopped
    assert read_artifact(root / "pilot.json")["status"] == "failed"


@pytest.mark.skipif(
    os.getenv("TRUSS_TABPFN_REAL_TEST") != "1",
    reason="Integración optativa con pesos locales, sin descarga automática",
)
def test_real_local_weights_round_trip(raw_table: pd.DataFrame, tmp_path: Path) -> None:
    from truss_research.datos.schemas import COLUMN_MAPPING

    settings = TabPFNParameters(n_estimators=1)
    verify_tabpfn(settings)
    table = raw_table.rename(columns=COLUMN_MAPPING)
    table["lr_height_m"] = np.linspace(0.3, 1.8, len(table))
    table["mass_kg"] = 1000.0 + np.arange(len(table)) * 10.0
    table["util_ratio"] = 0.2 + np.arange(len(table)) * 0.01
    table["z_disp_m"] = 0.01 + np.arange(len(table)) * 0.0002
    features = table.loc[:, list(INPUT_COLUMNS)]
    responses = table.loc[:, list(OUTPUT_COLUMNS)].to_numpy()
    model = build_predictor(PredictorConfig(model="tabpfn", tabpfn=settings))
    import torch

    state = torch.random.get_rng_state().clone()
    model.fit(features, responses)
    assert torch.equal(state, torch.random.get_rng_state())
    expected = model.predict(features.iloc[:3])
    save_model(model, tmp_path)
    np.testing.assert_allclose(
        load_model(tmp_path).predict(features.iloc[:3]), expected, rtol=1e-6, atol=1e-8
    )
    assert torch.equal(state, torch.random.get_rng_state())
    worker = mp.get_context("spawn").Process(
        target=check_real_reload, args=(tmp_path, (features.iloc[:3], expected))
    )
    worker.start()
    worker.join(timeout=60)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
        pytest.fail("La recarga del modelo real superó un minuto")
    assert worker.exitcode == 0


def check_real_reload(directory: Path, case: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, expected = case
    np.testing.assert_allclose(
        load_model(directory).predict(features), expected, rtol=1e-6, atol=1e-8
    )
