"""Equivalencia, dependencias y concurrencia real sobre datos sintéticos."""

from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pandas as pd
import psutil
import pytest

from truss_research.datos.artefactos import read_artifact
from truss_research.evaluacion.informe_costos import resource_inventory
from truss_research.optimizacion.contratos import AnalysisParameters
from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters, cpu_pool
from truss_research.optimizacion.estudio import run_study
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    load_study_config,
    open_session,
    study_lock,
)
from truss_research.surrogates.configuracion import (
    MODEL_NAMES,
    ModelParameters,
    PredictorConfig,
)
from truss_research.surrogates.experimentos import read_run, train_predictor

WORKER_STARTUP_SECONDS = 30
WORKER_POLL_SECONDS = 0.05


def tiny_parallel_config(data: Path, destination: Path) -> StudyConfig:
    return StudyConfig(
        destination,
        data_dir=data,
        models=("dummy", "ridge"),
        seeds=(42,),
        outer_folds=2,
        inner_folds=2,
        trials=0,
        catalog_size=4,
        budgets=(3,),
        episodes_per_seed=1,
        rl_steps=4,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=2,
        background_size=2,
        permutation_rounds=1,
        analysis=AnalysisParameters(catalog_sizes=(4, 8), bootstrap_repeats=10),
        execution=ExecutionParameters(cpu_workers=2),
    )


def wait_in_worker(marker: Path) -> int:
    process = psutil.Process()
    marker.write_text(str(process.pid), encoding="utf-8")
    sleep(WORKER_STARTUP_SECONDS)
    return process.pid


def test_pool_interruption_terminates_spawned_workers(tmp_path: Path) -> None:
    marker = tmp_path / "worker.pid"
    with pytest.raises(RuntimeError, match="interrupción simulada"):
        with cpu_pool(ExecutionParameters(cpu_workers=2)) as executor:
            executor.submit(wait_in_worker, marker)
            deadline = monotonic() + WORKER_STARTUP_SECONDS
            while not marker.exists() and monotonic() < deadline:
                sleep(WORKER_POLL_SECONDS)
            assert marker.exists(), "El trabajador no inició dentro del plazo"
            raise RuntimeError("interrupción simulada")
    assert not psutil.pid_exists(int(marker.read_text(encoding="utf-8")))


def test_failed_search_resume_preserves_completed_folds(
    design_data: Path, tmp_path: Path
) -> None:
    config = replace(
        tiny_parallel_config(design_data, tmp_path / "study"),
        models=("ridge",),
        trials=1,
    )
    with study_lock(config.output_dir):
        open_session(config, False)
    corrupted = config.output_dir / "studies/ridge_42.sqlite"
    corrupted.parent.mkdir(exist_ok=True)
    corrupted.write_bytes(b"base sqlite corrupta para simular fallo")
    with pytest.raises(RuntimeError, match="tareas fallidas"):
        run_study(config, "predictores", True)
    state = read_artifact(config.output_dir / "study.json")
    completed = {
        task["key"]: task
        for task in state["preparation_tasks"]
        if task["status"] == "complete"
    }
    assert len(completed) == 2
    signatures = {
        key: artifact_signatures(Path(task["directory"]))
        for key, task in completed.items()
    }
    corrupted.unlink()
    run_study(load_study_config(config.output_dir), "predictores", True)
    resumed = read_artifact(config.output_dir / "study.json")
    for task in resumed["preparation_tasks"]:
        if task["key"] in completed:
            assert task == completed[task["key"]]
            assert (
                artifact_signatures(Path(task["directory"])) == signatures[task["key"]]
            )
        else:
            assert len(task["attempts"]) == 2 and task["status"] == "complete"
    assert resource_inventory(config.output_dir).coverage.eq("complete").all()


def test_parallel_study_closure_benchmark_and_relocation(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion import benchmark
    from truss_research.evaluacion.auditoria_estudio import AuditConfig, audit_study
    from truss_research.evaluacion.informe_estudio import summarize_study
    from truss_research.evaluacion.recursos import MeasurementParameters
    from truss_research.evaluacion.sensibilidad import run_sensitivity

    exports = tmp_path / "exports"
    monkeypatch.setattr(benchmark, "benchmark_export_root", lambda study: exports)
    config = replace(
        tiny_parallel_config(design_data, tmp_path / "study"),
        measurement=MeasurementParameters(
            warmup=0, inference_repeats=1, selection_repeats=1, batch_sizes=(1, 4, 8)
        ),
    )
    run_study(config, "validacion")
    moved = tmp_path / "relocated"
    config.output_dir.rename(moved)
    config = load_study_config(moved)
    run_study(config, "todo", True)
    state = read_artifact(moved / "study.json")
    assert state["execution_runs"][-1]["phase"] == "todo"
    assert state["execution_runs"][-1]["status"] == "complete"
    frozen = read_artifact(moved / "freeze.json")
    assert set(
        task["key"]
        for task in state["preparation_tasks"]
        if not task["key"].startswith("test/")
    ) <= set(frozen["artifacts"])
    before = artifact_signatures(moved)
    sensitivity = run_sensitivity(moved, tmp_path / "sensitivity")
    assert (sensitivity / "catalog_summary.csv").is_file()
    result = benchmark.run_benchmark(moved, exports / "benchmark_001")
    report = summarize_study(moved, tmp_path / "report")
    audit_study(AuditConfig(moved, config.data_dir, tmp_path / "audit"))
    assert artifact_signatures(moved) == before
    resources = resource_inventory(moved)
    assert resources.coverage.eq("complete").all()
    jobs = pd.read_csv(result / "worker_jobs.csv")
    assert (
        len(jobs) == len(state["tasks"]) + len(state["preparation_tasks"])
        and jobs.success.all()
    )
    costs = pd.read_csv(result / "research_cost.csv")
    assert costs.campaign_wall_seconds.iloc[0] == pytest.approx(
        sum(run["wall_seconds"] for run in state["execution_runs"])
    )
    effective = pd.read_csv(report / "effective_costs.csv")
    task_seconds = sum(
        task["seconds"] for task in [*state["tasks"], *state["preparation_tasks"]]
    )
    assert effective.loc[
        effective.phase.eq("task_wall_seconds"), "seconds"
    ].sum() == pytest.approx(task_seconds)


def test_parallel_families_and_folds_match_serial_and_resume(
    design_data: Path, tmp_path: Path
) -> None:
    config = StudyConfig(
        tmp_path / "serial",
        data_dir=design_data,
        models=("ridge", "random_forest"),
        seeds=(42, 43),
        outer_folds=2,
        inner_folds=2,
        trials=1,
        catalog_size=4,
        budgets=(2, 3),
        episodes_per_seed=1,
        rl_steps=4,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=2,
        background_size=2,
        permutation_rounds=1,
        analysis=AnalysisParameters(catalog_sizes=(4, 8), bootstrap_repeats=10),
    )
    run_study(config, "predictores")
    parallel = replace(
        config,
        output_dir=tmp_path / "parallel",
        execution=ExecutionParameters(cpu_workers=2),
    )
    run_study(parallel, "predictores")
    serial_state = read_artifact(config.output_dir / "study.json")
    state = read_artifact(parallel.output_dir / "study.json")
    serial_tasks = {task["key"]: task for task in serial_state["tasks"]}
    tasks = {task["key"]: task for task in state["tasks"]}
    preparation = {task["key"]: task for task in state["preparation_tasks"]}
    assert len(preparation) == 6 and all(
        task["status"] == "complete" for task in preparation.values()
    )
    for key, task in tasks.items():
        serial_root, root = (
            Path(serial_tasks[key]["directory"]),
            Path(task["directory"]),
        )
        for filename in ("training_oof.csv", "validation_predictions.csv"):
            left, right = (
                pd.read_csv(serial_root / filename),
                pd.read_csv(root / filename),
            )
            np.testing.assert_allclose(
                left.filter(like="pred_").to_numpy(),
                right.filter(like="pred_").to_numpy(),
                rtol=1e-12,
                atol=1e-12,
            )
            assert left.iter_id.tolist() == right.iter_id.tolist()
        old, new = read_run(serial_root), read_run(root)
        assert old["config"]["parameters"] == new["config"]["parameters"]
        assert old["normalized_mae"] == pytest.approx(new["normalized_mae"])
        assert old["nested_trial_count"] == new["nested_trial_count"]
    events = [
        (task["key"], event)
        for task in [*tasks.values(), *preparation.values()]
        for event in task["worker_history"]
    ]
    pids = {event["pid"] for _, event in events}
    assert 1 < len(pids) <= 2
    overlaps = [
        (left_key, right_key)
        for left_key, left in events
        for right_key, right in events
        if left["pid"] != right["pid"]
        and max(left["started_at"], right["started_at"])
        < min(left["finished_at"], right["finished_at"])
    ]
    assert any(left.split("/")[1] != right.split("/")[1] for left, right in overlaps)
    assert any("outer_" in left and "outer_" in right for left, right in overlaps)
    for model in config.models:
        representative = tasks[f"predictor/{model}/42"]
        assert (
            tasks[f"predictor/{model}/43"]["worker_history"][0]["started_at"]
            >= representative["worker_history"][0]["finished_at"]
        )
        assert all(
            task["worker_history"][0]["finished_at"]
            <= representative["worker_history"][0]["started_at"]
            for key, task in preparation.items()
            if key.startswith(f"predictor/{model}/")
        )
    resources = resource_inventory(parallel.output_dir)
    assert resources.coverage.eq("complete").all()
    assert resources.loc[resources.phase.eq("task_total"), "task"].nunique() == 10
    before = artifact_signatures(parallel.output_dir)
    run_study(load_study_config(parallel.output_dir), "predictores", True)
    assert artifact_signatures(parallel.output_dir) == before
    run = state["execution_runs"][0]
    assert run["status"] == "complete" and run["cpu_workers"] == 2


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_execution_workers_require_positive_integer(workers: int | float) -> None:
    with pytest.raises(ValueError, match="entero positivo"):
        ExecutionParameters(cpu_workers=workers)


def test_all_cpu_families_support_parallel_plain_crossfit(
    design_data: Path, tmp_path: Path
) -> None:
    config = replace(
        tiny_parallel_config(design_data, tmp_path / "study"),
        models=tuple(name for name in MODEL_NAMES if name != "tabpfn"),
        inner_folds=None,
        parameters=ModelParameters(
            trees=3, depth=2, epochs=3, patience=2, hidden_width=8
        ),
    )
    run_study(config, "predictores")
    state = read_artifact(config.output_dir / "study.json")
    assert len(state["tasks"]) == len(config.models)
    for task in state["tasks"]:
        assert task["status"] == "complete"
        predictions = pd.read_csv(
            Path(task["directory"]) / "validation_predictions.csv"
        )
        assert np.isfinite(predictions.filter(like="pred_").to_numpy()).all()
    assert resource_inventory(config.output_dir).coverage.eq("complete").all()
    ridge = next(task for task in state["tasks"] if task["key"] == "predictor/ridge/42")
    reference = tmp_path / "reference"
    train_predictor(
        PredictorConfig(
            design_data,
            reference,
            model="ridge",
            folds=2,
            cv_seed=42,
            parameters=config.parameters,
        )
    )
    assert read_run(Path(ridge["directory"]))["cv_normalized_mae"] == pytest.approx(
        read_run(reference)["cv_normalized_mae"]
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(Path(ridge["directory"]) / "training_oof.csv"),
        pd.read_csv(reference / "training_oof.csv"),
    )
