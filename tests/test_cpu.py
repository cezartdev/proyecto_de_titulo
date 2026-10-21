"""Calibración breve, presión de RAM y equivalencia completa del pool de campaña."""

from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from test_paralelismo import tiny_parallel_config

from truss_research.datos.artefactos import read_artifact
from truss_research.evaluacion.benchmark_cpu import (
    CpuProfile,
    collect_futures,
    select_workers,
)
from truss_research.evaluacion.carga_cpu import (
    CALIBRATION_KINDS,
    CpuCalibration,
    CpuWork,
    execute_cpu_work,
)
from truss_research.evaluacion.informe_costos import resource_inventory
from truss_research.evaluacion.telemetria_cpu import CampaignMonitor
from truss_research.optimizacion import ejecucion_cpu
from truss_research.optimizacion.comparacion_paralela import (
    ComparisonCommand,
    validate_blocks,
)
from truss_research.optimizacion.contratos import EvolutionParameters, SelectionConfig
from truss_research.optimizacion.ejecucion_cpu import (
    ExecutionParameters,
    MemoryAdmission,
)
from truss_research.optimizacion.estudio import run_study
from truss_research.optimizacion.estudio_estado import artifact_signatures


@dataclass(frozen=True)
class MemorySnapshot:
    percent: float


def test_memory_admission_recovers_and_stops_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10.0]
    monkeypatch.setattr(ejecucion_cpu, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        ejecucion_cpu.psutil, "virtual_memory", lambda: MemorySnapshot(90.0)
    )
    gate = MemoryAdmission(ExecutionParameters())
    assert not gate.allow(1)
    clock[0] += ejecucion_cpu.MEMORY_WAIT_SECONDS * 2
    assert not gate.allow(1) and gate.blocked_since is None
    assert not gate.allow(0) and gate.blocked_since == clock[0]
    clock[0] += ejecucion_cpu.MEMORY_WAIT_SECONDS - 1
    assert not gate.allow(0)
    clock[0] += 1
    with pytest.raises(RuntimeError, match="Memoria insuficiente"):
        gate.allow(0)
    monkeypatch.setattr(
        ejecucion_cpu.psutil, "virtual_memory", lambda: MemorySnapshot(40.0)
    )
    assert gate.allow(0) and gate.blocked_since is None


@pytest.mark.parametrize(
    "parameters",
    [
        {"memory_limit_fraction": 1},
        {"memory_limit_fraction": True},
        {"telemetry_interval_seconds": 0},
    ],
)
def test_invalid_cpu_limits(parameters: dict[str, int | bool]) -> None:
    with pytest.raises(ValueError):
        ExecutionParameters(**parameters)


def test_cpu_selection_uses_complete_confirmation_and_tolerance() -> None:
    profiles = [
        CpuProfile(
            "confirmation", 6, 512, status="complete", durations=[1.02, 1.03, 1.04]
        ),
        CpuProfile(
            "confirmation", 8, 512, status="complete", durations=[0.99, 1.0, 1.01]
        ),
        CpuProfile("confirmation", 12, 512, status="incomplete", durations=[0.1]),
    ]
    assert select_workers(profiles, CpuCalibration()) == (6, False)
    assert select_workers(profiles[:1], CpuCalibration()) == (4, True)


def test_cpu_deadline_does_not_accept_unfinished_future(tmp_path: Path) -> None:
    from time import monotonic

    future: Future[int] = Future()
    monitor = CampaignMonitor(tmp_path / "unused.csv")
    with pytest.raises(TimeoutError, match="incompleta"):
        collect_futures([future], (monotonic() - 1, monitor, CpuCalibration()))
    assert not (tmp_path / "unused.csv").exists()


@pytest.mark.parametrize("kind", CALIBRATION_KINDS)
def test_small_cpu_workload_uses_real_models_and_finite_outputs(kind: str) -> None:
    protocol = CpuCalibration(trees=1, epochs=1, updates=1)
    result = execute_cpu_work(CpuWork(kind, 32, 42, protocol))
    assert result.wall_seconds > 0 and result.cpu_seconds >= 0
    assert np.isfinite(result.values).all()


def test_telemetry_records_occupancy_and_memory(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.csv"
    with CampaignMonitor(path, interval=10.0) as monitor:
        monitor.update((2, 5, 1, False))
        monitor.sample()
    table = pd.read_csv(path)
    assert table.active_workers.max() == 2
    assert table.available_ram_mib.gt(0).all()
    assert table.tree_cpu_seconds.ge(0).all()
    assert not monitor.errors


def test_full_parallel_campaign_matches_serial_and_reuses_one_pool(
    design_data: Path, tmp_path: Path
) -> None:
    parallel = replace(
        tiny_parallel_config(design_data, tmp_path / "parallel"),
        seeds=(42, 43),
        budgets=(3,),
        episodes_per_seed=2,
        evolution=EvolutionParameters(enabled=True, population_size=2),
        rl_steps=8,
        validation_interval=4,
    )
    serial = replace(
        parallel,
        output_dir=tmp_path / "serial",
        execution=ExecutionParameters(cpu_workers=1),
    )
    run_study(serial, "todo")
    run_study(parallel, "todo")
    left, right = (
        read_artifact(serial.output_dir / "study.json"),
        read_artifact(parallel.output_dir / "study.json"),
    )
    reference = {task["key"]: task for task in left["tasks"]}
    records = {task["key"]: task for task in right["tasks"]}
    assert reference.keys() == records.keys()
    for key, task in records.items():
        root, old = Path(task["directory"]), Path(reference[key]["directory"])
        if key.startswith("policy/"):
            expected, actual = (
                torch.load(old / "policy.pt", weights_only=True),
                torch.load(root / "policy.pt", weights_only=True),
            )
            for name in expected:
                torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
        for filename in ("summary.csv", "traces.csv", "shap_values.csv"):
            if not (root / filename).exists():
                continue
            expected, actual = pd.read_csv(old / filename), pd.read_csv(root / filename)
            columns = [
                name for name in expected.columns if not name.endswith("seconds")
            ]
            pd.testing.assert_frame_equal(
                expected[columns], actual[columns], rtol=1e-12, atol=1e-12
            )
    all_records = [*right["tasks"], *right["preparation_tasks"]]
    events = [
        (task["key"], event) for task in all_records for event in task["worker_history"]
    ]
    assert len({event["pid"] for _, event in events}) == 2
    policy_events = [event for key, event in events if key.startswith("policy/")]
    assert any(
        first["pid"] != second["pid"]
        and max(first["started_at"], second["started_at"])
        < min(first["finished_at"], second["finished_at"])
        for first in policy_events
        for second in policy_events
    )
    assert (
        len(
            [
                task
                for task in right["preparation_tasks"]
                if task["key"].startswith("decision/validation/block_")
            ]
        )
        == 4
    )
    assert resource_inventory(parallel.output_dir).coverage.eq("complete").all()
    preparation = {task["key"]: task for task in right["preparation_tasks"]}
    bank = joblib.load(
        Path(preparation["decision/validation/bank"]["directory"]) / "bank.joblib"
    )
    blocks = tuple(
        Path(task["directory"])
        for key, task in preparation.items()
        if key.startswith("decision/validation/block_")
    )
    settings = SelectionConfig(
        parallel.data_dir,
        Path(read_artifact(parallel.output_dir / "freeze.json")["winner"]),
        tmp_path / "unused",
        catalog_size=4,
        budgets=(3,),
        seeds=(42, 43),
        episodes_per_seed=2,
        evolution=parallel.evolution,
    )
    with pytest.raises(ValueError, match="cubren exactamente"):
        validate_blocks(ComparisonCommand(settings, blocks=(*blocks, blocks[0])), bank)
    with pytest.raises(ValueError, match="cubren exactamente"):
        validate_blocks(ComparisonCommand(settings, blocks=blocks[:-1]), bank)
    signatures = artifact_signatures(parallel.output_dir)
    run_study(parallel, "todo", True)
    assert artifact_signatures(parallel.output_dir) == signatures
