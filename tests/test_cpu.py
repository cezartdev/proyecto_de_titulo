"""Calibración breve, presión de RAM y equivalencia completa del pool de campaña."""

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from truss_research.evaluacion.telemetria_cpu import CampaignMonitor
from truss_research.optimizacion import ejecucion_cpu
from truss_research.optimizacion.ejecucion_cpu import (
    ExecutionParameters,
    MemoryAdmission,
)


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
