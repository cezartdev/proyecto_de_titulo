"""Recuperación de procesos y continuidad estricta del código operativo registrado."""

import os
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

import psutil
import pytest

from truss_research.optimizacion import ejecucion_cpu
from truss_research.optimizacion.ejecucion_cpu import (
    ExecutionParameters,
    MemoryAdmission,
    cpu_pool,
)

RETAINED: list[bytearray] = []
WORKER_TIMEOUT = 30
POLL_SECONDS = 0.02
ALLOCATION_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class MemorySnapshot:
    percent: float


def retain_memory(size: int) -> int:
    RETAINED.append(bytearray(size))
    return os.getpid()


def wait_for_marker(marker: Path) -> int:
    marker.write_text(str(os.getpid()), encoding="utf-8")
    sleep(0.5)
    return os.getpid()


def test_recovery_terminates_idle_worker_and_keeps_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with cpu_pool(ExecutionParameters(cpu_workers=2)) as pool:
        original = pool.submit(retain_memory, ALLOCATION_BYTES).result(
            timeout=WORKER_TIMEOUT
        )
        assert psutil.pid_exists(original)

        def snapshot() -> MemorySnapshot:
            return MemorySnapshot(90.0 if psutil.pid_exists(original) else 40.0)

        monkeypatch.setattr(ejecucion_cpu.psutil, "virtual_memory", snapshot)
        gate = MemoryAdmission(ExecutionParameters(cpu_workers=2))
        deadline = monotonic() + WORKER_TIMEOUT
        while not pool.admit(gate, 0) and monotonic() < deadline:
            sleep(POLL_SECONDS)
        assert not psutil.pid_exists(original)
        assert pool.capacity == 1 and len(pool.recoveries) == 1
        replacement = pool.submit(retain_memory, ALLOCATION_BYTES).result(
            timeout=WORKER_TIMEOUT
        )
        assert replacement != original and psutil.pid_exists(replacement)
        assert pool.recoveries[0]["ram_after_percent"] == 40.0
    assert not psutil.pid_exists(replacement) and psutil.pid_exists(os.getpid())


def test_recovery_never_recycles_a_running_job(tmp_path: Path) -> None:
    with cpu_pool(ExecutionParameters(cpu_workers=2)) as pool:
        marker = tmp_path / "running.pid"
        future = pool.submit(wait_for_marker, marker)
        deadline = monotonic() + WORKER_TIMEOUT
        while not marker.exists() and monotonic() < deadline:
            sleep(POLL_SECONDS)
        assert marker.exists()
        assert pool.recycle() is False
        assert future.result(timeout=WORKER_TIMEOUT) == int(marker.read_text())
        assert pool.capacity == 2 and not pool.recoveries


def test_external_pressure_stops_after_one_idle_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(ejecucion_cpu, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        ejecucion_cpu.psutil, "virtual_memory", lambda: MemorySnapshot(90.0)
    )
    with cpu_pool(ExecutionParameters(cpu_workers=2)) as pool:
        gate = MemoryAdmission(ExecutionParameters(cpu_workers=2))
        assert not pool.admit(gate, 1)
        clock[0] = 1000.0
        assert not pool.admit(gate, 0)
        assert len(pool.recoveries) == 1 and gate.blocked_since == clock[0]
        clock[0] += ejecucion_cpu.MEMORY_WAIT_SECONDS - 1
        assert not pool.admit(gate, 0) and len(pool.recoveries) == 1
        clock[0] += 1
        with pytest.raises(RuntimeError, match="sin tareas activas"):
            pool.admit(gate, 0)
