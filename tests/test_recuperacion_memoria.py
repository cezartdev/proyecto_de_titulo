"""Recuperación de procesos y continuidad estricta del código operativo registrado."""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic, sleep

import psutil
import pytest

from truss_research.datos.artefactos import read_artifact
from truss_research.optimizacion import compatibilidad_ejecucion as compatibility
from truss_research.optimizacion import ejecucion_cpu
from truss_research.optimizacion import estudio_estado as state
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


@pytest.fixture
def patched_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, str]:
    root = tmp_path / "code"
    filename = root / "src/truss_research/optimizacion/ejecucion_cpu.py"
    filename.parent.mkdir(parents=True)
    filename.write_text("original\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("project\n", encoding="utf-8")
    (root / "uv.lock").write_text("lock\n", encoding="utf-8")
    before = compatibility.source_inventory(root)
    previous = compatibility.inventory_signature(before)
    filename.write_text("recovered\n", encoding="utf-8")
    after = compatibility.source_inventory(root)
    current = compatibility.inventory_signature(after)
    patch = compatibility.ExecutionPatch(
        previous,
        current,
        compatibility.PATCH_SCOPE,
        "Recuperación de memoria del pool CPU",
        {
            filename.relative_to(root).as_posix(): {
                "before": before[filename.relative_to(root).as_posix()],
                "after": after[filename.relative_to(root).as_posix()],
            }
        },
    )
    path = root / compatibility.PATCH_FILE
    path.parent.mkdir()
    path.write_text(json.dumps({"patches": [asdict(patch)]}), encoding="utf-8")
    monkeypatch.setattr(compatibility, "REPOSITORY_ROOT", root)
    return root, previous, current


def test_compatibility_rejects_unregistered_and_hidden_changes(
    patched_sources: tuple[Path, str, str],
) -> None:
    root, previous, current = patched_sources
    assert compatibility.compatible_sources(current) == (previous,)
    assert not compatibility.compatible_sources("unregistered")
    scientific = root / "src/model.py"
    scientific.write_text("changed model\n", encoding="utf-8")
    changed = compatibility.inventory_signature(compatibility.source_inventory(root))
    assert not compatibility.compatible_sources(changed)
    path = root / compatibility.PATCH_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["patches"][0]["to_signature"] = changed
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reconstruye exactamente"):
        compatibility.compatible_sources(changed)


def test_resume_preserves_completed_tasks_and_original_provenance(
    design_data: Path,
    patched_sources: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, previous, current = patched_sources
    config = state.StudyConfig(
        root / "study", data_dir=design_data, models=("dummy",), seeds=(42,)
    )
    config.output_dir.mkdir()
    monkeypatch.setattr(state, "source_signature", lambda: previous)
    session = state.open_session(config, False)
    completed = root / "completed"
    completed.mkdir()
    (completed / "model.bin").write_bytes(b"original fitted model")
    session.tasks["predictor/dummy/42"] = state.TaskRecord(
        "predictor/dummy/42",
        status="complete",
        attempts=[str(completed)],
        directory=str(completed),
        signatures=state.artifact_signatures(completed),
    )
    session.tasks["pending"] = state.TaskRecord("pending")
    session.save()
    records = {key: asdict(task) for key, task in session.tasks.items()}
    monkeypatch.setattr(state, "source_signature", lambda: current)
    resumed = state.open_session(config, True)
    assert {key: asdict(task) for key, task in resumed.tasks.items()} == records
    assert resumed.source_history[0]["from_signature"] == previous
    resumed.save()
    assert len(state.open_session(config, True).source_history) == 1
    assert (
        read_artifact(config.output_dir / "study.json")["source_signature"] == current
    )
    (completed / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artefactos"):
        state.verify_task(resumed.tasks["predictor/dummy/42"])


def test_run_all_reuses_only_verified_operational_patch(
    patched_sources: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion import campana

    root, previous, current = patched_sources
    args = argparse.Namespace(
        mode="check",
        estudio=root / "check",
        config=Path("config/smoke.json"),
        cpu_workers=None,
    )
    flow = campana.Campaign(args)
    flow.root.mkdir(parents=True)
    output = root / "result.csv"
    output.write_text("verified", encoding="utf-8")
    monkeypatch.setattr(
        flow, "outputs", lambda stage: {str(output): campana.source_digest(output)}
    )
    monkeypatch.setattr(campana, "source_signature", lambda: previous)
    flow.complete("prepare", 1.0)
    monkeypatch.setattr(campana, "source_signature", lambda: current)
    assert flow.valid("prepare")
    monkeypatch.setattr(campana, "source_signature", lambda: "unregistered")
    assert not flow.valid("prepare")
    output.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="salida completa"):
        flow.valid("prepare")


def test_final_freeze_preserves_and_checks_source_history(
    design_data: Path,
    patched_sources: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from truss_research.optimizacion.estudio_cierre import (
        freeze_study,
        verify_frozen_study,
    )

    root, previous, current = patched_sources
    config = state.StudyConfig(
        root / "freeze_case", data_dir=design_data, models=("dummy",), seeds=(42,)
    )
    config.output_dir.mkdir()
    monkeypatch.setattr(state, "source_signature", lambda: previous)
    session = state.open_session(config, False)
    monkeypatch.setattr(state, "source_signature", lambda: current)
    session = state.open_session(config, True)
    freeze_study(session, config.output_dir)
    frozen = read_artifact(config.output_dir / "freeze.json")
    assert frozen["source_history"] == session.source_history
    verify_frozen_study(config.output_dir)
    session.source_history.clear()
    session.save()
    with pytest.raises(ValueError, match="congelado"):
        verify_frozen_study(config.output_dir)
    session.source_signature = previous
    session.save()
    with pytest.raises(ValueError, match="congelado"):
        state.open_session(config, True)
