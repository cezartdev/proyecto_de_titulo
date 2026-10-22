"""Recursos y benchmark sobre datos sintéticos, sin campañas de investigación."""

import json
from pathlib import Path

import numpy as np
import pytest

from truss_research.datos.artefactos import read_artifact
from truss_research.evaluacion.informe_costos import (
    resource_inventory,
)
from truss_research.evaluacion.recursos import (
    MeasurementParameters,
    ResourceMonitor,
    measure_phase,
    resource_session,
)


def test_monitor_reports_units_and_closes_after_exception(tmp_path: Path) -> None:
    path = tmp_path / "resources.json"
    with pytest.raises(RuntimeError, match="interrumpida"):
        with resource_session(path, MeasurementParameters(memory_interval_ms=1)):
            with measure_phase("fit"):
                allocation = bytearray(1024**2)
                assert len(allocation) == 1024**2
                raise RuntimeError("interrumpida")
    payload = read_artifact(path)
    assert payload["status"] == "failed"
    assert {row["phase"] for row in payload["phases"]} == {"task_total", "fit"}
    for phase in payload["phases"]:
        assert phase["finished"] and phase["status"] == "failed"
        assert phase["wall_seconds"] >= 0 and phase["cpu_seconds"] >= 0
        assert phase["peak_rss_mib"] >= phase["initial_rss_mib"] > 0
        assert phase["samples"] >= 2
    with measure_phase("outside_session"):
        pass


def test_monitor_separates_process_cpu_from_unmeasured_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = ResourceMonitor()
    monkeypatch.setattr(
        monitor.process, "children", lambda recursive: [monitor.process]
    )
    with monitor:
        pass
    assert monitor.snapshot()["cpu_coverage"] == "complete"
    assert monitor.snapshot()["cpu_tree_coverage"] == "unmeasured_children"
    assert monitor.snapshot()["children_seen"]


def test_monitor_excludes_startup_and_final_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from truss_research.evaluacion import recursos

    clock = [0.0]

    def sample() -> None:
        clock[0] += 5.0

    monitor = ResourceMonitor(interval_ms=100_000)
    monkeypatch.setattr(monitor, "sample", sample)
    monkeypatch.setattr(recursos, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(recursos, "process_time", lambda: clock[0])
    with monitor:
        clock[0] += 2.0
    assert clock[0] == 12.0
    assert monitor.wall_seconds == monitor.cpu_seconds == 2.0


@pytest.mark.parametrize(
    "values",
    [
        {"memory_interval_ms": 0},
        {"memory_interval_ms": True},
        {"memory_interval_ms": -1},
        {"memory_interval_ms": 1.5},
        {"memory_interval_ms": "1"},
    ],
)
def test_measurement_parameters_reject_invalid_values(
    values: dict[str, int | float | str],
) -> None:
    with pytest.raises(ValueError):
        MeasurementParameters(**values)


def test_inventory_preserves_missing_and_partial_resources(tmp_path: Path) -> None:
    from truss_research.optimizacion.estudio_estado import STUDY_SCHEMA_VERSION

    path = tmp_path / "study.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": STUDY_SCHEMA_VERSION,
                "tasks": [
                    {
                        "key": "predictor/dummy/42",
                        "attempts": ["attempt_001"],
                        "measurement_files": [str(tmp_path / "missing.json")],
                        "measurement_signatures": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    inventory = resource_inventory(tmp_path)
    assert inventory.coverage.eq("missing").all()
    assert inventory.wall_seconds.isna().all()
    with ResourceMonitor() as monitor:
        pass
    partial = tmp_path / "partial.json"
    snapshot = {
        "status": "running",
        "phases": [{"phase": "task_total", **monitor.snapshot(), "finished": False}],
    }
    partial.write_text(json.dumps(snapshot), encoding="utf-8")
    state = json.loads(path.read_text(encoding="utf-8"))
    state["tasks"][0]["measurement_files"] = [str(partial)]
    path.write_text(json.dumps(state), encoding="utf-8")
    inventory = resource_inventory(tmp_path)
    assert inventory.coverage.eq("incomplete").all()
    assert np.isfinite(inventory.wall_seconds).all()


def test_cli_benchmark_missing_study_has_no_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from truss_research.__main__ import main

    missing = tmp_path / "missing"
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "truss_research",
            "medir-costos",
            "--estudio",
            str(missing),
            "--destino",
            str(output),
        ],
    )
    assert main() == 1
    assert not missing.exists() and not output.exists()
