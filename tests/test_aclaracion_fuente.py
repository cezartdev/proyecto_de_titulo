"""Pruebas cortas de generaciones, criterio de servicio y reinicio controlado."""

import json
from pathlib import Path

import numpy as np
import pytest

from truss_research.evaluacion import reinicio
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.estudio_estado import study_lock
from truss_research.optimizacion.protocolo import load_protocol


def test_fixed_three_objective_directions() -> None:
    values = np.array([[1000, 0.5, 0.01], [900, 0.6, 0.009]])
    space = ObjectiveSpace.fit(values)
    np.testing.assert_array_equal(space.front(values), values[1:])
    assert space.hypervolume(values) > 0


def test_protocol_limits_propagate_and_unknown_fields_fail(tmp_path: Path) -> None:
    protocol = tmp_path / "study.json"
    protocol.write_text(json.dumps({"limits": {"serviceability_divisor": 360}}))
    config = load_protocol(protocol, (tmp_path / "study", tmp_path / "data"))
    assert config.limits.displacement_limit_m == 20 / 360
    protocol.write_text(json.dumps({"limits": {"unknown": 1}}))
    with pytest.raises(ValueError):
        load_protocol(protocol, (tmp_path / "study", tmp_path / "data"))


def test_reset_preserves_sources_weights_and_other_studies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reinicio, "WORKSPACE", tmp_path)
    study = tmp_path / "artifacts/estudio"
    protected = tmp_path / "artifacts/pretrained/tabpfn"
    protocol = tmp_path / "config.json"
    protocol.write_text(json.dumps({"tabpfn": {"checkpoint_dir": str(protected)}}))
    keep = [
        tmp_path / "data/raw/source.xlsx",
        protected / "weights.ckpt",
        tmp_path / "artifacts/otro/keep.txt",
    ]
    discard = [
        study / "study.json",
        tmp_path / "data/processed_evolution/full_clean.csv",
        protected / "pilots/pilot.json",
    ]
    for path in [*keep, *discard]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    reinicio.reset_study(study, protocol, True)
    assert all(path.exists() for path in discard)
    with study_lock(study), pytest.raises(ValueError, match="activa"):
        reinicio.reset_study(study, protocol)
    assert all(path.exists() for path in discard)
    reinicio.reset_study(study, protocol)
    assert all(path.exists() for path in keep)
    assert not any(path.exists() for path in discard)
    for invalid in [tmp_path, tmp_path / "data/raw", tmp_path / "artifacts", protected]:
        with pytest.raises(ValueError):
            reinicio.reset_targets(invalid, protocol)
