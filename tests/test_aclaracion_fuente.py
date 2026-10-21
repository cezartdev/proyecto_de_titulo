"""Pruebas cortas de generaciones, criterio de servicio y reinicio controlado."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.particion_agrupada import forward_generation_folds
from truss_research.datos.particion_equilibrada import (
    chronological_generation_split,
    verify_group_exclusion,
)
from truss_research.datos.preparacion import prepare_dataset
from truss_research.datos.procesados import load_structural_limits
from truss_research.datos.schemas import PreparationConfig
from truss_research.evaluacion import reinicio
from truss_research.evaluacion.pareto import ObjectiveSpace, meets_thresholds
from truss_research.evaluacion.protocolo_analisis import GeneralizationParameters
from truss_research.evaluacion.sensibilidad import recompute_reference
from truss_research.optimizacion.catalogos import SelectionBank
from truss_research.optimizacion.contratos import EpisodeSpec
from truss_research.optimizacion.estudio_estado import study_lock
from truss_research.optimizacion.protocolo import load_protocol


@pytest.fixture(scope="module")
def source_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("fuente") / "processed"
    prepare_dataset(
        PreparationConfig(destination=destination, calibration_fraction=0.2)
    )
    return destination


def test_generation_contract_and_real_temporal_counts(source_data: Path) -> None:
    table = pd.read_csv(source_data / "full_clean.csv")
    membership = pd.read_csv(source_data / "design_membership.csv")
    assert len(membership) == 5000 and len(table) == 1273
    assert "is_base" not in table and "problem" not in table
    assert membership.groupby("generation").size().eq(50).all()
    partition = chronological_generation_split(
        table, membership, (42, GeneralizationParameters())
    )
    assert {name: len(value) for name, value in partition.tables.items()} == {
        "train": 855,
        "calibration": 132,
        "validation": 70,
        "test": 108,
    }
    assert len(partition.excluded) == 108
    assert set(partition.tables["test"].web_config) == {1, 2}
    verify_group_exclusion(partition.tables, membership)
    training = partition.tables["train"]
    spans = membership.groupby("representative_iter_id").generation.agg(["min", "max"])
    folds = forward_generation_folds(training, membership, 3)
    assert [(len(fit), len(held)) for fit, held in folds] == [
        (273, 171),
        (473, 172),
        (678, 141),
    ]
    for fit, held in folds:
        assert (
            spans.loc[training.iloc[fit].iter_id, "max"].max()
            < spans.loc[training.iloc[held].iter_id, "min"].min()
        )


def test_serviceability_boundary_and_unbounded_reference(source_data: Path) -> None:
    table = pd.read_csv(source_data / "full_clean.csv")
    limits = load_structural_limits(source_data)
    assert limits.displacement_limit_m == 20 / 700
    assert meets_thresholds(table[list(OUTPUT_COLUMNS)].to_numpy(), limits).sum() == 835
    edge = np.array(
        [[1000, 0.9, 20 / 700], [1000, 0.9, np.nextafter(20 / 700, np.inf)]]
    )
    assert meets_thresholds(edge, limits).tolist() == [True, False]
    original = replace(limits, serviceability_divisor=None)
    assert meets_thresholds(edge, original).all()
    space = ObjectiveSpace.fit(table[list(OUTPUT_COLUMNS)].to_numpy(), original)
    assert np.isfinite(space.reference).all()
    assert space.reference_physical[2] > table.z_disp_m.max()


def test_fixed_three_objective_directions() -> None:
    values = np.array([[1000, 0.5, 0.01], [900, 0.6, 0.009]])
    space = ObjectiveSpace.fit(values)
    np.testing.assert_array_equal(space.front(values), values[1:])
    assert space.hypervolume(values) > 0


def test_serviceability_recomputes_same_queries_without_new_labels() -> None:
    table = pd.DataFrame(
        [[1000, 0.9, 0.04], [1200, 0.8, 0.01]],
        index=["a", "b"],
        columns=OUTPUT_COLUMNS,
    )
    space = ObjectiveSpace.fit(table.to_numpy())
    bank = SelectionBank(
        table,
        table,
        (EpisodeSpec(("a", "b"), 0, 2, 42, "episode"),),
        space,
        "data",
        "predictor",
    )
    traces = pd.DataFrame(
        {
            "episode_id": ["episode"] * 2,
            "comparison_id": ["random"] * 2,
            "query": [1, 2],
            "iter_id": ["a", "b"],
            "strategy": ["random"] * 2,
            "policy_training_seed": [None, None],
        }
    )
    original_space = ObjectiveSpace.fit(
        table.to_numpy(), replace(space.limits, serviceability_divisor=None)
    )
    primary = recompute_reference(bank, traces, space).iloc[0]
    original = recompute_reference(bank, traces, original_space).iloc[0]
    assert primary.observed_failures == 1 and original.observed_failures == 0
    assert primary.queries == original.queries == 2
    assert traces.iter_id.tolist() == ["a", "b"]


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
        tmp_path / "data/processed_designs/full_clean.csv",
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
