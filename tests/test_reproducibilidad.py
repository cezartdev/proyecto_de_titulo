"""Portabilidad, planificación sin efectos y exclusión estricta de pliegues."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.__main__ import main
from truss_research.datos.artefactos import dataset_identity, read_artifact
from truss_research.datos.procesados import load_design_split
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.optimizacion.protocolo import load_protocol, study_plan
from truss_research.surrogates.configuracion import PredictorConfig
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


def test_atomic_artifact_survives_block_longer_than_previous_retry_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.datos import artefactos

    original = Path.replace
    delays: list[float] = []

    def temporarily_blocked(path: Path, target: Path) -> Path:
        if len(delays) < 8:
            raise PermissionError("bloqueo prolongado simulado")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", temporarily_blocked)
    monkeypatch.setattr(artefactos, "sleep", delays.append)
    target = tmp_path / "run.json"
    write_json(target, {"value": 42})
    assert read_artifact(target)["value"] == 42
    assert len(delays) == 8 and sum(delays) > 0.62
    assert max(delays) <= artefactos.REPLACE_MAX_RETRY_SECONDS


def test_digest_session_reads_once_and_invalidates_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from truss_research.datos import preparacion

    target = tmp_path / "artifact.bin"
    target.write_bytes(b"original")
    original = hashlib.file_digest
    calls: list[str] = []

    def measured(stream, algorithm):
        calls.append(algorithm)
        return original(stream, algorithm)

    monkeypatch.setattr(hashlib, "file_digest", measured)
    with preparacion.digest_session():
        first = preparacion.source_digest(target)
        assert preparacion.source_digest(target) == first
        assert len(calls) == 1
        target.write_bytes(b"modified artifact")
        assert preparacion.source_digest(target) != first
        assert len(calls) == 2
    with preparacion.digest_session():
        preparacion.source_digest(target)
    assert len(calls) == 3


def test_digest_rejects_file_changed_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from truss_research.datos import preparacion

    target = tmp_path / "artifact.bin"
    target.write_bytes(b"original")
    original = hashlib.file_digest

    def changed(stream, algorithm):
        digest = original(stream, algorithm)
        target.write_bytes(b"changed during verification")
        return digest

    monkeypatch.setattr(hashlib, "file_digest", changed)
    with preparacion.digest_session(), pytest.raises(ValueError, match="durante"):
        preparacion.source_digest(target)


def test_operational_revision_preserves_identity_and_rejects_other_sources(
    prepared_temporal: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.datos.preparacion import source_digest
    from truss_research.evaluacion.busqueda_evolutiva import campaign_identity
    from truss_research.optimizacion import estudio_estado as state

    config = state.StudyConfig(tmp_path / "study", data_dir=prepared_temporal)
    config.output_dir.mkdir()
    monkeypatch.setattr(state, "source_signature", lambda: "previous")
    state.open_session(config, False)
    identity = campaign_identity(config, {})
    monkeypatch.setattr(state, "source_signature", lambda: "io_revision")
    with pytest.raises(ValueError, match="código"):
        state.open_session(config, True)
    receipt = config.output_dir / "maintenance/compatibility.json"
    receipt.parent.mkdir()
    write_json(
        receipt,
        {
            "scope": "io_only",
            "previous_source_signature": "previous",
            "current_source_signature": "io_revision",
        },
    )
    saved = state.read_study(config.output_dir)
    saved["source_compatibility"] = {
        "file": "maintenance/compatibility.json",
        "signature": source_digest(receipt),
    }
    write_json(config.output_dir / "study.json", saved)
    session = state.open_session(config, True)
    session.save()
    assert session.source_signature == "previous"
    assert state.study_source_signature(config.output_dir) == "previous"
    assert campaign_identity(config, {}) == identity
    assert (
        state.read_study(config.output_dir)["source_compatibility"]
        == saved["source_compatibility"]
    )
    monkeypatch.setattr(state, "source_signature", lambda: "unreviewed_science_change")
    with pytest.raises(ValueError, match="revisión operativa"):
        state.open_session(config, True)
    monkeypatch.setattr(state, "source_signature", lambda: "io_revision")
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="evidencia"):
        state.open_session(config, True)


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
    assert result["primary_component"] == "predictor_comparison"
    assert result["policy_transitions"] == 1_000_000
    assert result["execution"]["numeric_threads_per_worker"] == 1
    assert not target.exists()


def test_profiles_and_unknown_settings(tmp_path: Path) -> None:
    locations = (tmp_path / "study", tmp_path / "data")
    full = load_protocol(Path("config/study.json"), locations)
    assert full.label == "posterior" and full.budgets == (32, 64, 96)
    assert full.experiment == "evolution" and full.split_protocol == "first_appearance"
    smoke = load_protocol(Path("config/smoke.json"), locations)
    assert smoke.label == "comprobacion" and smoke.rl_steps == 100
    assert (
        full.execution.cpu_workers in (4, 6, 8, 12) and smoke.execution.cpu_workers == 2
    )
    nested = load_protocol(Path("config/study.json"), locations)
    assert nested.nested_folds == 3
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


def test_relative_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    payload = {
        "config": {"data_dir": str(tmp_path.parent / "data")},
        "directory": str(tmp_path / "task"),
    }
    write_json(path, payload)
    assert str(tmp_path) not in path.read_text(encoding="utf-8")
    assert read_artifact(path) == payload
