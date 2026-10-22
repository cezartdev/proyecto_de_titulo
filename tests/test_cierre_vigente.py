"""Regresiones del cierre vigente sin datos privados ni entrenamientos de campaña."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.informe_costos import export_resource_report
from truss_research.evaluacion.recorrido import pipeline_stages
from truss_research.optimizacion.estudio import validate_training_protocol
from truss_research.optimizacion.estudio_estado import StudySession, TaskRecord
from truss_research.optimizacion.protocolo import load_protocol
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.especializacion import TargetwiseRegressor
from truss_research.surrogates.experimentos import predictor_devices, write_json
from truss_research.surrogates.tabpfn import TabPFNParameters


@pytest.mark.parametrize(
    "payload",
    [
        {"study": {"experiment": "catalog"}},
        {"study": {"catalog_size": 32}},
        {"evolution": {"proposal_pool": 8}},
        {"rl": {"algorithms": ["ppo"]}},
        {"adaptive": {}},
        {"analysis": {"learning_curves": {"enabled": True}}},
    ],
)
def test_retired_protocol_fields_are_rejected(
    tmp_path: Path, payload: dict[str, JsonValue]
) -> None:
    source = tmp_path / "retired.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_protocol(source, (tmp_path / "study", tmp_path / "data"))


def test_budget_exhaustion_fails_before_training(
    prepared_temporal: Path, tmp_path: Path
) -> None:
    config = replace(
        load_protocol(
            Path("config/smoke.json"), (tmp_path / "study", prepared_temporal)
        ),
        models=("dummy", "ridge"),
    )
    validate_training_protocol(config)
    with pytest.raises(ValueError, match="agota un catálogo"):
        validate_training_protocol(
            replace(
                config,
                budgets=(128,),
                proposals=replace(config.proposals, max_evaluations=256),
            )
        )
    assert not config.output_dir.exists()


def test_devices_describe_actual_targetwise_components() -> None:
    cuda = PredictorConfig(model="tabpfn", tabpfn=TabPFNParameters(device="cuda"))
    cpu = PredictorConfig(model="ridge")
    settings = PredictorConfig(model="targetwise", tabpfn=cuda.tabpfn)
    regressor = TargetwiseRegressor(settings)
    component = Pipeline([("regressor", DummyRegressor())])
    regressor.components_ = [(cpu, component), (cuda, component), (cpu, component)]
    assert predictor_devices(Pipeline([("regressor", regressor)]), settings) == {
        "mass_kg": "cpu",
        "util_ratio": "cuda",
        "z_disp_m": "cpu",
    }
    assert set(predictor_devices(component, cuda).values()) == {"cuda"}


def test_full_pipeline_includes_shap_before_exports() -> None:
    stages = pipeline_stages()
    assert stages.index("final") < stages.index("explain") < stages.index("report")
    assert "explain" not in pipeline_stages(mode="check")


def test_resource_export_keeps_all_attempts(
    prepared_temporal: Path, tmp_path: Path
) -> None:
    study, report = tmp_path / "study", tmp_path / "report"
    study.mkdir()
    report.mkdir()
    config = load_protocol(Path("config/smoke.json"), (study, prepared_temporal))
    task = TaskRecord("predictor/ridge/42", status="complete")
    for number in (1, 2):
        destination = study / f"attempt_{number}.resources.json"
        write_json(
            destination,
            {
                "phases": [
                    {
                        "phase": "task_total",
                        "wall_seconds": float(number),
                        "cpu_seconds": number / 2,
                        "peak_rss_mib": 100 * number,
                        "finished": number == 2,
                        "cpu_coverage": "complete",
                        "memory_coverage": "sampled",
                    }
                ]
            },
        )
        task.attempts.append(str(study / f"attempt_{number}"))
        task.measurement_files.append(str(destination))
        task.measurement_signatures[str(destination)] = source_digest(destination)
    session = StudySession(config, "synthetic", "synthetic", {}, tasks={task.key: task})
    session.save()
    assert export_resource_report(study, report) is False
    row = pd.read_csv(report / "resource_tasks.csv").iloc[0]
    assert row.attempts == 2 and row.wall_seconds == 3 and row.cpu_seconds == 1.5
    assert row.peak_rss_mib == 200
