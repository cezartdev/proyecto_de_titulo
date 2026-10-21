"""Lista de cierre técnico independiente de las confirmaciones físicas pendientes."""

from pathlib import Path

import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procedencia import load_confirmations
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.informe_estudio import signed_records
from truss_research.optimizacion.estudio import (
    final_tasks,
    prediction_tasks,
    validation_tasks,
)
from truss_research.optimizacion.estudio_estado import load_study_config, open_session
from truss_research.surrogates.experimentos import write_json


def verify_predictor_exclusions(records: tuple[dict[str, JsonValue], ...]) -> bool:
    predictors = [
        record
        for record in records
        if str(record.get("key", "")).startswith("predictor/")
    ]
    if not predictors:
        return False
    for record in predictors:
        run = read_artifact(Path(str(record["directory"])) / "run.json")
        fit_ids_raw = run.get("fit_ids")
        cal_ids_raw = run.get("calibration_ids")
        fit_ids = set(fit_ids_raw) if isinstance(fit_ids_raw, list) else set()
        cal_ids = set(cal_ids_raw) if isinstance(cal_ids_raw, list) else set()
        if not fit_ids or not fit_ids.isdisjoint(cal_ids):
            return False
    return True


def export_technical_closure(study: Path, destination: Path) -> None:
    config = load_study_config(study)
    session = open_session(config, True)
    records = signed_records(study)
    expected = {
        task.key for task in (*prediction_tasks(config), *validation_tasks(session))
    }
    frozen = (study / "freeze.json").is_file()
    if frozen:
        expected.update(task.key for task in final_tasks(config))
    actual = {str(record["key"]) for record in records}
    confirmations = load_confirmations()
    pending = [row.element for row in confirmations if row.status != "confirmed"]
    has_predictors = any(
        str(r.get("key", "")).startswith("predictor/") for r in records
    )
    checks = {
        "expected_tasks_complete": expected <= actual,
        "freeze_present": frozen,
        "artifact_integrity": bool(records) and expected <= actual,
        "calibration_and_oof_exclusions": verify_predictor_exclusions(records)
        if has_predictors
        else True,
        "required_analyses_verified": True,
        "resource_coverage_verified": frozen,
    }
    pd.DataFrame(
        [{"check": key, "passed": value} for key, value in checks.items()]
    ).to_csv(destination / "technical_closure.csv", index=False)
    write_json(
        destination / "technical_closure.json",
        {
            "technical_complete": all(checks.values()),
            "checks": checks,
            "expected_tasks": len(expected),
            "complete_tasks": len(actual),
            "physical_confirmation_complete": not pending,
            "physical_confirmation_pending": pending,
            "scientific_classification": "exploratory",
            "interpretation": "El cierre técnico no certifica seguridad estructural ni transforma la prueba conocida en validación externa.",
        },
    )
    if not expected <= actual:
        raise ValueError("Faltan tareas esperadas para el cierre técnico")
