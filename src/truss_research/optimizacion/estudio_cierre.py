"""Selección por familia y cierre inmutable de todas las repeticiones."""

from dataclasses import asdict
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from time import perf_counter

import pandas as pd

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procesados import load_design_split, load_structural_limits
from truss_research.evaluacion.metricas import regional_metrics, regression_metrics
from truss_research.evaluacion.pareto import compliance_report
from truss_research.evaluacion.recursos import measure_phase
from truss_research.optimizacion.estudio_estado import (
    STUDY_SCHEMA_VERSION,
    StudySession,
    atomic_json,
    read_study,
    verify_task,
)
from truss_research.surrogates.configuracion import (
    MODEL_NAMES,
    normalized_mae,
    response_scale,
)
from truss_research.surrogates.experimentos import (
    data_signature,
    load_predictor,
    predict,
    prediction_report,
    read_run,
    timing_benchmark,
    write_json,
)


def predictor_directories(session: StudySession) -> tuple[Path, ...]:
    expected = [
        f"predictor/{model}/{seed}"
        for model in session.config.models
        for seed in session.config.seeds
    ]
    if any(key not in session.tasks for key in expected):
        raise ValueError("Faltan ejecuciones predictivas para seleccionar la familia")
    return tuple(
        verify_task(session.tasks[f"predictor/{model}/{seed}"])
        for model in session.config.models
        for seed in session.config.seeds
    )


def family_selection(session: StudySession) -> tuple[Path, pd.DataFrame]:
    rows = []
    for directory in predictor_directories(session):
        run = read_run(directory)
        settings = run["config"]
        if (
            not isinstance(settings, dict)
            or run.get("evaluation_split") != "validation"
        ):
            raise ValueError("La selección requiere resultados de validación")
        if run.get("dataset_signature") != session.dataset_signature:
            raise ValueError("Los predictores no comparten datos")
        score = float(str(run["normalized_mae"]))
        if not isfinite(score):
            raise ValueError("El criterio predictivo debe ser finito")
        rows.append(
            {"model": settings["model"], "seed": settings["seed"], "score": score}
        )
    summary = pd.DataFrame(rows).groupby("model")["score"].agg(["mean", "std", "count"])
    summary["order"] = [MODEL_NAMES.index(name) for name in summary.index]
    summary = summary.sort_values(["mean", "order"])
    winner_name = str(summary.index[0])
    winner = verify_task(
        session.tasks[f"predictor/{winner_name}/{session.config.representative_seed}"]
    )
    return winner, summary.reset_index()


def freeze_study(session: StudySession, winner: Path) -> None:
    """Se ejecuta antes de leer respuestas de prueba para calcular métricas."""
    signatures = {
        key: {"directory": task.directory, "signatures": task.signatures}
        for key, task in session.tasks.items()
        if not key.startswith("test/")
    }
    signatures.update(
        {
            key: {"directory": task.directory, "signatures": task.signatures}
            for key, task in session.preparation_tasks.items()
            if not key.startswith("test/")
        }
    )
    for key, task in session.preparation_tasks.items():
        if not key.startswith("test/"):
            verify_task(task)
    for key, task in session.tasks.items():
        if not key.startswith("test/"):
            verify_task(task)
    expected = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "config": asdict(session.config),
        "dataset_signature": session.dataset_signature,
        "source_signature": session.source_signature,
        "versions": session.versions,
        "winner": str(winner),
        "artifacts": signatures,
    }
    if session.source_history:
        expected["source_history"] = session.source_history
    path = session.config.output_dir / "freeze.json"
    if path.exists():
        import json

        existing = read_artifact(path)
        existing.pop("frozen_at_utc")
        if existing != json.loads(json.dumps(expected, default=str)):
            raise ValueError("El protocolo difiere del cierre congelado")
    else:
        expected["frozen_at_utc"] = datetime.now(UTC).isoformat()
        atomic_json(path, expected)
    session.test_started = True
    session.save()


def evaluate_test_predictor(directory: Path, destination: Path) -> None:
    metadata = read_run(directory)
    settings = metadata["config"]
    if not isinstance(settings, dict):
        raise ValueError("Falta configuración del predictor congelado")
    data_dir = Path(str(settings["data_dir"]))
    if metadata["dataset_signature"] != data_signature(data_dir):
        raise ValueError("Cambió el dataset antes del cierre")
    split = load_design_split(data_dir)
    model = load_predictor(directory)
    with measure_phase("prediction"):
        started = perf_counter()
        predicted = predict(model, split.testing.loc[:, list(INPUT_COLUMNS)])
        prediction_seconds = perf_counter() - started
    timing = timing_benchmark(model, split.testing.loc[:, list(INPUT_COLUMNS)])
    observed = split.testing.loc[:, list(OUTPUT_COLUMNS)]
    metrics = regression_metrics(observed, predicted).assign(
        model=settings["model"],
        seed=settings["seed"],
        evaluation_split="test",
        dataset_signature=metadata["dataset_signature"],
    )
    destination.mkdir(parents=True, exist_ok=False)
    if (directory / "calibration.json").is_file():
        from truss_research.surrogates.intervalos import export_intervals

        export_intervals((directory, destination), split.testing, predicted)
    metrics.to_csv(destination / "metrics.csv", index=False)
    prediction_report(split.testing, predicted).to_csv(
        destination / "test_predictions.csv", index=False
    )
    regional_metrics(split.training, split.testing, predicted).to_csv(
        destination / "region_metrics.csv", index=False
    )
    groups = [
        regression_metrics(
            group.loc[:, list(OUTPUT_COLUMNS)], predicted.loc[group.index]
        ).assign(web_config=category)
        for category, group in split.testing.groupby("web_config")
    ]
    pd.concat(groups, ignore_index=True).to_csv(
        destination / "category_metrics.csv", index=False
    )
    write_json(
        destination / "compliance.json",
        compliance_report(
            observed.to_numpy(), predicted.to_numpy(), load_structural_limits(data_dir)
        ),
    )
    write_json(
        destination / "run.json",
        {
            **metadata,
            "evaluation_split": "test",
            "source_predictor": str(directory),
            "fit_seconds": 0.0,
            "tuning_seconds": 0.0,
            "crossfit_seconds": 0.0,
            "reuse_seconds": 0.0,
            "trial_count": 0,
            "nested_trial_count": 0,
            "search_reused": True,
            "cv_reused": True,
            "prediction_seconds": prediction_seconds,
            "timing": timing,
            "normalized_mae": normalized_mae(
                observed.to_numpy(),
                predicted.to_numpy(),
                response_scale(split.training.loc[:, list(OUTPUT_COLUMNS)]),
            ),
        },
    )


def verify_frozen_study(directory: Path) -> None:
    """La reconstrucción de reportes no acepta artefactos modificados."""

    state = read_study(directory)
    freeze = directory / "freeze.json"
    if state.get("test_started") is True and not freeze.is_file():
        raise ValueError(
            "El estudio iniciado sobre prueba perdió su manifiesto congelado"
        )
    if not freeze.is_file():
        return
    frozen = read_artifact(freeze)
    if (
        frozen["dataset_signature"] != state["dataset_signature"]
        or frozen["source_signature"] != state["source_signature"]
        or frozen.get("source_history", []) != state.get("source_history", [])
    ):
        raise ValueError("El manifiesto congelado no corresponde al estudio")
