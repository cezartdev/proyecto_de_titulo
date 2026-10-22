"""Selección por familia y cierre inmutable de todas las repeticiones."""

from pathlib import Path
from time import perf_counter

import pandas as pd

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import load_design_split, load_structural_limits
from truss_research.evaluacion.metricas import regional_metrics, regression_metrics
from truss_research.evaluacion.pareto import compliance_report
from truss_research.evaluacion.recursos import measure_phase
from truss_research.surrogates.configuracion import (
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
