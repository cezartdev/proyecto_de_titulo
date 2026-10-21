"""Ajuste y evaluación explícitos para controles y curvas sin contaminar particiones."""

from dataclasses import asdict

import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    StructuralLimits,
)
from truss_research.datos.preparacion import source_digest
from truss_research.datos.schemas import TableSplit
from truss_research.evaluacion.diagnosticos_incertidumbre import uncertainty_diagnostics
from truss_research.evaluacion.metricas import regression_metrics
from truss_research.surrogates.calibracion import SplitConformal
from truss_research.surrogates.configuracion import (
    SCHEMA_VERSION,
    PredictorConfig,
    normalized_mae,
)
from truss_research.surrogates.experimentos import (
    data_signature,
    predict,
    tune_predictor,
    write_json,
)
from truss_research.surrogates.modelos import build_predictor
from truss_research.surrogates.persistencia import save_model


def fit_retained(
    config: PredictorConfig, split: TableSplit, tune: bool
) -> tuple[Pipeline, SplitConformal]:
    if split.calibration is None:
        raise ValueError("Falta calibración separada del ajuste")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    effective, trials = (
        tune_predictor(config, split.training) if tune else (config, pd.DataFrame())
    )
    model = build_predictor(effective)
    model.fit(
        split.training.loc[:, list(INPUT_COLUMNS)],
        split.training.loc[:, list(OUTPUT_COLUMNS)],
    )
    path = config.output_dir / "model.joblib"
    save_model(model, config.output_dir)
    predicted = predict(model, split.calibration.loc[:, list(INPUT_COLUMNS)])
    calibrator = SplitConformal(
        split.calibration.loc[:, list(OUTPUT_COLUMNS)].to_numpy() - predicted.to_numpy()
    )
    for coverage in calibrator.coverages:
        calibrator.widths(coverage)
    write_json(
        config.output_dir / "run.json",
        {
            "config": asdict(effective),
            "schema_version": SCHEMA_VERSION,
            "role": "retained_predictor",
            "winner_eligible": True,
            "fit_ids": split.training.iter_id.tolist(),
            "calibration_ids": split.calibration.iter_id.tolist(),
            "trial_count": len(trials),
            "tuning": tune,
            "model_signature": source_digest(path),
            "dataset_signature": data_signature(config.data_dir),
            "targetwise_selection": getattr(
                model.named_steps["regressor"], "provenance_", []
            ),
            "family_search": getattr(
                model.named_steps["regressor"], "family_search_", []
            ),
            "family_trial_count": getattr(
                model.named_steps["regressor"], "family_trials_", 0
            ),
        },
    )
    write_json(
        config.output_dir / "calibration.json",
        {
            "residuals": calibrator.residuals.tolist(),
            "coverages": list(calibrator.coverages),
        },
    )
    return model, calibrator


def assess_retained(
    fit: tuple[pd.DataFrame, Pipeline, SplitConformal, StructuralLimits],
    table: pd.DataFrame,
    scales: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    training, model, calibrator, limits = fit
    predicted = predict(model, table.loc[:, list(INPUT_COLUMNS)])
    metrics = regression_metrics(table.loc[:, list(OUTPUT_COLUMNS)], predicted)
    intervals, feasibility = uncertainty_diagnostics(
        training, (table, predicted), (calibrator, limits)
    )
    score = normalized_mae(
        table.loc[:, list(OUTPUT_COLUMNS)].to_numpy(), predicted.to_numpy(), scales
    )
    return metrics, intervals, feasibility, score
