"""Ajuste diagnóstico único sin Optuna, OOF ni acceso a métricas de prueba."""

from dataclasses import asdict

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import load_design_split
from truss_research.evaluacion.metricas import regression_metrics
from truss_research.evaluacion.recursos import measure_phase, timed
from truss_research.surrogates.configuracion import (
    PredictorConfig,
    normalized_mae,
    response_scale,
)
from truss_research.surrogates.experimentos import (
    data_signature,
    predict,
    prediction_report,
    write_json,
)
from truss_research.surrogates.modelos import build_predictor
from truss_research.surrogates.persistencia import save_model


def fit_diagnostic(config: PredictorConfig) -> None:
    if config.model == "targetwise" or config.trials or config.nested_folds is not None:
        raise ValueError(
            "El ajuste directo requiere una familia fija sin búsqueda ni OOF"
        )
    if config.output_dir.exists():
        raise ValueError("El destino del ajuste directo ya existe")
    with measure_phase("data_loading"):
        split = load_design_split(config.data_dir)
        features = split.training.loc[:, list(INPUT_COLUMNS)]
        responses = split.training.loc[:, list(OUTPUT_COLUMNS)]
        validation = split.validation.loc[:, list(INPUT_COLUMNS)]
    model = build_predictor(config)
    with measure_phase("fit"):
        _, fitting = timed(lambda: model.fit(features, responses))
    with measure_phase("prediction"):
        predicted, prediction = timed(lambda: predict(model, validation))
        observed = split.validation.loc[:, list(OUTPUT_COLUMNS)]
        metrics = regression_metrics(observed, predicted)
        score = normalized_mae(
            observed.to_numpy(), predicted.to_numpy(), response_scale(responses)
        )
    with measure_phase("export"):
        config.output_dir.mkdir(parents=True, exist_ok=False)
        save_model(model, config.output_dir)
        metrics.to_csv(config.output_dir / "metrics.csv", index=False)
        prediction_report(split.validation, predicted).to_csv(
            config.output_dir / "validation_predictions.csv", index=False
        )
        write_json(
            config.output_dir / "diagnostic.json",
            {
                "kind": "fit_only_diagnostic",
                "config": asdict(config),
                "dataset_signature": data_signature(config.data_dir),
                "training_rows": len(features),
                "validation_rows": len(validation),
                "fit_calls": 1,
                "trials": 0,
                "oof": False,
                "test_used": False,
                "eligible_for_campaign": False,
                "normalized_mae": score,
                "fitting": fitting,
                "prediction": prediction,
            },
        )
