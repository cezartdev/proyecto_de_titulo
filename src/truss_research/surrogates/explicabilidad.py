"""Explicaciones SHAP en unidades originales, con referencia solo de entrenamiento."""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import load_design_split
from truss_research.evaluacion.recursos import measured
from truss_research.surrogates.configuracion import SCHEMA_VERSION, FloatArray
from truss_research.surrogates.experimentos import (
    data_signature,
    load_predictor,
    predict,
    read_run,
    warning_records,
    write_json,
)

BACKGROUND_SIZE = 100

PERMUTATION_ROUNDS = 10

ADDITIVITY_TOLERANCE = 1e-4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplanationConfig:
    data_dir: Path
    predictor_dir: Path
    output_dir: Path
    seed: int = 42
    background_size: int = BACKGROUND_SIZE
    permutation_rounds: int = PERMUTATION_ROUNDS
    device: str | None = None
    evaluation_limit: int | None = None

    def __post_init__(self) -> None:
        if self.device not in (None, "cpu", "cuda"):
            raise ValueError("El dispositivo de explicación debe ser cpu o cuda")
        if self.evaluation_limit is not None and (
            type(self.evaluation_limit) is not int or self.evaluation_limit < 1
        ):
            raise ValueError("El límite del piloto debe ser positivo")
        if min(self.background_size, self.permutation_rounds) < 1:
            raise ValueError("Referencia y rondas deben ser positivas")


def permutation_values(
    model: Pipeline, data: tuple[pd.DataFrame, pd.DataFrame], config: ExplanationConfig
) -> tuple[FloatArray, FloatArray]:
    from truss_research.surrogates.permutaciones import resumable_permutations

    return resumable_permutations(model, data, config)


@measured("explanation")
def explain_model(config: ExplanationConfig) -> pd.DataFrame:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = explain_values(config)
    metadata = read_run(config.output_dir)
    metadata["warnings"] = warning_records(captured)
    write_json(config.output_dir / "run.json", metadata)
    return result


def tree_values(
    model: Pipeline, data: tuple[pd.DataFrame, pd.DataFrame]
) -> tuple[FloatArray, FloatArray]:
    import shap

    background, evaluation = data
    regressor = model.named_steps["regressor"]
    if "preprocessing" not in model.named_steps:
        from truss_research.surrogates.modelos import categorical_features

        table = categorical_features(evaluation, regressor.config.model)
        native_values = []
        native_bases = []
        for estimator in regressor.estimators_:
            explanation = shap.TreeExplainer(
                estimator, feature_perturbation="tree_path_dependent"
            )
            native_values.append(
                np.asarray(explanation.shap_values(table, check_additivity=False))
            )
            native_bases.append(
                np.full(
                    len(table), float(np.asarray(explanation.expected_value).squeeze())
                )
            )
        return np.stack(native_values, axis=-1), np.stack(native_bases, axis=-1)

    transformer = model.named_steps["preprocessing"]
    encoded_background, encoded = (
        transformer.transform(background),
        transformer.transform(evaluation),
    )
    values = np.empty((len(evaluation), len(INPUT_COLUMNS), len(OUTPUT_COLUMNS)))
    bases = np.empty((len(evaluation), len(OUTPUT_COLUMNS)))
    numeric = [name for name in INPUT_COLUMNS if name != "web_config"]
    for target, estimator in enumerate(model.named_steps["regressor"].estimators_):
        explanation = shap.TreeExplainer(
            estimator, data=encoded_background, feature_perturbation="interventional"
        )
        contributions = np.asarray(
            explanation.shap_values(encoded, check_additivity=False)
        )
        if contributions.shape != encoded.shape:
            raise ValueError("Dimensiones SHAP transformadas incompatibles")
        values[:, INPUT_COLUMNS.index("web_config"), target] = contributions[
            :, transformer.output_indices_["web_config"]
        ].sum(axis=1)
        numeric_values = contributions[:, transformer.output_indices_["numeric"]]
        for column, name in enumerate(numeric):
            values[:, INPUT_COLUMNS.index(name), target] = numeric_values[:, column]
        bases[:, target] = float(np.asarray(explanation.expected_value).squeeze())
    return values, bases


def validate_contributions(
    values: FloatArray, bases: FloatArray, prediction_shape: tuple[int, int]
) -> None:
    if (
        values.shape != (prediction_shape[0], len(INPUT_COLUMNS), prediction_shape[1])
        or bases.shape != prediction_shape
        or not np.isfinite(values).all()
        or not np.isfinite(bases).all()
    ):
        raise ValueError("Dimensiones o valores SHAP inválidos")


def explain_values(config: ExplanationConfig) -> pd.DataFrame:
    from time import perf_counter

    started = perf_counter()
    if config.output_dir.exists():
        raise ValueError("El destino ya existe; elegir una ejecución nueva")

    metadata = read_run(config.predictor_dir)
    if metadata.get("dataset_signature") != data_signature(config.data_dir):
        raise ValueError("El predictor no corresponde a la partición solicitada")
    split = load_design_split(config.data_dir)
    model = load_predictor(config.predictor_dir, config.device)
    from truss_research.surrogates.persistencia import (
        prediction_device,
        prediction_devices,
    )

    execution_device = prediction_device(model)
    training = split.training.loc[:, list(INPUT_COLUMNS)]
    evaluation = split.validation.loc[:, list(INPUT_COLUMNS)]
    if config.evaluation_limit is not None:
        evaluation = evaluation.iloc[: config.evaluation_limit]
    background = training.sample(
        min(config.background_size, len(training)), random_state=config.seed
    )
    predicted = predict(model, evaluation).to_numpy()
    values = np.empty((len(evaluation), len(INPUT_COLUMNS), len(OUTPUT_COLUMNS)))
    bases = np.empty((len(evaluation), len(OUTPUT_COLUMNS)))
    method = "permutation"
    configuration = metadata.get("config")
    is_forest = isinstance(configuration, dict) and configuration.get("model") in (
        "random_forest",
        "catboost",
        "xgboost",
        "lightgbm",
    )
    tree_exception = None
    if is_forest:
        try:
            values, bases = tree_values(model, (background, evaluation))
            validate_contributions(values, bases, predicted.shape)
            method = (
                "tree_interventional"
                if configuration["model"] == "random_forest"
                else "tree_path_dependent"
            )
        except (ValueError, RuntimeError, TypeError, NotImplementedError) as error:
            tree_exception = f"{type(error).__name__}: {error}"
            logger.warning(
                "TreeExplainer falló: %s; se usarán permutaciones", tree_exception
            )
            values, bases = permutation_values(model, (background, evaluation), config)
            method = "permutation_after_tree_error"
    else:
        values, bases = permutation_values(model, (background, evaluation), config)
    tree_error = (
        np.max(np.abs(bases + values.sum(axis=1) - predicted), axis=0).tolist()
        if is_forest and tree_exception is None
        else None
    )
    if (
        is_forest
        and tree_exception is None
        and not np.allclose(
            bases + values.sum(axis=1),
            predicted,
            rtol=ADDITIVITY_TOLERANCE,
            atol=ADDITIVITY_TOLERANCE,
        )
    ):
        logger.warning(
            "TreeExplainer no reconstruye las respuestas (errores %s); se utilizarán permutaciones",
            tree_error,
        )
        values, bases = permutation_values(model, (background, evaluation), config)
        method = "permutation_after_tree_additivity_failure"
    validate_contributions(values, bases, predicted.shape)
    if not np.allclose(
        bases + values.sum(axis=1),
        predicted,
        rtol=ADDITIVITY_TOLERANCE,
        atol=ADDITIVITY_TOLERANCE,
    ):
        raise ValueError(
            "Las contribuciones SHAP no reconstruyen las predicciones en unidades originales"
        )
    rows = [
        {
            "iter_id": str(split.validation.loc[index, "iter_id"]),
            "variable": variable,
            "target": target,
            "shap_value": float(values[row, column, output]),
            "base_value": float(bases[row, output]),
            "feature_value": float(evaluation.loc[index, variable]),
        }
        for row, index in enumerate(evaluation.index)
        for column, variable in enumerate(INPUT_COLUMNS)
        for output, target in enumerate(OUTPUT_COLUMNS)
    ]
    table = pd.DataFrame(rows)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(config.output_dir / "shap_values.csv", index=False)
    table.assign(abs_shap=lambda frame: frame["shap_value"].abs()).groupby(
        ["target", "variable"]
    )["abs_shap"].mean().to_csv(config.output_dir / "global_importance.csv")
    write_json(
        config.output_dir / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "tree_additivity_error_by_target": tree_error,
            "method": method,
            "requested_method": (
                "tree_interventional"
                if configuration["model"] == "random_forest"
                else "tree_path_dependent"
            )
            if is_forest
            else "permutation",
            "fallback_reason": tree_exception
            or (
                "tree_additivity_failure"
                if method == "permutation_after_tree_additivity_failure"
                else None
            ),
            "final_additivity_error_by_target": np.max(
                np.abs(bases + values.sum(axis=1) - predicted), axis=0
            ).tolist(),
            "output_space": "original_physical_units",
            "dataset_signature": metadata["dataset_signature"],
            "predictor_signature": metadata.get("model_signature"),
            "background_ids": split.training.loc[background.index, "iter_id"].tolist(),
            "evaluation_ids": split.validation.loc[
                evaluation.index, "iter_id"
            ].tolist(),
            "execution_device": execution_device,
            "prediction_components": prediction_devices(model),
            "evaluation_limit": config.evaluation_limit,
            "seed": config.seed,
            "permutation_rounds": config.permutation_rounds,
            "explanation_seconds": perf_counter() - started,
            "interpretation": "Contribuciones predictivas; no evidencia causal ni certificación física",
        },
    )
    return table
