"""Explicaciones SHAP en unidades originales, con referencia solo de entrenamiento."""

import logging
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import JsonValue, load_design_split
from truss_research.evaluacion.recursos import measured
from truss_research.surrogates.atribuciones import linear_values, tree_values
from truss_research.surrogates.configuracion import SCHEMA_VERSION, FloatArray
from truss_research.surrogates.experimentos import (
    data_signature,
    load_predictor,
    read_run,
    warning_records,
    write_json,
)

BACKGROUND_SIZE = 100

PERMUTATION_ROUNDS = 10
PERMUTATION_BATCH_SIZE = 128
TREE_MODELS = frozenset(("random_forest", "catboost", "xgboost", "lightgbm"))

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
    batch_size: int = PERMUTATION_BATCH_SIZE
    target_index: int | None = None

    def __post_init__(self) -> None:
        if self.device not in (None, "cpu", "cuda"):
            raise ValueError("El dispositivo de explicación debe ser cpu o cuda")
        if self.evaluation_limit is not None and (
            type(self.evaluation_limit) is not int or self.evaluation_limit < 1
        ):
            raise ValueError("El límite del piloto debe ser positivo")
        if any(
            type(value) is not int or value < 1
            for value in (
                self.background_size,
                self.permutation_rounds,
                self.batch_size,
            )
        ):
            raise ValueError("Referencia, rondas y lotes deben ser enteros positivos")
        if self.target_index is not None and (
            type(self.target_index) is not int
            or not 0 <= self.target_index < len(OUTPUT_COLUMNS)
        ):
            raise ValueError("Índice de respuesta inválido")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Semilla de explicación inválida")


@dataclass(frozen=True)
class AttributionJob:
    model: Pipeline
    data: tuple[pd.DataFrame, pd.DataFrame]
    config: ExplanationConfig
    family: str
    predicted: FloatArray


@dataclass
class AttributionResult:
    values: FloatArray
    bases: FloatArray
    method: str
    requested_method: str
    fallback_reason: str | None = None
    tree_error: list[float] | None = None
    components: list[dict[str, JsonValue]] = field(default_factory=list)
    prediction_calls: int = 1
    prediction_rows: int = 0
    permutation_seconds: float = 0.0


def checked_attributions(
    job: AttributionJob, result: AttributionResult
) -> AttributionResult:
    validate_contributions(result.values, result.bases, job.predicted.shape)
    if not np.allclose(
        result.bases + result.values.sum(axis=1),
        job.predicted,
        rtol=ADDITIVITY_TOLERANCE,
        atol=ADDITIVITY_TOLERANCE,
    ):
        raise ValueError(
            "Las contribuciones SHAP no reconstruyen las predicciones en unidades "
            f"originales; familia={job.family}, respuesta={job.config.target_index}, "
            "error máximo por respuesta="
            f"{np.max(np.abs(result.bases + result.values.sum(axis=1) - job.predicted), axis=0).tolist()}"
        )
    return result


def permuted_attributions(job: AttributionJob) -> AttributionResult:
    from truss_research.surrogates.permutaciones import checkpoint_metrics

    values, bases = permutation_values(job.model, job.data, job.config)
    checked_attributions(
        job, AttributionResult(values, bases, "permutation", "permutation")
    )
    costs = checkpoint_metrics(job.config)
    return checked_attributions(
        job,
        AttributionResult(
            values,
            bases,
            "permutation",
            "permutation",
            prediction_calls=1 + int(costs["prediction_calls"]),
            prediction_rows=len(job.data[1]) + int(costs["prediction_rows"]),
            permutation_seconds=float(costs["elapsed"]),
        ),
    )


def forest_attributions(job: AttributionJob) -> AttributionResult:
    requested = (
        "tree_interventional"
        if "preprocessing" in job.model.named_steps
        else "tree_path_dependent"
    )
    try:
        values, bases = tree_values(job.model, job.data)
        validate_contributions(values, bases, job.predicted.shape)
    except (ValueError, RuntimeError, TypeError, NotImplementedError) as error:
        reason = f"{type(error).__name__}: {error}"
        logger.warning("TreeExplainer falló: %s; se usarán permutaciones", reason)
        result = permuted_attributions(job)
        result.method, result.requested_method = (
            "permutation_after_tree_error",
            requested,
        )
        result.fallback_reason = reason
        return result
    tree_error = np.max(
        np.abs(bases + values.sum(axis=1) - job.predicted), axis=0
    ).tolist()
    if not np.allclose(
        bases + values.sum(axis=1),
        job.predicted,
        rtol=ADDITIVITY_TOLERANCE,
        atol=ADDITIVITY_TOLERANCE,
    ):
        logger.warning(
            "TreeExplainer no reconstruye las respuestas; se usarán permutaciones"
        )
        result = permuted_attributions(job)
        result.method, result.requested_method = (
            "permutation_after_tree_additivity_failure",
            requested,
        )
        result.fallback_reason, result.tree_error = (
            "tree_additivity_failure",
            tree_error,
        )
        return result
    return AttributionResult(
        values,
        bases,
        requested,
        requested,
        tree_error=tree_error,
        prediction_rows=len(job.data[1]),
    )


def composite_attributions(job: AttributionJob) -> AttributionResult:
    from truss_research.surrogates.permutaciones import explanation_predictions

    results: list[AttributionResult] = []
    components: list[dict[str, JsonValue]] = []
    regressor = job.model.named_steps["regressor"]
    for index, (settings, model) in enumerate(regressor.components_):
        config = replace(
            job.config,
            predictor_dir=job.config.predictor_dir / "components" / str(index),
            output_dir=job.config.output_dir.with_name(
                job.config.output_dir.name + f"_target_{index}"
            ),
            target_index=index,
        )
        predicted = explanation_predictions(model, job.data[1], config.batch_size)
        result = checked_attributions(
            AttributionJob(model, job.data, config, settings.model, predicted),
            compute_attributions(
                AttributionJob(model, job.data, config, settings.model, predicted)
            ),
        )
        results.append(result)
        components.append(
            {
                "target": OUTPUT_COLUMNS[index],
                "model": settings.model,
                "method": result.method,
                "requested_method": result.requested_method,
                "fallback_reason": result.fallback_reason,
            }
        )
    return AttributionResult(
        np.concatenate([item.values for item in results], axis=-1),
        np.concatenate([item.bases for item in results], axis=-1),
        "targetwise_components",
        "targetwise_components",
        components=components,
        prediction_calls=1 + sum(item.prediction_calls for item in results),
        prediction_rows=len(job.data[1])
        + sum(item.prediction_rows for item in results),
        permutation_seconds=sum(item.permutation_seconds for item in results),
    )


def compute_attributions(job: AttributionJob) -> AttributionResult:
    if job.family == "targetwise":
        return checked_attributions(job, composite_attributions(job))
    if job.family in TREE_MODELS:
        return forest_attributions(job)
    if job.family == "dummy":
        values = np.zeros(
            (len(job.data[1]), len(INPUT_COLUMNS), job.predicted.shape[1])
        )
        return checked_attributions(
            job,
            AttributionResult(
                values,
                np.broadcast_to(job.predicted[0], job.predicted.shape).copy(),
                "constant",
                "constant",
                prediction_rows=len(job.data[1]),
            ),
        )
    if job.family == "ridge":
        values, bases = linear_values(job.model, job.data)
        return checked_attributions(
            job,
            AttributionResult(
                values,
                bases,
                "linear_independent",
                "linear_independent",
                prediction_rows=len(job.data[1]),
            ),
        )
    return permuted_attributions(job)


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

    from truss_research.surrogates.permutaciones import explanation_predictions

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
    predicted = explanation_predictions(model, evaluation, config.batch_size)
    configuration = metadata["config"]
    result = compute_attributions(
        AttributionJob(
            model,
            (background, evaluation),
            config,
            str(configuration["model"]),
            predicted,
        )
    )
    values, bases = result.values, result.bases
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
            "method": result.method,
            "requested_method": result.requested_method,
            "fallback_reason": result.fallback_reason,
            "tree_additivity_error_by_target": result.tree_error,
            "component_methods": result.components,
            "counted_prediction_calls": result.prediction_calls,
            "counted_prediction_rows": result.prediction_rows,
            "permutation_completed_seconds": result.permutation_seconds,
            "prediction_count_scope": "logical prediction calls and requested rows before fixed-batch padding; includes reconstruction checks; checkpoints count completed designs, interrupted uncommitted work excluded",
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
            "full_validation_designs": len(split.validation),
            "full_training_designs": len(split.training),
            "batch_size": config.batch_size,
            "prediction_batch_policy": "TabPFN uses fixed batches padded with the last row; padding predictions are discarded; other families keep their original inference",
            "seed": config.seed,
            "permutation_rounds": config.permutation_rounds,
            "explanation_seconds": perf_counter() - started,
            "interpretation": "Contribuciones predictivas; no evidencia causal ni certificación física",
            "background_semantics": "Independent/linear/interventional usan la referencia muestreada; tree_path_dependent usa frecuencias internas del árbol. Targetwise declara método por respuesta.",
        },
    )
    return table
