"""Permutaciones SHAP con checkpoint atómico y continuidad del generador aleatorio."""

import hashlib
import json
import logging
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.preparacion import source_digest
from truss_research.surrogates.configuracion import FloatArray
from truss_research.surrogates.persistencia import model_signatures

if TYPE_CHECKING:
    from truss_research.surrogates.explicabilidad import ExplanationConfig

CHECKPOINT_VERSION = 3
logger = logging.getLogger(__name__)


def contains_tabpfn(model: Pipeline) -> bool:
    from truss_research.surrogates.especializacion import TargetwiseRegressor
    from truss_research.surrogates.tabpfn import TabPFNRegressor

    regressor = model.named_steps["regressor"]
    if isinstance(regressor, TargetwiseRegressor):
        return any(contains_tabpfn(component) for _, component in regressor.components_)
    return isinstance(regressor, TabPFNRegressor)


def explanation_predictions(
    model: Pipeline, features: pd.DataFrame, batch_size: int
) -> FloatArray:
    """Fija el tamaño de inferencia TabPFN para comparar máscaras y reconstrucción."""
    if not contains_tabpfn(model):
        return np.asarray(model.predict(features)).reshape(len(features), -1)
    predictions: list[FloatArray] = []
    for start in range(0, len(features), batch_size):
        batch = features.iloc[start : start + batch_size]
        padded = pd.concat(
            [batch, batch.iloc[[-1] * (batch_size - len(batch))]], ignore_index=True
        )
        values = np.asarray(model.predict(padded)).reshape(batch_size, -1)
        predictions.append(values[: len(batch)])
    return np.concatenate(predictions)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not_installed"


def checkpoint_identity(
    data: tuple[pd.DataFrame, pd.DataFrame], config: "ExplanationConfig"
) -> str:
    payload = {
        "version": CHECKPOINT_VERSION,
        "implementation": source_digest(Path(__file__)),
        "config": {
            key: str(value)
            for key, value in asdict(config).items()
            if key != "output_dir"
        },
        "models": model_signatures(config.predictor_dir),
        "versions": {
            name: package_version(name) for name in ("shap", "numpy", "torch", "tabpfn")
        },
        "data": [table.to_csv(index=True, float_format="%.17g") for table in data],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def resumable_permutations(
    model: Pipeline,
    data: tuple[pd.DataFrame, pd.DataFrame],
    config: "ExplanationConfig",
) -> tuple[FloatArray, FloatArray]:
    background, evaluation = data
    output_count = 1 if config.target_index is not None else len(OUTPUT_COLUMNS)
    counts = {"prediction_calls": 0, "prediction_rows": 0}

    def predict_array(array: FloatArray) -> FloatArray:
        counts["prediction_calls"] += 1
        counts["prediction_rows"] += len(array)
        values = explanation_predictions(
            model, pd.DataFrame(array, columns=INPUT_COLUMNS), config.batch_size
        )
        values = values.reshape(len(array), output_count)
        if not np.isfinite(values).all():
            raise ValueError("El predictor SHAP devolvió valores no finitos")
        return values

    explainer = shap.Explainer(
        predict_array,
        shap.maskers.Independent(background.to_numpy(), max_samples=len(background)),
        algorithm="permutation",
        seed=config.seed,
    )
    checkpoint = config.output_dir.with_suffix(".shap.npz")
    identity = checkpoint_identity(data, config)
    values = np.empty((0, len(INPUT_COLUMNS), output_count))
    bases = np.empty((0, output_count))
    elapsed = 0.0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if saved["identity"].item() != identity:
                raise ValueError(
                    "El checkpoint SHAP no corresponde al modelo, datos o configuración"
                )
            values, bases = saved["values"], saved["bases"]
            elapsed = float(saved["elapsed"])
            counts = {key: int(saved[key]) for key in counts}
            np.random.set_state(
                (
                    "MT19937",
                    saved["rng_keys"],
                    int(saved["rng_pos"]),
                    int(saved["rng_gauss"]),
                    float(saved["rng_cached"]),
                )
            )
        if (
            values.shape != (len(bases), len(INPUT_COLUMNS), output_count)
            or bases.shape != (len(values), output_count)
            or len(values) > len(evaluation)
            or not np.isfinite(values).all()
            or not np.isfinite(bases).all()
        ):
            raise ValueError("El checkpoint SHAP tiene dimensiones o valores inválidos")
        logger.info("SHAP reanudado: %s/%s diseños", len(values), len(evaluation))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for index in range(len(values), len(evaluation)):
        started = perf_counter()
        result = explainer(
            evaluation.iloc[index : index + 1].to_numpy(),
            max_evals=config.permutation_rounds * (2 * len(INPUT_COLUMNS) + 1),
            batch_size=config.batch_size,
            silent=True,
        )
        values = np.concatenate(
            (
                values,
                np.asarray(result.values).reshape(1, len(INPUT_COLUMNS), output_count),
            )
        )
        bases = np.concatenate(
            (bases, np.asarray(result.base_values).reshape(1, output_count))
        )
        if not np.isfinite(values).all() or not np.isfinite(bases).all():
            raise ValueError(
                "SHAP produjo contribuciones no finitas; no se guarda el checkpoint"
            )
        elapsed += perf_counter() - started
        rng = np.random.get_state()
        temporary = checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                identity=identity,
                values=values,
                bases=bases,
                elapsed=elapsed,
                **counts,
                rng_keys=rng[1],
                rng_pos=rng[2],
                rng_gauss=rng[3],
                rng_cached=rng[4],
            )
        temporary.replace(checkpoint)
        logger.info(
            "SHAP %s: %s/%s diseños, %.1f s acumulados, restante estimado %.1f s",
            config.device or "modelo",
            index + 1,
            len(evaluation),
            elapsed,
            elapsed / (index + 1) * (len(evaluation) - index - 1),
        )
    return values, bases


def checkpoint_metrics(config: "ExplanationConfig") -> dict[str, int | float]:
    """Trabajo confirmado por checkpoints; excluye diseños interrumpidos sin guardar."""
    with np.load(
        config.output_dir.with_suffix(".shap.npz"), allow_pickle=False
    ) as saved:
        return {
            "prediction_calls": int(saved["prediction_calls"]),
            "prediction_rows": int(saved["prediction_rows"]),
            "elapsed": float(saved["elapsed"]),
        }
