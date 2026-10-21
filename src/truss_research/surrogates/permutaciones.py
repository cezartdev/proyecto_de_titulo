"""Permutaciones SHAP con checkpoint atómico y continuidad del generador aleatorio."""

import hashlib
import json
import logging
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.surrogates.configuracion import FloatArray
from truss_research.surrogates.experimentos import predict
from truss_research.surrogates.persistencia import model_signatures

if TYPE_CHECKING:
    from truss_research.surrogates.explicabilidad import ExplanationConfig

CHECKPOINT_VERSION = 1
logger = logging.getLogger(__name__)


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
        "config": {
            key: str(value)
            for key, value in asdict(config).items()
            if key != "output_dir"
        },
        "models": model_signatures(config.predictor_dir),
        "versions": {
            name: package_version(name) for name in ("shap", "numpy", "torch", "tabpfn")
        },
        "data": [table.to_json(orient="split", double_precision=15) for table in data],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def resumable_permutations(
    model: Pipeline,
    data: tuple[pd.DataFrame, pd.DataFrame],
    config: "ExplanationConfig",
) -> tuple[FloatArray, FloatArray]:
    background, evaluation = data

    def predict_array(array: FloatArray) -> FloatArray:
        return predict(model, pd.DataFrame(array, columns=INPUT_COLUMNS)).to_numpy()

    explainer = shap.Explainer(
        predict_array,
        shap.maskers.Independent(background.to_numpy()),
        algorithm="permutation",
        seed=config.seed,
    )
    checkpoint = config.output_dir.with_suffix(".shap.npz")
    identity = checkpoint_identity(data, config)
    values = np.empty((0, len(INPUT_COLUMNS), len(OUTPUT_COLUMNS)))
    bases = np.empty((0, len(OUTPUT_COLUMNS)))
    elapsed = 0.0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if saved["identity"].item() != identity:
                raise ValueError(
                    "El checkpoint SHAP no corresponde al modelo, datos o configuración"
                )
            values, bases = saved["values"], saved["bases"]
            elapsed = float(saved["elapsed"])
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
            values.shape != (len(bases), len(INPUT_COLUMNS), len(OUTPUT_COLUMNS))
            or bases.shape != (len(values), len(OUTPUT_COLUMNS))
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
            silent=True,
        )
        values = np.concatenate((values, np.asarray(result.values)))
        bases = np.concatenate((bases, np.asarray(result.base_values)))
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
