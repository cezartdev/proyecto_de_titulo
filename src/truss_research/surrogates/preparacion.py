"""Búsquedas y pliegues independientes para el planificador global de CPU."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue, load_design_split
from truss_research.evaluacion.recursos import measure_phase
from truss_research.surrogates.configuracion import ModelParameters, PredictorConfig
from truss_research.surrogates.experimentos import (
    data_signature,
    tune_predictor,
    write_json,
)
from truss_research.surrogates.persistencia import copy_model, save_model
from truss_research.surrogates.pliegues import eligible_indices
from truss_research.surrogates.validacion import fit_outer_fold


@dataclass(frozen=True)
class PreparedPredictor:
    parameters: ModelParameters
    trials: pd.DataFrame
    oof: pd.DataFrame
    score: float
    provenance: list[dict[str, JsonValue]]
    tuning_seconds: float
    crossfit_seconds: float
    models: tuple[Path, ...] = ()
    refit_trees: tuple[int, ...] = ()


def preparation_identity(config: PredictorConfig) -> str:
    payload = asdict(config)
    for name in ("output_dir", "study_path", "preparation_dir", "reuse_dir"):
        payload.pop(name)
    payload["data_dir"] = data_signature(config.data_dir)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def prepare_search(config: PredictorConfig, destination: Path) -> None:
    table = load_design_split(config.data_dir).training
    tuning = (
        replace(
            config,
            folds=config.final_tuning_folds or config.nested_folds,
            nested_folds=None,
        )
        if config.final_tuning_folds or config.nested_folds
        else config
    )
    with measure_phase("tuning"):
        start = perf_counter()
        effective, trials = tune_predictor(tuning, table)
        seconds = perf_counter() - start
    destination.mkdir(parents=True, exist_ok=False)
    write_json(
        destination / "run.json",
        {
            "identity": preparation_identity(config),
            "parameters": asdict(effective.parameters),
            "refit_trees": effective.refit_trees,
            "tuning_seconds": seconds,
            "trials": trials.astype(object)
            .where(pd.notna(trials), None)
            .to_dict(orient="records"),
            "trial_columns": list(trials.columns),
        },
    )
    trials.to_csv(destination / "trials.csv", index=False)


def prepare_outer_fold(config: PredictorConfig, destination: Path, number: int) -> None:
    table = load_design_split(config.data_dir).training
    with measure_phase("crossfit"):
        start = perf_counter()
        result = fit_outer_fold(config, table, number)
        seconds = perf_counter() - start
    destination.mkdir(parents=True, exist_ok=False)
    held = table.iloc[list(result.held_indices)]
    save_model(result.model, destination)
    predictions = pd.DataFrame(
        result.values, columns=[f"pred_{name}" for name in OUTPUT_COLUMNS]
    )
    predictions.insert(0, "iter_id", held.iter_id.to_numpy())
    predictions["fold"] = number
    predictions.to_csv(destination / "oof.csv", index=False)
    write_json(
        destination / "run.json",
        {
            "score": result.score,
            "crossfit_seconds": seconds,
            "provenance": result.provenance,
        },
    )


def assemble_preparation(
    config: PredictorConfig, sources: tuple[Path, tuple[Path, ...]], destination: Path
) -> None:
    search, folds = sources
    metadata = read_artifact(search / "run.json")
    if metadata["identity"] != preparation_identity(config):
        raise ValueError("La preparación pertenece a otro protocolo")
    table = load_design_split(config.data_dir).training
    predictions = pd.concat(
        [pd.read_csv(path / "oof.csv") for path in folds], ignore_index=True
    )
    table = table.iloc[eligible_indices(config, table)]
    if not predictions.iter_id.is_unique or set(predictions.iter_id) != set(
        table.iter_id
    ):
        raise ValueError("Los pliegues no cubren exactamente entrenamiento")
    predictions = predictions.set_index("iter_id").loc[table.iter_id].reset_index()
    runs = [read_artifact(path / "run.json") for path in folds]
    destination.mkdir(parents=True, exist_ok=False)
    for number, path in enumerate(folds):
        copy_model(path, destination / f"fold_{number}")
    predictions.to_csv(destination / "oof.csv", index=False)
    write_json(
        destination / "run.json",
        {
            **metadata,
            "score": float(np.mean([float(run["score"]) for run in runs])),
            "crossfit_seconds": sum(float(run["crossfit_seconds"]) for run in runs),
            "provenance": [run["provenance"] for run in runs]
            if config.nested_folds is not None
            else [],
            "fold_warnings": [
                {"fold": number, "warnings": run["provenance"]["warnings"]}
                for number, run in enumerate(runs)
            ]
            if config.nested_folds is None
            else [],
            "oof_signature": source_digest(destination / "oof.csv"),
        },
    )


def load_prepared_predictor(config: PredictorConfig) -> PreparedPredictor:
    if config.preparation_dir is None:
        raise ValueError("Falta preparación predictiva")
    root = config.preparation_dir
    metadata = read_artifact(root / "run.json")
    if metadata["identity"] != preparation_identity(config) or metadata[
        "oof_signature"
    ] != source_digest(root / "oof.csv"):
        raise ValueError(
            "La preparación predictiva cambió o pertenece a otro protocolo"
        )
    table = load_design_split(config.data_dir).training
    table = table.iloc[eligible_indices(config, table)]
    oof = pd.read_csv(root / "oof.csv")
    if oof.iter_id.tolist() != table.iter_id.tolist():
        raise ValueError("Las OOF preparadas no están alineadas")
    oof.index = table.index
    oof.attrs["fold_warnings"] = metadata["fold_warnings"]
    return PreparedPredictor(
        ModelParameters(**cast(dict[str, int | float], metadata["parameters"])),
        pd.DataFrame(metadata["trials"], columns=metadata["trial_columns"]),
        oof,
        float(metadata["score"]),
        cast(list[dict[str, JsonValue]], metadata["provenance"]),
        float(metadata["tuning_seconds"]),
        float(metadata["crossfit_seconds"]),
        tuple(sorted(root.glob("fold_[0-9]*"))),
        tuple(metadata.get("refit_trees", ())),
    )
