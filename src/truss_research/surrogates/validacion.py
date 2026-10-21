"""Cross-fitting anidado: selección y ajuste excluyen cada pliegue externo."""

import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import JsonValue
from truss_research.surrogates.configuracion import (
    FloatArray,
    PredictorConfig,
    normalized_mae,
    response_scale,
)
from truss_research.surrogates.convergencia import convergence_records
from truss_research.surrogates.experimentos import (
    predict,
    tune_predictor,
    warning_records,
)
from truss_research.surrogates.modelos import build_predictor
from truss_research.surrogates.parada_temprana import stopping_records
from truss_research.surrogates.pliegues import predictor_folds


@dataclass(frozen=True)
class NestedPredictions:
    predictions: pd.DataFrame
    score: float
    provenance: list[dict[str, JsonValue]]
    models: tuple[Pipeline, ...] = ()


@dataclass(frozen=True)
class OuterFoldResult:
    number: int
    held_indices: tuple[int, ...]
    values: FloatArray
    score: float
    provenance: dict[str, JsonValue]
    model: Pipeline


def fit_outer_fold(
    config: PredictorConfig, table: pd.DataFrame, number: int
) -> OuterFoldResult:
    seed = config.cv_seed if config.cv_seed is not None else config.seed
    fit_idx, held_idx = predictor_folds(config, table)[number]
    fitting, held = table.iloc[fit_idx], table.iloc[held_idx]
    path = (
        config.study_path.with_name(f"{config.study_path.stem}_outer_{number}.sqlite")
        if config.study_path
        else None
    )
    inner = replace(
        config,
        folds=config.nested_folds or config.folds,
        nested_folds=None,
        study_path=path,
    )
    effective, trials = (
        tune_predictor(inner, fitting)
        if config.nested_folds is not None
        else (config, pd.DataFrame(columns=["number", "state", "value"]))
    )
    model = build_predictor(effective)
    with (
        threadpool_limits(limits=1),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        model.fit(
            fitting.loc[:, list(INPUT_COLUMNS)],
            fitting.loc[:, list(OUTPUT_COLUMNS)],
        )
        predicted = predict(model, held.loc[:, list(INPUT_COLUMNS)]).to_numpy()
    score = normalized_mae(
        held.loc[:, list(OUTPUT_COLUMNS)].to_numpy(),
        predicted,
        response_scale(fitting.loc[:, list(OUTPUT_COLUMNS)]),
    )
    provenance = {
        "outer_fold": number,
        "fit_ids": fitting["iter_id"].tolist(),
        "held_ids": held["iter_id"].tolist(),
        "inner_folds": config.nested_folds,
        "trial_count": len(trials),
        "cv_seed": seed,
        "parameters": asdict(effective.parameters),
        "refit_trees": effective.refit_trees,
        "early_stopping": stopping_records(model),
        "targetwise_selection": getattr(
            model.named_steps["regressor"], "provenance_", []
        ),
        "targetwise_family_trials": getattr(
            model.named_steps["regressor"], "family_trials_", 0
        ),
        "targetwise_family_search": getattr(
            model.named_steps["regressor"], "family_search_", []
        ),
        "warnings": warning_records(captured),
        "convergence": convergence_records(model),
        "inner_trial_diagnostics": trials.astype(object)
        .where(pd.notna(trials), None)
        .to_dict(orient="records"),
    }
    return OuterFoldResult(
        number,
        tuple(int(value) for value in held_idx),
        predicted,
        score,
        provenance,
        model,
    )


def nested_cross_fitted(
    config: PredictorConfig, table: pd.DataFrame
) -> NestedPredictions:
    if config.nested_folds is None:
        raise ValueError("Especificar pliegues internos para el cross-fitting anidado")
    values = np.full((len(table), len(OUTPUT_COLUMNS)), np.nan)
    folds = np.full(len(table), -1, dtype=int)
    provenance: list[dict[str, JsonValue]] = []
    scores: list[float] = []
    models: list[Pipeline] = []
    for number, (_fit_idx, held_idx) in enumerate(predictor_folds(config, table)):
        result = fit_outer_fold(config, table, number)
        models.append(result.model)
        values[held_idx], folds[held_idx] = result.values, number
        scores.append(result.score)
        provenance.append(result.provenance)
    covered = folds >= 0
    output = pd.DataFrame(
        values[covered],
        index=table.index[covered],
        columns=[f"pred_{name}" for name in OUTPUT_COLUMNS],
    )
    output.insert(0, "iter_id", table["iter_id"])
    output["fold"] = folds[covered]
    return NestedPredictions(output, float(np.mean(scores)), provenance, tuple(models))
