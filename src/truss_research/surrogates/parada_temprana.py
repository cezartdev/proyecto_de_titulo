"""Selección interna de árboles y reajuste de boosting sin datos de evaluación externa."""

import logging
from dataclasses import dataclass, replace
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import JsonValue
from truss_research.surrogates.configuracion import FloatArray, PredictorConfig

if TYPE_CHECKING:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from optuna.trial import FrozenTrial
    from xgboost import XGBRegressor


BOOSTING_MODELS = frozenset(("catboost", "xgboost", "lightgbm"))

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoostFit:
    features: pd.DataFrame
    responses: FloatArray
    evaluation: tuple[pd.DataFrame, FloatArray] | None = None


@dataclass(frozen=True)
class BoostIterations:
    executed: int
    retained: int


def fit_catboost(
    estimator: "CatBoostRegressor", data: BoostFit, config: PredictorConfig
) -> BoostIterations:
    enabled = (
        data.evaluation is not None and config.parameters.early_stopping_rounds > 0
    )
    if enabled:
        estimator.fit(
            data.features,
            data.responses,
            eval_set=data.evaluation,
            early_stopping_rounds=config.parameters.early_stopping_rounds,
            use_best_model=True,
            verbose=False,
        )
    else:
        estimator.fit(data.features, data.responses)
    history = estimator.get_evals_result()["learn"]
    return BoostIterations(len(next(iter(history.values()))), estimator.tree_count_)


def fit_xgboost(
    estimator: "XGBRegressor", data: BoostFit, config: PredictorConfig
) -> BoostIterations:
    from xgboost.callback import EarlyStopping

    enabled = (
        data.evaluation is not None and config.parameters.early_stopping_rounds > 0
    )
    if not enabled:
        estimator.fit(data.features, data.responses)
        count = estimator.get_booster().num_boosted_rounds()
        return BoostIterations(count, count)
    estimator.set_params(
        eval_metric="mae",
        callbacks=[
            EarlyStopping(
                rounds=config.parameters.early_stopping_rounds, save_best=True
            )
        ],
    )
    estimator.fit(
        data.features, data.responses, eval_set=[data.evaluation], verbose=False
    )
    history = estimator.evals_result()["validation_0"]["mae"]
    return BoostIterations(len(history), estimator.get_booster().num_boosted_rounds())


def fit_lightgbm(
    estimator: "LGBMRegressor", data: BoostFit, config: PredictorConfig
) -> BoostIterations:
    from lightgbm import early_stopping

    enabled = (
        data.evaluation is not None and config.parameters.early_stopping_rounds > 0
    )
    if not enabled:
        estimator.fit(data.features, data.responses)
        return BoostIterations(estimator.n_iter_, estimator.n_iter_)
    estimator.set_params(metric="l1")
    estimator.fit(
        data.features,
        data.responses,
        eval_X=data.evaluation[0],
        eval_y=data.evaluation[1],
        callbacks=[
            early_stopping(
                config.parameters.early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )
    history = estimator.evals_result_["valid_0"]["l1"]
    retained = estimator.booster_.current_iteration()
    return BoostIterations(len(history), retained)


def fit_boost(
    estimator: "CatBoostRegressor | XGBRegressor | LGBMRegressor",
    data: BoostFit,
    config: PredictorConfig,
) -> dict[str, JsonValue]:
    started = perf_counter()
    fitters = {
        "catboost": fit_catboost,
        "xgboost": fit_xgboost,
        "lightgbm": fit_lightgbm,
    }
    counts = fitters[config.model](estimator, data, config)
    enabled = (
        data.evaluation is not None and config.parameters.early_stopping_rounds > 0
    )
    stopped = enabled and counts.executed < config.parameters.trees
    record: dict[str, JsonValue] = {
        "executed_trees": counts.executed,
        "retained_trees": counts.retained,
        "maximum_trees": config.parameters.trees,
        "early_stopped": stopped,
        "stop_reason": "patience" if stopped else "limit",
        "fit_seconds": perf_counter() - started,
    }
    if enabled:
        logger.info("%s: %s", config.model, record)
    return record


def fit_search_fold(
    model: Pipeline, config: PredictorConfig, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    fitting, held = tables
    target = config.target_index
    responses = fitting.loc[:, list(OUTPUT_COLUMNS)].to_numpy(dtype=float)
    evaluation = held.loc[:, list(OUTPUT_COLUMNS)].to_numpy(dtype=float)
    if config.model in BOOSTING_MODELS and config.parameters.early_stopping_rounds:
        model.named_steps["regressor"].fit_evaluated(
            BoostFit(
                fitting.loc[:, list(INPUT_COLUMNS)],
                responses if target is None else responses[:, target],
                (
                    held.loc[:, list(INPUT_COLUMNS)],
                    evaluation if target is None else evaluation[:, target],
                ),
            )
        )
        return
    model.fit(
        fitting.loc[:, list(INPUT_COLUMNS)],
        responses if target is None else responses[:, target],
    )


def selected_config(config: PredictorConfig, trial: "FrozenTrial") -> PredictorConfig:
    from truss_research.surrogates.configuracion import ModelParameters

    effective = replace(
        config,
        parameters=ModelParameters(**trial.user_attrs["effective_parameters"]),
        refit_trees=(),
    )
    records = trial.user_attrs.get("early_stopping", [])
    if (
        effective.model not in BOOSTING_MODELS
        or not effective.parameters.early_stopping_rounds
    ):
        return effective
    if not records:
        raise ValueError("Faltan las iteraciones internas del ensayo ganador")
    counts = tuple(
        int(median(fold[column]["retained_trees"] for fold in records))
        for column in range(len(records[0]))
    )
    return replace(effective, refit_trees=counts)


def stopping_records(model: Pipeline) -> list[dict[str, JsonValue]]:
    return getattr(model.named_steps["regressor"], "early_stopping_", [])
