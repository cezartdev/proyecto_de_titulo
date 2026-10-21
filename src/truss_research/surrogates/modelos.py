"""Fábrica de regresores y adaptador de categorías nativas, sin ajustes al importar."""

from collections.abc import Callable
from dataclasses import replace
from typing import Protocol, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from truss_research.cerchas.schemas import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    WEB_CONFIG_VALUES,
)
from truss_research.datos.procesados import JsonValue
from truss_research.surrogates.configuracion import FloatArray, PredictorConfig
from truss_research.surrogates.parada_temprana import (
    BOOSTING_MODELS,
    BoostFit,
    fit_boost,
)

SCALED_MODELS = frozenset(("ridge", "svr", "gpr", "mlp"))

NATIVE_MODELS = BOOSTING_MODELS

MAXIMUM_LIGHTGBM_LEAVES = 31

MATERN_SMOOTHNESS = 2.5


class Regressor(Protocol):
    def fit(self, features: pd.DataFrame, responses: FloatArray) -> Self: ...
    def predict(self, features: pd.DataFrame) -> FloatArray: ...


def validate_features(features: pd.DataFrame) -> pd.DataFrame:
    if tuple(features.columns) != INPUT_COLUMNS or features.empty:
        raise ValueError("Se requieren exactamente las siete entradas ordenadas")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Las entradas deben ser finitas")
    if not features["web_config"].isin(WEB_CONFIG_VALUES).all():
        raise ValueError("Web Config no pertenece al contrato del dataset")
    return features


def categorical_features(features: pd.DataFrame, model: str) -> pd.DataFrame:
    table = validate_features(features).copy()
    values = table["web_config"].astype(int)
    table["web_config"] = (
        values.astype(str)
        if model == "catboost"
        else pd.Categorical(values, categories=WEB_CONFIG_VALUES)
    )
    return table


def native_factory(config: PredictorConfig) -> Regressor:
    p = config.parameters
    if config.model == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=p.trees,
            eval_metric="MAE",
            depth=p.depth,
            learning_rate=p.boosting_rate,
            l2_leaf_reg=p.regularization,
            cat_features=["web_config"],
            random_seed=config.seed,
            thread_count=1,
            task_type="GPU" if config.device == "cuda" else "CPU",
            verbose=False,
            allow_writing_files=False,
        )
    if config.model == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=p.trees,
            max_depth=p.depth,
            learning_rate=p.boosting_rate,
            reg_lambda=p.regularization,
            enable_categorical=True,
            tree_method="hist",
            device=config.device,
            random_state=config.seed,
            n_jobs=1,
        )
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        device_type="cpu",
        n_estimators=p.trees,
        max_depth=p.depth,
        num_leaves=min(MAXIMUM_LIGHTGBM_LEAVES, 2**p.depth),
        learning_rate=p.boosting_rate,
        reg_lambda=p.regularization,
        random_state=config.seed,
        n_jobs=1,
        verbosity=-1,
    )


class NativeBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, config: PredictorConfig) -> None:
        self.config = config

    def fit(self, features: pd.DataFrame, responses: FloatArray) -> Self:
        return self.fit_evaluated(BoostFit(features, np.asarray(responses)))

    def fit_evaluated(self, data: BoostFit) -> Self:
        features, responses, evaluation = data.features, data.responses, data.evaluation
        table = categorical_features(features, self.config.model)
        values = np.asarray(responses, dtype=float)
        self.estimators_: list[Regressor] = []
        self.early_stopping_: list[dict[str, JsonValue]] = []
        for column in range(len(OUTPUT_COLUMNS)):
            config = self.config
            if config.refit_trees and evaluation is None:
                config = replace(
                    config,
                    parameters=replace(
                        config.parameters, trees=config.refit_trees[column]
                    ),
                    refit_trees=(),
                )
            estimator = native_factory(config)
            held = (
                None
                if evaluation is None
                else (
                    categorical_features(evaluation[0], config.model),
                    np.asarray(evaluation[1])[:, column],
                )
            )
            record = fit_boost(
                estimator, BoostFit(table, values[:, column], held), config
            )
            self.early_stopping_.append(record)
            self.estimators_.append(estimator)
        self.n_features_in_ = len(INPUT_COLUMNS)
        return self

    def predict(self, features: pd.DataFrame) -> FloatArray:
        table = categorical_features(features, self.config.model)
        return np.column_stack([model.predict(table) for model in self.estimators_])


def preprocessing(config: PredictorConfig) -> ColumnTransformer:
    numeric = [name for name in INPUT_COLUMNS if name != "web_config"]
    return ColumnTransformer(
        [
            (
                "web_config",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["web_config"],
            ),
            (
                "numeric",
                StandardScaler() if config.model in SCALED_MODELS else "passthrough",
                numeric,
            ),
        ]
    )


def classic_regressor(config: PredictorConfig) -> BaseEstimator:
    from truss_research.surrogates.convergencia import GprOptimizer

    p = config.parameters
    factories: dict[str, Callable[[], BaseEstimator]] = {
        "dummy": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: MultiOutputRegressor(Ridge(alpha=p.regularization)),
        "random_forest": lambda: MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=p.trees,
                max_depth=p.depth,
                min_samples_leaf=p.min_samples_leaf,
                random_state=config.seed,
                n_jobs=1,
            )
        ),
        "svr": lambda: MultiOutputRegressor(
            TransformedTargetRegressor(
                regressor=SVR(C=p.svr_c, epsilon=p.svr_epsilon),
                transformer=StandardScaler(),
            )
        ),
        "gpr": lambda: MultiOutputRegressor(
            GaussianProcessRegressor(
                kernel=ConstantKernel() * Matern(nu=MATERN_SMOOTHNESS),
                alpha=p.gpr_noise,
                normalize_y=True,
                optimizer=GprOptimizer(p),
                random_state=config.seed,
            )
        ),
    }
    if config.model == "mlp":
        from truss_research.surrogates.mlp import TorchRegressor

        return TorchRegressor(p, config.seed, config.device)
    return factories[config.model]()


def build_predictor(config: PredictorConfig) -> Pipeline:
    from truss_research.surrogates.mlp_temporal import TemporalMLP, temporal_predictor

    if config.model == "mlp" and temporal_predictor(config):
        return Pipeline([("regressor", TemporalMLP(config))])
    if config.model == "tabpfn":
        from truss_research.surrogates.tabpfn import TabPFNRegressor

        return Pipeline([("regressor", TabPFNRegressor(config.tabpfn, config.seed))])
    if config.model == "targetwise":
        from truss_research.surrogates.especializacion import TargetwiseRegressor

        return Pipeline([("regressor", TargetwiseRegressor(config))])
    if config.model in NATIVE_MODELS:
        return Pipeline([("regressor", NativeBoostRegressor(config))])
    return Pipeline(
        [
            ("preprocessing", preprocessing(config)),
            ("regressor", classic_regressor(config)),
        ]
    )
