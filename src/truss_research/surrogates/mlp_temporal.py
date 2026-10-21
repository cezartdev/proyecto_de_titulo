"""Reserva por generaciones dentro del ajuste; transformadores sin acceso al futuro."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.procesados import manifest_protocol, read_manifest
from truss_research.surrogates.configuracion import FloatArray, PredictorConfig
from truss_research.surrogates.mlp import VALIDATION_FRACTION, TorchRegressor


def temporal_predictor(config: PredictorConfig) -> bool:
    if not (config.data_dir / "manifest.json").is_file():
        return False
    return manifest_protocol(read_manifest(config.data_dir)) == "first_appearance"


class TemporalMLP(RegressorMixin, BaseEstimator):
    def __init__(self, config: PredictorConfig) -> None:
        self.config = config

    def fit(self, features: pd.DataFrame, responses: FloatArray) -> Self:
        from truss_research.surrogates.modelos import preprocessing

        known = pd.read_csv(
            self.config.data_dir / "train.csv", float_precision="round_trip"
        )
        generations = features.merge(
            known[[*INPUT_COLUMNS, "generation"]],
            on=list(INPUT_COLUMNS),
            how="left",
            validate="one_to_one",
        ).generation
        if generations.isna().any():
            raise ValueError("La reserva MLP carece de procedencia temporal")
        ordered = np.array(sorted(generations.unique()))
        if len(ordered) < 2:
            raise ValueError("MLP temporal requiere al menos dos generaciones internas")
        cutoff = ordered[max(1, int(len(ordered) * (1 - VALIDATION_FRACTION)))]
        fitting = np.flatnonzero(generations.lt(cutoff))
        stopping = np.flatnonzero(generations.ge(cutoff))
        self.transformer_ = preprocessing(self.config)
        train_x = self.transformer_.fit_transform(features.iloc[fitting])
        stop_x = self.transformer_.transform(features.iloc[stopping])
        values = np.asarray(responses).reshape(len(features), -1)
        self.regressor_ = TorchRegressor(
            self.config.parameters, self.config.seed, self.config.device
        )
        self.regressor_.fit_reserved(
            (train_x, values[fitting]), (stop_x, values[stopping])
        )
        self.early_stopping_ = self.regressor_.early_stopping_
        self.early_stopping_[0]["evaluation"] = "internal_forward_generation_reserve"
        self.early_stopping_[0]["training_max_generation"] = int(
            generations.iloc[fitting].max()
        )
        self.early_stopping_[0]["stopping_min_generation"] = int(
            generations.iloc[stopping].min()
        )
        self.n_features_in_ = len(INPUT_COLUMNS)
        return self

    def predict(self, features: pd.DataFrame) -> FloatArray:
        values = self.regressor_.predict(self.transformer_.transform(features))
        return values.ravel() if self.config.target_index is not None else values
