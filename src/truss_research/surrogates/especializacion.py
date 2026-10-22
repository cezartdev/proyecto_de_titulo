"""Sustituto compuesto: selección interna de familias y ajuste por respuesta."""

import warnings
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.procesados import JsonValue
from truss_research.surrogates.configuracion import (
    FloatArray,
    PredictorConfig,
)
from truss_research.surrogates.mlp import TorchRegressor
from truss_research.surrogates.modelos import (
    NATIVE_MODELS,
    categorical_features,
    classic_regressor,
    native_factory,
    preprocessing,
)
from truss_research.surrogates.parada_temprana import (
    BoostFit,
    fit_boost,
    fit_search_fold,
    selected_config,
    stopping_records,
)
from truss_research.surrogates.pliegues import predictor_folds
from truss_research.surrogates.poda import FoldProgress, report_fold


class ScalarTorchRegressor(TorchRegressor):
    def fit(self, features: FloatArray, responses: FloatArray) -> Self:
        super().fit(features, np.asarray(responses).reshape(-1, 1))
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        return super().predict(features).ravel()


class NativeScalarRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, config: PredictorConfig) -> None:
        self.config = config

    def fit(self, features: pd.DataFrame, responses: FloatArray) -> Self:
        return self.fit_evaluated(BoostFit(features, np.asarray(responses)))

    def fit_evaluated(self, data: BoostFit) -> Self:
        features, responses, evaluation = data.features, data.responses, data.evaluation
        config = self.config
        if config.refit_trees and evaluation is None:
            count = (
                config.refit_trees[0]
                if len(config.refit_trees) == 1
                else config.refit_trees[config.target_index]
            )
            config = replace(
                config,
                parameters=replace(config.parameters, trees=count),
                refit_trees=(),
            )
        self.estimator_ = native_factory(config)
        table = categorical_features(features, config.model)
        values = np.asarray(responses).ravel()
        self.early_stopping_: list[dict[str, JsonValue]] = []
        held = (
            None
            if evaluation is None
            else (
                categorical_features(evaluation[0], config.model),
                np.asarray(evaluation[1]).ravel(),
            )
        )
        self.early_stopping_.append(
            fit_boost(self.estimator_, BoostFit(table, values, held), config)
        )
        self.n_features_in_ = len(INPUT_COLUMNS)
        return self

    def predict(self, features: pd.DataFrame) -> FloatArray:
        return np.asarray(
            self.estimator_.predict(categorical_features(features, self.config.model))
        ).ravel()


def scalar_predictor(config: PredictorConfig) -> Pipeline:
    from truss_research.surrogates.mlp_temporal import TemporalMLP, temporal_predictor

    if config.model == "mlp" and temporal_predictor(config):
        return Pipeline([("regressor", TemporalMLP(config))])
    if config.model == "tabpfn":
        from truss_research.surrogates.tabpfn import TabPFNRegressor

        return Pipeline(
            [
                (
                    "regressor",
                    TabPFNRegressor(config.tabpfn, config.seed, config.target_index),
                )
            ]
        )
    if config.model in NATIVE_MODELS:
        return Pipeline([("regressor", NativeScalarRegressor(config))])
    estimator = classic_regressor(config)
    if isinstance(estimator, MultiOutputRegressor):
        estimator = estimator.estimator
    if isinstance(estimator, TorchRegressor):
        estimator = ScalarTorchRegressor(config.parameters, config.seed)
    return Pipeline(
        [("preprocessing", preprocessing(config)), ("regressor", estimator)]
    )


def scalar_search(
    config: PredictorConfig, table: pd.DataFrame, target: int
) -> tuple[PredictorConfig, pd.DataFrame]:
    from truss_research.surrogates.experimentos import (
        managed_study,
        save_sampler,
        suggested_parameters,
        warning_records,
    )

    if config.trials == 0 or config.model in ("dummy", "tabpfn"):
        return config, pd.DataFrame(columns=["number", "state", "value"])
    import optuna

    config = replace(config, target_index=target)
    folds = tuple(predictor_folds(config, table))
    with managed_study(config, table) as study:

        def objective(trial: optuna.Trial) -> float:
            p = suggested_parameters(trial, config)
            trial.set_user_attr("effective_parameters", asdict(p))
            scores = []
            stopping = []
            for fitting, held in folds:
                train, valid = table.iloc[fitting], table.iloc[held]
                candidate = replace(config, parameters=p, refit_trees=())
                model = scalar_predictor(candidate)
                response = train[OUTPUT_COLUMNS[target]].to_numpy()
                from truss_research.surrogates.convergencia import NumericalFitError

                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        fit_search_fold(model, candidate, (train, valid))
                except NumericalFitError as error:
                    trial.set_user_attr("convergence_failure", error.attempts)
                    raise
                trial.set_user_attr(
                    f"fold_{len(scores)}_warnings", warning_records(captured)
                )
                values = np.asarray(
                    model.predict(valid.loc[:, list(INPUT_COLUMNS)])
                ).ravel()
                scores.append(
                    float(
                        np.mean(
                            np.abs(valid[OUTPUT_COLUMNS[target]].to_numpy() - values)
                        )
                        / (response.std() or 1.0)
                    )
                )
                stopping.append(stopping_records(model))
                trial.set_user_attr("early_stopping", stopping)
                report_fold(
                    trial, FoldProgress(tuple(scores), len(folds)), config.pruning
                )
            return float(np.mean(scores))

        for trial in study.trials:
            if trial.state.name == "RUNNING":
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        study.optimize(
            objective,
            n_trials=max(0, config.trials - len(study.trials)),
            catch=(ValueError, RuntimeError),
            callbacks=[save_sampler] if config.study_path else [],
        )
        successful = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not successful:
            raise ValueError("Todos los ensayos por respuesta fallaron")
        return selected_config(config, study.best_trial), study.trials_dataframe()


class TargetwiseRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, config: PredictorConfig) -> None:
        self.config = config

    def candidates(
        self, table: pd.DataFrame
    ) -> list[tuple[PredictorConfig, FloatArray]]:
        from truss_research.surrogates.experimentos import (
            cross_fitted_predictions,
            tune_predictor,
        )

        candidates = []
        for name in self.config.candidate_models:
            path = self.search_path(f"family_{name}")
            config = replace(
                self.config,
                model=name,
                trials=0 if name == "tabpfn" else self.config.trials,
                folds=self.config.nested_folds
                or self.config.final_tuning_folds
                or self.config.folds,
                nested_folds=None,
                reuse_dir=None,
                preparation_dir=None,
                study_path=path,
            )
            effective, trials = tune_predictor(config, table)
            evaluation_seed = (
                effective.cv_seed if effective.cv_seed is not None else effective.seed
            ) + 1
            oof, _ = cross_fitted_predictions(
                replace(effective, cv_seed=evaluation_seed), table
            )
            oof.attrs.pop("fold_models", None)
            error = np.abs(
                table.loc[oof.index, list(OUTPUT_COLUMNS)].to_numpy()
                - oof.loc[:, [f"pred_{n}" for n in OUTPUT_COLUMNS]].to_numpy()
            ).mean(axis=0)
            candidates.append((effective, error))
            self.family_trials_ += len(trials)
            self.family_search_.append(
                {
                    "family": name,
                    "target_mae": error.tolist(),
                    "trial_count": len(trials),
                    "trials": trials.astype(object)
                    .where(pd.notna(trials), None)
                    .to_dict(orient="records"),
                }
            )
        return candidates

    def search_path(self, label: str) -> Path | None:
        path = self.config.study_path
        return (
            path.with_name(f"{path.stem}_{label}.sqlite") if path is not None else None
        )

    def fit(self, features: pd.DataFrame, responses: FloatArray) -> Self:
        from truss_research.surrogates.experimentos import load_predictor

        table = features.copy()
        table.loc[:, list(OUTPUT_COLUMNS)] = np.asarray(responses)
        table["iter_id"] = [f"fit_{index}" for index in table.index]
        from truss_research.surrogates.mlp_temporal import temporal_predictor

        if temporal_predictor(self.config):
            source = pd.read_csv(
                self.config.data_dir / "train.csv", float_precision="round_trip"
            )
            matched = features.merge(
                source[[*INPUT_COLUMNS, "iter_id", "generation", "first_generation"]],
                on=list(INPUT_COLUMNS),
                how="left",
                validate="one_to_one",
            )
            if matched.generation.isna().any():
                raise ValueError("Falta procedencia temporal del compuesto")
            for column in ("iter_id", "generation", "first_generation"):
                table[column] = matched[column].to_numpy()
        self.family_trials_ = 0
        self.family_search_: list[dict[str, JsonValue]] = []
        self.components_: list[tuple[PredictorConfig, Pipeline]] = []
        self.provenance_: list[dict[str, JsonValue]] = []
        reused = self.config.reuse_dir is not None
        reference = (
            load_predictor(self.config.reuse_dir).named_steps["regressor"]
            if reused
            else None
        )
        started = perf_counter()
        candidates = [] if reused else self.candidates(table)
        self.selection_seconds_ = perf_counter() - started
        for target, name in enumerate(OUTPUT_COLUMNS):
            if reused:
                effective = replace(
                    reference.components_[target][0],
                    seed=self.config.seed,
                    study_path=None,
                )
                trials = pd.DataFrame()
            else:
                started = perf_counter()
                selected = min(
                    range(len(candidates)), key=lambda i: (candidates[i][1][target], i)
                )
                selected_config = replace(
                    candidates[selected][0],
                    study_path=self.search_path(f"target_{target}"),
                )
                effective, trials = scalar_search(selected_config, table, target)
                self.selection_seconds_ += perf_counter() - started
            effective = replace(effective, target_index=target)
            model = scalar_predictor(effective)
            model.fit(features, np.asarray(responses)[:, target])
            self.components_.append((effective, model))
            self.provenance_.append(
                {
                    "target": name,
                    "family": effective.model,
                    "parameters": asdict(effective.parameters),
                    "refit_trees": effective.refit_trees,
                    "early_stopping": stopping_records(model),
                    "trial_count": len(trials),
                    "search_reused": reused,
                    "trials": trials.astype(object)
                    .where(pd.notna(trials), None)
                    .to_dict(orient="records"),
                }
            )
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, features: pd.DataFrame) -> FloatArray:
        return np.column_stack(
            [
                np.asarray(model.predict(features)).ravel()
                for _, model in self.components_
            ]
        )
