"""Ajuste, cross-fitting y persistencia de experimentos predictivos nuevos."""

import base64
import hashlib
import json
import logging
import pickle
import platform
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.artefactos import (
    dataset_identity,
    portable_payload,
    read_artifact,
    replace_artifact,
)
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import (
    JsonValue,
    load_design_split,
    load_structural_limits,
)
from truss_research.evaluacion.metricas import (
    crossfit_metrics,
    regional_metrics,
    regression_metrics,
)
from truss_research.evaluacion.pareto import compliance_report
from truss_research.evaluacion.recursos import measure_phase
from truss_research.surrogates.configuracion import (
    BASE_MODEL_NAMES,
    SCHEMA_VERSION,
    TIMING_REPEATS,
    ModelName,
    ModelParameters,
    PredictorConfig,
    PredictorResult,
    normalized_mae,
    response_scale,
)
from truss_research.surrogates.convergencia import (
    NumericalFitError,
    convergence_records,
)
from truss_research.surrogates.modelos import build_predictor, validate_features
from truss_research.surrogates.parada_temprana import (
    BOOSTING_MODELS,
    fit_search_fold,
    selected_config,
    stopping_records,
)
from truss_research.surrogates.persistencia import (
    copy_model,
    load_model,
    model_signatures,
    save_model,
)
from truss_research.surrogates.poda import (
    FoldProgress,
    PruningParameters,
    create_pruner,
    pruning_summary,
    report_fold,
)
from truss_research.surrogates.tabpfn import TabPFNParameters

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"

OOF_FILE = "training_oof.csv"

SVR_C_RANGE = (0.1, 100.0)

RIDGE_REGULARIZATION_RANGE = (1e-6, 1e3)

SVR_EPSILON_RANGE = (0.001, 0.2)

GPR_NOISE_RANGE = (1e-8, 0.1)

MLP_WIDTHS = (32, 64, 128)

MLP_REGULARIZATION_RANGE = (1e-6, 0.01)

MLP_RATE_RANGE = (1e-4, 0.01)

TREE_COUNT_RANGE = (50, 300)

TREE_COUNT_STEP = 50

TREE_DEPTH_RANGE = (3, 10)

TREE_LEAF_RANGE = (1, 10)

BOOST_REGULARIZATION_RANGE = (0.01, 10.0)

BOOST_RATE_RANGE = (0.01, 0.2)

if TYPE_CHECKING:
    import optuna
    from optuna import Trial


def write_json(path: Path, payload: JsonValue) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            portable_payload(json.loads(json.dumps(payload, default=str)), path),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    replace_artifact(temporary, path)


def read_run(directory: Path) -> dict[str, JsonValue]:
    return read_artifact(directory / "run.json")


def data_signature(directory: Path) -> str:
    return dataset_identity(directory)


def predict(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    values = np.asarray(model.predict(validate_features(features)), dtype=float)
    if (
        values.shape != (len(features), len(OUTPUT_COLUMNS))
        or not np.isfinite(values).all()
    ):
        raise ValueError("El predictor devolvió respuestas no finitas o mal alineadas")
    return pd.DataFrame(values, columns=OUTPUT_COLUMNS, index=features.index)


def load_predictor(directory: Path, device: str | None = None) -> Pipeline:
    metadata = read_run(directory)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("El predictor no corresponde al formato vigente")
    expected = metadata.get("model_signature")
    if expected is not None and source_digest(directory / MODEL_FILE) != expected:
        raise ValueError("La firma del predictor no coincide con el manifiesto")
    if metadata.get("model_artifacts") != model_signatures(directory):
        raise ValueError("Los estados del predictor cambiaron")
    return load_model(directory, device)


def config_from_run(directory: Path) -> PredictorConfig:
    payload = read_run(directory).get("config")
    if not isinstance(payload, dict) or not isinstance(payload.get("parameters"), dict):
        raise ValueError("La ejecución no contiene configuración predictiva versionada")
    parameters = ModelParameters(**cast(dict[str, int | float], payload["parameters"]))
    return PredictorConfig(
        data_dir=Path(str(payload["data_dir"])),
        output_dir=directory,
        model=cast(ModelName, payload["model"]),
        seed=int(str(payload["seed"])),
        folds=int(str(payload["folds"])),
        trials=0,
        parameters=parameters,
        tabpfn=TabPFNParameters(**payload.get("tabpfn", {})),
        pruning=PruningParameters(**payload.get("pruning", {})),
        refit_trees=tuple(payload.get("refit_trees", ())),
        device=str(payload.get("device", "cpu")),
        cv_seed=cast(int | None, payload.get("cv_seed")),
        nested_folds=cast(int | None, payload.get("nested_folds")),
        final_tuning_folds=cast(int | None, payload.get("final_tuning_folds")),
        candidate_models=tuple(
            cast(
                list[ModelName], payload.get("candidate_models", list(BASE_MODEL_NAMES))
            )
        ),
    )


def cross_fitted_predictions(
    config: PredictorConfig, table: pd.DataFrame, trial: "optuna.Trial | None" = None
) -> tuple[pd.DataFrame, float]:
    from truss_research.surrogates.pliegues import predictor_folds

    values = np.full((len(table), len(OUTPUT_COLUMNS)), np.nan, dtype=float)
    fold_ids = np.full(len(table), -1, dtype=int)
    scores: list[float] = []
    diagnostics: list[dict[str, JsonValue]] = []
    models: list[Pipeline] = []
    folds = tuple(predictor_folds(config, table))
    for number, (fit_idx, held_idx) in enumerate(folds):
        fitting, held = table.iloc[fit_idx], table.iloc[held_idx]
        model = build_predictor(config)
        with (
            threadpool_limits(limits=1),
            warnings.catch_warnings(record=True) as captured,
        ):
            warnings.simplefilter("always")
            if trial is not None:
                fit_search_fold(model, config, (fitting, held))
            else:
                model.fit(
                    fitting.loc[:, list(INPUT_COLUMNS)],
                    fitting.loc[:, list(OUTPUT_COLUMNS)],
                )
            predicted = predict(model, held.loc[:, list(INPUT_COLUMNS)]).to_numpy()
        models.append(model)
        diagnostics.append(
            {
                "fold": number,
                "warnings": warning_records(captured),
                "convergence": convergence_records(model),
            }
        )
        values[held_idx] = predicted
        fold_ids[held_idx] = number
        scores.append(
            normalized_mae(
                held.loc[:, list(OUTPUT_COLUMNS)].to_numpy(),
                predicted,
                response_scale(fitting.loc[:, list(OUTPUT_COLUMNS)]),
            )
        )
        if trial is not None:
            trial.set_user_attr(
                "early_stopping", [stopping_records(fitted) for fitted in models]
            )
            trial.set_user_attr("fold_warnings", diagnostics)
            report_fold(trial, FoldProgress(tuple(scores), len(folds)), config.pruning)
    covered = fold_ids >= 0
    output = pd.DataFrame(
        values[covered],
        index=table.index[covered],
        columns=[f"pred_{name}" for name in OUTPUT_COLUMNS],
    )
    output.insert(0, "iter_id", table["iter_id"])
    output["fold"] = fold_ids[covered]
    output.attrs["fold_warnings"] = diagnostics
    output.attrs["fold_models"] = tuple(models)
    return output, float(np.mean(scores))


def suggested_parameters(trial: "Trial", config: PredictorConfig) -> ModelParameters:
    p = config.parameters
    if config.model == "ridge":
        return replace(
            p,
            regularization=trial.suggest_float(
                "regularization", *RIDGE_REGULARIZATION_RANGE, log=True
            ),
        )
    if config.model == "svr":
        return replace(
            p,
            svr_c=trial.suggest_float("svr_c", *SVR_C_RANGE, log=True),
            svr_epsilon=trial.suggest_float(
                "svr_epsilon", *SVR_EPSILON_RANGE, log=True
            ),
        )
    if config.model == "gpr":
        return replace(
            p, gpr_noise=trial.suggest_float("gpr_noise", *GPR_NOISE_RANGE, log=True)
        )
    if config.model == "mlp":
        return replace(
            p,
            hidden_width=trial.suggest_categorical("hidden_width", MLP_WIDTHS),
            mlp_weight_decay=trial.suggest_float(
                "mlp_weight_decay", *MLP_REGULARIZATION_RANGE, log=True
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", *MLP_RATE_RANGE, log=True
            ),
        )
    if config.model == "random_forest":
        return replace(
            p,
            trees=trial.suggest_int("trees", *TREE_COUNT_RANGE, step=TREE_COUNT_STEP),
            depth=trial.suggest_int("depth", *TREE_DEPTH_RANGE),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", *TREE_LEAF_RANGE),
        )
    return replace(
        p,
        trees=trial.suggest_int("trees", *TREE_COUNT_RANGE, step=TREE_COUNT_STEP),
        depth=trial.suggest_int("depth", *TREE_DEPTH_RANGE),
        regularization=trial.suggest_float(
            "regularization", *BOOST_REGULARIZATION_RANGE, log=True
        ),
        boosting_rate=trial.suggest_float("boosting_rate", *BOOST_RATE_RANGE, log=True),
    )


def tune_predictor(
    config: PredictorConfig, table: pd.DataFrame
) -> tuple[PredictorConfig, pd.DataFrame]:
    if config.trials == 0 or config.model in ("dummy", "targetwise", "tabpfn"):
        return config, pd.DataFrame(columns=["number", "state", "value"])
    import optuna

    with managed_study(config, table) as study:

        def objective(trial: optuna.Trial) -> float:
            parameters = suggested_parameters(trial, config)
            logger.info(
                "Ensayo %s: familia=%s semilla=%s parámetros=%s",
                trial.number,
                config.model,
                config.seed,
                asdict(parameters),
            )
            trial.set_user_attr("effective_parameters", asdict(parameters))
            try:
                candidate = replace(config, parameters=parameters, refit_trees=())
                oof, score = (
                    cross_fitted_predictions(candidate, table, trial)
                    if config.pruning.enabled
                    or config.model == "mlp"
                    or (
                        config.model in BOOSTING_MODELS
                        and config.parameters.early_stopping_rounds
                    )
                    else cross_fitted_predictions(candidate, table)
                )
            except NumericalFitError as error:
                trial.set_user_attr("convergence_failure", error.attempts)
                raise
            trial.set_user_attr("fold_warnings", oof.attrs["fold_warnings"])
            return score

        unfinished = [trial for trial in study.trials if trial.state.name == "RUNNING"]
        for trial in unfinished:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        remaining = config.trials - len(study.trials)
        study.optimize(
            objective,
            n_trials=max(0, remaining),
            catch=(ValueError, RuntimeError),
            callbacks=[save_sampler] if config.study_path is not None else None,
        )
        successful = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not successful:
            raise ValueError(
                "Todos los ensayos fallaron; revisar los diagnósticos de Optuna"
            )
        return selected_config(config, study.best_trial), study.trials_dataframe()


@contextmanager
def managed_study(
    config: PredictorConfig, table: pd.DataFrame
) -> Iterator["optuna.Study"]:
    import optuna

    storage = None
    if config.study_path is not None:
        config.study_path.parent.mkdir(parents=True, exist_ok=True)
        storage = optuna.storages.RDBStorage(
            "sqlite:///" + config.study_path.resolve().as_posix()
        )
    try:
        study = persistent_study(config, table, storage)
        try:
            yield study
        finally:
            if config.study_path is not None:
                write_json(
                    config.study_path.with_suffix(".pruning.json"),
                    {"parameters": asdict(config.pruning), **pruning_summary(study)},
                )
    finally:
        if storage is not None:
            try:
                storage.remove_session()
            finally:
                storage.engine.dispose()


def persistent_study(
    config: PredictorConfig,
    table: pd.DataFrame,
    storage: "optuna.storages.RDBStorage | None",
) -> "optuna.Study":
    """Recupera intentos terminados y rechaza estudios de otro protocolo."""
    import optuna

    settings = asdict(config)
    for name in ("output_dir", "study_path", "preparation_dir"):
        settings.pop(name)
    settings["data_dir"] = data_signature(config.data_dir)
    settings["training_rows"] = table["iter_id"].tolist()
    settings["search_ranges"] = {
        name: value for name, value in globals().items() if name.endswith("_RANGE")
    }
    signature = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()
    study = optuna.create_study(
        direction="minimize",
        study_name="predictor",
        storage=storage,
        load_if_exists=storage is not None,
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=create_pruner(config.pruning),
    )
    previous = study.user_attrs.get("protocol_signature")
    if previous is not None and previous != signature:
        raise ValueError("El estudio Optuna pertenece a otra configuración")
    study.set_user_attr("protocol_signature", signature)
    if len(study.trials) > config.trials:
        raise ValueError("El estudio Optuna excede el presupuesto de intentos")
    snapshot = study.user_attrs.get("sampler_snapshot")
    if snapshot is not None:
        sampler = pickle.loads(base64.b64decode(snapshot))
        if not isinstance(sampler, optuna.samplers.TPESampler):
            raise ValueError("El estudio no contiene el muestreador esperado")
        study.sampler = sampler
    return study


def save_sampler(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
    """La instantánea vive en la misma base local que los ensayos registrados."""
    snapshot = base64.b64encode(pickle.dumps(study.sampler)).decode("ascii")
    study.set_user_attr("sampler_snapshot", snapshot)


def timing_benchmark(model: Pipeline, features: pd.DataFrame) -> dict[str, float | int]:
    batches: list[float] = []
    individuals: list[float] = []
    with threadpool_limits(limits=1):
        model.predict(features)
        for _ in range(TIMING_REPEATS):
            start = perf_counter()
            model.predict(features)
            batches.append(perf_counter() - start)
            start = perf_counter()
            model.predict(features.iloc[:1])
            individuals.append(perf_counter() - start)
    return {
        "batch_median_seconds": float(np.median(batches)),
        "individual_median_seconds": float(np.median(individuals)),
        "batch_rows": len(features),
        "repeats": TIMING_REPEATS,
    }


def prediction_report(table: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        name
        for name in ("iter_id", "generation", "solution", "web_config")
        if name in table
    ]
    return table.loc[:, [*identifiers, *OUTPUT_COLUMNS]].join(
        predicted.add_prefix("pred_")
    )


def train_predictor(config: PredictorConfig) -> PredictorResult:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_predictor(config)
    metadata = read_run(config.output_dir)
    metadata["warnings"] = warning_records(captured)
    write_json(config.output_dir / "run.json", metadata)
    for item in metadata["warnings"]:
        logger.warning("%s (%s): %s", item["category"], item["count"], item["message"])
    return result


def warning_records(
    captured: list[warnings.WarningMessage],
) -> list[dict[str, JsonValue]]:
    counts = Counter((item.category.__name__, str(item.message)) for item in captured)
    return [
        {"category": category, "message": message, "count": count}
        for (category, message), count in counts.items()
    ]


def reuse_predictor(config: PredictorConfig) -> PredictorConfig:
    if config.reuse_dir is None:
        return config
    reference = read_run(config.reuse_dir)
    previous = config_from_run(config.reuse_dir)
    if reference["dataset_signature"] != data_signature(config.data_dir) or (
        previous.model,
        previous.folds,
        previous.cv_seed,
        previous.nested_folds,
    ) != (config.model, config.folds, config.cv_seed, config.nested_folds):
        raise ValueError(
            "La búsqueda compartida no corresponde a la familia, datos o pliegues"
        )
    if reference["model_signature"] != source_digest(
        config.reuse_dir / MODEL_FILE
    ) or reference["oof_signature"] != source_digest(config.reuse_dir / OOF_FILE):
        raise ValueError("La ejecución compartida perdió su integridad")
    return replace(
        config,
        parameters=previous.parameters,
        refit_trees=previous.refit_trees,
        trials=0,
    )


def fit_predictor(config: PredictorConfig) -> PredictorResult:
    if config.output_dir.exists():
        raise ValueError("El destino ya existe; elegir una ejecución nueva")
    signature = data_signature(config.data_dir)
    split = load_design_split(config.data_dir)
    from truss_research.surrogates.preparacion import load_prepared_predictor

    prepared = (
        load_prepared_predictor(config) if config.preparation_dir is not None else None
    )
    tuning_config = (
        replace(
            config,
            folds=config.final_tuning_folds or config.nested_folds,
            nested_folds=None,
        )
        if config.final_tuning_folds or config.nested_folds
        else config
    )
    with measure_phase(
        "preparation_load"
        if prepared is not None
        else ("reuse" if config.reuse_dir is not None else "tuning")
    ):
        start = perf_counter()
        effective, trials = (
            (
                replace(
                    config,
                    parameters=prepared.parameters,
                    refit_trees=prepared.refit_trees,
                ),
                prepared.trials,
            )
            if prepared is not None
            else (
                (
                    reuse_predictor(config),
                    pd.DataFrame(columns=["number", "state", "value"]),
                )
                if config.reuse_dir is not None
                else tune_predictor(tuning_config, split.training)
            )
        )
        tuning_seconds = (
            prepared.tuning_seconds if prepared is not None else perf_counter() - start
        )
    provenance: list[dict[str, JsonValue]] = []
    fold_models: tuple[Pipeline, ...] = ()
    with measure_phase(
        "preparation_load"
        if prepared is not None
        else ("reuse" if config.reuse_dir is not None else "crossfit")
    ):
        crossfit_start = perf_counter()
        if prepared is not None:
            oof, cv_score, provenance = (
                prepared.oof,
                prepared.score,
                prepared.provenance,
            )
        elif config.reuse_dir is not None:
            reference = read_run(config.reuse_dir)
            oof = pd.read_csv(config.reuse_dir / OOF_FILE)
            from truss_research.surrogates.pliegues import eligible_indices

            eligible = split.training.iloc[eligible_indices(config, split.training)]
            oof.index = eligible.index
            if oof["iter_id"].tolist() != eligible["iter_id"].tolist():
                raise ValueError("Las predicciones compartidas no están alineadas")
            cv_score = float(str(reference["cv_normalized_mae"]))
        elif config.nested_folds is not None:
            from truss_research.surrogates.validacion import nested_cross_fitted

            nested = nested_cross_fitted(config, split.training)
            fold_models = nested.models
            oof, cv_score, provenance = (
                nested.predictions,
                nested.score,
                nested.provenance,
            )
        else:
            oof, cv_score = cross_fitted_predictions(
                replace(effective, folds=config.folds), split.training
            )
            fold_models = oof.attrs.pop("fold_models", ())
        crossfit_seconds = (
            prepared.crossfit_seconds
            if prepared is not None
            else perf_counter() - crossfit_start
        )
    model = build_predictor(effective)
    with measure_phase("fit"), threadpool_limits(limits=1):
        start = perf_counter()
        model.fit(
            split.training.loc[:, list(INPUT_COLUMNS)],
            split.training.loc[:, list(OUTPUT_COLUMNS)],
        )
        fit_seconds = perf_counter() - start
    with measure_phase("prediction"):
        prediction_start = perf_counter()
        predicted = predict(model, split.validation.loc[:, list(INPUT_COLUMNS)])
        prediction_seconds = perf_counter() - prediction_start
    metrics = regression_metrics(
        split.validation.loc[:, list(OUTPUT_COLUMNS)], predicted
    )
    metrics = metrics.assign(model=config.model, dataset_signature=signature)
    score = normalized_mae(
        split.validation.loc[:, list(OUTPUT_COLUMNS)].to_numpy(),
        predicted.to_numpy(),
        response_scale(split.training.loc[:, list(OUTPUT_COLUMNS)]),
    )
    grouped = [
        regression_metrics(
            group.loc[:, list(OUTPUT_COLUMNS)], predicted.loc[group.index]
        ).assign(web_config=key)
        for key, group in split.validation.groupby("web_config")
    ]
    timing = timing_benchmark(model, split.validation.loc[:, list(INPUT_COLUMNS)])
    if signature != data_signature(config.data_dir):
        raise ValueError("Los datos cambiaron durante el experimento")
    config.output_dir.mkdir(parents=True)
    save_model(model, config.output_dir)
    member_sources = (
        prepared.models
        if prepared is not None
        else tuple(sorted(config.reuse_dir.glob("fold_[0-9]*")))
        if config.reuse_dir is not None
        else ()
    )
    for number, source in enumerate(member_sources):
        copy_model(source, config.output_dir / f"fold_{number}")
    for number, member in enumerate(fold_models):
        save_model(member, config.output_dir / f"fold_{number}")
    if split.calibration is not None:
        from truss_research.surrogates.intervalos import calibrate_predictor

        with measure_phase("calibration"):
            calibrate_predictor(model, split, config.output_dir)
    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    report = prediction_report(split.validation, predicted)
    report.to_csv(config.output_dir / "validation_predictions.csv", index=False)
    pd.concat(grouped, ignore_index=True).to_csv(
        config.output_dir / "category_metrics.csv", index=False
    )
    regional_metrics(split.training, split.validation, predicted).to_csv(
        config.output_dir / "region_metrics.csv", index=False
    )
    fold_metrics = crossfit_metrics(split.training, oof)
    fold_metrics.to_csv(config.output_dir / "cv_fold_metrics.csv", index=False)
    write_json(
        config.output_dir / "compliance.json",
        compliance_report(
            split.validation.loc[:, list(OUTPUT_COLUMNS)].to_numpy(),
            predicted.to_numpy(),
            load_structural_limits(config.data_dir),
        ),
    )
    oof.to_csv(config.output_dir / OOF_FILE, index=False)
    write_json(
        config.output_dir / "oof_coverage.json",
        {
            "eligible_ids": oof.iter_id.tolist(),
            "excluded_ids": split.training.loc[
                ~split.training.iter_id.isin(oof.iter_id), "iter_id"
            ].tolist(),
            "reason": "progressive_warmup_or_purge",
        },
    )
    if provenance:
        write_json(config.output_dir / "nested_folds.json", {"folds": provenance})
    trials.to_csv(config.output_dir / "trials.csv", index=False)
    from truss_research.evaluacion.costos import machine_context

    component_selection = getattr(
        model.named_steps["regressor"], "selection_seconds_", 0.0
    )
    fit_seconds = max(0.0, fit_seconds - component_selection)
    tuning_seconds += component_selection

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            **asdict(config),
            "parameters": asdict(effective.parameters),
            "refit_trees": effective.refit_trees,
            "data_dir": str(config.data_dir.resolve()),
            "output_dir": str(config.output_dir.resolve()),
        },
        "early_stopping": stopping_records(model),
        "requested_trials": config.trials,
        "search_reused": config.reuse_dir is not None,
        "cv_reused": config.reuse_dir is not None,
        "convergence": convergence_records(model),
        "cv_fold_warnings": oof.attrs.get("fold_warnings", []),
        "tuning_fold_warnings": [
            value if isinstance(value, list) else []
            for value in trials["user_attrs_fold_warnings"]
        ]
        if "user_attrs_fold_warnings" in trials
        else [],
        "dataset_signature": signature,
        "model_signature": source_digest(config.output_dir / MODEL_FILE),
        "model_artifacts": model_signatures(config.output_dir),
        "oof_signature": source_digest(config.output_dir / OOF_FILE),
        "oof_protocol": "nested"
        if config.nested_folds is not None
        else "selected_parameters_oof",
        "input_columns": list(INPUT_COLUMNS),
        "output_columns": list(OUTPUT_COLUMNS),
        "evaluation_split": "validation",
        "fit_rows": len(split.training),
        "targetwise_selection": getattr(
            model.named_steps["regressor"], "provenance_", []
        ),
        "targetwise_family_search": getattr(
            model.named_steps["regressor"], "family_search_", []
        ),
        "targetwise_family_trials": getattr(
            model.named_steps["regressor"], "family_trials_", 0
        ),
        "fit_ids": split.training.iter_id.tolist(),
        "calibration_ids": split.calibration.iter_id.tolist()
        if split.calibration is not None
        else [],
        "fold_model_signatures": {
            path.name: model_signatures(path)
            for path in sorted(config.output_dir.glob("fold_[0-9]*"))
        },
        "normalized_mae": score,
        "cv_normalized_mae": cv_score,
        "cv_fold_normalized_mae": fold_metrics.groupby("fold")["normalized_mae"]
        .first()
        .tolist(),
        "response_scale": response_scale(
            split.training.loc[:, list(OUTPUT_COLUMNS)]
        ).tolist(),
        "constant_training_targets": split.training.loc[:, list(OUTPUT_COLUMNS)]
        .nunique()
        .eq(1)
        .tolist(),
        "fit_seconds": fit_seconds,
        "crossfit_seconds": crossfit_seconds if config.reuse_dir is None else 0.0,
        "reuse_seconds": tuning_seconds + crossfit_seconds
        if config.reuse_dir is not None
        else 0.0,
        "prediction_seconds": prediction_seconds,
        "trial_count": len(trials)
        + int(getattr(model.named_steps["regressor"], "family_trials_", 0))
        + sum(
            int(row["trial_count"])
            for row in getattr(model.named_steps["regressor"], "provenance_", [])
        ),
        "nested_trial_count": sum(
            int(str(fold["trial_count"]))
            + int(fold.get("targetwise_family_trials", 0))
            + sum(
                int(row["trial_count"]) for row in fold.get("targetwise_selection", [])
            )
            for fold in provenance
        ),
        "tuning_seconds": tuning_seconds if config.reuse_dir is None else 0.0,
        "timing": timing,
        "timing_repeats": TIMING_REPEATS,
        "device": "cpu",
        "threads": 1,
        "machine": machine_context(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "optuna",
                "pymoo",
                "shap",
                "catboost",
                "xgboost",
                "lightgbm",
                "joblib",
                "threadpoolctl",
            )
        },
    }
    write_json(config.output_dir / "run.json", metadata)
    if split.calibration is not None:
        from truss_research.surrogates.intervalos import export_intervals

        export_intervals(
            (config.output_dir, config.output_dir), split.validation, predicted
        )
    logger.info("Predictor %s: MAE normalizado de validación %.6f", config.model, score)
    return PredictorResult(model, metrics, report, config.output_dir, score)
