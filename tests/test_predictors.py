"""Predictores, cross-fitting, persistencia y explicaciones sin contaminación."""

from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from test_tabpfn import fake_tabpfn as fake_tabpfn

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.particion import training_folds
from truss_research.datos.procesados import load_design_split
from truss_research.surrogates.configuracion import (
    MODEL_NAMES,
    ModelName,
    ModelParameters,
    PredictorConfig,
)
from truss_research.surrogates.experimentos import (
    load_predictor,
    predict,
    read_run,
    train_predictor,
)
from truss_research.surrogates.explicabilidad import ExplanationConfig, explain_model
from truss_research.surrogates.modelos import build_predictor


def nonlinear_responses(features: pd.DataFrame) -> pd.DataFrame:
    """Respuestas suaves distintas para probar persistencia sin degeneración lineal.

    La tabla general sirve para partición y cross-fitting, pero sus tres respuestas
    son proporcionales y perfectamente lineales. Aquí cada salida depende de la
    altura y de la categoría, conservando las unidades y valores positivos.
    """
    height = features["lr_height_m"].to_numpy()
    category = features["web_config"].to_numpy()
    return pd.DataFrame(
        {
            "mass_kg": 1000.0 + 100.0 * np.sin(2.0 * height) + 10.0 * category,
            "util_ratio": 0.5 + 0.1 * np.cos(3.0 * height) + 0.01 * category,
            "z_disp_m": 0.03 + 0.005 * np.sin(4.0 * height) + 0.001 * category,
        },
        index=features.index,
    ).loc[:, list(OUTPUT_COLUMNS)]


@pytest.mark.usefixtures("fake_tabpfn")
@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.filterwarnings("error::sklearn.exceptions.ConvergenceWarning")
def test_all_model_families_round_trip(
    name: ModelName, design_data: Path, tmp_path: Path
) -> None:
    split = load_design_split(design_data)
    parameters = ModelParameters(
        trees=3, depth=2, epochs=3, patience=2, hidden_width=8, regularization=0.0001
    )
    model = build_predictor(PredictorConfig(model=name, parameters=parameters))
    features = split.training.loc[:, list(INPUT_COLUMNS)]
    model.fit(features, nonlinear_responses(features))
    expected = predict(model, split.validation.loc[:, list(INPUT_COLUMNS)])
    assert expected.shape == (10, 3)
    assert np.isfinite(expected.to_numpy()).all()
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(
        predict(joblib.load(path), split.validation.loc[:, list(INPUT_COLUMNS)]),
        expected,
    )


def test_oof_dummy_excludes_its_fold_and_validation(
    design_data: Path, tmp_path: Path
) -> None:
    config = PredictorConfig(design_data, tmp_path / "run", model="dummy", folds=2)
    result = train_predictor(config)
    split = load_design_split(design_data)
    oof = pd.read_csv(result.output_dir / "training_oof.csv").set_index("iter_id")
    for number, (fit_indices, held_indices) in enumerate(
        training_folds(split.training, 2, config.seed)
    ):
        means = (
            split.training.iloc[fit_indices]
            .loc[:, list(OUTPUT_COLUMNS)]
            .mean()
            .to_numpy()
        )
        held_ids = split.training.iloc[held_indices]["iter_id"]
        predictions = oof.loc[held_ids, [f"pred_{name}" for name in OUTPUT_COLUMNS]]
        np.testing.assert_allclose(predictions, np.tile(means, (len(held_ids), 1)))
        assert oof.loc[held_ids, "fold"].eq(number).all()
    assert set(oof.index).isdisjoint(split.testing["iter_id"])
    assert {"bias", "p95_absolute_error", "count"} <= set(result.metrics)
    assert read_run(result.output_dir)["evaluation_split"] == "validation"
    with pytest.raises(ValueError, match="ya existe"):
        train_predictor(config)
    (result.output_dir / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="firma"):
        load_predictor(result.output_dir)


@pytest.mark.parametrize(
    "name", ["dummy", "random_forest", "catboost", "xgboost", "lightgbm"]
)
def test_shap_reconstructs_original_predictions(
    name: ModelName, design_data: Path, tmp_path: Path
) -> None:
    run = tmp_path / "predictor"
    train_predictor(
        PredictorConfig(
            design_data, run, model=name, folds=2, parameters=ModelParameters(trees=3)
        )
    )
    output = tmp_path / "shap"
    values = explain_model(
        ExplanationConfig(
            design_data, run, output, background_size=4, permutation_rounds=1
        )
    )
    assert set(values["variable"]) == set(INPUT_COLUMNS)
    assert set(values["target"]) == set(OUTPUT_COLUMNS)
    training_ids = set(load_design_split(design_data).training["iter_id"])
    assert set(read_run(output)["background_ids"]) <= training_ids


def test_optuna_records_effective_parameters(design_data: Path, tmp_path: Path) -> None:
    result = train_predictor(
        PredictorConfig(design_data, tmp_path / "tuned", model="svr", folds=2, trials=1)
    )
    assert len(pd.read_csv(result.output_dir / "trials.csv")) == 1
    assert read_run(result.output_dir)["requested_trials"] == 1


def test_ridge_search_is_reused_without_tuning_or_crossfit(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.surrogates.experimentos as experiments
    import truss_research.surrogates.validacion as validation

    config = PredictorConfig(
        design_data,
        tmp_path / "representative",
        model="ridge",
        trials=1,
        folds=2,
        nested_folds=2,
        cv_seed=42,
    )
    first = train_predictor(config)
    metadata = read_run(first.output_dir)
    assert metadata["trial_count"] == 1 and metadata["nested_trial_count"] == 2
    assert 1e-6 <= metadata["config"]["parameters"]["regularization"] <= 1e3

    def forbidden(*args: PredictorConfig | pd.DataFrame) -> None:
        raise AssertionError("Las semillas reutilizan la búsqueda y las OOF")

    monkeypatch.setattr(experiments, "tune_predictor", forbidden)
    monkeypatch.setattr(validation, "nested_cross_fitted", forbidden)
    repeated = train_predictor(
        replace(
            config, output_dir=tmp_path / "replica", seed=43, reuse_dir=first.output_dir
        )
    )
    second = read_run(repeated.output_dir)
    assert second["search_reused"] and second["cv_reused"]
    assert second["trial_count"] == second["nested_trial_count"] == 0
    assert second["config"]["parameters"] == metadata["config"]["parameters"]
    np.testing.assert_allclose(
        first.predictions[[f"pred_{name}" for name in OUTPUT_COLUMNS]],
        repeated.predictions[[f"pred_{name}" for name in OUTPUT_COLUMNS]],
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(first.output_dir / "training_oof.csv"),
        pd.read_csv(repeated.output_dir / "training_oof.csv"),
    )


def test_bootstrap_uses_designs_and_preserves_pairing() -> None:
    from truss_research.evaluacion.metricas import paired_prediction_errors

    frames = []
    for model, error in (("dummy", 2.0), ("ridge", 1.0)):
        table = pd.DataFrame(
            {
                "iter_id": ["a", "b", "c"],
                "split": "validation",
                "model": model,
                "seed": 42,
            }
        )
        for target in OUTPUT_COLUMNS:
            table[target] = [10.0, 20.0, 30.0]
            table[f"pred_{target}"] = table[target] + error
        frames.append(table)
    table = pd.concat(frames, ignore_index=True)
    first = paired_prediction_errors(table, np.ones(3))
    repeated = paired_prediction_errors(
        pd.concat([table, table.assign(seed=43)]), np.ones(3)
    )
    pd.testing.assert_frame_equal(first, repeated)
    assert first["count"].eq(3).all()
    ridge = first[first["model"].eq("ridge")]
    assert ridge["mean_difference"].eq(-1).all()
    assert ridge["ci_lower"].eq(-1).all() and ridge["ci_upper"].eq(-1).all()
