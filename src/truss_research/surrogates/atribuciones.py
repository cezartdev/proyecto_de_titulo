"""Atribuciones analíticas y de árboles agrupadas en las siete entradas originales."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.surrogates.configuracion import FloatArray


def grouped_values(values: FloatArray, transformer: ColumnTransformer) -> FloatArray:
    """Reagrupa las categorías one-hot sin asignar ordinalidad a la topología."""
    grouped = np.zeros((len(values), len(INPUT_COLUMNS), values.shape[-1]))
    grouped[:, INPUT_COLUMNS.index("web_config"), :] = values[
        :, transformer.output_indices_["web_config"], :
    ].sum(axis=1)
    numeric = [name for name in INPUT_COLUMNS if name != "web_config"]
    for column, name in enumerate(numeric):
        grouped[:, INPUT_COLUMNS.index(name), :] = values[
            :, transformer.output_indices_["numeric"], :
        ][:, column, :]
    return grouped


def linear_values(
    model: Pipeline, data: tuple[pd.DataFrame, pd.DataFrame]
) -> tuple[FloatArray, FloatArray]:
    """SHAP lineal independiente en el espacio transformado y unidades de respuesta."""
    import shap

    transformer = model.named_steps["preprocessing"]
    regressor = model.named_steps["regressor"]
    estimators = (
        regressor.estimators_
        if isinstance(regressor, MultiOutputRegressor)
        else [regressor]
    )
    background, evaluation = (transformer.transform(table) for table in data)
    explanations = [
        shap.LinearExplainer(
            estimator, shap.maskers.Independent(background, max_samples=len(background))
        )(evaluation)
        for estimator in estimators
    ]
    values = np.stack([np.asarray(item.values) for item in explanations], axis=-1)
    bases = np.column_stack(
        [np.broadcast_to(item.base_values, (len(evaluation),)) for item in explanations]
    )
    return grouped_values(values, transformer), bases


def tree_values(
    model: Pipeline, data: tuple[pd.DataFrame, pd.DataFrame]
) -> tuple[FloatArray, FloatArray]:
    """Explica árboles multirespuesta o componentes escalares de Targetwise."""
    import shap

    background, evaluation = data
    regressor = model.named_steps["regressor"]
    if "preprocessing" not in model.named_steps:
        from truss_research.surrogates.modelos import categorical_features

        table = categorical_features(evaluation, regressor.config.model)
        estimators = getattr(regressor, "estimators_", None)
        estimators = estimators if estimators is not None else [regressor.estimator_]
        explainers = [
            shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
            for estimator in estimators
        ]
        return (
            np.stack(
                [
                    np.asarray(item.shap_values(table, check_additivity=False))
                    for item in explainers
                ],
                axis=-1,
            ),
            np.column_stack(
                [
                    np.full(
                        len(table), float(np.asarray(item.expected_value).squeeze())
                    )
                    for item in explainers
                ]
            ),
        )
    transformer = model.named_steps["preprocessing"]
    encoded_background, encoded = (transformer.transform(table) for table in data)
    estimators = (
        regressor.estimators_
        if isinstance(regressor, MultiOutputRegressor)
        else [regressor]
    )
    explainers = [
        shap.TreeExplainer(
            estimator,
            data=shap.maskers.Independent(
                encoded_background, max_samples=len(encoded_background)
            ),
            feature_perturbation="interventional",
        )
        for estimator in estimators
    ]
    values = np.stack(
        [
            np.asarray(item.shap_values(encoded, check_additivity=False))
            for item in explainers
        ],
        axis=-1,
    )
    bases = np.column_stack(
        [
            np.full(len(encoded), float(np.asarray(item.expected_value).squeeze()))
            for item in explainers
        ]
    )
    return grouped_values(values, transformer), bases
