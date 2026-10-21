"""Evaluación por respuesta sin mezclar unidades físicas en una única métrica."""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS

METRICS_FILENAME: Final[str] = "metrics.csv"

METRIC_COLUMNS: Final[tuple[str, ...]] = ("mae", "rmse", "r2")

EXTENDED_METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "bias",
    "p95_absolute_error",
    "count",
)


def regression_metrics(observed: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    if (
        tuple(observed.columns) != OUTPUT_COLUMNS
        or tuple(predicted.columns) != OUTPUT_COLUMNS
    ):
        raise ValueError("Las respuestas deben respetar el orden de OUTPUT_COLUMNS")
    if len(observed) < 1 or not observed.index.equals(predicted.index):
        raise ValueError("Se requieren observaciones con índices alineados")
    if (
        not np.isfinite(observed.to_numpy()).all()
        or not np.isfinite(predicted.to_numpy()).all()
    ):
        raise ValueError("Las respuestas observadas y predichas deben ser finitas")
    rows: list[dict[str, str | float | None]] = []
    for target in OUTPUT_COLUMNS:
        rows.append(
            {
                "target": target,
                "mae": float(mean_absolute_error(observed[target], predicted[target])),
                "rmse": float(
                    root_mean_squared_error(observed[target], predicted[target])
                ),
                "r2": float(r2_score(observed[target], predicted[target]))
                if observed[target].nunique() > 1
                else None,
                "bias": float((predicted[target] - observed[target]).mean()),
                "p95_absolute_error": float(
                    (predicted[target] - observed[target]).abs().quantile(0.95)
                ),
                "count": len(observed),
            }
        )
    return pd.DataFrame(rows).astype({"r2": "float64"})


def load_comparison(run_directories: Sequence[Path]) -> pd.DataFrame:
    """Carga resultados compatibles sin entrenar ni abrir modelos serializados."""
    if not run_directories:
        raise ValueError("Indicar al menos una ejecución para comparar")
    tables: list[pd.DataFrame] = []
    required = {"target", "model", "dataset_signature", *METRIC_COLUMNS}
    for directory in run_directories:
        table = pd.read_csv(directory / METRICS_FILENAME)
        if (
            not required.issubset(table.columns)
            or tuple(table["target"]) != OUTPUT_COLUMNS
        ):
            raise ValueError(f"Métricas incompletas o incompatibles en {directory}")
        if table["dataset_signature"].isna().any():
            raise ValueError(f"Falta identificación del dataset en {directory}")
        table["run"] = directory.name
        for column in EXTENDED_METRIC_COLUMNS:
            table[column] = table[column] if column in table else np.nan
        table["evaluation_split"] = (
            table["evaluation_split"] if "evaluation_split" in table else "validation"
        )
        tables.append(table)
    comparison = pd.concat(tables, ignore_index=True)
    if comparison["dataset_signature"].nunique() != 1:
        raise ValueError("No comparar ejecuciones con datos o particiones diferentes")
    if comparison["evaluation_split"].nunique() != 1:
        raise ValueError("No mezclar resultados de validación y prueba")
    return comparison


REGION_QUANTILES = (0.25, 0.5, 0.75)

BOOTSTRAP_REPEATS = 2_000

BOOTSTRAP_SEED = 42

BOOTSTRAP_QUANTILES = (0.025, 0.975)


def paired_prediction_errors(
    table: pd.DataFrame, scales: npt.NDArray[np.float64]
) -> pd.DataFrame:
    """Remuestrea diseños pareados; promedia pérdidas entre semillas sin multiplicar n."""
    rows: list[dict[str, str | int | float]] = []
    columns = [
        "split",
        "model",
        "reference_model",
        "target",
        "count",
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "bootstrap_repeats",
    ]
    for split, group in table.groupby("split"):
        losses = group.loc[:, ["model", "seed", "iter_id"]].copy()
        error = np.abs(
            group.loc[:, list(OUTPUT_COLUMNS)].to_numpy()
            - group.loc[:, [f"pred_{name}" for name in OUTPUT_COLUMNS]].to_numpy()
        )
        losses.loc[:, list(OUTPUT_COLUMNS)] = error
        losses["normalized_mae"] = (error / scales).mean(axis=1)
        averaged = losses.groupby(["iter_id", "model"])[
            [*OUTPUT_COLUMNS, "normalized_mae"]
        ].mean()
        for target in (*OUTPUT_COLUMNS, "normalized_mae"):
            matrix = averaged[target].unstack("model")
            if matrix.isna().any().any():
                raise ValueError("Los modelos deben evaluarse sobre los mismos diseños")
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            indices = rng.integers(
                0, len(matrix), size=(BOOTSTRAP_REPEATS, len(matrix))
            )
            for reference in (name for name in ("dummy", "ridge") if name in matrix):
                for model in matrix.columns.drop(reference):
                    difference = (matrix[model] - matrix[reference]).to_numpy()
                    lower, upper = np.quantile(
                        difference[indices].mean(axis=1), BOOTSTRAP_QUANTILES
                    )
                    rows.append(
                        {
                            "split": split,
                            "model": model,
                            "reference_model": reference,
                            "target": target,
                            "count": len(matrix),
                            "mean_difference": float(difference.mean()),
                            "ci_lower": float(lower),
                            "ci_upper": float(upper),
                            "bootstrap_repeats": BOOTSTRAP_REPEATS,
                        }
                    )
    return pd.DataFrame(rows, columns=columns)


def regional_metrics(
    training: pd.DataFrame, validation: pd.DataFrame, predicted: pd.DataFrame
) -> pd.DataFrame:
    reports: list[pd.DataFrame] = []
    for variable in (name for name in INPUT_COLUMNS if name != "web_config"):
        internal = np.unique(training[variable].quantile(REGION_QUANTILES).to_numpy())
        boundaries = np.r_[-np.inf, internal, np.inf]
        regions = pd.cut(validation[variable], bins=boundaries, labels=False)
        for region, group in validation.groupby(regions):
            metrics = regression_metrics(
                group.loc[:, list(OUTPUT_COLUMNS)], predicted.loc[group.index]
            )
            reports.append(
                metrics.assign(
                    variable=variable,
                    region=int(region),
                    lower_bound=boundaries[int(region)],
                    upper_bound=boundaries[int(region) + 1],
                )
            )
    return pd.concat(reports, ignore_index=True)


def crossfit_metrics(training: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    training = training.set_index("iter_id", drop=False)
    oof = oof.set_index("iter_id", drop=False)
    reports: list[pd.DataFrame] = []
    for fold, held in oof.groupby("fold"):
        predicted = held.loc[:, [f"pred_{name}" for name in OUTPUT_COLUMNS]].copy()
        predicted.columns = OUTPUT_COLUMNS
        fitting = training.drop(index=held.index).loc[:, list(OUTPUT_COLUMNS)]
        if "first_generation" in training:
            fitting = training.loc[
                training.generation.lt(training.loc[held.index, "generation"].min()),
                list(OUTPUT_COLUMNS),
            ]
        scale = fitting.std(ddof=0).to_numpy()
        scale = np.where(scale > 0, scale, 1.0)
        observed = training.loc[held.index, list(OUTPUT_COLUMNS)]
        predicted.index = observed.index
        score = float(
            np.mean(np.abs(observed.to_numpy() - predicted.to_numpy()) / scale)
        )
        reports.append(
            regression_metrics(observed, predicted).assign(
                fold=int(fold), normalized_mae=score
            )
        )
    return pd.concat(reports, ignore_index=True)
