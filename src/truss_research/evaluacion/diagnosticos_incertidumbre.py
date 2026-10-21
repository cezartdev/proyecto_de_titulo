"""Cobertura descriptiva y errores de factibilidad en diseños retenidos."""

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    WEB_CONFIG_VALUES,
    StructuralLimits,
)
from truss_research.datos.distancias import design_distances
from truss_research.evaluacion.pareto import meets_thresholds
from truss_research.surrogates.calibracion import SplitConformal

DISTANCE_QUANTILES = (0.25, 0.50, 0.75)


def uncertainty_diagnostics(
    training: pd.DataFrame,
    held: tuple[pd.DataFrame, pd.DataFrame],
    calibration: tuple[SplitConformal, StructuralLimits],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibrator, limits = calibration
    table, predicted = held
    features = training.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float)
    scales = tuple(np.ptp(features, axis=0))
    distances = design_distances(
        table.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float), features, scales
    ).min(axis=1)
    neighbors = design_distances(features, features, scales)
    np.fill_diagonal(neighbors, np.inf)
    thresholds = np.quantile(neighbors.min(axis=1), DISTANCE_QUANTILES)
    widths = calibrator.widths(0.90)
    intervals = calibrator.table(predicted, table)
    metadata = table.loc[:, ["iter_id", "generation", "web_config"]].copy()
    metadata["distance"] = distances
    metadata["distance_region"] = np.searchsorted(thresholds, distances, side="right")
    metadata["near_util_limit"] = (
        np.abs(table.util_ratio.to_numpy() - limits.utilization_limit) <= widths[1]
    )
    metadata["near_disp_limit"] = (
        False
        if limits.displacement_limit_m is None
        else np.abs(table.z_disp_m.to_numpy() - limits.displacement_limit_m)
        <= widths[2]
    )
    intervals = intervals.merge(
        metadata, left_on="row_index", right_index=True, validate="many_to_one"
    )
    actual = meets_thresholds(table.loc[:, list(OUTPUT_COLUMNS)].to_numpy(), limits)
    point = meets_thresholds(predicted.to_numpy(), limits)
    upper = predicted.to_numpy().copy()
    upper[:, 1:] += widths[1:]
    safe = meets_thresholds(upper, limits) & point
    feasibility = metadata.assign(
        actual_feasible=actual,
        predicted_feasible=point,
        safe_predicted_feasible=safe,
        false_feasible=point & ~actual,
        false_infeasible=~point & actual,
        safe_false_feasible=safe & ~actual,
    )
    return intervals, feasibility


def summarize_coverage(
    intervals: pd.DataFrame, membership: pd.DataFrame | None = None
) -> pd.DataFrame:
    keys = [
        key
        for key in (
            "evaluation_seed",
            "experiment",
            "model",
            "split",
            "train_size",
            "subset_seed",
        )
        if key in intervals
    ]
    rows = []
    for dimension in (
        None,
        "web_config",
        "distance_region",
        "generation",
        "near_util_limit",
        "near_disp_limit",
    ):
        dimensions = [
            *keys,
            "target",
            "nominal_coverage",
            *([dimension] if dimension else []),
        ]
        observed = intervals
        if dimension == "generation" and membership is not None:
            observed = intervals.drop(columns="generation").merge(
                membership[["representative_iter_id", "generation"]].drop_duplicates(),
                left_on="iter_id",
                right_on="representative_iter_id",
                validate="many_to_many",
            )
        summary = (
            observed.groupby(dimensions, dropna=False)
            .agg(
                count=("covered", "size"),
                coverage=("covered", "mean"),
                mean_width=("width", "mean"),
            )
            .reset_index()
        )
        if dimension == "web_config":
            categories = pd.DataFrame({"web_config": WEB_CONFIG_VALUES})
            expected = (
                intervals[[*keys, "target", "nominal_coverage"]]
                .drop_duplicates()
                .merge(categories, how="cross")
            )
            summary = expected.merge(
                summary, on=dimensions, how="left", validate="one_to_one"
            )
            summary["count"] = summary["count"].fillna(0).astype(int)
        summary["absent"] = summary["count"].eq(0)
        summary["stratum"] = dimension or "global"
        summary["stratum_value"] = (
            summary.pop(dimension).astype(str) if dimension else "all"
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)
