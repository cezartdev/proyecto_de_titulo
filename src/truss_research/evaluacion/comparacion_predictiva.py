"""Comparación pareada exploratoria, condicionada a una trayectoria y ajustes fijos."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.particion import IndexArray
from truss_research.datos.procesados import JsonValue, load_design_split
from truss_research.evaluacion.busqueda_evolutiva import predictor_records
from truss_research.evaluacion.metricas import regression_metrics
from truss_research.evaluacion.robustez import (
    export_error_diagnostics,
    export_evidence_context,
)
from truss_research.optimizacion.contratos import AnalysisParameters
from truss_research.optimizacion.estudio_estado import load_study_config, verify_task
from truss_research.surrogates.configuracion import FloatArray, response_scale
from truss_research.surrogates.experimentos import read_run, write_json

INTERVAL_QUANTILES = (0.025, 0.975)
PRIMARY_CONTROLS = ("dummy", "ridge")
NORMALIZED_TARGET = "normalized_mae"
FEW_BLOCKS_THRESHOLD = 10


def paired_losses(table: pd.DataFrame, scales: FloatArray) -> pd.DataFrame:
    """Promedia pérdidas entre semillas; cada diseño aporta una observación pareada."""
    keys = ["model", "seed", "iter_id"]
    if table.empty or table.duplicated(keys).any():
        raise ValueError(
            "Predicciones vacías o duplicadas por modelo, semilla y diseño"
        )
    if (
        np.shape(scales) != (len(OUTPUT_COLUMNS),)
        or not np.isfinite(scales).all()
        or (scales <= 0).any()
    ):
        raise ValueError("Las escalas deben ser tres valores finitos positivos")
    columns = [*OUTPUT_COLUMNS, *[f"pred_{name}" for name in OUTPUT_COLUMNS]]
    if not np.isfinite(table[columns].to_numpy()).all():
        raise ValueError("Las predicciones y observaciones deben ser finitas")
    metadata = ["generation", "web_config", *OUTPUT_COLUMNS]
    if table.groupby("iter_id")[metadata].nunique().gt(1).any().any():
        raise ValueError("Las observaciones o su procedencia difieren entre modelos")
    identities = set(table.iter_id)
    seed_sets: list[frozenset[int]] = []
    for _, family in table.groupby("model"):
        seed_sets.append(frozenset(family.seed))
        if any(set(group.iter_id) != identities for _, group in family.groupby("seed")):
            raise ValueError("Cada modelo y semilla debe cubrir los mismos diseños")
    if len(set(seed_sets)) != 1:
        raise ValueError("Las familias requieren las mismas semillas")
    losses = table[keys + ["generation", "web_config"]].copy()
    errors = np.abs(
        table[list(OUTPUT_COLUMNS)].to_numpy()
        - table[[f"pred_{name}" for name in OUTPUT_COLUMNS]].to_numpy()
    )
    losses[list(OUTPUT_COLUMNS)] = errors
    losses[NORMALIZED_TARGET] = (errors / scales).mean(axis=1)
    return losses.groupby(
        ["iter_id", "model", "generation", "web_config"], as_index=False
    )[[*OUTPUT_COLUMNS, NORMALIZED_TARGET]].mean()


def bootstrap_means(
    values: FloatArray, groups: IndexArray, settings: AnalysisParameters
) -> FloatArray:
    """Remuestrea bloques completos y pondera por sus tamaños, sin contar semillas."""
    _, labels = np.unique(groups, return_inverse=True)
    count = int(labels.max()) + 1
    totals = np.zeros((count, values.shape[1]))
    np.add.at(totals, labels, values)
    sizes = np.bincount(labels)
    rng = np.random.default_rng(settings.bootstrap_seed)
    weights = rng.multinomial(
        count, np.full(count, 1 / count), size=settings.bootstrap_repeats
    )
    return (weights @ totals) / (weights @ sizes)[:, None]


def predictive_contrasts(
    losses: pd.DataFrame, settings: AnalysisParameters
) -> pd.DataFrame:
    records: list[dict[str, JsonValue]] = []
    metadata = losses.drop_duplicates("iter_id").set_index("iter_id")
    for target in (*OUTPUT_COLUMNS, NORMALIZED_TARGET):
        matrix = losses.pivot(
            index="iter_id", columns="model", values=target
        ).sort_index()
        if matrix.isna().any().any():
            raise ValueError("Faltan pérdidas pareadas de algún modelo")
        generations = metadata.loc[matrix.index, "generation"].to_numpy(dtype=int)
        groups = {
            "generation_block": (generations - generations.min())
            // settings.predictive_block_generations,
            "design_iid_sensitivity": np.arange(len(matrix)),
        }
        for method, labels in groups.items():
            blocks = len(np.unique(labels))
            sampled = bootstrap_means(matrix.to_numpy(), labels, settings)
            for left, right in combinations(matrix.columns, 2):
                reference, model = (
                    (left, right) if left in PRIMARY_CONTROLS else (right, left)
                )
                difference = (
                    sampled[:, matrix.columns.get_loc(model)]
                    - sampled[:, matrix.columns.get_loc(reference)]
                )
                lower, upper = np.quantile(difference, INTERVAL_QUANTILES)
                records.append(
                    {
                        "model": model,
                        "reference_model": reference,
                        "target": target,
                        "contrast": "primary_control"
                        if reference in PRIMARY_CONTROLS
                        else "secondary_family",
                        "resampling": method,
                        "count": len(matrix),
                        "blocks": blocks,
                        "block_generations": settings.predictive_block_generations
                        if method == "generation_block"
                        else None,
                        "mean_difference": float(
                            (matrix[model] - matrix[reference]).mean()
                        ),
                        "ci_lower": float(lower) if blocks > 1 else None,
                        "ci_upper": float(upper) if blocks > 1 else None,
                        "bootstrap_repeats": settings.bootstrap_repeats,
                        "interval_status": "insufficient_blocks"
                        if blocks < 2
                        else "exploratory_few_blocks"
                        if blocks < FEW_BLOCKS_THRESHOLD
                        else "exploratory",
                        "scope": "pointwise_conditional_single_trajectory_no_multiplicity_adjustment",
                    }
                )
    return pd.DataFrame(records)


def seed_metrics(table: pd.DataFrame, scales: FloatArray) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for (split, model, seed), group in table.groupby(["split", "model", "seed"]):
        observed = group[list(OUTPUT_COLUMNS)]
        predicted = group[[f"pred_{name}" for name in OUTPUT_COLUMNS]].copy()
        predicted.columns = OUTPUT_COLUMNS
        metrics = regression_metrics(observed, predicted)
        metrics["normalized_mae"] = metrics.mae.to_numpy() / scales
        records.append(metrics.assign(split=split, model=model, seed=seed))
    return pd.concat(records, ignore_index=True)


def export_predictive_comparison(study: Path, destination: Path) -> None:
    """Lee predicciones firmadas; no entrena, selecciona ni modifica el estudio."""
    config = load_study_config(study)
    split = load_design_split(config.data_dir)
    scales = response_scale(split.training[list(OUTPUT_COLUMNS)])
    frames: list[pd.DataFrame] = []
    intervals: list[pd.DataFrame] = []
    categories: list[pd.DataFrame] = []
    regions: list[pd.DataFrame] = []
    inference: list[dict[str, JsonValue]] = []
    for reference in predictor_records(study):
        source = verify_task(reference.record)
        roots = {
            "validation": source,
            "test": study
            / "evolution"
            / "test_predictors"
            / f"{reference.model}_{reference.seed}",
        }
        for name, root in roots.items():
            categories.append(
                pd.read_csv(root / "category_metrics.csv").assign(
                    split=name, model=reference.model, seed=reference.seed
                )
            )
            regions.append(
                pd.read_csv(root / "region_metrics.csv").assign(
                    split=name, model=reference.model, seed=reference.seed
                )
            )
            metadata = read_run(root)
            inference.append(
                {
                    "split": name,
                    "model": reference.model,
                    "seed": reference.seed,
                    "prediction_seconds": metadata["prediction_seconds"],
                    **metadata["timing"],
                }
            )
            frames.append(
                pd.read_csv(root / f"{name}_predictions.csv").assign(
                    split=name, model=reference.model, seed=reference.seed
                )
            )
            if (root / "prediction_intervals.csv").is_file():
                intervals.append(
                    pd.read_csv(root / "prediction_intervals.csv").assign(
                        split=name, model=reference.model, seed=reference.seed
                    )
                )
    table = pd.concat(frames, ignore_index=True)
    pd.concat(categories, ignore_index=True).to_csv(
        destination / "predictive_category_metrics.csv", index=False
    )
    pd.concat(regions, ignore_index=True).to_csv(
        destination / "predictive_region_metrics.csv", index=False
    )
    pd.DataFrame(inference).to_csv(
        destination / "predictive_inference_costs.csv", index=False
    )
    export_prediction_tables(table, (scales, config.analysis), destination)
    if intervals:
        pd.concat(intervals).groupby(
            ["split", "model", "seed", "target", "nominal_coverage", "web_config"]
        ).agg(
            observed_coverage=("covered", "mean"),
            mean_width=("width", "mean"),
            count=("covered", "size"),
        ).to_csv(destination / "predictive_interval_coverage.csv")
    export_evidence_context(split, destination)


def export_prediction_tables(
    table: pd.DataFrame,
    context: tuple[FloatArray, AnalysisParameters],
    destination: Path,
) -> None:
    scales, settings = context
    contrasts: list[pd.DataFrame] = []
    for name, group in table.groupby("split"):
        losses = paired_losses(group, scales)
        contrasts.append(predictive_contrasts(losses, settings).assign(split=name))
        losses.groupby(["model", "generation"])[
            [*OUTPUT_COLUMNS, NORMALIZED_TARGET]
        ].agg(["mean", "count"]).to_csv(
            destination / f"predictive_{name}_generation_errors.csv"
        )
    pd.concat(contrasts, ignore_index=True).to_csv(
        destination / "predictive_paired_errors.csv", index=False
    )
    metrics = seed_metrics(table, scales)
    metrics.to_csv(destination / "predictive_seed_metrics.csv", index=False)
    criterion = metrics.groupby(["split", "model", "seed"])["normalized_mae"].mean()
    criterion.groupby(["split", "model"]).agg(
        ["mean", "std", "min", "max", "count"]
    ).rename(columns={"count": "adjustment_seeds"}).to_csv(
        destination / "predictive_model_summary.csv"
    )
    metrics.groupby(["split", "model", "target"])[
        ["mae", "rmse", "r2", "bias", "p95_absolute_error", "normalized_mae"]
    ].agg(["mean", "std", "min", "max"]).to_csv(
        destination / "predictive_metric_summary.csv"
    )
    table.groupby(["split", "model", "iter_id"])[
        [f"pred_{name}" for name in OUTPUT_COLUMNS]
    ].agg(["mean", "std", "min", "max", "count"]).to_csv(
        destination / "predictive_seed_dispersion.csv"
    )
    export_error_diagnostics(table, destination)
    write_json(
        destination / "predictive_protocol.json",
        {
            "primary_component": "predictor_comparison",
            "primary_controls": list(PRIMARY_CONTROLS),
            "response_scales_training_std": scales.tolist(),
            "response_weights": [1 / len(OUTPUT_COLUMNS)] * len(OUTPUT_COLUMNS),
            "block_generations": settings.predictive_block_generations,
            "bootstrap_repeats": settings.bootstrap_repeats,
            "bootstrap_seed": settings.bootstrap_seed,
            "interval_coverage": 0.95,
            "significance_tests": False,
            "equivalence_claim": False,
            "scope": "exploratory_conditional_fixed_fits_single_previously_examined_trajectory",
            "seed_dispersion_scope": "final_fits_not_independent_structural_problems",
        },
    )
